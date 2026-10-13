(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The host *)

external arch : unit -> string = "caml_rig_arch"
external error_message : int -> string = "caml_rig_host_error_message"

let em_x86_64 = 62
let em_aarch64 = 183

(* The host's ELF machine, 0 for an instruction set rig links no code for. *)
let host =
  match arch () with "x86_64" -> em_x86_64 | "arm64" -> em_aarch64 | _ -> 0

let machine_name m = if m = em_x86_64 then "x86_64" else "arm64"

(* Refusals: a link raises [Refused] where it finds the object unfit, and
   returns it as its [Error]. *)

exception Refused of string

let refusef fmt = Printf.ksprintf (fun m -> raise (Refused m)) fmt

(* Errors *)

let err_unsupported k at =
  refusef "relocation of type %d at 0x%x is unsupported" k at

(* The errors by which the system refuses the process executable memory, and
   their remedies. On macOS, mapping MAP_JIT memory fails with EPERM without the
   entitlement; making memory executable fails with EACCES under a policy such
   as SELinux's execmem. Windows reports its own codes. *)
let eperm = 1
let eacces = 13

let err_map size align e =
  let remedy =
    if Sys.win32 || e <> eperm then ""
    else
      "; sign the program with the com.apple.security.cs.allow-jit entitlement"
  in
  refusef
    "mapping %d bytes of executable memory at a multiple of %d bytes: %s%s" size
    align (error_message e) remedy

let err_protect size e =
  let remedy =
    if Sys.win32 || e <> eacces then ""
    else
      "; the system's security policy, such as SELinux's execmem, refuses \
       executable memory to the process"
  in
  refusef "making %d bytes executable: %s%s" size (error_message e) remedy

(* Code memory: a mapping at a multiple of an alignment, which the collector
   unmaps. [base] is its address, or the system's error, negated, if mapping
   failed; [install] is [0] or that error. *)

external map : int -> int -> Def.code = "caml_rig_host_map"
external base : Def.code -> int = "caml_rig_host_base"
external install : Def.code -> bytes -> int = "caml_rig_host_install"
external process_symbol : string -> int = "caml_rig_host_symbol"

(* Objects *)

let et_rel = 1
let shf_write = 0x1
let shf_alloc = 0x2
let shf_execinstr = 0x4

(* Relocations, from the x86-64 psABI and aaelf64. *)

let r_x86_64_pc32 = 2
let r_x86_64_plt32 = 4
let r_aarch64_adr_prel_pg_hi21 = 275
let r_aarch64_add_abs_lo12_nc = 277
let r_aarch64_ldst8_abs_lo12_nc = 278
let r_aarch64_jump26 = 282
let r_aarch64_call26 = 283
let r_aarch64_ldst16_abs_lo12_nc = 284
let r_aarch64_ldst32_abs_lo12_nc = 285
let r_aarch64_ldst64_abs_lo12_nc = 286
let r_aarch64_ldst128_abs_lo12_nc = 299

(* Every relocation patches 4 bytes: an x86_64 displacement or an arm64
   instruction. *)
let field_bytes = 4

(* Fields *)

let fits bits x = -(1 lsl (bits - 1)) <= x && x < 1 lsl (bits - 1)
let page_of a = a land lnot 0xfff
let set32 b at x = Bytes.set_int32_le b at (Int32.of_int x)

(* Replaces the [width] bits from [lo] of the instruction at [at] by [x]'s low
   bits. *)
let set_bits b at ~lo ~width x =
  let insn = Int32.to_int (Bytes.get_int32_le b at) in
  let mask = ((1 lsl width) - 1) lsl lo in
  set32 b at (insn land lnot mask lor ((x lsl lo) land mask))

(* Slots

   A slot follows the image for each symbol the object refers to and does not
   define: a stub at its start that jumps to the word at its end, which holds
   the symbol's address. On x86_64 the stub is [jmp [rip + 2]], whose 6 bytes
   end 2 before the word; on arm64 [ldr x17, #8; br x17]. A branch goes through
   the stub when the symbol is out of its reach. *)

let slot_bytes = 16

(* The symbols [o] refers to and does not define, by name: each one's address
   in the process and its slot's index. *)
let externals (o : Rig_elf.t) =
  let externals = Hashtbl.create 8 in
  let target name =
    match process_symbol name with
    | 0 -> refusef "symbol %S is defined by no library of the process" name
    | a -> a
  in
  let add ({ symbol = { name; place }; _ } : Rig_elf.relocation) =
    match place with
    | Undefined when not (Hashtbl.mem externals name) ->
        Hashtbl.add externals name (target name, Hashtbl.length externals)
    | _ -> ()
  in
  List.iter add o.relocations;
  externals

let write_slot b ~at a =
  if host = em_x86_64 then begin
    Bytes.set_uint16_le b at 0x25ff;
    set32 b (at + 2) 2
  end
  else begin
    set32 b at 0x58000051;
    set32 b (at + 4) 0xd61f0220
  end;
  Bytes.set_int64_le b (at + 8) (Int64.of_int a)

(* Relocating *)

let checked ~at bits x ~target =
  if fits bits x then x
  else
    refusef
      "relocation at 0x%x reaches 0x%x, which does not fit its %d-bit field" at
      target bits

(* A branch lands at [S + A] on arm64 and at [S + A + 4] on x86_64, whose
   displacement counts from the field's end. Out of reach, it goes to [stub], if
   there is one and its addend lands it on the stub's start. *)
let branch_addend = if host = em_x86_64 then -4 else 0

let branch ~at ~p ~stub bits s a =
  let x = s + a - p in
  match stub with
  | Some l when (not (fits bits x)) && a = branch_addend ->
      checked ~at bits (l + a - p) ~target:(s + a)
  | _ -> checked ~at bits x ~target:(s + a)

(* ADRP's 21-bit immediate, the page distance [x], in two fields. *)
let adrp b ~at x ~target =
  let x = checked ~at 33 x ~target asr 12 in
  set_bits b at ~lo:29 ~width:2 x;
  set_bits b at ~lo:5 ~width:19 (x asr 2)

let lo12 b ~at x shift =
  set_bits b at ~lo:10 ~width:12 ((x land 0xfff) lsr shift)

(* Patches the field at [at] of the relocation of type [k] and addend [a] in
   [b], the image at [base], for the symbol at [s]. *)
let relocate b ~base ~stub ~at ~k ~a s =
  let p = base + at in
  if host = em_x86_64 then
    if k = r_x86_64_pc32 then
      set32 b at (checked ~at 32 (s + a - p) ~target:(s + a))
    else if k = r_x86_64_plt32 then set32 b at (branch ~at ~p ~stub 32 s a)
    else err_unsupported k at
  else if k = r_aarch64_call26 || k = r_aarch64_jump26 then
    set_bits b at ~lo:0 ~width:26 (branch ~at ~p ~stub 28 s a asr 2)
  else if k = r_aarch64_adr_prel_pg_hi21 then
    adrp b ~at (page_of (s + a) - page_of p) ~target:(s + a)
  else if k = r_aarch64_add_abs_lo12_nc || k = r_aarch64_ldst8_abs_lo12_nc then
    lo12 b ~at (s + a) 0
  else if k = r_aarch64_ldst16_abs_lo12_nc then lo12 b ~at (s + a) 1
  else if k = r_aarch64_ldst32_abs_lo12_nc then lo12 b ~at (s + a) 2
  else if k = r_aarch64_ldst64_abs_lo12_nc then lo12 b ~at (s + a) 3
  else if k = r_aarch64_ldst128_abs_lo12_nc then lo12 b ~at (s + a) 4
  else err_unsupported k at

(* Linking *)

let check (o : Rig_elf.t) =
  let machine m =
    if m = em_x86_64 || m = em_aarch64 then
      strf "machine %d (%s)" m (machine_name m)
    else strf "machine %d" m
  in
  let writable (s : Rig_elf.section) =
    s.size > 0 && s.flags land shf_write <> 0 && s.flags land shf_alloc <> 0
  in
  if host <> em_x86_64 && host <> em_aarch64 then
    refusef "the host's machine is neither x86_64 nor arm64";
  if o.bits <> 64 then refusef "the object is 32-bit, expected 64-bit";
  if o.kind <> et_rel then
    refusef "the object is of type %d, expected ET_REL (1)" o.kind;
  if o.machine <> host then
    refusef "the object is for %s, expected %s" (machine o.machine)
      (machine_name host);
  match Iarray.find_opt writable o.sections with
  | Some s ->
      refusef
        "section %S is writable and not empty; pass variables through the \
         parameters"
        s.name
  | None -> ()

(* The address of each named symbol of an executable section of [o], linked
   at [base]. *)
let functions (o : Rig_elf.t) ~base =
  let code (s : Rig_elf.symbol) =
    match s.place with
    | Image { section; offset }
      when s.name <> ""
           && (Iarray.get o.sections section).flags land shf_execinstr <> 0 ->
        Some (s.name, base + offset)
    | _ -> None
  in
  List.filter_map code (Iarray.to_list o.symbols)

let image (o : Rig_elf.t) size =
  let b = Bytes.make size '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  b

(* [o] linked at [base]: its image, then its slots from [slots_at], then its
   relocations applied. *)
let write (o : Rig_elf.t) externals ~slots_at ~size ~base =
  let b = image o size in
  let slot i = slots_at + (slot_bytes * i) in
  Hashtbl.iter (fun _ (a, i) -> write_slot b ~at:(slot i) a) externals;
  let patch (r : Rig_elf.relocation) =
    let at = r.offset in
    if at + field_bytes > o.size then
      refusef "relocation at 0x%x patches past the image's end" at;
    let a =
      match r.addend with
      | Explicit a -> a
      | Implicit _ ->
          refusef "relocation at 0x%x has its addend in its field (SHT_REL)" at
    in
    let s, stub =
      match r.symbol.place with
      | Image { offset; _ } -> (base + offset, None)
      | Absolute v -> (v, None)
      | Undefined ->
          let a, i = Hashtbl.find externals r.symbol.name in
          (a, Some (base + slot i))
      | Outside _ -> refusef "symbol %S lies outside the image" r.symbol.name
    in
    relocate b ~base ~stub ~at ~k:r.kind ~a s
  in
  List.iter patch o.relocations;
  b

let link_exn obj =
  let o =
    match Rig_elf.of_string obj with Ok o -> o | Error e -> raise (Refused e)
  in
  check o;
  let externals = externals o in
  let n = Hashtbl.length externals in
  if o.size > max_int - (slot_bytes * (n + 1)) then
    refusef "the image's %d bytes and %d stubs exceed max_int" o.size n;
  let slots_at = (o.size + slot_bytes - 1) / slot_bytes * slot_bytes in
  let size = Int.max 1 (slots_at + (slot_bytes * n)) in
  let code = map size o.align in
  let base = base code in
  if base < 0 then err_map size o.align (-base);
  let b = write o externals ~slots_at ~size ~base in
  match install code b with
  | 0 -> { Def.code; functions = functions o ~base }
  | e -> err_protect size (-e)

let link obj = match link_exn obj with c -> Ok c | exception Refused e -> Error e

let entry (c : Def.linked) f = List.assoc_opt f c.functions
