(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Refusals: a link raises [Refused] where it finds the object unfit, and
   returns it as its [Error]. *)

exception Refused of string

let refusef fmt = Printf.ksprintf (fun m -> raise (Refused m)) fmt

(* Errors *)

let err_unsupported k at =
  refusef "relocation of type %d at 0x%x is unsupported" k at

(* The host *)

external host_machine : unit -> int = "caml_device_host_machine"
external error_message : int -> string = "caml_device_host_error_message"
external workers : unit -> int = "caml_device_host_workers"

let em_x86_64 = 62
let em_aarch64 = 183
let host = host_machine ()

(* The least page of every supported host: a mapping starts at a multiple of
   it. *)
let max_align = 4096
let machine_name m = if m = em_x86_64 then "x86_64" else "arm64"

(* Code memory: a mapping the collector unmaps. [base] is its address, or the
   system's error, negated, if mapping failed; [install] is [0] or that
   error. *)

type mapping

external map : int -> mapping = "caml_device_host_map"
external base : mapping -> int = "caml_device_host_base"
external install : mapping -> bytes -> int = "caml_device_host_install"
external process_symbol : string -> int = "caml_device_host_symbol"

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

(* The symbols [o] refers to and does not define, by name: each one's address in
   the process and its slot's index. *)
let externals (o : Device_elf.t) =
  let externals = Hashtbl.create 8 in
  let add ({ symbol = { name; place }; _ } : Device_elf.relocation) =
    match place with
    | Undefined when not (Hashtbl.mem externals name) -> (
        match process_symbol name with
        | 0 -> refusef "symbol %S is defined by no library of the process" name
        | a -> Hashtbl.add externals name (a, Hashtbl.length externals))
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

type t = { mapping : mapping; address : int }

let address p = p.address

let check (o : Device_elf.t) =
  let machine m =
    if m = em_x86_64 || m = em_aarch64 then
      strf "machine %d (%s)" m (machine_name m)
    else strf "machine %d" m
  in
  let writable (s : Device_elf.section) =
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
  if o.align > max_align then
    refusef "a section asks for an alignment of %d bytes, expected at most %d"
      o.align max_align;
  match Iarray.find_opt writable o.sections with
  | Some s ->
      refusef
        "section %S is writable and not empty; pass variables through the \
         buffers"
        s.name
  | None -> ()

let entry_offset (o : Device_elf.t) entry =
  let code (s : Device_elf.symbol) =
    match s.place with
    | Image { section; _ } ->
        s.name = entry
        && (Iarray.get o.sections section).flags land shf_execinstr <> 0
    | _ -> false
  in
  match Iarray.find_opt code o.symbols with
  | Some { place = Image { offset; _ }; _ } -> offset
  | _ -> refusef "entry %S is no symbol of an executable section" entry

let image (o : Device_elf.t) size =
  let b = Bytes.make size '\000' in
  let put (s : Device_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  b

(* [o] linked at [base]: its image, then its slots from [slots_at], then its
   relocations applied. *)
let write (o : Device_elf.t) externals ~slots_at ~size ~base =
  let b = image o size in
  let slot i = slots_at + (slot_bytes * i) in
  Hashtbl.iter (fun _ (a, i) -> write_slot b ~at:(slot i) a) externals;
  let patch (r : Device_elf.relocation) =
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
          let v, i = Hashtbl.find externals r.symbol.name in
          (v, Some (base + slot i))
      | Outside _ -> refusef "symbol %S lies outside the image" r.symbol.name
    in
    relocate b ~base ~stub ~at ~k:r.kind ~a s
  in
  List.iter patch o.relocations;
  b

let link_exn ~entry obj =
  let o =
    match Device_elf.of_string obj with
    | Ok o -> o
    | Error e -> raise (Refused e)
  in
  check o;
  let start = entry_offset o entry in
  let externals = externals o in
  let n = Hashtbl.length externals in
  if o.size > max_int - (slot_bytes * (n + 1)) then
    refusef "the image's %d bytes and %d stubs exceed max_int" o.size n;
  let slots_at = (o.size + slot_bytes - 1) / slot_bytes * slot_bytes in
  let size = Int.max 1 (slots_at + (slot_bytes * n)) in
  let mapping = map size in
  let base = base mapping in
  if base < 0 then
    refusef "mapping %d bytes of executable memory: %s" size
      (error_message (-base));
  let b = write o externals ~slots_at ~size ~base in
  match install mapping b with
  | 0 -> { mapping; address = base + start }
  | e -> refusef "making %d bytes executable: %s" size (error_message (-e))

let link ~entry obj =
  match link_exn ~entry obj with p -> Ok p | exception Refused e -> Error e

(* Calling *)

type split = { extent : int; blocks : int; lo : int; hi : int }

external call_entry :
  int -> mapping -> int array -> int array -> split option -> unit
  = "caml_device_host_call"

let check_index what i n =
  if i < 0 || i >= n then
    invalid_argf "Device_host.call: split.%s %d is no index of %d values" what i
      n

let check_split s n =
  if s.extent < 0 then
    invalid_argf "Device_host.call: split.extent %d is negative" s.extent;
  if s.blocks < 1 then
    invalid_argf "Device_host.call: split.blocks %d is below 1" s.blocks;
  check_index "lo" s.lo n;
  check_index "hi" s.hi n;
  if s.lo = s.hi then
    invalid_argf "Device_host.call: split.lo and split.hi are both %d" s.lo

let call ?split p buffers values =
  (match split with
  | Some s -> check_split s (Array.length values)
  | None -> ());
  call_entry p.address p.mapping buffers values split
