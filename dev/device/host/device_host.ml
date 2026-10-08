(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The host *)

external host_machine : unit -> int = "caml_device_host_machine"
external page_size : unit -> int = "caml_device_host_page_size"
external error_message : int -> string = "caml_device_host_error_message"

external workers : unit -> (int[@untagged])
  = "caml_device_host_workers_byte" "caml_device_host_workers"
[@@noalloc]

let em_x86_64 = 62
let em_aarch64 = 183
let host = host_machine ()
let page = page_size ()
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
let r_x86_64_gotpcrel = 9
let r_x86_64_pc64 = 24
let r_x86_64_gotpcrelx = 41
let r_x86_64_rex_gotpcrelx = 42
let r_aarch64_prel64 = 260
let r_aarch64_prel32 = 261
let r_aarch64_prel16 = 262
let r_aarch64_adr_prel_pg_hi21 = 275
let r_aarch64_add_abs_lo12_nc = 277
let r_aarch64_ldst8_abs_lo12_nc = 278
let r_aarch64_jump26 = 282
let r_aarch64_call26 = 283
let r_aarch64_ldst16_abs_lo12_nc = 284
let r_aarch64_ldst32_abs_lo12_nc = 285
let r_aarch64_ldst64_abs_lo12_nc = 286
let r_aarch64_ldst128_abs_lo12_nc = 299
let r_aarch64_adr_got_page = 311
let r_aarch64_ld64_got_lo12_nc = 312

let is_branch k =
  k = r_x86_64_plt32 || k = r_aarch64_call26 || k = r_aarch64_jump26

let is_got k =
  k = r_x86_64_gotpcrel || k = r_x86_64_gotpcrelx || k = r_x86_64_rex_gotpcrelx
  || k = r_aarch64_adr_got_page
  || k = r_aarch64_ld64_got_lo12_nc

(* Fields *)

let fits bits x = -(1 lsl (bits - 1)) <= x && x < 1 lsl (bits - 1)

(* aaelf64 checks a 16- or 32-bit place-relative datum as signed or unsigned. *)
let fits_either bits x = -(1 lsl (bits - 1)) <= x && x < 1 lsl bits
let page_of a = a land lnot 0xfff
let set16 b at x = Bytes.set_int16_le b at x
let set32 b at x = Bytes.set_int32_le b at (Int32.of_int x)
let set64 b at x = Bytes.set_int64_le b at (Int64.of_int x)

(* Replaces the [width] bits from [lo] of the instruction at [at] by [x]'s low
   bits. *)
let set_bits b at ~lo ~width x =
  let insn = Int32.to_int (Bytes.get_int32_le b at) in
  let mask = ((1 lsl width) - 1) lsl lo in
  set32 b at (insn land lnot mask lor ((x lsl lo) land mask))

(* Refusals: a link raises [Refused] where it finds the object unfit, and
   returns it as its [Error]. *)

exception Refused of string

let refusef fmt = Printf.ksprintf (fun m -> raise (Refused m)) fmt

(* Symbols *)

(* Where a symbol is: at an offset of the image, or at an address. *)
type value = Offset of int | Address of int

let absolute ~base = function Offset o -> base + o | Address a -> a

(* The value of the symbol [s] of [o]. [externals] holds the values of the
   undefined symbols already looked up. *)
let value (o : Device_elf.t) externals (s : Device_elf.symbol) =
  match s.place with
  | Image { offset; _ } -> Offset offset
  | Absolute a -> Address a
  | Outside _ -> refusef "symbol %S lies outside the image" s.name
  | Undefined -> (
      match Hashtbl.find_opt externals s.name with
      | Some v -> v
      | None ->
          let v =
            match Device_elf.symbol o s.name with
            | Some offset -> Offset offset
            | None -> (
                match process_symbol s.name with
                | 0 ->
                    refusef "symbol %S is defined by no library of the process"
                      s.name
                | a -> Address a)
          in
          Hashtbl.add externals s.name v;
          v)

let addend (r : Device_elf.relocation) =
  match r.addend with
  | Explicit a -> a
  | Implicit ->
      refusef "relocation at 0x%x has its addend in its field (SHT_REL)"
        r.offset

(* Slots

   A slot follows the image for each address that a GOT reference, or a branch
   to a symbol outside the image, names: a stub at its start that jumps to the
   word at its end, which holds the address. On x86_64 the stub is [jmp [rip +
   2]], whose 6 bytes end 2 before the word; on arm64 [ldr x17, #8; br x17]. A
   branch goes through the stub when the symbol is out of its reach; a GOT
   reference takes the word. *)

let slot_bytes = 16
let word_at = 8

(* The value a GOT reference's word holds: [S], or [S + A] on arm64, which adds
   the addend to the word, [GDAT(S + A)]. *)
let word a v =
  if host = em_x86_64 then v
  else match v with Offset o -> Offset (o + a) | Address s -> Address (s + a)

(* The slots of [o]'s relocations, numbered from 0 by the value of their
   word. *)
let slots (o : Device_elf.t) externals =
  let slots = Hashtbl.create 8 in
  let add (r : Device_elf.relocation) =
    let a = addend r and v = value o externals r.symbol in
    let w =
      if is_got r.kind then Some (word a v)
      else match v with Address _ when is_branch r.kind -> Some v | _ -> None
    in
    match w with
    | Some w when not (Hashtbl.mem slots w) ->
        Hashtbl.add slots w (Hashtbl.length slots)
    | _ -> ()
  in
  List.iter add o.relocations;
  slots

let write_slot b ~at a =
  if host = em_x86_64 then begin
    Bytes.set_uint16_le b at 0x25ff;
    set32 b (at + 2) 2
  end
  else begin
    set32 b at 0x58000051;
    set32 b (at + 4) 0xd61f0220
  end;
  set64 b (at + word_at) a

(* Relocating *)

let field_bytes k =
  if k = r_x86_64_pc64 || k = r_aarch64_prel64 then 8
  else if k = r_aarch64_prel16 then 2
  else 4

let refuse_reach ~at bits ~target =
  refusef "relocation at 0x%x reaches 0x%x, beyond its %d bits" at target bits

let checked ~at bits x ~target =
  if fits bits x then x else refuse_reach ~at bits ~target

let checked_either ~at bits x ~target =
  if fits_either bits x then x else refuse_reach ~at bits ~target

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
   [b], the image at [base]. [s] is the address its formula takes: its symbol's,
   or for a GOT reference its word's, [G]. *)
let relocate b ~base ~stub ~at ~k ~a s =
  let p = base + at in
  if host = em_x86_64 then
    if k = r_x86_64_pc32 || is_got k then
      set32 b at (checked ~at 32 (s + a - p) ~target:(s + a))
    else if k = r_x86_64_pc64 then set64 b at (s + a - p)
    else if k = r_x86_64_plt32 then set32 b at (branch ~at ~p ~stub 32 s a)
    else refusef "relocation of type %d at 0x%x is unsupported" k at
  else if k = r_aarch64_prel64 then set64 b at (s + a - p)
  else if k = r_aarch64_prel32 then
    set32 b at (checked_either ~at 32 (s + a - p) ~target:(s + a))
  else if k = r_aarch64_prel16 then
    set16 b at (checked_either ~at 16 (s + a - p) ~target:(s + a))
  else if k = r_aarch64_call26 || k = r_aarch64_jump26 then
    set_bits b at ~lo:0 ~width:26 (branch ~at ~p ~stub 28 s a asr 2)
  else if k = r_aarch64_adr_prel_pg_hi21 then
    adrp b ~at (page_of (s + a) - page_of p) ~target:(s + a)
  else if k = r_aarch64_adr_got_page then
    adrp b ~at (page_of s - page_of p) ~target:s
  else if k = r_aarch64_ld64_got_lo12_nc then lo12 b ~at s 3
  else if k = r_aarch64_add_abs_lo12_nc || k = r_aarch64_ldst8_abs_lo12_nc then
    lo12 b ~at (s + a) 0
  else if k = r_aarch64_ldst16_abs_lo12_nc then lo12 b ~at (s + a) 1
  else if k = r_aarch64_ldst32_abs_lo12_nc then lo12 b ~at (s + a) 2
  else if k = r_aarch64_ldst64_abs_lo12_nc then lo12 b ~at (s + a) 3
  else if k = r_aarch64_ldst128_abs_lo12_nc then lo12 b ~at (s + a) 4
  else refusef "relocation of type %d at 0x%x is unsupported" k at

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
  if o.bits <> 64 then refusef "the object is 32-bit";
  if o.kind <> et_rel then
    refusef "the object is of type %d, not relocatable" o.kind;
  if o.machine <> host then
    refusef "the object is for %s, expected %s" (machine o.machine)
      (machine_name host);
  if o.align > page then
    refusef "a section asks for an alignment of %d bytes, above the page's %d"
      o.align page;
  match Iarray.find_opt writable o.sections with
  | Some s -> refusef "section %s is writable" s.name
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

(* [o] linked into [mapping] at [base]: its image, then its slots from
   [slots_at], then its relocations applied. *)
let write (o : Device_elf.t) externals ~slots ~slots_at ~size ~base =
  let b = image o size in
  let slot w = base + slots_at + (slot_bytes * Hashtbl.find slots w) in
  let write_word w i =
    write_slot b ~at:(slots_at + (slot_bytes * i)) (absolute ~base w)
  in
  Hashtbl.iter write_word slots;
  let patch (r : Device_elf.relocation) =
    let at = r.offset and k = r.kind and a = addend r in
    let v = value o externals r.symbol in
    if at + field_bytes k > o.size then
      refusef "relocation at 0x%x patches past the image's end" at;
    if is_got k then
      relocate b ~base ~stub:None ~at ~k ~a (slot (word a v) + word_at)
    else
      let stub =
        match v with Address _ when is_branch k -> Some (slot v) | _ -> None
      in
      relocate b ~base ~stub ~at ~k ~a (absolute ~base v)
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
  let externals = Hashtbl.create 8 in
  let slots = slots o externals in
  let n = Hashtbl.length slots in
  if o.size > max_int - (slot_bytes * (n + 1)) then
    refusef "the image's %d bytes and %d slots pass max_int" o.size n;
  let slots_at = (o.size + slot_bytes - 1) / slot_bytes * slot_bytes in
  let size = Int.max 1 (slots_at + (slot_bytes * n)) in
  let mapping = map size in
  let base = base mapping in
  if base < 0 then
    refusef "mapping %d bytes of executable memory: %s" size
      (error_message (-base));
  let b = write o externals ~slots ~slots_at ~size ~base in
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
