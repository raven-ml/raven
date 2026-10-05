(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Elf = Nx_device_elf

external code_alloc : int -> nativeint = "caml_nx_device_code_alloc"

external code_install : nativeint -> bytes -> unit
  = "caml_nx_device_code_install"

external code_free : nativeint -> int -> unit = "caml_nx_device_code_free"
external symbol_address : string -> nativeint = "caml_nx_device_symbol"

let fail fmt = Printf.ksprintf failwith fmt

(* Objects *)

let et_rel = 1
let em_x86_64 = 62
let em_aarch64 = 183

let machine_name m =
  if m = em_x86_64 then "x86_64"
  else if m = em_aarch64 then "aarch64"
  else Printf.sprintf "machine %d" m

(* Allocated and writable: data the program would write, in memory that is never
   writable. *)
let shf_write_alloc = 0x3

(* Relocations *)

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

let is_branch kind =
  kind = r_x86_64_plt32 || kind = r_aarch64_call26 || kind = r_aarch64_jump26

(* A branch to an external symbol out of direct range goes through a slot after
   the image, which jumps to the symbol's absolute address. *)
let slot_bytes = 16
let fits bits v = v >= -(1 lsl (bits - 1)) && v < 1 lsl (bits - 1)
let get32 b at = Int32.to_int (Bytes.get_int32_le b at) land 0xffff_ffff
let set32 b at v = Bytes.set_int32_le b at (Int32.of_int v)

(* [insn] with the [width] bits from [lo] replaced by [v]'s low bits. *)
let field insn ~lo ~width v =
  let mask = ((1 lsl width) - 1) lsl lo in
  insn land lnot mask lor ((v lsl lo) land mask)

let page a = a land lnot 0xfff

(* The slot at [at] made to jump to [dst], whose address is its second 8 bytes:
   [jmp [rip + 2]] on x86_64, whose 6 bytes end 2 before it, and [ldr x17, #8;
   br x17] on aarch64. *)
let write_slot b ~machine ~at dst =
  if machine = em_x86_64 then begin
    Bytes.set_uint16_le b at 0x25ff;
    set32 b (at + 2) 2
  end
  else begin
    set32 b at 0x58000051;
    set32 b (at + 4) 0xd61f0220
  end;
  Bytes.set_int64_le b (at + 8) (Int64.of_int dst)

(* [o]'s image linked at [base], in [size] bytes with its slots from [slots_at].
   [externals] resolves the symbols [o] does not define. *)
let link (o : Elf.t) ~machine ~base ~size ~slots_at externals =
  let b = Bytes.make size '\000' in
  Bytes.blit_string o.image 0 b 0 (String.length o.image);
  let next_slot = ref slots_at in
  let slot dst =
    let at = !next_slot in
    next_slot := at + slot_bytes;
    write_slot b ~machine ~at dst;
    base + at
  in
  let relocate (r : Elf.relocation) =
    if r.at < 0 || r.at > String.length o.image - 4 then
      fail "a relocation at 0x%x outside the image" r.at;
    let p = base + r.at and a = r.addend and k = r.kind in
    let s, external_ =
      match r.target with
      | Offset t -> (base + t, false)
      | External name -> (externals name, true)
    in
    let checked bits d =
      if fits bits d then d else fail "a reference at 0x%x out of range" r.at
    in
    (* S + A - P for a branch that lands at [dst]; out of range, the branch
       lands on a slot that jumps to [dst]. *)
    let branch bits ~dst =
      let d = s + a - p in
      checked bits
        (if fits bits d || not external_ then d else d - dst + slot dst)
    in
    let insn = get32 b r.at in
    let lo12 shift =
      set32 b r.at
        (field insn ~lo:10 ~width:12 (((s + a) land 0xfff) lsr shift))
    in
    let unsupported () =
      fail "a relocation of unsupported type %d at 0x%x" k r.at
    in
    if machine = em_x86_64 then
      if k = r_x86_64_pc32 then set32 b r.at (checked 32 (s + a - p))
      else if k = r_x86_64_plt32 then
        (* The addend makes up for the field's distance to the next instruction,
           which the displacement counts from: the branch lands at S. *)
        set32 b r.at (branch 32 ~dst:s)
      else unsupported ()
    else if k = r_aarch64_call26 || k = r_aarch64_jump26 then
      let d = branch 28 ~dst:(s + a) in
      set32 b r.at (field insn ~lo:0 ~width:26 (d asr 2))
    else if k = r_aarch64_adr_prel_pg_hi21 then
      let d = checked 33 (page (s + a) - page p) asr 12 in
      set32 b r.at
        (field
           (field insn ~lo:29 ~width:2 (d land 3))
           ~lo:5 ~width:19 (d asr 2))
    else if k = r_aarch64_add_abs_lo12_nc || k = r_aarch64_ldst8_abs_lo12_nc
    then lo12 0
    else if k = r_aarch64_ldst16_abs_lo12_nc then lo12 1
    else if k = r_aarch64_ldst32_abs_lo12_nc then lo12 2
    else if k = r_aarch64_ldst64_abs_lo12_nc then lo12 3
    else if k = r_aarch64_ldst128_abs_lo12_nc then lo12 4
    else unsupported ()
  in
  List.iter relocate o.relocations;
  b

(* Loading *)

let round_up n a = (n + a - 1) / a * a

let load_exn machine ~binary ~entry =
  let o = Elf.load binary in
  if o.kind <> et_rel then fail "the object is not relocatable";
  if o.machine <> machine then
    fail "the object is for %s, not %s" (machine_name o.machine)
      (machine_name machine);
  let start =
    match Elf.symbol o entry with
    | Some e -> e
    | None -> fail "the object has no function %s" entry
  in
  List.iter
    (fun (s : Elf.section) ->
      if s.flags land shf_write_alloc = shf_write_alloc && s.size > 0 then
        fail "the object has writable data (%s)" s.name)
    o.sections;
  let resolved = Hashtbl.create 8 in
  let externals name =
    match Hashtbl.find_opt resolved name with
    | Some a -> a
    | None -> (
        match symbol_address name with
        | 0n -> fail "the object refers to an undefined symbol %s" name
        | a ->
            let a = Nativeint.to_int a in
            Hashtbl.add resolved name a;
            a)
  in
  let slots =
    List.length
      (List.filter
         (fun (r : Elf.relocation) ->
           is_branch r.kind
           && match r.target with External _ -> true | Offset _ -> false)
         o.relocations)
  in
  let slots_at = round_up (String.length o.image) slot_bytes in
  let size = Int.max 1 (slots_at + (slots * slot_bytes)) in
  let base = code_alloc size in
  let free () = code_free base size in
  match
    code_install base
      (link o ~machine ~base:(Nativeint.to_int base) ~size ~slots_at externals)
  with
  | () -> (Nativeint.add base (Nativeint.of_int start), free)
  | exception e ->
      free ();
      raise e

let load_for machine ~binary ~entry =
  match load_exn machine ~binary ~entry with
  | loaded -> Ok loaded
  | exception Failure why -> Error why

let load =
  match Host_arch.architecture with
  | "amd64" -> Some (load_for em_x86_64)
  | "arm64" -> Some (load_for em_aarch64)
  | _ -> None
