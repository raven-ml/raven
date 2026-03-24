(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

module Image = struct
  type t = { mutable data : Bytes.t; mutable len : int }

  let of_bytes ?(extra_capacity = 0) bytes =
    let len = Bytes.length bytes in
    let data = Bytes.make (len + extra_capacity) '\000' in
    Bytes.blit bytes 0 data 0 len;
    { data; len }

  let ensure t size =
    let capacity = Bytes.length t.data in
    if size <= capacity then ()
    else
      let rec grow cap = if cap >= size then cap else grow (cap * 2) in
      let next = Bytes.make (grow (max 1 capacity)) '\000' in
      Bytes.blit t.data 0 next 0 t.len;
      t.data <- next

  let append_bytes t src =
    let off = t.len in
    let src_len = Bytes.length src in
    let needed = off + src_len in
    ensure t needed;
    Bytes.blit src 0 t.data off src_len;
    t.len <- needed;
    off

  let get_u32 t off =
    let b0 = Char.code (Bytes.get t.data off) in
    let b1 = Char.code (Bytes.get t.data (off + 1)) in
    let b2 = Char.code (Bytes.get t.data (off + 2)) in
    let b3 = Char.code (Bytes.get t.data (off + 3)) in
    b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24)

  let set_u32 t off v =
    let byte n = Char.chr ((v lsr n) land 0xFF) in
    Bytes.set t.data off (byte 0);
    Bytes.set t.data (off + 1) (byte 8);
    Bytes.set t.data (off + 2) (byte 16);
    Bytes.set t.data (off + 3) (byte 24)

  let to_bytes t = Bytes.sub t.data 0 t.len
end

type reloc = { offset : int; target : nativeint; r_type : int; addend : int }

type t = {
  image : Bytes.t;
  relocs : reloc list;
  entry_offset : int;
  extra_capacity : int;
}

let r_x86_64_pc32 = 2
let r_x86_64_plt32 = 4
let r_aarch64_adr_prel_pg_hi21 = 275
let r_aarch64_add_abs_lo12_nc = 277
let r_aarch64_jump26 = 282
let r_aarch64_call26 = 283
let r_aarch64_ldst16_abs_lo12_nc = 284
let r_aarch64_ldst32_abs_lo12_nc = 285
let r_aarch64_ldst64_abs_lo12_nc = 286
let r_aarch64_ldst128_abs_lo12_nc = 299
let alloc_size t = Bytes.length t.image + t.extra_capacity
let entry_offset t = t.entry_offset
let mask_bits n = if n <= 0 then 0L else Int64.(sub (shift_left 1L n) 1L)

let getbits x lo hi =
  let width = hi - lo + 1 in
  Int64.(to_int (logand (shift_right_logical x lo) (mask_bits width)))

let i2u32 x = Int64.to_int (Int64.logand x 0xFFFFFFFFL)

let resolve_reloc ~link_symbol sections reloc =
  let symbol = reloc.Elf.symbol in
  let target =
    if symbol.shndx = 0 then link_symbol symbol.name
    else
      let section = sections.(symbol.shndx) in
      Nativeint.of_int (section.Elf.addr + symbol.value)
  in
  {
    offset = reloc.offset;
    target;
    r_type = reloc.r_type;
    addend = reloc.addend;
  }

let prepare ~link_symbol ~entry elf =
  let sections = Elf.sections elf in
  let relocs_rev, extra_capacity =
    List.fold_left
      (fun (acc, cap) r ->
        let reloc = resolve_reloc ~link_symbol sections r in
        let cap =
          if reloc.r_type = r_aarch64_call26 || reloc.r_type = r_aarch64_jump26
          then cap + 16
          else cap
        in
        (reloc :: acc, cap))
      ([], 0) (Elf.relocs elf)
  in
  {
    image = Elf.image elf;
    relocs = List.rev relocs_rev;
    entry_offset = Elf.find_symbol_offset elf entry;
    extra_capacity;
  }

let load ~link_symbol ~entry obj =
  let elf = Elf.load obj in
  prepare ~link_symbol ~entry elf

(* Patches a single relocation in the loaded ELF image at runtime. Handles
   x86_64 PC-relative (PC32, PLT32) and ARM64 page-relative (ADRP, ADD/LDSTn
   lo12, CALL26/JUMP26) relocation types. For ARM64 CALL26/JUMP26 targets
   beyond the +/-128 MiB direct-branch range, emits a trampoline stub (LDR X17
   + BR X17 + 8-byte absolute address) appended to the image. *)
let apply_reloc image ~base reloc =
  let open Int64 in
  let ploc = reloc.offset in
  let base_i64 = of_nativeint base in
  let ploc_i64 = of_int ploc in
  let tgt = add (of_nativeint reloc.target) (of_int reloc.addend) in
  let patch_lo12 ~shift =
    let instr = Image.get_u32 image ploc in
    let patched = instr lor (getbits tgt shift 11 lsl 10) in
    Image.set_u32 image ploc patched
  in
  let rt = reloc.r_type in
  if rt = r_x86_64_pc32 then Image.set_u32 image ploc (i2u32 (sub tgt ploc_i64))
  else if rt = r_x86_64_plt32 then
    Image.set_u32 image ploc (i2u32 (sub tgt (add ploc_i64 base_i64)))
  else if rt = r_aarch64_adr_prel_pg_hi21 then begin
    let instr = Image.get_u32 image ploc in
    let rel_pg =
      sub
        (logand tgt (lognot 0xFFFL))
        (logand (add base_i64 ploc_i64) (lognot 0xFFFL))
    in
    let patched =
      instr lor (getbits rel_pg 12 13 lsl 29) lor (getbits rel_pg 14 32 lsl 5)
    in
    Image.set_u32 image ploc patched
  end
  else if rt = r_aarch64_add_abs_lo12_nc then patch_lo12 ~shift:0
  else if rt = r_aarch64_ldst16_abs_lo12_nc then patch_lo12 ~shift:1
  else if rt = r_aarch64_ldst32_abs_lo12_nc then patch_lo12 ~shift:2
  else if rt = r_aarch64_ldst64_abs_lo12_nc then patch_lo12 ~shift:3
  else if rt = r_aarch64_ldst128_abs_lo12_nc then patch_lo12 ~shift:4
  else if rt = r_aarch64_call26 || rt = r_aarch64_jump26 then begin
    let delta = sub tgt (add base_i64 ploc_i64) in
    let lo = of_int (-((1 lsl 25) * 4)) in
    let hi = of_int (((1 lsl 25) - 1) * 4) in
    if compare delta lo >= 0 && compare delta hi <= 0 then
      let instr = Image.get_u32 image ploc in
      let patched = instr lor getbits delta 2 27 in
      Image.set_u32 image ploc patched
    else
      let tramp = Bytes.make 16 '\000' in
      let tramp_img = Image.of_bytes tramp in
      Image.set_u32 tramp_img 0 0x58000051;
      Image.set_u32 tramp_img 4 0xD61F0220;
      let bytes =
        Bytes.init 8 (fun i ->
            Char.chr (to_int (logand (shift_right_logical tgt (i * 8)) 0xFFL)))
      in
      Bytes.blit bytes 0 tramp 8 8;
      let tramp_off = Image.append_bytes image tramp in
      let instr = Image.get_u32 image ploc in
      let patched = instr lor getbits (of_int (tramp_off - ploc)) 2 27 in
      Image.set_u32 image ploc patched
  end
  else invalid_arg "unknown relocation type"

let link ~base t =
  let image = Image.of_bytes ~extra_capacity:t.extra_capacity t.image in
  List.iter (apply_reloc image ~base) t.relocs;
  Image.to_bytes image
