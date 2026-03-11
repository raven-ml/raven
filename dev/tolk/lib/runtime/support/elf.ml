(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Image = struct
  type t = { mutable data : Bytes.t; mutable len : int }

  let create len = { data = Bytes.make len '\000'; len }

  let ensure t size =
    let capacity = Bytes.length t.data in
    if size <= capacity then ()
    else
      let rec grow cap = if cap >= size then cap else grow (cap * 2) in
      let next_cap = if capacity = 0 then size else grow capacity in
      let next = Bytes.make next_cap '\000' in
      Bytes.blit t.data 0 next 0 t.len;
      t.data <- next

  let extend_zero t len =
    let needed = t.len + len in
    ensure t needed;
    t.len <- needed

  let set_bytes t off src =
    let src_len = Bytes.length src in
    let needed = off + src_len in
    ensure t needed;
    Bytes.blit src 0 t.data off src_len;
    if needed > t.len then t.len <- needed

  let append_bytes t src =
    let off = t.len in
    set_bytes t off src;
    off

  let align t alignment =
    let rem = t.len mod alignment in
    if rem <> 0 then extend_zero t (alignment - rem)

  let length t = t.len

  let set_zero t off len =
    let needed = off + len in
    ensure t needed;
    Bytes.fill t.data off len '\000';
    if needed > t.len then t.len <- needed

  let to_bytes t = Bytes.sub t.data 0 t.len
end

type section_header = {
  sh_name : int;
  sh_type : int;
  sh_flags : Int64.t;
  mutable sh_addr : int;
  sh_offset : int;
  sh_size : int;
  sh_link : int;
  sh_info : int;
  sh_addralign : int;
  sh_entsize : int;
}

type raw_section = { name : string; header : section_header; content : Bytes.t }
type section = { name : string; addr : int; size : int; content : Bytes.t }
type symbol = { name : string; shndx : int; value : int }
type reloc = { offset : int; symbol : symbol; r_type : int; addend : int }

type t = {
  image : Bytes.t;
  sections : section array;
  symbols : symbol array;
  relocs : reloc list;
}

let array_find_opt f a =
  let len = Array.length a in
  let rec aux i =
    if i >= len then None else if f a.(i) then Some a.(i) else aux (i + 1)
  in
  aux 0

let sht_null = 0
let sht_symtab = 2
let sht_rela = 4
let sht_nobits = 8
let sht_rel = 9
let shf_alloc = 0x2L
let image t = t.image
let sections t = t.sections
let relocs t = t.relocs
let u8 bytes off = Char.code (Bytes.get bytes off)

let u16 bytes off =
  let b0 = u8 bytes off in
  let b1 = u8 bytes (off + 1) in
  b0 lor (b1 lsl 8)

let u32 bytes off =
  let b0 = u8 bytes off in
  let b1 = u8 bytes (off + 1) in
  let b2 = u8 bytes (off + 2) in
  let b3 = u8 bytes (off + 3) in
  b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24)

let u64 bytes off =
  let open Int64 in
  let lo = of_int (u32 bytes off) in
  let hi = of_int (u32 bytes (off + 4)) in
  logor lo (shift_left hi 32)

let strtab_get bytes off =
  let rec find_end idx =
    if idx >= Bytes.length bytes then idx
    else if Bytes.get bytes idx = '\000' then idx
    else find_end (idx + 1)
  in
  let last = find_end off in
  Bytes.sub_string bytes off (last - off)

let read_headers obj =
  if Bytes.length obj < 64 then invalid_arg "invalid ELF";
  if
    Bytes.get obj 0 <> '\x7f'
    || Bytes.get obj 1 <> 'E'
    || Bytes.get obj 2 <> 'L'
    || Bytes.get obj 3 <> 'F'
  then invalid_arg "invalid ELF";
  let class_ = u8 obj 4 in
  let data = u8 obj 5 in
  if class_ <> 2 || data <> 1 then invalid_arg "unsupported ELF format";
  let e_type = u16 obj 16 in
  if e_type <> 1 then invalid_arg "unsupported ELF type";
  let e_shoff = Int64.to_int (u64 obj 40) in
  let e_shentsize = u16 obj 58 in
  let e_shnum = u16 obj 60 in
  let e_shstrndx = u16 obj 62 in
  let headers =
    Array.init e_shnum (fun i ->
        let off = e_shoff + (i * e_shentsize) in
        let sh_name = u32 obj off in
        let sh_type = u32 obj (off + 4) in
        let sh_flags = u64 obj (off + 8) in
        let sh_addr = Int64.to_int (u64 obj (off + 16)) in
        let sh_offset = Int64.to_int (u64 obj (off + 24)) in
        let sh_size = Int64.to_int (u64 obj (off + 32)) in
        let sh_link = u32 obj (off + 40) in
        let sh_info = u32 obj (off + 44) in
        let sh_addralign = Int64.to_int (u64 obj (off + 48)) in
        let sh_entsize = Int64.to_int (u64 obj (off + 56)) in
        {
          sh_name;
          sh_type;
          sh_flags;
          sh_addr;
          sh_offset;
          sh_size;
          sh_link;
          sh_info;
          sh_addralign;
          sh_entsize;
        })
  in
  let sh_strtab =
    let hdr = headers.(e_shstrndx) in
    Bytes.sub obj hdr.sh_offset hdr.sh_size
  in
  Array.map
    (fun header ->
      let name = strtab_get sh_strtab header.sh_name in
      let content =
        if header.sh_type = sht_nobits then Bytes.create 0
        else Bytes.sub obj header.sh_offset header.sh_size
      in
      { name; header; content })
    headers

let is_alloc_section section =
  Int64.logand section.header.sh_flags shf_alloc <> 0L

let build_image ?(force_section_align = 1) sections =
  let max_fixed =
    Array.fold_left
      (fun acc s ->
        if is_alloc_section s && s.header.sh_addr <> 0 then
          max acc (s.header.sh_addr + s.header.sh_size)
        else acc)
      0 sections
  in
  let image = Image.create max_fixed in
  Array.iter
    (fun s ->
      if not (is_alloc_section s) then ()
      else if s.header.sh_addr <> 0 then
        if s.header.sh_type = sht_nobits then
          Image.set_zero image s.header.sh_addr s.header.sh_size
        else Image.set_bytes image s.header.sh_addr s.content
      else begin
        let align = max force_section_align (max s.header.sh_addralign 1) in
        Image.align image align;
        s.header.sh_addr <- Image.length image;
        if s.header.sh_type = sht_nobits then
          Image.extend_zero image s.header.sh_size
        else ignore (Image.append_bytes image s.content)
      end)
    sections;
  image

let symtab sections =
  array_find_opt (fun s -> s.header.sh_type = sht_symtab) sections

let read_symbols sections =
  match symtab sections with
  | None -> [||]
  | Some sym_sec ->
      let strtab = sections.(sym_sec.header.sh_link).content in
      let entsize = max sym_sec.header.sh_entsize 24 in
      let count = sym_sec.header.sh_size / entsize in
      Array.init count (fun i ->
          let off = i * entsize in
          let st_name = u32 sym_sec.content off in
          let st_shndx = u16 sym_sec.content (off + 6) in
          let st_value = Int64.to_int (u64 sym_sec.content (off + 8)) in
          let name = strtab_get strtab st_name in
          { name; shndx = st_shndx; value = st_value })

let read_relocs sections symbols =
  let acc = ref [] in
  Array.iter
    (fun rel_sec ->
      if rel_sec.header.sh_type <> sht_rel && rel_sec.header.sh_type <> sht_rela
      then ()
      else
        let target : raw_section = sections.(rel_sec.header.sh_info) in
        if not (String.equal target.name ".eh_frame") then begin
          let entsize =
            if rel_sec.header.sh_entsize <> 0 then rel_sec.header.sh_entsize
            else if rel_sec.header.sh_type = sht_rel then 16
            else 24
          in
          let count = rel_sec.header.sh_size / entsize in
          for i = 0 to count - 1 do
            let off = i * entsize in
            let r_offset = Int64.to_int (u64 rel_sec.content off) in
            let r_info = u64 rel_sec.content (off + 8) in
            let addend =
              if rel_sec.header.sh_type = sht_rela then
                Int64.to_int (u64 rel_sec.content (off + 16))
              else 0
            in
            let sym_idx = Int64.(to_int (shift_right_logical r_info 32)) in
            if sym_idx < 0 || sym_idx >= Array.length symbols then
              invalid_arg "invalid relocation symbol";
            let symbol = symbols.(sym_idx) in
            let r_type = Int64.(to_int (logand r_info 0xFFFFFFFFL)) in
            acc :=
              {
                offset = target.header.sh_addr + r_offset;
                symbol;
                r_type;
                addend;
              }
              :: !acc
          done
        end)
    sections;
  List.rev !acc

let public_section raw =
  let content =
    if raw.header.sh_type = sht_nobits then Bytes.make raw.header.sh_size '\000'
    else raw.content
  in
  {
    name = raw.name;
    addr = raw.header.sh_addr;
    size = raw.header.sh_size;
    content;
  }

let load ?(force_section_align = 1) obj =
  let sections = read_headers obj in
  let image = build_image ~force_section_align sections in
  let symbols = read_symbols sections in
  let relocs = read_relocs sections symbols in
  {
    image = Image.to_bytes image;
    sections = Array.map public_section sections;
    symbols;
    relocs;
  }

let find_section (t : t) name =
  array_find_opt (fun (s : section) -> s.name = name) t.sections

let find_symbol_offset t name =
  match array_find_opt (fun (s : symbol) -> s.name = name) t.symbols with
  | None -> invalid_arg ("missing symbol: " ^ name)
  | Some sym ->
      if sym.shndx = sht_null then invalid_arg ("symbol is undefined: " ^ name)
      else
        let section = t.sections.(sym.shndx) in
        section.addr + sym.value
