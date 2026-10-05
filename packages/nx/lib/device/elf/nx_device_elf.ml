(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type place = Undefined | Defined of { section : int; offset : int }
type symbol = { name : string; place : place }

type section = {
  name : string;
  kind : int;
  flags : int;
  offset : int;
  size : int;
  contents : string;
}

type target = Offset of int | External of string
type relocation = { at : int; target : target; kind : int; addend : int }

type t = {
  kind : int;
  machine : int;
  abi_version : int;
  flags : int;
  image : string;
  sections : section list;
  symbols : symbol array;
  relocations : relocation list;
}

let sht_progbits = 1
let sht_symtab = 2
let sht_rela = 4
let sht_nobits = 8
let sht_rel = 9

(* A section header. *)
type header = {
  h_name : string;
  h_kind : int;
  h_flags : int;
  h_addr : int;
  h_size : int;
  h_contents : string;
  h_link : int;
  h_info : int;
  h_align : int;
  h_entsize : int;
}

type raw_symbol = { s_name : string; s_shndx : int; s_value : int }

let load ?(align = 1) obj =
  let fail fmt =
    Printf.ksprintf (fun m -> failwith ("Nx_device_elf.load: " ^ m)) fmt
  in
  let len = String.length obj in
  let check off n =
    if off < 0 || n < 0 || off > len - n then fail "truncated"
  in
  let u16 off =
    check off 2;
    String.get_uint16_le obj off
  in
  let u32 off =
    check off 4;
    Int32.to_int (String.get_int32_le obj off) land 0xffff_ffff
  in
  let u64 off =
    check off 8;
    Int64.to_int (String.get_int64_le obj off)
  in
  let sub off n =
    check off n;
    String.sub obj off n
  in
  if len < 64 || String.sub obj 0 4 <> "\x7fELF" then fail "not an ELF object";
  if obj.[4] <> '\002' || obj.[5] <> '\001' then
    fail "not a 64-bit little-endian object";
  let shoff = u64 40 and shnum = u16 60 and shstrndx = u16 62 in
  let raw_header i =
    let h = shoff + (64 * i) in
    (u32 h, u32 (h + 4), u64 (h + 16), u64 (h + 24), u64 (h + 32), h)
  in
  let cstring tab off =
    check (tab + off) 0;
    match String.index_from_opt obj (tab + off) '\000' with
    | Some e -> String.sub obj (tab + off) (e - tab - off)
    | None -> fail "an unterminated name"
  in
  let _, _, _, names, _, _ = raw_header shstrndx in
  let headers =
    Array.init shnum (fun i ->
        let name, kind, addr, offset, size, h = raw_header i in
        {
          h_name = cstring names name;
          h_kind = kind;
          h_flags = u64 (h + 8);
          h_addr = addr;
          h_size = size;
          h_contents = (if kind = sht_nobits then "" else sub offset size);
          h_link = u32 (h + 40);
          h_info = u32 (h + 44);
          h_align = Int.max 1 (u64 (h + 48));
          h_entsize = u64 (h + 56);
        })
  in
  (* Sections with an address go at it; the others follow, aligned. *)
  let fixed =
    Array.fold_left
      (fun m h ->
        if h.h_kind = sht_progbits && h.h_addr <> 0 then
          Int.max m (h.h_addr + String.length h.h_contents)
        else m)
      0 headers
  in
  let image = Buffer.create fixed in
  Buffer.add_string image (String.make fixed '\000');
  let appended =
    Array.map
      (fun h ->
        if h.h_kind <> sht_progbits || h.h_addr <> 0 then None
        else
          let a = Int.max h.h_align align in
          let at = (Buffer.length image + a - 1) / a * a in
          Buffer.add_string image
            (String.make (at - Buffer.length image) '\000');
          Buffer.add_string image h.h_contents;
          Some at)
      headers
  in
  let image = Buffer.to_bytes image in
  Array.iter
    (fun h ->
      if h.h_kind = sht_progbits && h.h_addr <> 0 then
        Bytes.blit_string h.h_contents 0 image h.h_addr
          (String.length h.h_contents))
    headers;
  (* A value is relative to its section when the section was appended, and an
     image offset already when the section has an address. *)
  let resolve shndx value =
    if shndx > 0 && shndx < shnum then
      match appended.(shndx) with Some at -> at + value | None -> value
    else value
  in
  let symbols_of h =
    let strtab = headers.(h.h_link).h_contents in
    let entsize = Int.max 24 h.h_entsize in
    List.init
      (String.length h.h_contents / entsize)
      (fun k ->
        let e = k * entsize in
        let name = Int32.to_int (String.get_int32_le h.h_contents e) in
        let s_name =
          match String.index_from_opt strtab name '\000' with
          | Some z -> String.sub strtab name (z - name)
          | None -> fail "an unterminated symbol name"
        in
        {
          s_name;
          s_shndx = String.get_uint16_le h.h_contents (e + 6);
          s_value = Int64.to_int (String.get_int64_le h.h_contents (e + 8));
        })
  in
  let table =
    match Array.find_opt (fun h -> h.h_kind = sht_symtab) headers with
    | Some h -> Array.of_list (symbols_of h)
    | None -> [||]
  in
  let relocations_of h =
    let rela = h.h_kind = sht_rela in
    let entsize = Int.max (if rela then 24 else 16) h.h_entsize in
    List.init
      (String.length h.h_contents / entsize)
      (fun k ->
        let e = k * entsize in
        let get off = String.get_int64_le h.h_contents (e + off) in
        let info = get 8 in
        let sym = Int64.to_int (Int64.shift_right_logical info 32) in
        if sym >= Array.length table then fail "a relocation's symbol %d" sym;
        let s = table.(sym) in
        {
          at = resolve h.h_info (Int64.to_int (get 0));
          target =
            (if s.s_shndx = 0 then External s.s_name
             else Offset (resolve s.s_shndx s.s_value));
          kind = Int64.to_int (Int64.logand info 0xffff_ffffL);
          addend = (if rela then Int64.to_int (get 16) else 0);
        })
  in
  let relocations =
    Array.to_list headers
    |> List.concat_map (fun h ->
        let target =
          if h.h_info > 0 && h.h_info < shnum then headers.(h.h_info).h_name
          else ""
        in
        if (h.h_kind = sht_rel || h.h_kind = sht_rela) && target <> ".eh_frame"
        then relocations_of h
        else [])
  in
  let sections =
    Array.to_list
      (Array.mapi
         (fun i h ->
           let offset =
             match appended.(i) with Some at -> at | None -> h.h_addr
           in
           {
             name = h.h_name;
             kind = h.h_kind;
             flags = h.h_flags;
             offset;
             size = h.h_size;
             contents = h.h_contents;
           })
         headers)
  in
  {
    kind = u16 16;
    machine = u16 18;
    abi_version = Char.code obj.[8];
    flags = u32 48;
    image = Bytes.to_string image;
    sections;
    symbols =
      Array.map
        (fun s ->
          let place =
            if s.s_shndx = 0 || s.s_shndx >= shnum || s.s_shndx >= 0xff00 then
              Undefined
            else
              Defined
                { section = s.s_shndx; offset = resolve s.s_shndx s.s_value }
          in
          { name = s.s_name; place })
        table;
    relocations;
  }

let symbol o name =
  Array.find_map
    (fun s ->
      match s.place with
      | Defined { offset; _ } when s.name = name && name <> "" -> Some offset
      | _ -> None)
    o.symbols
