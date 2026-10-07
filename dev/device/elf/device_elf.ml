(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type place =
  | Undefined
  | Absolute of int
  | Image of { section : int; offset : int }
  | Outside of int

type symbol = { name : string; place : place }

type section = {
  name : string;
  kind : int;
  flags : int;
  offset : int option;
  size : int;
  contents : string;
}

type relocation = { offset : int; kind : int; addend : int; symbol : symbol }

type t = {
  kind : int;
  machine : int;
  os_abi : int;
  abi_version : int;
  flags : int;
  image : string;
  sections : section iarray;
  symbols : symbol iarray;
  relocations : relocation list;
}

(* The format's constants, from the System V gABI. *)

let ehdr_size = 64
let shdr_size = 64
let sym_size = 24
let rel_size = 16
let rela_size = 24
let elfclass64 = 2
let elfdata2lsb = 1
let sht_null = 0
let sht_progbits = 1
let sht_symtab = 2
let sht_strtab = 3
let sht_rela = 4
let sht_nobits = 8
let sht_rel = 9
let sht_dynsym = 11
let sht_symtab_shndx = 18
let shf_alloc = 0x2
let shn_undef = 0
let shn_loreserve = 0xff00
let shn_abs = 0xfff1
let shn_xindex = 0xffff

(* Errors. A malformed object raises [Malformed] anywhere in the reading;
   [of_string] turns it into [Error]. *)

exception Malformed of string

let fail fmt = Printf.ksprintf (fun m -> raise (Malformed m)) fmt

(* Little-endian reads of [s] at [off], refused past its end. *)

let check s off n =
  if off < 0 || off > String.length s - n then
    fail "the object is truncated: %d bytes at %d lie past its end" n off

let u8 s off =
  check s off 1;
  Char.code s.[off]

let u16 s off =
  check s off 2;
  String.get_uint16_le s off

let u32 s off =
  check s off 4;
  Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

(* A signed 64-bit field, such as an addend, modulo the integers. *)
let i64 s off =
  check s off 8;
  Int64.to_int (String.get_int64_le s off)

(* An unsigned 64-bit field: an offset, size, address or flags. *)
let u64 s off =
  check s off 8;
  let v = String.get_int64_le s off in
  if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
    fail "the 64-bit field %Lu at %d is out of range" v off;
  Int64.to_int v

(* Section headers *)

type header = {
  name_at : int;
  h_kind : int;
  h_flags : int;
  addr : int;
  at : int;
  h_size : int;
  link : int;
  info : int;
  align : int;
  entsize : int;
}

let header obj off =
  {
    name_at = u32 obj off;
    h_kind = u32 obj (off + 4);
    h_flags = u64 obj (off + 8);
    addr = u64 obj (off + 16);
    at = u64 obj (off + 24);
    h_size = u64 obj (off + 32);
    link = u32 obj (off + 40);
    info = u32 obj (off + 44);
    align = u64 obj (off + 48);
    entsize = u64 obj (off + 56);
  }

let has_bytes h = h.h_kind <> sht_null && h.h_kind <> sht_nobits
let held h = h.h_kind = sht_progbits && h.h_flags land shf_alloc <> 0
let is_pow2 n = n > 0 && n land (n - 1) = 0

(* The headers of [obj], with the count and the names' index extended through
   section 0 when they do not fit the ELF header. *)
let headers obj =
  let shoff = u64 obj 40 and entsize = u16 obj 58 in
  let shnum = u16 obj 60 and shstrndx = u16 obj 62 in
  if shoff = 0 then ([||], shn_undef)
  else begin
    if entsize < shdr_size then fail "section headers of %d bytes" entsize;
    let first = header obj shoff in
    let count = if shnum = 0 then first.h_size else shnum in
    let names = if shstrndx = shn_xindex then first.link else shstrndx in
    if count > (String.length obj - shoff) / entsize then
      fail "the object is truncated: %d section headers lie past its end" count;
    (Array.init count (fun i -> header obj (shoff + (i * entsize))), names)
  end

(* The string at [off] of the string table [tab]. *)
let string_at tab off =
  if off = 0 && tab = "" then ""
  else begin
    if off < 0 || off >= String.length tab then
      fail "a name at %d lies past its string table" off;
    match String.index_from_opt tab off '\000' with
    | Some e -> String.sub tab off (e - off)
    | None -> fail "the name at %d is unterminated" off
  end

(* Layout *)

let too_long () =
  fail "the image would be longer than %d bytes" Sys.max_string_length

(* [n] rounded up to a multiple of [a], an image offset. *)
let round_up n a =
  if n mod a = 0 then n
  else if a > Sys.max_string_length then too_long ()
  else n + a - (n mod a)

(* Each held section's image offset, and the image's length. Sections go at
   their addresses if one has an address, else follow each other. *)
let layout ~align hs =
  let addressed = Array.exists (fun h -> held h && h.addr <> 0) hs in
  let length = ref 0 and spans = ref [] in
  let place h =
    if not (held h) then None
    else
      let off =
        if addressed then h.addr else round_up !length (Int.max align h.align)
      in
      if off > Sys.max_string_length - h.h_size then too_long ();
      length := Int.max !length (off + h.h_size);
      if h.h_size > 0 then spans := (off, h.h_size) :: !spans;
      Some off
  in
  let offsets = Array.map place hs in
  (* Sorted by offset, a section with bytes starts at or past the end of the one
     before it. *)
  let rec disjoint last = function
    | [] -> ()
    | (off, n) :: rest ->
        if off < last then fail "two sections overlap at image offset %d" off;
        disjoint (off + n) rest
  in
  disjoint 0 (List.sort compare !spans);
  (offsets, !length)

(* Reading *)

let read ~align obj =
  if
    String.length obj < ehdr_size
    || not (String.starts_with ~prefix:"\x7fELF" obj)
  then fail "not an ELF object";
  if u8 obj 4 <> elfclass64 || u8 obj 5 <> elfdata2lsb then
    fail "not a 64-bit little-endian ELF object";
  let hs, names_index = headers obj in
  let count = Array.length hs in
  let section_of i what =
    if i < 0 || i >= count then
      fail "%s refers to section %d, which the object does not have" what i;
    hs.(i)
  in
  Array.iter
    (fun h ->
      if h.align <> 0 && not (is_pow2 h.align) then
        fail "a section's alignment of %d is not a power of two" h.align)
    hs;
  let contents =
    Array.map
      (fun h ->
        if not (has_bytes h) then ""
        else begin
          check obj h.at h.h_size;
          String.sub obj h.at h.h_size
        end)
      hs
  in
  let strings i what =
    let h = section_of i what in
    if h.h_kind <> sht_strtab then
      fail "%s links to section %d, not a string table" what i;
    contents.(i)
  in
  let names =
    if names_index = shn_undef then None
    else Some (strings names_index "the ELF header")
  in
  let offsets, length = layout ~align hs in
  let image = Bytes.make length '\000' in
  Array.iteri
    (fun i h ->
      Option.iter
        (fun off -> Bytes.blit_string obj h.at image off h.h_size)
        offsets.(i))
    hs;
  (* The place of a symbol in section [index] at [value]. *)
  let place index value =
    let h = section_of index "a symbol" in
    match offsets.(index) with
    | Some off when value - h.addr >= 0 && value - h.addr <= h.h_size ->
        Image { section = index; offset = off + value - h.addr }
    | _ -> Outside index
  in
  (* The symbol table of section [i], and the extended indexes of its symbols at
     [SHN_XINDEX], from the [SHT_SYMTAB_SHNDX] section that links to it. *)
  let table i =
    let h = hs.(i) and syms = contents.(i) in
    let names = strings h.link "a symbol table" in
    let entsize = Int.max sym_size h.entsize in
    let extended k =
      match
        Array.find_index (fun x -> x.h_kind = sht_symtab_shndx && x.link = i) hs
      with
      | Some x -> u32 contents.(x) (4 * k)
      | None ->
          fail "symbol %d has an extended section index and no table of them" k
    in
    Iarray.init
      (String.length syms / entsize)
      (fun k ->
        let e = k * entsize in
        let name = string_at names (u32 syms e) in
        let index = u16 syms (e + 6) and value = i64 syms (e + 8) in
        let place =
          if index = shn_undef then Undefined
          else if index = shn_abs then Absolute value
          else if index = shn_xindex then place (extended k) value
          else if index >= shn_loreserve then Outside index
          else place index value
        in
        { name; place })
  in
  let tables = Array.init count (fun i -> lazy (table i)) in
  let symbols =
    let find kind = Array.find_index (fun h -> h.h_kind = kind) hs in
    match (find sht_symtab, find sht_dynsym) with
    | Some i, _ | None, Some i -> Lazy.force tables.(i)
    | None, None -> Iarray.of_list []
  in
  (* The relocations of section [r], which patch the image at [at]'s results. *)
  let entries r at =
    let h = hs.(r) and rels = contents.(r) in
    let rela = h.h_kind = sht_rela in
    let entsize = Int.max (if rela then rela_size else rel_size) h.entsize in
    let symbol sym =
      if sym = 0 then { name = ""; place = Absolute 0 }
      else begin
        let link = section_of h.link "a relocation section" in
        if link.h_kind <> sht_symtab && link.h_kind <> sht_dynsym then
          fail "a relocation section links to section %d, not a symbol table"
            h.link;
        let syms = Lazy.force tables.(h.link) in
        if sym >= Iarray.length syms then
          fail "a relocation refers to symbol %d, which its table lacks" sym;
        Iarray.get syms sym
      end
    in
    List.init
      (String.length rels / entsize)
      (fun k ->
        let e = k * entsize in
        let info = String.get_int64_le rels (e + 8) in
        {
          offset = at (u64 rels e);
          kind = Int64.to_int (Int64.logand info 0xffff_ffffL);
          addend = (if rela then i64 rels (e + 16) else 0);
          symbol = symbol (Int64.to_int (Int64.shift_right_logical info 32));
        })
  in
  (* A dynamic relocation's offset is an address; any other's lies in the
     section it patches. *)
  let relocations_of r h =
    if h.h_kind <> sht_rel && h.h_kind <> sht_rela then []
    else if h.info = 0 then
      entries r (fun a ->
          if a >= length then
            fail "a relocation patches address %d, past the image" a;
          a)
    else
      let target = section_of h.info "a relocation section" in
      if target.h_flags land shf_alloc = 0 then []
      else
        match offsets.(h.info) with
        | None ->
            fail
              "a relocation patches section %d, which the image does not hold"
              h.info
        | Some off ->
            entries r (fun a ->
                let o = a - target.addr in
                if o < 0 || o >= target.h_size then
                  fail
                    "a relocation's offset %d lies past the end of section %d" a
                    h.info;
                off + o)
  in
  let section i h =
    {
      name = Option.fold ~none:"" ~some:(fun n -> string_at n h.name_at) names;
      kind = h.h_kind;
      flags = h.h_flags;
      offset = offsets.(i);
      size = h.h_size;
      contents = contents.(i);
    }
  in
  {
    kind = u16 obj 16;
    machine = u16 obj 18;
    os_abi = u8 obj 7;
    abi_version = u8 obj 8;
    flags = u32 obj 48;
    image = Bytes.unsafe_to_string image;
    sections = Iarray.of_array (Array.mapi section hs);
    symbols;
    relocations = List.concat (List.mapi relocations_of (Array.to_list hs));
  }

let of_string ?(align = 1) obj =
  if not (is_pow2 align) then
    invalid_arg
      (Printf.sprintf
         "Device_elf.of_string: align %d is not a positive power of two" align);
  match read ~align obj with o -> Ok o | exception Malformed m -> Error m

let symbol o name =
  if name = "" then None
  else
    Iarray.find_map
      (fun (s : symbol) ->
        match s.place with
        | Image { offset; _ } when s.name = name -> Some offset
        | _ -> None)
      o.symbols
