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

let ei_nident = 16
let elfclass32 = 1
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

(* The two classes differ in the width of a {e word}, an address, offset or
   size, and in where the ELF header and a symbol put their fields. A section
   header has the same fields in the same order in both. *)
type form = {
  word : int;
  ehdr : int;
  e_shoff : int;
  e_flags : int;
  e_shentsize : int;
  shdr : int;
  sym : int;
  st_value : int;
  st_shndx : int;
}

let elf32 =
  {
    word = 4;
    ehdr = 52;
    e_shoff = 32;
    e_flags = 36;
    e_shentsize = 46;
    shdr = 40;
    sym = 16;
    st_value = 4;
    st_shndx = 14;
  }

let elf64 =
  {
    word = 8;
    ehdr = 64;
    e_shoff = 40;
    e_flags = 48;
    e_shentsize = 58;
    shdr = 64;
    sym = 24;
    st_value = 8;
    st_shndx = 6;
  }

(* Errors. A malformed object raises [Malformed] anywhere in the reading;
   [of_string] turns it into [Error]. *)

exception Malformed of string

let fail fmt = Printf.ksprintf (fun m -> raise (Malformed m)) fmt

(* Little-endian reads of [s] at [off], refused past its end. They run for every
   field of the object and are inlined, which saves a call per field. *)

let[@inline] check s off n =
  if off < 0 || off > String.length s - n then
    fail "the object is truncated: %d bytes at %d lie past its end" n off

let[@inline] u8 s off =
  check s off 1;
  Char.code s.[off]

let[@inline] u16 s off =
  check s off 2;
  String.get_uint16_le s off

let[@inline] u32 s off =
  check s off 4;
  Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let[@inline] i32 s off =
  check s off 4;
  Int32.to_int (String.get_int32_le s off)

(* A 64-bit field that does not fit an [int] is refused. A signed one is an
   addend or a symbol's value. *)
let[@inline] i64 s off =
  check s off 8;
  let v = String.get_int64_le s off in
  let n = Int64.to_int v in
  if Int64.of_int n <> v then
    fail "the 64-bit field %Ld at %d is out of range" v off;
  n

(* An unsigned one is an offset, size, address or flags. *)
let[@inline] u64 s off =
  check s off 8;
  let v = String.get_int64_le s off in
  if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
    fail "the 64-bit field %Lu at %d is out of range" v off;
  Int64.to_int v

(* An unsigned word of the form [f]. *)
let[@inline] word f s off = if f.word = 8 then u64 s off else u32 s off

(* Section headers *)

(* A section header, its fields named as in the format. *)
type header = {
  sh_name : int;
  sh_type : int;
  sh_flags : int;
  sh_addr : int;
  sh_offset : int;
  sh_size : int;
  sh_link : int;
  sh_info : int;
  sh_addralign : int;
  sh_entsize : int;
}

(* Past [sh_name] and [sh_type], a header holds four words, [sh_link] and
   [sh_info], then two words. *)
let header f obj off =
  let w k = off + 8 + (k * f.word) in
  {
    sh_name = u32 obj off;
    sh_type = u32 obj (off + 4);
    sh_flags = word f obj (w 0);
    sh_addr = word f obj (w 1);
    sh_offset = word f obj (w 2);
    sh_size = word f obj (w 3);
    sh_link = u32 obj (w 4);
    sh_info = u32 obj (w 4 + 4);
    sh_addralign = word f obj (w 4 + 8);
    sh_entsize = word f obj (w 5 + 8);
  }

(* The null section's size is the section count when [e_shnum] is 0, so it has
   no bytes whatever its size. *)
let has_bytes h = h.sh_type <> sht_null && h.sh_type <> sht_nobits
let held h = h.sh_type = sht_progbits && h.sh_flags land shf_alloc <> 0
let is_pow2 n = n > 0 && n land (n - 1) = 0

(* The headers of [obj], with the count and the names' index extended through
   section 0 when they do not fit the ELF header. *)
let headers f obj =
  let shoff = word f obj f.e_shoff and entsize = u16 obj f.e_shentsize in
  let shnum = u16 obj (f.e_shentsize + 2) in
  let shstrndx = u16 obj (f.e_shentsize + 4) in
  if shoff = 0 then ([||], shn_undef)
  else begin
    if entsize < f.shdr then fail "section headers of %d bytes" entsize;
    let first = header f obj shoff in
    let count = if shnum = 0 then first.sh_size else shnum in
    let names = if shstrndx = shn_xindex then first.sh_link else shstrndx in
    if count > (String.length obj - shoff) / entsize then
      fail "the object is truncated: %d section headers lie past its end" count;
    (Array.init count (fun i -> header f obj (shoff + (i * entsize))), names)
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

(* The longest image read, 1 GiB. Real images are at most tens of megabytes, so
   a longer one comes from a corrupted address or alignment, and refusing it
   keeps a small object from making the reader allocate gigabytes. *)
let max_image = 1 lsl 30
let too_long () = fail "the image would be longer than %d bytes" max_image

(* [n] rounded up to a multiple of [a], an image offset. *)
let round_up n a =
  if n mod a = 0 then n
  else if a > max_image then too_long ()
  else n + a - (n mod a)

(* Whether the sections go at their addresses, each held section's image offset,
   and the image's length. Sections go at their addresses if one has an address,
   else follow each other. *)
let layout ~align hs =
  let addressed = Array.exists (fun h -> held h && h.sh_addr <> 0) hs in
  let offsets = Array.make (Array.length hs) None in
  let length = ref 0 and spans = ref [] in
  Array.iteri
    (fun i h ->
      if held h then begin
        let off =
          if addressed then h.sh_addr
          else round_up !length (Int.max align h.sh_addralign)
        in
        if off > max_image - h.sh_size then too_long ();
        if h.sh_addralign > 1 && off mod h.sh_addralign <> 0 then
          fail "section %d at %d is not a multiple of its alignment %d" i off
            h.sh_addralign;
        offsets.(i) <- Some off;
        length := Int.max !length (off + h.sh_size);
        if h.sh_size > 0 then spans := (off, h.sh_size) :: !spans
      end)
    hs;
  (* Sorted by offset, a section with bytes starts at or past the end of the one
     before it. *)
  let rec disjoint last = function
    | [] -> ()
    | (off, n) :: rest ->
        if off < last then fail "two sections overlap at image offset %d" off;
        disjoint (off + n) rest
  in
  disjoint 0 (List.sort compare !spans);
  (addressed, offsets, !length)

(* Reading *)

let read ~align obj =
  if
    String.length obj < ei_nident
    || not (String.starts_with ~prefix:"\x7fELF" obj)
  then fail "not an ELF object";
  let f =
    match u8 obj 4 with
    | c when c = elfclass64 -> elf64
    | c when c = elfclass32 -> elf32
    | c -> fail "an ELF object of unknown class %d" c
  in
  if u8 obj 5 <> elfdata2lsb then fail "not a little-endian ELF object";
  if String.length obj < f.ehdr then
    fail "the object is truncated: its ELF header lies past its end";
  let hs, names_index = headers f obj in
  let count = Array.length hs in
  let section_of i what =
    if i < 0 || i >= count then
      fail "%s refers to section %d, which the object does not have" what i;
    hs.(i)
  in
  Array.iter
    (fun h ->
      if h.sh_addralign <> 0 && not (is_pow2 h.sh_addralign) then
        fail "a section's alignment of %d is not a power of two" h.sh_addralign)
    hs;
  let contents =
    Array.map
      (fun h ->
        if not (has_bytes h) then ""
        else begin
          check obj h.sh_offset h.sh_size;
          String.sub obj h.sh_offset h.sh_size
        end)
      hs
  in
  let strings i what =
    let h = section_of i what in
    if h.sh_type <> sht_strtab then
      fail "%s links to section %d, not a string table" what i;
    contents.(i)
  in
  let names =
    if names_index = shn_undef then None
    else Some (strings names_index "the ELF header")
  in
  let addressed, offsets, length = layout ~align hs in
  let image = Bytes.make length '\000' in
  Array.iteri
    (fun i h ->
      Option.iter
        (fun off -> Bytes.blit_string obj h.sh_offset image off h.sh_size)
        offsets.(i))
    hs;
  (* The place of a symbol in section [index] at [value]. *)
  let place index value =
    let h = section_of index "a symbol" in
    match offsets.(index) with
    | Some off when value - h.sh_addr >= 0 && value - h.sh_addr <= h.sh_size ->
        Image { section = index; offset = off + value - h.sh_addr }
    | _ -> Outside index
  in
  (* The symbol table of section [i], and the extended indexes of its symbols at
     [SHN_XINDEX], from the [SHT_SYMTAB_SHNDX] section that links to it. *)
  let read_table i =
    let h = hs.(i) and syms = contents.(i) in
    let names = strings h.sh_link "a symbol table" in
    let entsize = Int.max f.sym h.sh_entsize in
    let shndx =
      Array.find_index
        (fun x -> x.sh_type = sht_symtab_shndx && x.sh_link = i)
        hs
    in
    let extended k =
      match shndx with
      | Some x when 4 * (k + 1) <= String.length contents.(x) ->
          u32 contents.(x) (4 * k)
      | _ -> fail "symbol %d has an extended section index the object lacks" k
    in
    Iarray.init
      (String.length syms / entsize)
      (fun k ->
        let e = k * entsize in
        let name = string_at names (u32 syms e) in
        let index = u16 syms (e + f.st_shndx) in
        let value =
          if f.word = 8 then i64 syms (e + f.st_value)
          else u32 syms (e + f.st_value)
        in
        let place =
          if index = shn_undef then Undefined
          else if index = shn_abs then Absolute value
          else if index = shn_xindex then place (extended k) value
          else if index >= shn_loreserve then Outside index
          else place index value
        in
        { name; place })
  in
  let tables = Array.make count None in
  let table i =
    match tables.(i) with
    | Some t -> t
    | None ->
        let t = read_table i in
        tables.(i) <- Some t;
        t
  in
  let symbols =
    let find kind = Array.find_index (fun h -> h.sh_type = kind) hs in
    match (find sht_symtab, find sht_dynsym) with
    | Some i, _ | None, Some i -> table i
    | None, None -> Iarray.of_list []
  in
  (* The relocations of section [r], which patch the image at [at]'s results. *)
  let entries r at =
    let h = hs.(r) and rels = contents.(r) in
    let rela = h.sh_type = sht_rela in
    let entsize = Int.max ((if rela then 3 else 2) * f.word) h.sh_entsize in
    let symbol sym =
      if sym = 0 then { name = ""; place = Absolute 0 }
      else begin
        let link = section_of h.sh_link "a relocation section" in
        if link.sh_type <> sht_symtab && link.sh_type <> sht_dynsym then
          fail "a relocation section links to section %d, not a symbol table"
            h.sh_link;
        let syms = table h.sh_link in
        if sym >= Iarray.length syms then
          fail "a relocation refers to symbol %d, which its table lacks" sym;
        Iarray.get syms sym
      end
    in
    (* [r_info] holds the type in its low 32 bits and the symbol in its high
       ones, or in 32 bits the type in its low 8 bits and the symbol above. *)
    let r_type e =
      if f.word = 8 then u32 rels (e + 8) else u32 rels (e + 4) land 0xff
    in
    let r_sym e =
      if f.word = 8 then u32 rels (e + 12) else u32 rels (e + 4) lsr 8
    in
    let r_addend e =
      if not rela then 0
      else if f.word = 8 then i64 rels (e + 16)
      else i32 rels (e + 8)
    in
    List.init
      (String.length rels / entsize)
      (fun k ->
        let e = k * entsize in
        {
          offset = at (word f rels e);
          kind = r_type e;
          addend = r_addend e;
          symbol = symbol (r_sym e);
        })
  in
  (* The allocated sections the image does not hold, as address ranges, which
     mean something in an object whose sections have addresses. *)
  let lacking () =
    if not addressed then []
    else
      List.filter_map
        (fun h ->
          if h.sh_flags land shf_alloc = 0 || held h then None
          else Some (h.sh_addr, h.sh_size))
        (Array.to_list hs)
  in
  (* A dynamic relocation's offset is an address; any other's lies in the
     section it patches. *)
  let relocations_of r h =
    if h.sh_type <> sht_rel && h.sh_type <> sht_rela then []
    else if h.sh_info = 0 then
      let lacking = lacking () in
      let lacks a =
        List.exists (fun (at, n) -> a >= at && a - at < n) lacking
      in
      entries r (fun a ->
          if a >= length || lacks a then
            fail
              "a relocation patches address %d, which the image does not hold" a;
          a)
    else
      let target = section_of h.sh_info "a relocation section" in
      if target.sh_flags land shf_alloc = 0 then []
      else
        match offsets.(h.sh_info) with
        | None ->
            fail
              "a relocation patches section %d, which the image does not hold"
              h.sh_info
        | Some off ->
            entries r (fun a ->
                let o = a - target.sh_addr in
                if o < 0 || o >= target.sh_size then
                  fail
                    "a relocation's offset %d lies past the end of section %d" a
                    h.sh_info;
                off + o)
  in
  let section i h =
    {
      name = Option.fold ~none:"" ~some:(fun n -> string_at n h.sh_name) names;
      kind = h.sh_type;
      flags = h.sh_flags;
      offset = offsets.(i);
      size = h.sh_size;
      contents = contents.(i);
    }
  in
  {
    kind = u16 obj 16;
    machine = u16 obj 18;
    os_abi = u8 obj 7;
    abi_version = u8 obj 8;
    flags = u32 obj f.e_flags;
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

(* A loop: the AMD loader looks up each kernel it loads. *)
let symbol o name =
  let n = Iarray.length o.symbols in
  let rec find i =
    if i = n then None
    else
      let s : symbol = Iarray.unsafe_get o.symbols i in
      match s.place with
      | Image { offset; _ } when String.equal s.name name -> Some offset
      | _ -> find (i + 1)
  in
  if name = "" then None else find 0
