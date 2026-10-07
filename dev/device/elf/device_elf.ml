(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

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
  at : int;
  length : int;
}

type relocation = { offset : int; kind : int; addend : int; symbol : symbol }

type t = {
  kind : int;
  machine : int;
  os_abi : int;
  abi_version : int;
  flags : int;
  address : int;
  file : string;
  size : int;
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
let shf_tls = 0x400
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

(* Whether a section takes memory of its own: it is allocated, and not a
   thread-local section without bytes ([.tbss]), a template each thread copies,
   whose addresses the next section takes. *)
let takes_memory kind flags =
  flags land shf_alloc <> 0 && not (kind = sht_nobits && flags land shf_tls <> 0)

(* The sections of code and data a loader's memory holds: program sections and
   sections without bytes that take memory. *)
let allocated_as kind flags =
  (kind = sht_progbits || kind = sht_nobits) && takes_memory kind flags

let is_pow2 n = n > 0 && n land (n - 1) = 0

(* The headers of [obj], with the count and the names' index extended through
   section 0 when they do not fit the ELF header. *)
let headers f obj =
  let shoff = word f obj f.e_shoff and entsize = u16 obj f.e_shentsize in
  let shnum = u16 obj (f.e_shentsize + 2) in
  let shstrndx = u16 obj (f.e_shentsize + 4) in
  if shoff = 0 then ([||], shn_undef)
  else begin
    if entsize < f.shdr then
      fail "section headers are %d bytes, expected at least %d" entsize f.shdr;
    let first = header f obj shoff in
    let count = if shnum = 0 then first.sh_size else shnum in
    let names = if shstrndx = shn_xindex then first.sh_link else shstrndx in
    if count > (String.length obj - shoff) / entsize then
      fail "the object is truncated: %d section headers lie past its end" count;
    (Array.init count (fun i -> header f obj (shoff + (i * entsize))), names)
  end

(* Where a section's bytes start in the object, and how many it has there. *)
let at_of h = if has_bytes h then h.sh_offset else 0
let length_of h = if has_bytes h then h.sh_size else 0

(* The string at [off] of the string table [tab], a header of [obj]. [named]
   counts the bytes copied into names so far, which may not exceed the object's
   length: names may share bytes of their table, and copying each would
   otherwise take memory quadratic in the object. *)
let string_at obj named tab off =
  let first = at_of tab and n = length_of tab in
  if off = 0 && n = 0 then ""
  else begin
    if off < 0 || off >= n then
      fail "a name at %d lies past its string table" off;
    match String.index_from_opt obj (first + off) '\000' with
    | Some e when e < first + n ->
        let len = e - first - off in
        named := !named + len;
        if !named > String.length obj then
          raise (Malformed "the names are longer than the object");
        String.sub obj (first + off) len
    | _ -> fail "the name at %d is unterminated" off
  end

let by_start (a, _) (b, _) = Int.compare a b

(* Sorted by start, each span [(at, n)] with [n > 0] starts at or past the end
   of the one before it. Spans usually come sorted, and are sorted otherwise
   with a merge sort, whose memory is linear. *)
let disjoint ~where spans =
  let rec sorted i =
    i >= Array.length spans
    || (fst spans.(i - 1) <= fst spans.(i) && sorted (i + 1))
  in
  if not (sorted 1) then Array.stable_sort by_start spans;
  ignore
    (Array.fold_left
       (fun last (at, n) ->
         if n = 0 then last
         else if at < last then
           fail "two sections share bytes at %d of %s" at where
         else at + n)
       0 spans)

(* Layout *)

let too_long () = fail "the image would be longer than %d bytes" max_int

(* [n] rounded up to a multiple of [a], an image offset. *)
let round_up n a =
  if n mod a = 0 then n
  else if n > max_int - a then too_long ()
  else n + a - (n mod a)

(* The address of image offset 0 when sections go at their addresses: the lowest
   held section's, rounded down to their largest alignment, so that each keeps
   its alignment in the image. *)
let start held hs =
  let low = ref max_int and align = ref 1 in
  Array.iteri
    (fun i h ->
      if held.(i) then begin
        low := Int.min !low h.sh_addr;
        align := Int.max !align h.sh_addralign
      end)
    hs;
  !low - (!low land (!align - 1))

(* Whether the sections go at their addresses, the address the image starts at,
   each held section's image offset, and the image's length. Sections go at
   their addresses if one has an address, else follow each other. *)
let layout ~align held hs =
  let addressed =
    Array.exists2 (fun held h -> held && h.sh_addr <> 0) held hs
  in
  let address = if addressed then start held hs else 0 in
  let offsets = Array.make (Array.length hs) None in
  let size = ref 0 and spans = ref [] in
  Array.iteri
    (fun i h ->
      if held.(i) then begin
        let off =
          if addressed then h.sh_addr - address
          else round_up !size (Int.max align h.sh_addralign)
        in
        if off > max_int - h.sh_size then too_long ();
        if h.sh_addralign > 1 && off mod h.sh_addralign <> 0 then
          fail "section %d at %d is not a multiple of its alignment %d" i off
            h.sh_addralign;
        offsets.(i) <- Some off;
        size := Int.max !size (off + h.sh_size);
        if h.sh_size > 0 then spans := (off, h.sh_size) :: !spans
      end)
    hs;
  disjoint ~where:"the image" (Array.of_list (List.rev !spans));
  (addressed, address, offsets, !size)

(* [at + n], or [max_int] past it: the end of an address range. *)
let end_of at n = if at > max_int - n then max_int else at + n

(* The sections taking memory that the image does not hold, as address ranges
   sorted and merged into disjoint [(start, end)] pairs, for a binary search. *)
let lacking held hs =
  let spans = ref [] in
  Array.iteri
    (fun i h ->
      if takes_memory h.sh_type h.sh_flags && (not held.(i)) && h.sh_size > 0
      then spans := (h.sh_addr, end_of h.sh_addr h.sh_size) :: !spans)
    hs;
  let spans = Array.of_list !spans in
  Array.stable_sort by_start spans;
  let merge acc (s, e) =
    match acc with
    | (s', e') :: rest when s <= e' -> (s', Int.max e e') :: rest
    | _ -> (s, e) :: acc
  in
  Array.of_list (List.rev (Array.fold_left merge [] spans))

(* Whether one of [ranges], disjoint and sorted, holds [a]. *)
let holds ranges a =
  let rec search lo hi =
    if lo >= hi then false
    else
      let mid = (lo + hi) / 2 in
      let s, e = ranges.(mid) in
      if a < s then search lo mid
      else if a >= e then search (mid + 1) hi
      else true
  in
  search 0 (Array.length ranges)

(* Reading *)

let read ~align ?held obj =
  if
    String.length obj < ei_nident
    || not (String.starts_with ~prefix:"\x7fELF" obj)
  then fail "not an ELF object";
  let f =
    match u8 obj 4 with
    | c when c = elfclass64 -> elf64
    | c when c = elfclass32 -> elf32
    | c -> fail "unknown ELF class %d, expected 1 or 2" c
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
        fail "a section's alignment of %d is not a power of two" h.sh_addralign;
      if has_bytes h then check obj h.sh_offset h.sh_size)
    hs;
  (* The gABI puts no byte of the object in two sections. Holding to it keeps
     the entries of tables that would share bytes from being decoded twice. *)
  disjoint ~where:"the object"
    (Array.map (fun h -> (h.sh_offset, length_of h)) hs);
  let named = ref 0 in
  let strings i what =
    let h = section_of i what in
    if h.sh_type <> sht_strtab then
      fail "%s links to section %d, which is no string table" what i;
    h
  in
  let names =
    if names_index = shn_undef then None
    else Some (strings names_index "the ELF header")
  in
  let section h offset =
    {
      name =
        (match names with
        | None -> ""
        | Some n -> string_at obj named n h.sh_name);
      kind = h.sh_type;
      flags = h.sh_flags;
      offset;
      size = h.sh_size;
      at = at_of h;
      length = length_of h;
    }
  in
  (* The caller's predicate sees each section before the layout, without an
     offset; its names are read once. *)
  let unplaced, kept =
    match held with
    | None -> (None, Array.map (fun h -> allocated_as h.sh_type h.sh_flags) hs)
    | Some held ->
        let unplaced = Array.map (fun h -> section h None) hs in
        (Some unplaced, Array.map held unplaced)
  in
  let addressed, address, offsets, size = layout ~align kept hs in
  (* The place of a symbol in section [index] at [value]. *)
  let place index value =
    let h = section_of index "a symbol" in
    match offsets.(index) with
    | Some off when value - h.sh_addr >= 0 && value - h.sh_addr <= h.sh_size ->
        Image { section = index; offset = off + value - h.sh_addr }
    | _ -> Outside index
  in
  (* Each symbol table's [SHT_SYMTAB_SHNDX] section, the first linked to it. *)
  let shndx = Array.make count None in
  Array.iter
    (fun h ->
      if
        h.sh_type = sht_symtab_shndx
        && h.sh_link < count
        && Option.is_none shndx.(h.sh_link)
      then shndx.(h.sh_link) <- Some h)
    hs;
  (* The symbol table of section [i], with the extended indexes of its symbols
     at [SHN_XINDEX]. *)
  let read_table i =
    let h = hs.(i) in
    let names = strings h.sh_link "a symbol table" in
    let entsize = Int.max f.sym h.sh_entsize in
    let extended k =
      match shndx.(i) with
      | Some x when 4 * (k + 1) <= length_of x -> u32 obj (at_of x + (4 * k))
      | _ -> fail "symbol %d has an extended section index the object lacks" k
    in
    Iarray.init
      (length_of h / entsize)
      (fun k ->
        let e = at_of h + (k * entsize) in
        let name = string_at obj named names (u32 obj e) in
        let index = u16 obj (e + f.st_shndx) in
        let value =
          if f.word = 8 then i64 obj (e + f.st_value)
          else u32 obj (e + f.st_value)
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
    let h = hs.(r) in
    let rela = h.sh_type = sht_rela in
    let entsize = Int.max ((if rela then 3 else 2) * f.word) h.sh_entsize in
    let symbol sym =
      if sym = 0 then { name = ""; place = Absolute 0 }
      else begin
        let link = section_of h.sh_link "a relocation section" in
        if link.sh_type <> sht_symtab && link.sh_type <> sht_dynsym then
          fail
            "a relocation section links to section %d, which is no symbol table"
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
      if f.word = 8 then u32 obj (e + 8) else u32 obj (e + 4) land 0xff
    in
    let r_sym e =
      if f.word = 8 then u32 obj (e + 12) else u32 obj (e + 4) lsr 8
    in
    let r_addend e =
      if not rela then 0
      else if f.word = 8 then i64 obj (e + 16)
      else i32 obj (e + 8)
    in
    List.init
      (length_of h / entsize)
      (fun k ->
        let e = at_of h + (k * entsize) in
        {
          offset = at (word f obj e);
          kind = r_type e;
          addend = r_addend e;
          symbol = symbol (r_sym e);
        })
  in
  (* The allocated memory the image lacks, as addresses, which mean something in
     an object whose sections have addresses. *)
  let lacks = lazy (if addressed then lacking kept hs else [||]) in
  (* A dynamic relocation's offset is an address; any other's lies in the
     section it patches. *)
  let relocations_of r h =
    if h.sh_type <> sht_rel && h.sh_type <> sht_rela then []
    else if h.sh_info = 0 then
      let lacks = Lazy.force lacks in
      entries r (fun a ->
          if a < address || a - address >= size || holds lacks a then
            fail
              "a relocation patches address %d, which the image does not hold" a;
          a - address)
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
  let sections =
    match unplaced with
    | None -> Array.mapi (fun i h -> section h offsets.(i)) hs
    | Some unplaced ->
        (* Placed in place: a section the image does not hold keeps its
           record. *)
        Array.iteri
          (fun i (s : section) ->
            match offsets.(i) with
            | None -> ()
            | offset -> unplaced.(i) <- { s with offset })
          unplaced;
        unplaced
  in
  {
    kind = u16 obj 16;
    machine = u16 obj 18;
    os_abi = u8 obj 7;
    abi_version = u8 obj 8;
    flags = u32 obj f.e_flags;
    address;
    file = obj;
    size;
    sections = Iarray.of_array sections;
    symbols;
    relocations = List.concat (List.mapi relocations_of (Array.to_list hs));
  }

let allocated (s : section) = allocated_as s.kind s.flags

let of_string ?(align = 1) ?held obj =
  if not (is_pow2 align) then
    invalid_argf "Device_elf.of_string: align %d is not a positive power of two"
      align;
  match read ~align ?held obj with
  | o -> Ok o
  | exception Malformed m -> Error m

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
