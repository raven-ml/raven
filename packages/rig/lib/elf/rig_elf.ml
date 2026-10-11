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

type addend = Explicit of int | Implicit of { at : int; length : int }
type relocation = { offset : int; kind : int; addend : addend; symbol : symbol }

type t = {
  bits : int;
  kind : int;
  machine : int;
  os_abi : int;
  abi_version : int;
  flags : int;
  address : int;
  align : int;
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

(* x86-64's, from its psABI. *)

let em_x86_64 = 62
let sht_x86_64_unwind = 0x70000001

(* The two classes differ in the width of a {e word}, an address, offset or
   size, and in where the ELF header and a symbol put their fields. A section
   header has the same fields in the same order in both. *)
type form = {
  word : int;
  e_shoff : int;
  e_flags : int;
  e_shentsize : int;
  sym : int;
  st_value : int;
  st_shndx : int;
}

let elf32 =
  {
    word = 4;
    e_shoff = 32;
    e_flags = 36;
    e_shentsize = 46;
    sym = 16;
    st_value = 4;
    st_shndx = 14;
  }

let elf64 =
  {
    word = 8;
    e_shoff = 40;
    e_flags = 48;
    e_shentsize = 58;
    sym = 24;
    st_value = 8;
    st_shndx = 6;
  }

(* Errors. A malformed object raises [Malformed], or [Invalid_argument] where a
   read past the object's end or a table's meets OCaml's bounds checks;
   [of_string] turns both into [Error]. What the caller's [held] raises
   crosses the reading as [Held], and [of_string] raises it again. *)

exception Malformed of string
exception Held of exn * Printexc.raw_backtrace

let fail fmt = Printf.ksprintf (fun m -> raise (Malformed m)) fmt
let truncated = "the object is truncated, or refers to data it does not have"

(* Little-endian reads of [s] at [off]. *)

let[@inline] u16 s off = String.get_uint16_le s off
let[@inline] u32 s off =
  Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
let[@inline] i32 s off = Int32.to_int (String.get_int32_le s off)

(* A 64-bit field that does not fit an [int] is refused. A signed one is an
   addend or a symbol's value. *)
let[@inline] i64 s off =
  let v = String.get_int64_le s off in
  let n = Int64.to_int v in
  if Int64.of_int n <> v then
    fail "the 64-bit field %Ld at %d is out of range" v off;
  n

(* An unsigned one is an offset, size, address or flags. *)
let[@inline] u64 s off =
  let v = String.get_int64_le s off in
  let n = Int64.to_int v in
  if n < 0 || Int64.of_int n <> v then
    fail "the 64-bit field %Lu at %d is out of range" v off;
  n

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
  let w = f.word in
  {
    sh_name = u32 obj off;
    sh_type = u32 obj (off + 4);
    sh_flags = word f obj (off + 8);
    sh_addr = word f obj (off + 8 + w);
    sh_offset = word f obj (off + 8 + (2 * w));
    sh_size = word f obj (off + 8 + (3 * w));
    sh_link = u32 obj (off + 8 + (4 * w));
    sh_info = u32 obj (off + 12 + (4 * w));
    sh_addralign = word f obj (off + 16 + (4 * w));
    sh_entsize = word f obj (off + 16 + (5 * w));
  }

(* The null section's size is the section count when [e_shnum] is 0, so it has
   no bytes whatever its size. *)
let has_bytes h = h.sh_type <> sht_null && h.sh_type <> sht_nobits

(* Whether a section takes memory of its own: it is allocated, and not a
   thread-local section without bytes ([.tbss]), a template each thread copies,
   whose addresses the next section takes. *)
let takes_memory kind flags =
  flags land shf_alloc <> 0 && not (kind = sht_nobits && flags land shf_tls <> 0)

(* The sections of code and data a loader's memory holds, in an object for
   [machine]: program sections, sections without bytes that take memory, and
   x86-64's unwind tables. *)
let allocated_as machine kind flags =
  (kind = sht_progbits || kind = sht_nobits
  || (machine = em_x86_64 && kind = sht_x86_64_unwind))
  && takes_memory kind flags

let is_pow2 n = n > 0 && n land (n - 1) = 0

(* The headers of [obj], with the count and the names' index extended through
   section 0 when they do not fit the ELF header. The count is checked against
   the object's length before the headers are allocated. *)
let headers f obj =
  let shoff = word f obj f.e_shoff and entsize = u16 obj f.e_shentsize in
  let shnum = u16 obj (f.e_shentsize + 2) in
  let shstrndx = u16 obj (f.e_shentsize + 4) in
  if shoff = 0 then ([||], shn_undef)
  else begin
    let first = header f obj shoff in
    let count = if shnum = 0 then first.sh_size else shnum in
    let names = if shstrndx = shn_xindex then first.sh_link else shstrndx in
    if entsize = 0 || count > (String.length obj - shoff) / entsize then
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
        (* A raise of its own: [fail] here makes every read 13% slower
           (amd_128_gfx1100.hsaco, M1 Max). *)
        if !named > String.length obj then
          raise
            (Malformed
               (Printf.sprintf "the names total more than the object's %d bytes"
                  (String.length obj)));
        String.sub obj (first + off) len
    | _ -> fail "the name at %d is unterminated" off
  end

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
let start low align = low - (low land (align - 1))

(* The address the image starts at, the image's alignment, each held section's
   image offset, and the image's length. Sections go at their addresses if one
   has an address, else follow each other. *)
let layout ~align held hs =
  let count = Array.length hs in
  let addressed = ref false and low = ref max_int and largest = ref 1 in
  for i = 0 to count - 1 do
    if held.(i) then begin
      let h = hs.(i) in
      if h.sh_addr <> 0 then addressed := true;
      low := Int.min !low h.sh_addr;
      largest := Int.max !largest h.sh_addralign
    end
  done;
  let addressed = !addressed in
  let address = if addressed then start !low !largest else 0 in
  let offsets = Array.make count None in
  let size = ref 0 in
  for i = 0 to count - 1 do
    if held.(i) then begin
      let h = hs.(i) in
      let off =
        if addressed then h.sh_addr - address
        else round_up !size (Int.max align h.sh_addralign)
      in
      if off > max_int - h.sh_size then too_long ();
      offsets.(i) <- Some off;
      size := Int.max !size (off + h.sh_size)
    end
  done;
  let image_align = if addressed then !largest else Int.max align !largest in
  (address, image_align, offsets, !size)

(* The held sections with bytes in the object, by image offset: their indexes,
   in which [holding] finds the section of an image offset. *)
let with_bytes offsets hs =
  let idx = ref [] in
  for i = Array.length hs - 1 downto 0 do
    if Option.is_some offsets.(i) && length_of hs.(i) > 0 then idx := i :: !idx
  done;
  let idx = Array.of_list !idx in
  let off i = Option.get offsets.(i) in
  Array.stable_sort (fun i j -> Int.compare (off i) (off j)) idx;
  idx

(* The section of [idx] whose bytes hold image offset [x], or [-1]. *)
let holding offsets hs idx x =
  let lo = ref 0 and hi = ref (Array.length idx) in
  while !lo < !hi do
    let mid = (!lo + !hi) / 2 in
    if Option.get offsets.(idx.(mid)) <= x then lo := mid + 1 else hi := mid
  done;
  if !lo = 0 then -1
  else
    let i = idx.(!lo - 1) in
    if x < Option.get offsets.(i) + length_of hs.(i) then i else -1

(* The implicit addend of a field at image offset [x] of section [h], placed at
   [off], which has bytes there. *)
let implicit h off x =
  let o = x - off in
  Implicit { at = h.sh_offset + o; length = h.sh_size - o }

(* Reading *)

let read ~align ?held obj =
  if
    String.length obj < ei_nident
    || not (String.starts_with ~prefix:"\x7fELF" obj)
  then fail "not an ELF object";
  let f =
    match Char.code obj.[4] with
    | c when c = elfclass64 -> elf64
    | c when c = elfclass32 -> elf32
    | c -> fail "unknown ELF class %d, expected 1 or 2" c
  in
  if Char.code obj.[5] <> elfdata2lsb then
    fail "not a little-endian ELF object";
  let machine = u16 obj 18 in
  let hs, names_index = headers f obj in
  let count = Array.length hs in
  let symtab = ref (-1) and dynsym = ref (-1) in
  for i = count - 1 downto 0 do
    let h = hs.(i) in
    if h.sh_addralign <> 0 && not (is_pow2 h.sh_addralign) then
      fail "a section's alignment of %d is not a power of two" h.sh_addralign;
    if has_bytes h && h.sh_offset > String.length obj - h.sh_size then
      fail "the object is truncated: section %d lies past its end" i;
    if h.sh_type = sht_symtab then symtab := i
    else if h.sh_type = sht_dynsym then dynsym := i
  done;
  let named = ref 0 in
  let names = if names_index = shn_undef then None else Some hs.(names_index) in
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
    | None ->
        (None, Array.map (fun h -> allocated_as machine h.sh_type h.sh_flags) hs)
    | Some held ->
        let unplaced = Array.map (fun h -> section h None) hs in
        let held s =
          try held s
          with e -> raise (Held (e, Printexc.get_raw_backtrace ()))
        in
        (Some unplaced, Array.map held unplaced)
  in
  let address, image_align, offsets, size = layout ~align kept hs in
  (* The place of a symbol in section [index] at [value]. *)
  let place index value =
    let h = hs.(index) in
    match offsets.(index) with
    | Some off when value - h.sh_addr >= 0 && value - h.sh_addr <= h.sh_size ->
        Image { section = index; offset = off + value - h.sh_addr }
    | _ -> Outside index
  in
  (* The symbol table of section [i], with the extended indexes of its symbols
     at [SHN_XINDEX], in the first [SHT_SYMTAB_SHNDX] section linked to it. *)
  let read_table i =
    let h = hs.(i) in
    let names = hs.(h.sh_link) in
    let entsize = Int.max f.sym h.sh_entsize in
    let shndx =
      lazy
        (Array.find_opt
           (fun x -> x.sh_type = sht_symtab_shndx && x.sh_link = i)
           hs)
    in
    let extended k =
      match Lazy.force shndx with
      | Some x when 4 * (k + 1) <= length_of x -> u32 obj (at_of x + (4 * k))
      | _ -> fail "symbol %d has an extended section index the object lacks" k
    in
    let first = at_of h in
    Iarray.init
      (length_of h / entsize)
      (fun k ->
        let e = first + (k * entsize) in
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
    if !symtab >= 0 then table !symtab
    else if !dynsym >= 0 then table !dynsym
    else Iarray.of_list []
  in
  (* The relocations of section [r], which patch the image at [at]'s results,
     prepended to [acc]. Its symbol table is read at the first relocation that
     names a symbol. *)
  let entries r at field acc =
    let h = hs.(r) in
    let rela = h.sh_type = sht_rela in
    let entsize = Int.max ((if rela then 3 else 2) * f.word) h.sh_entsize in
    let syms = lazy (table h.sh_link) and first = at_of h and acc = ref acc in
    for k = (length_of h / entsize) - 1 downto 0 do
      let e = first + (k * entsize) in
      let offset = at (word f obj e) in
      (* [r_info] holds the type in its low 32 bits and the symbol in its high
         ones, or in 32 bits the type in its low 8 bits and the symbol above. *)
      let kind =
        if f.word = 8 then u32 obj (e + 8) else u32 obj (e + 4) land 0xff
      in
      let sym =
        if f.word = 8 then u32 obj (e + 12) else u32 obj (e + 4) lsr 8
      in
      let addend =
        if not rela then field offset
        else if f.word = 8 then Explicit (i64 obj (e + 16))
        else Explicit (i32 obj (e + 8))
      in
      let symbol =
        if sym = 0 then { name = ""; place = Absolute 0 }
        else Iarray.get (Lazy.force syms) sym
      in
      acc := { offset; kind; addend; symbol } :: !acc
    done;
    !acc
  in
  (* The held sections with bytes, in which a dynamic relocation without an
     addend finds its field by address. *)
  let by_offset = lazy (with_bytes offsets hs) in
  (* A dynamic relocation's offset is an address; any other's lies in the
     section it patches. *)
  let relocations_of r h acc =
    if h.sh_type <> sht_rel && h.sh_type <> sht_rela then acc
    else if h.sh_info = 0 then
      entries r
        (fun a ->
          if a < address || a - address >= size then
            fail
              "a relocation patches address 0x%x, which the image does not hold"
              a;
          a - address)
        (fun x ->
          let i = holding offsets hs (Lazy.force by_offset) x in
          if i < 0 then
            fail
              "a relocation without an addend patches address 0x%x, where the \
               object has no bytes"
              (x + address);
          implicit hs.(i) (Option.get offsets.(i)) x)
        acc
    else
      let target = hs.(h.sh_info) in
      if target.sh_flags land shf_alloc = 0 then acc
      else
        match offsets.(h.sh_info) with
        | None ->
            fail
              "a relocation patches section %d, which the image does not hold"
              h.sh_info
        | Some off ->
            entries r
              (fun a ->
                let o = a - target.sh_addr in
                if o < 0 || o >= target.sh_size then
                  fail "a relocation's offset %d lies outside section %d" a
                    h.sh_info;
                off + o)
              (fun x ->
                if not (has_bytes target) then
                  fail
                    "a relocation without an addend patches section %d, which \
                     has no bytes"
                    h.sh_info;
                implicit target off x)
              acc
  in
  (* Built from the last section's last entry, so the list is made once. *)
  let relocations = ref [] in
  for r = count - 1 downto 0 do
    relocations := relocations_of r hs.(r) !relocations
  done;
  let relocations = !relocations in
  let sections =
    match unplaced with
    | None -> Iarray.init count (fun i -> section hs.(i) offsets.(i))
    | Some unplaced ->
        (* A section the image does not hold keeps its record. *)
        Iarray.init count (fun i ->
            match offsets.(i) with
            | None -> unplaced.(i)
            | offset -> { (unplaced.(i) : section) with offset })
  in
  {
    bits = 8 * f.word;
    kind = u16 obj 16;
    machine;
    os_abi = Char.code obj.[7];
    abi_version = Char.code obj.[8];
    flags = u32 obj f.e_flags;
    address;
    align = image_align;
    file = obj;
    size;
    sections;
    symbols;
    relocations;
  }

let allocated ~machine (s : section) = allocated_as machine s.kind s.flags

let of_string ?(align = 1) ?held obj =
  if not (is_pow2 align) then
    invalid_argf "Rig_elf.of_string: align %d is not a positive power of two"
      align;
  match read ~align ?held obj with
  | o -> Ok o
  | exception Malformed m -> Error m
  | exception Invalid_argument _ -> Error truncated
  | exception Held (e, bt) -> Printexc.raise_with_backtrace e bt

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
