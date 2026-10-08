(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ELF reader over objects a small writer builds, over real GPU and host
   objects, and over corrupted objects. *)

open Windtrap
module Elf = Rig_elf

let strf = Printf.sprintf

(* The format's numbers, from the System V gABI *)

let sht_progbits = 1
let sht_symtab = 2
let sht_strtab = 3
let sht_rela = 4
let sht_nobits = 8
let sht_rel = 9
let sht_dynsym = 11
let sht_init_array = 14
let sht_symtab_shndx = 18
let sht_x86_64_unwind = 0x70000001
let em_x86_64 = 62
let em_cuda = 190
let shf_write = 0x1
let shf_alloc = 0x2
let shf_execinstr = 0x4
let shf_tls = 0x400
let shn_undef = 0
let shn_loreserve = 0xff00
let shn_abs = 0xfff1
let shn_common = 0xfff2
let shn_xindex = 0xffff
let ehdr_size = 64
let shdr_size = 64

(* An ELF writer *)

type sh = {
  name : string;
  kind : int;
  flags : int;
  addr : int;
  contents : string;
  size : int;
  link : int;
  info : int;
  align : int;
  entsize : int;
}

(* A section, by default an allocated program section: the kind the image holds.
   A [nobits] section has a [size] and no contents. *)
let section ?(kind = sht_progbits) ?(flags = shf_alloc) ?(addr = 0) ?size
    ?(link = 0) ?(info = 0) ?(align = 1) ?(entsize = 0) name contents =
  let size = Option.value size ~default:(String.length contents) in
  { name; kind; flags; addr; contents; size; link; info; align; entsize }

let bss ?addr name size =
  section ~kind:sht_nobits ~flags:(shf_alloc lor shf_write) ?addr ~size name ""

(* A thread-local section without bytes: allocated, and still out of the
   image. *)
let tbss ?addr name size =
  section ~kind:sht_nobits
    ~flags:(shf_alloc lor shf_write lor shf_tls)
    ?addr ~size name ""

let note name contents = section ~flags:0 name contents

type header = {
  kind : int;
  machine : int;
  os_abi : int;
  abi_version : int;
  flags : int;
}

let relocatable =
  { kind = 1; machine = 62; os_abi = 0; abi_version = 0; flags = 0 }

let add_section_header b ~name (s : sh) ~offset =
  Buffer.add_int32_le b (Int32.of_int name);
  Buffer.add_int32_le b (Int32.of_int s.kind);
  Buffer.add_int64_le b (Int64.of_int s.flags);
  Buffer.add_int64_le b (Int64.of_int s.addr);
  Buffer.add_int64_le b (Int64.of_int offset);
  Buffer.add_int64_le b (Int64.of_int s.size);
  Buffer.add_int32_le b (Int32.of_int s.link);
  Buffer.add_int32_le b (Int32.of_int s.info);
  Buffer.add_int64_le b (Int64.of_int s.align);
  Buffer.add_int64_le b (Int64.of_int s.entsize)

(* The object with [sections] at indexes 1, 2, ..., after the null section, and
   its section names in a last [.shstrtab]. With more sections than the header
   counts, or with [extended], the count and the names' index go in the null
   section. *)
let write ?(header = relocatable) ?(extended = false) sections =
  let names = Buffer.create 256 in
  Buffer.add_char names '\000';
  let name_of n =
    if n = "" then 0
    else
      let at = Buffer.length names in
      Buffer.add_string names n;
      Buffer.add_char names '\000';
      at
  in
  let section_names = List.map (fun (s : sh) -> name_of s.name) sections in
  let shstrtab_name = name_of ".shstrtab" in
  let shstrtab = Buffer.contents names in
  let sections =
    sections @ [ section ~kind:sht_strtab ~flags:0 ".shstrtab" shstrtab ]
  in
  let names = section_names @ [ shstrtab_name ] in
  let body = Buffer.create 4096 in
  let headers =
    List.map2
      (fun (s : sh) name ->
        let at = ehdr_size + Buffer.length body in
        if s.kind <> sht_nobits then Buffer.add_string body s.contents;
        (s, name, at))
      sections names
  in
  let count = List.length sections + 1 in
  let shstrndx = count - 1 in
  let extended = extended || count >= shn_loreserve in
  let far_names = shstrndx >= shn_loreserve in
  let b =
    Buffer.create (ehdr_size + Buffer.length body + (count * shdr_size))
  in
  Buffer.add_string b "\x7fELF\002\001\001";
  Buffer.add_uint8 b header.os_abi;
  Buffer.add_uint8 b header.abi_version;
  Buffer.add_string b (String.make 7 '\000');
  Buffer.add_uint16_le b header.kind;
  Buffer.add_uint16_le b header.machine;
  Buffer.add_int32_le b 1l;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b (Int64.of_int (ehdr_size + Buffer.length body));
  Buffer.add_int32_le b (Int32.of_int header.flags);
  Buffer.add_uint16_le b ehdr_size;
  Buffer.add_uint16_le b 0;
  Buffer.add_uint16_le b 0;
  Buffer.add_uint16_le b shdr_size;
  Buffer.add_uint16_le b (if extended then 0 else count);
  Buffer.add_uint16_le b (if far_names then shn_xindex else shstrndx);
  Buffer.add_buffer b body;
  let null =
    section ~kind:0 ~flags:0
      ~size:(if extended then count else 0)
      ~link:(if far_names then shstrndx else 0)
      "" ""
  in
  add_section_header b ~name:0 null ~offset:0;
  List.iter
    (fun (s, name, offset) -> add_section_header b ~name s ~offset)
    headers;
  Buffer.contents b

(* Symbols *)

type sym = { name : string; shndx : int; value : int }

let undefined name = { name; shndx = shn_undef; value = 0 }
let defined name shndx value = { name; shndx; value }

let add_symbol b ~name ~shndx ~value =
  Buffer.add_int32_le b (Int32.of_int name);
  Buffer.add_uint8 b 0x10;
  Buffer.add_uint8 b 0;
  Buffer.add_uint16_le b shndx;
  Buffer.add_int64_le b (Int64.of_int value);
  Buffer.add_int64_le b 0L

(* A symbol table's entries, after the null symbol, and its string table. *)
let symbols syms =
  let names = Buffer.create 64 and table = Buffer.create 256 in
  Buffer.add_char names '\000';
  add_symbol table ~name:0 ~shndx:0 ~value:0;
  List.iter
    (fun s ->
      let name =
        if s.name = "" then 0
        else
          let at = Buffer.length names in
          Buffer.add_string names s.name;
          Buffer.add_char names '\000';
          at
      in
      add_symbol table ~name ~shndx:s.shndx ~value:s.value)
    syms;
  (Buffer.contents table, Buffer.contents names)

let symtab ?(kind = sht_symtab) ?(name = ".symtab") ~link entries =
  section ~kind ~flags:0 ~link ~align:8 ~entsize:24 name entries

let strtab ?(name = ".strtab") names =
  section ~kind:sht_strtab ~flags:0 name names

(* Relocations *)

let r_info sym kind = Int64.(logor (shift_left (of_int sym) 32) (of_int kind))

let rela entries =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind, addend) ->
      Buffer.add_int64_le b (Int64.of_int offset);
      Buffer.add_int64_le b (r_info sym kind);
      Buffer.add_int64_le b (Int64.of_int addend))
    entries;
  Buffer.contents b

let rel entries =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind) ->
      Buffer.add_int64_le b (Int64.of_int offset);
      Buffer.add_int64_le b (r_info sym kind))
    entries;
  Buffer.contents b

let rela_section ?(name = ".rela") ~link ~info entries =
  section ~kind:sht_rela ~flags:0 ~link ~info ~align:8 ~entsize:24 name
    (rela entries)

let rel_section ?(name = ".rel") ~link ~info entries =
  section ~kind:sht_rel ~flags:0 ~link ~info ~align:8 ~entsize:16 name
    (rel entries)

(* Patching one field of a written object *)

let patch obj at width v =
  let b = Bytes.of_string obj in
  (match width with
  | 1 -> Bytes.set_uint8 b at v
  | 2 -> Bytes.set_uint16_le b at v
  | 4 -> Bytes.set_int32_le b at (Int32.of_int v)
  | _ -> Bytes.set_int64_le b at (Int64.of_int v));
  Bytes.to_string b

let shoff obj = Int64.to_int (String.get_int64_le obj 40)

(* The byte offsets of a section header's fields *)
let sh_name = 0
let sh_addr = 16
let sh_offset = 24
let sh_size = 32
let sh_link = 40
let sh_info = 44
let sh_addralign = 48

let patch_section obj i field width v =
  patch obj (shoff obj + (i * shdr_size) + field) width v

(* Reading *)

let pp_place ppf (p : Elf.place) =
  match p with
  | Undefined -> Format.pp_print_string ppf "Undefined"
  | Absolute v -> Format.fprintf ppf "Absolute %#x" v
  | Image { section; offset } ->
      Format.fprintf ppf "Image { section = %d; offset = %#x }" section offset
  | Outside i -> Format.fprintf ppf "Outside %d" i

let pp_symbol ppf (s : Elf.symbol) =
  Format.fprintf ppf "{ name = %S; place = %a }" s.name pp_place s.place

let symbol = Testable.make ~pp:pp_symbol ~equal:( = )

let elf_section =
  Testable.make
    ~pp:(fun ppf (s : Elf.section) ->
      Format.fprintf ppf
        "{ name = %S; kind = %d; flags = %#x; offset = %a; size = %d; at = %d; \
         length = %d }"
        s.name s.kind s.flags
        (Format.pp_print_option
           ~none:(fun ppf () -> Format.pp_print_string ppf "None")
           (fun ppf -> Format.fprintf ppf "Some %#x"))
        s.offset s.size s.at s.length)
    ~equal:( = )

let pp_addend ppf : Elf.addend -> unit = function
  | Explicit a -> Format.fprintf ppf "Explicit %d" a
  | Implicit { at; length } ->
      Format.fprintf ppf "Implicit { at = %d; length = %d }" at length

let elf_addend = Testable.make ~pp:pp_addend ~equal:( = )

let relocation =
  Testable.make
    ~pp:(fun ppf (r : Elf.relocation) ->
      Format.fprintf ppf "{ offset = %#x; kind = %d; addend = %a; symbol = %a }"
        r.offset r.kind pp_addend r.addend pp_symbol r.symbol)
    ~equal:( = )

let pp_t ppf (o : Elf.t) =
  Format.fprintf ppf "<object: image of %d bytes, %d sections>" o.size
    (Iarray.length o.sections)

let read ?align obj =
  require_ok ~pp:Format.pp_print_string (Elf.of_string ?align obj)

let refused ?align obj = require_error ~pp:pp_t (Elf.of_string ?align obj)

(* [f ()] and the bytes it allocates. *)
let allocated f =
  let before = Gc.allocated_bytes () in
  let r = f () in
  (r, Gc.allocated_bytes () -. before)

(* The most bytes reading may allocate for an object of [n] bytes. The real
   objects take at most 3.2 per byte, and failing takes 424 bytes; an object of
   nothing but small table entries takes more per byte. *)
let linear n = (32. *. Float.of_int n) +. 4096.

let in_linear_memory obj =
  let r, bytes = allocated (fun () -> Elf.of_string obj) in
  less ~msg:"bytes allocated" float_exact
    ~than:(linear (String.length obj))
    bytes;
  r

let section_named (o : Elf.t) name =
  require_some ~msg:name
    (Iarray.find_opt (fun (s : Elf.section) -> s.name = name) o.sections)

let symbol_named (o : Elf.t) name =
  require_some ~msg:name
    (Iarray.find_opt (fun (s : Elf.symbol) -> s.name = name) o.symbols)

let syms (o : Elf.t) = Iarray.to_list o.symbols

(* The image, as the documentation writes it into [bytes]. *)
let image (o : Elf.t) =
  let b = Bytes.make o.size '\000' in
  let put (s : Elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  Bytes.to_string b

(* A section's bytes in the object. *)
let contents (o : Elf.t) (s : Elf.section) = String.sub o.file s.at s.length
let sym_entry name place : Elf.symbol = { name; place }
let null_symbol = sym_entry "" Undefined
let no_symbol = sym_entry "" (Absolute 0)

(* Sorted by start, the spans [(at, n, msg)] share no byte. *)
let disjoint where spans =
  ignore
    (List.fold_left
       (fun last (at, n, msg) ->
         at_least ~msg:(msg ^ " in " ^ where) int ~than:last at;
         at + n)
       0 (List.sort compare spans))

(* The equations of [Elf.t]'s documentation. They never build the image, whose
   size a corrupted address can make any [int]. *)
let invariants (o : Elf.t) =
  let count = Iarray.length o.sections in
  let in_file = ref [] and in_image = ref [] in
  Iarray.iteri
    (fun i (s : Elf.section) ->
      let msg = strf "section %d %S" i s.name in
      at_least ~msg int ~than:0 s.at;
      at_least ~msg int ~than:0 s.length;
      at_most ~msg int ~than:(String.length o.file) (s.at + s.length);
      if s.length > 0 then in_file := (s.at, s.length, msg) :: !in_file;
      match s.offset with
      | None -> ()
      | Some off ->
          at_least ~msg int ~than:0 off;
          if s.kind = sht_nobits then equal ~msg int 0 s.length
          else equal ~msg int s.size s.length;
          at_most ~msg int ~than:o.size (off + s.size);
          if s.size > 0 then in_image := (off, s.size, msg) :: !in_image)
    o.sections;
  disjoint "the object" !in_file;
  disjoint "the image" !in_image;
  equal ~msg:"the image's alignment is a power of two" int 0
    (o.align land (o.align - 1));
  at_least ~msg:"the image's alignment" int ~than:1 o.align;
  equal ~msg:"its address is a multiple of its alignment" int 0
    (o.address land (o.align - 1));
  let place (s : Elf.symbol) =
    match s.place with
    | Image { section; offset } ->
        let msg = Format.asprintf "%a" pp_symbol s in
        at_least ~msg int ~than:0 section;
        less ~msg int ~than:count section;
        let sec = Iarray.get o.sections section in
        let off = require_some ~msg sec.offset in
        at_least ~msg int ~than:off offset;
        at_most ~msg int ~than:(off + sec.size) offset
    | Undefined | Absolute _ | Outside _ -> ()
  in
  Iarray.iter place o.symbols;
  List.iter
    (fun (r : Elf.relocation) ->
      let msg = strf "relocation at %#x" r.offset in
      at_least ~msg int ~than:0 r.offset;
      less ~msg int ~than:o.size r.offset;
      place r.symbol)
    o.relocations

(* Header *)

let test_header () =
  let header =
    {
      kind = 2;
      machine = 224;
      os_abi = 64;
      abi_version = 4;
      flags = 0xff00_004e;
    }
  in
  let o = read (write ~header [ section ".text" "ABCD" ]) in
  equal ~msg:"its class" int 64 o.bits;
  equal ~msg:"its type" int 2 o.kind;
  equal ~msg:"its machine" int 224 o.machine;
  equal ~msg:"its OS ABI" int 64 o.os_abi;
  equal ~msg:"its ABI version" int 4 o.abi_version;
  equal ~msg:"its flags, all 32 bits unsigned" int 0xff00_004e o.flags

(* Layout *)

let test_appended () =
  let o =
    read
      (write
         [
           section ~align:4 ".text" "ABCDE";
           note ".comment" "not loaded";
           section ~align:16 ".data" "01234567";
           section ~align:0 ".rodata" "xy";
         ])
  in
  equal ~msg:"text, data at 16, rodata right after it" string
    ("ABCDE" ^ String.make 11 '\000' ^ "01234567" ^ "xy")
    (image o);
  equal ~msg:"offsets"
    (list (option int))
    [ None; Some 0; None; Some 16; Some 24; None ]
    (List.map (fun (s : Elf.section) -> s.offset) (Iarray.to_list o.sections))

let test_align () =
  let obj = write [ section ".text" "ABC"; section ".data" "DEF" ] in
  equal ~msg:"each section at a multiple of align" string
    ("ABC" ^ String.make 125 '\000' ^ "DEF")
    (image (read ~align:128 obj));
  let obj = write [ section ".text" "ABC"; section ~align:64 ".data" "DEF" ] in
  let o = read ~align:2 obj in
  equal ~msg:"the larger of align and the section's alignment" string
    ("ABC" ^ String.make 61 '\000' ^ "DEF")
    (image o);
  equal ~msg:"which the image asks of its address" int 64 o.align;
  equal ~msg:"or align, if larger" int 128 (read ~align:128 obj).align

let test_no_padding () =
  let o = read ~align:128 (write [ section ".a" "A"; section ".b" "BCD" ]) in
  equal ~msg:"the image ends where its last section ends" int 131 o.size;
  let o = read (write [ section ".a" "A"; section ~align:64 ".empty" "" ]) in
  equal ~msg:"an empty last section ends the image at its offset" int 64 o.size;
  let o = read (write [ note ".comment" "x" ]) in
  equal ~msg:"no section, no image" string "" (image o)

let test_addressed () =
  let obj =
    write
      [
        section ~addr:0x10 ~align:16 ".rodata" "RO";
        note ".comment" "not loaded";
        section ~addr:0x08 ".text" "ABCD";
        bss ~addr:0x40 ".bss" 16;
      ]
  in
  let o = read obj in
  equal
    ~msg:"each at its address, zeros in the gaps, before the first and in .bss"
    string
    (String.make 8 '\000' ^ "ABCD" ^ String.make 4 '\000' ^ "RO"
    ^ String.make (0x50 - 0x12) '\000')
    (image o);
  equal ~msg:"the image asks for its sections' largest alignment" int 16 o.align;
  equal ~msg:"align has no effect" string (image o)
    (image (read ~align:4096 obj));
  equal ~msg:"not even on the image's alignment" int 16
    (read ~align:4096 obj).align;
  let o =
    read
      (write
         [ section ~addr:0 ".text" "ABCD"; section ~addr:0x100 ".data" "EF" ])
  in
  equal ~msg:"a section at address 0 of an addressed object stays at 0"
    (option int) (Some 0) (section_named o ".text").offset;
  equal ~msg:"and the other at its address" (option int) (Some 0x100)
    (section_named o ".data").offset

let test_bss () =
  let o =
    read
      (write
         [
           section ".text" "ABCD";
           { (bss ".bss" 64) with align = 16 };
           tbss ".tbss" 8;
         ])
  in
  equal
    ~msg:
      "a section with no bytes takes its size in the image, at its alignment, \
       and no byte of the object"
    elf_section
    {
      name = ".bss";
      kind = sht_nobits;
      flags = shf_alloc lor shf_write;
      offset = Some 16;
      size = 64;
      at = 0;
      length = 0;
    }
    (section_named o ".bss");
  equal ~msg:"its bytes are zeros" string
    ("ABCD" ^ String.make 76 '\000')
    (image o);
  equal ~msg:"a thread-local one stays out" (option int) None
    (section_named o ".tbss").offset

(* A loader whose memory does not hold a section says so: it stays out, with its
   symbols, and the others lay out without it. *)
let test_held () =
  let entries, names = symbols [ defined "t" 3 0; defined "c" 1 2 ] in
  let obj =
    write
      [
        section ".text" "ABCD";
        note ".note" "N";
        { (bss ".shared" 64) with align = 16 };
        bss ".global" 8;
        symtab ~link:6 entries;
        strtab names;
      ]
  in
  let held (s : Elf.section) =
    Elf.allocated ~machine:relocatable.machine s && s.name <> ".shared"
  in
  let o = require_ok ~pp:Format.pp_print_string (Elf.of_string ~held obj) in
  invariants o;
  equal ~msg:"the section the loader does not hold stays out" (option int) None
    (section_named o ".shared").offset;
  equal ~msg:"the next one follows the code" (option int) (Some 4)
    (section_named o ".global").offset;
  equal ~msg:"its symbols are outside" symbol
    (sym_entry "t" (Outside 3))
    (symbol_named o "t");
  equal ~msg:"by default, ELF's rule holds both"
    (list (option int))
    [ Some 16; Some 80 ]
    (List.map
       (fun n -> (section_named (read obj) n).offset)
       [ ".shared"; ".global" ])

let test_allocated () =
  let sections =
    [
      section ".text" "AB";
      bss ".bss" 4;
      tbss ".tbss" 4;
      note ".note" "N";
      section ~kind:sht_init_array ~flags:(shf_alloc lor shf_write)
        ".init_array" (String.make 8 '\000');
      section ~kind:sht_x86_64_unwind ".eh_frame" "E";
    ]
  in
  let names = List.map (fun (s : sh) -> s.name) sections in
  let o = read (write sections) in
  equal ~msg:"allocated program sections and ones without bytes, not .tbss"
    (list (pair string bool))
    [
      (".text", true);
      (".bss", true);
      (".tbss", false);
      (".note", false);
      (".init_array", false);
      (".eh_frame", true);
    ]
    (List.map
       (fun n -> (n, Elf.allocated ~machine:em_x86_64 (section_named o n)))
       names);
  equal ~msg:"x86-64's unwind type means another on another machine" bool false
    (Elf.allocated ~machine:em_cuda (section_named o ".eh_frame"));
  let o =
    read (write ~header:{ relocatable with machine = em_cuda } sections)
  in
  equal ~msg:"and the image of an object for it lacks the section" (option int)
    None (section_named o ".eh_frame").offset

let test_sections () =
  let o =
    read (write [ section ~flags:(shf_alloc lor shf_execinstr) ".text" "AB" ])
  in
  equal ~msg:"every section by index, from the null section" (list elf_section)
    [
      {
        name = "";
        kind = 0;
        flags = 0;
        offset = None;
        size = 0;
        at = 0;
        length = 0;
      };
      {
        name = ".text";
        kind = sht_progbits;
        flags = shf_alloc lor shf_execinstr;
        offset = Some 0;
        size = 2;
        at = ehdr_size;
        length = 2;
      };
      {
        name = ".shstrtab";
        kind = sht_strtab;
        flags = 0;
        offset = None;
        size = 17;
        at = ehdr_size + 2;
        length = 17;
      };
    ]
    (Iarray.to_list o.sections);
  equal ~msg:"their bytes" (list string)
    [ ""; "AB"; "\000.text\000.shstrtab\000" ]
    (List.map (contents o) (Iarray.to_list o.sections))

let test_no_names () =
  let obj = write [ section ".text" "AB" ] in
  let o = read (patch obj 62 2 0) in
  equal ~msg:"an object without section names names each \"\"" (list string)
    [ ""; ""; "" ]
    (List.map (fun (s : Elf.section) -> s.name) (Iarray.to_list o.sections))

(* Symbols *)

let test_places () =
  let entries, names =
    symbols
      [
        undefined "ext";
        defined "abs" shn_abs 0x1234;
        defined "start" 2 0;
        defined "end" 2 4;
        defined "past" 2 5;
        defined "counter" 3 0;
        defined "common" shn_common 8;
        defined "comment" 4 0;
      ]
  in
  let o =
    read
      (write
         [
           section ~align:8 ".data" "01234";
           section ~align:8 ".text" "ABCD";
           tbss ".tbss" 4;
           note ".comment" "c";
           symtab ~link:6 entries;
           strtab names;
         ])
  in
  equal ~msg:"every entry by index, from the null symbol" (list symbol)
    [
      null_symbol;
      sym_entry "ext" Undefined;
      sym_entry "abs" (Absolute 0x1234);
      sym_entry "start" (Image { section = 2; offset = 8 });
      sym_entry "end" (Image { section = 2; offset = 12 });
      sym_entry "past" (Outside 2);
      sym_entry "counter" (Outside 3);
      sym_entry "common" (Outside shn_common);
      sym_entry "comment" (Outside 4);
    ]
    (syms o)

let test_addressed_symbols () =
  let entries, names =
    symbols [ defined "k" 1 0x104; defined "k.kd" 2 0x200 ]
  in
  let o =
    read
      (write
         [
           section ~addr:0x100 ".text" "ABCDEFGH";
           section ~addr:0x200 ".rodata" "KD";
           symtab ~link:4 entries;
           strtab names;
         ])
  in
  equal ~msg:"the image starts at the first section's address" int 0x100
    o.address;
  equal ~msg:"a symbol's value is its address less the image's" (list symbol)
    [
      null_symbol;
      sym_entry "k" (Image { section = 1; offset = 0x4 });
      sym_entry "k.kd" (Image { section = 2; offset = 0x100 });
    ]
    (syms o)

let test_lookup () =
  let entries, names =
    symbols
      [
        undefined "k";
        defined "k" 3 0;
        defined "k" 1 2;
        defined "j" 1 1;
        defined "j" 1 3;
        defined "" 1 0;
      ]
  in
  let o =
    read
      (write
         [
           section ".text" "ABCD";
           symtab ~link:4 entries;
           tbss ".tbss" 4;
           strtab names;
         ])
  in
  equal ~msg:"the first one in the image, past undefined and outside ones"
    (option int) (Some 2) (Elf.symbol o "k");
  equal ~msg:"the first by index" (option int) (Some 1) (Elf.symbol o "j");
  equal ~msg:"a name no symbol has" (option int) None (Elf.symbol o "x");
  equal ~msg:"the empty name, though a nameless symbol is in the image"
    (option int) None (Elf.symbol o "")

let test_no_symbols () =
  let o = read (write [ section ".text" "ABCD" ]) in
  equal ~msg:"no symbol table" (list symbol) [] (syms o);
  equal ~msg:"no symbol" (option int) None (Elf.symbol o "f")

(* An executable stripped of its symbol table keeps the dynamic one, which a
   loader reads its symbols from. *)
let test_dynamic_symbols () =
  let text = section ~addr:0x100 ".text" "ABCD" in
  let dyn, dynnames = symbols [ defined "k.kd" 1 0x102 ] in
  let dynamic =
    [
      symtab ~kind:sht_dynsym ~name:".dynsym" ~link:3 dyn;
      strtab ~name:".dynstr" dynnames;
    ]
  in
  equal ~msg:"the dynamic symbols of an object with no symbol table"
    (option int) (Some 0x2)
    (Elf.symbol (read (write (text :: dynamic))) "k.kd");
  let static, names = symbols [ defined "k.kd" 1 0x101 ] in
  let o =
    read (write ((text :: dynamic) @ [ symtab ~link:5 static; strtab names ]))
  in
  equal ~msg:"the symbol table over the dynamic one" (option int) (Some 0x1)
    (Elf.symbol o "k.kd")

(* Relocations *)

let test_relocations () =
  let entries, names = symbols [ defined "data" 2 4; undefined "ext" ] in
  let o =
    read
      (write
         [
           section ~align:4 ".text" "ABCDEF";
           section ~align:8 ".data" "01234567";
           symtab ~link:4 entries;
           strtab names;
           rela_section ~link:3 ~info:2 [ (0, 1, 1, -8) ];
           rela_section ~link:3 ~info:1 [ (2, 2, 4, -4); (5, 0, 8, 7) ];
           rel_section ~link:3 ~info:2 [ (6, 2, 9) ];
           rela_section ~link:3 ~info:5 [ (0, 1, 1, 0) ];
         ])
  in
  let data = Elf.Image { section = 2; offset = 12 } in
  let data_at = (section_named o ".data").at in
  equal ~msg:"in the order of their sections, then of their entries"
    (list relocation)
    [
      {
        offset = 8;
        kind = 1;
        addend = Explicit (-8);
        symbol = sym_entry "data" data;
      };
      {
        offset = 2;
        kind = 4;
        addend = Explicit (-4);
        symbol = sym_entry "ext" Undefined;
      };
      { offset = 5; kind = 8; addend = Explicit 7; symbol = no_symbol };
      {
        offset = 14;
        kind = 9;
        addend = Implicit { at = data_at + 6; length = 2 };
        symbol = sym_entry "ext" Undefined;
      };
    ]
    o.relocations

let test_own_table () =
  let static, names = symbols [ defined "static" 1 0 ] in
  let dyn, dynnames = symbols [ defined "dynamic" 1 2 ] in
  let o =
    read
      (write
         [
           section ".text" "ABCD";
           symtab ~link:3 static;
           strtab names;
           symtab ~kind:sht_dynsym ~name:".dynsym" ~link:5 dyn;
           strtab ~name:".dynstr" dynnames;
           rela_section ~link:4 ~info:1 [ (0, 1, 1, 0) ];
           rela_section ~link:2 ~info:1 [ (1, 1, 1, 0) ];
         ])
  in
  equal ~msg:"each through the table its section links to" (list symbol)
    [
      sym_entry "dynamic" (Image { section = 1; offset = 2 });
      sym_entry "static" (Image { section = 1; offset = 0 });
    ]
    (List.map (fun (r : Elf.relocation) -> r.symbol) o.relocations);
  equal ~msg:"while the object's symbols are its symbol table" (list symbol)
    [ null_symbol; sym_entry "static" (Image { section = 1; offset = 0 }) ]
    (syms o)

let test_unloaded_relocations () =
  let entries, names = symbols [ defined "f" 1 0 ] in
  let o =
    read
      (write
         [
           section ".text" "ABCD";
           note ".debug_info" "debugging";
           symtab ~link:4 entries;
           strtab names;
           rela_section ~link:3 ~info:2 [ (0, 1, 1, 0); (4, 1, 1, 0) ];
         ])
  in
  equal ~msg:"relocations of a section that does not occupy memory"
    (list relocation) [] o.relocations

let test_dynamic_relocations () =
  let dyn, dynnames = symbols [ defined "k" 1 0x100 ] in
  let obj ~at =
    write
      [
        section ~addr:0x100 ".text" "ABCD";
        section ~addr:0x200 ".data" "EFGH";
        symtab ~kind:sht_dynsym ~name:".dynsym" ~link:4 dyn;
        strtab ~name:".dynstr" dynnames;
        rela_section ~name:".rela.dyn" ~link:3 ~info:0 [ (at, 1, 1, 0) ];
      ]
  in
  let offsets at =
    List.map (fun (r : Elf.relocation) -> r.offset) (read (obj ~at)).relocations
  in
  equal ~msg:"one in a section patches its address less the image's" (list int)
    [ 0x2 ] (offsets 0x102);
  equal ~msg:"one in the zeros between sections too" (list int) [ 0x80 ]
    (offsets 0x180);
  equal ~msg:"the image's first byte" (list int) [ 0 ] (offsets 0x100);
  equal ~msg:"and its last" (list int) [ 0x103 ] (offsets 0x203);
  ignore (refused (obj ~at:0xff));
  ignore (refused (obj ~at:0x204))

(* A relocation without an addend has it in the field it patches: the object's
   bytes there, which a section without bytes and a gap between sections
   lack. *)
let test_implicit () =
  let dynamic ~at =
    write
      [
        section ~addr:0x100 ".text" "ABCD";
        section ~addr:0x200 ".data" "EFGH";
        bss ~addr:0x300 ".bss" 8;
        rel_section ~name:".rel.dyn" ~link:0 ~info:0 [ (at, 0, 8) ];
      ]
  in
  let o = read (dynamic ~at:0x201) in
  equal ~msg:"a dynamic one's field, found by its address" (list elf_addend)
    [ Implicit { at = (section_named o ".data").at + 1; length = 3 } ]
    (List.map (fun (r : Elf.relocation) -> r.addend) o.relocations);
  ignore (refused (dynamic ~at:0x180));
  ignore (refused (dynamic ~at:0x304));
  ignore
    (refused
       (write
          [
            section ".text" "ABCD";
            bss ".bss" 8;
            rel_section ~link:0 ~info:2 [ (0, 0, 8) ];
          ]))

(* A thread-local section without bytes takes no memory, so the section after it
   may start at its address, as a linker lays out [.tbss] and [.init_array]. *)
let test_tbss_addresses () =
  let dyn, dynnames = symbols [] in
  let offsets ~data =
    let o =
      read
        (write
           [
             section ~addr:0x100 ".text" "ABCD";
             tbss ~addr:0x200 ".tbss" 16;
             section ~addr:data ".data" "EFGH";
             symtab ~kind:sht_dynsym ~name:".dynsym" ~link:5 dyn;
             strtab ~name:".dynstr" dynnames;
             rela_section ~name:".rela.dyn" ~link:4 ~info:0 [ (0x202, 0, 8, 0) ];
           ])
    in
    List.map (fun (r : Elf.relocation) -> r.offset) o.relocations
  in
  equal ~msg:"one at its addresses patches the section that takes them"
    (list int) [ 0x102 ] (offsets ~data:0x200);
  equal ~msg:"or the zeros between sections where none does" (list int)
    [ 0x102 ] (offsets ~data:0x300)

(* Extended section numbering *)

let test_extended_header () =
  let entries, names = symbols [ defined "f" 1 2 ] in
  let o =
    read
      (write ~extended:true
         [ section ".text" "ABCD"; symtab ~link:3 entries; strtab names ])
  in
  equal ~msg:"the section count in the null section" int 5
    (Iarray.length o.sections);
  equal ~msg:"with the symbols it reads through it" (option int) (Some 2)
    (Elf.symbol o "f")

let test_xindex () =
  let entries, names =
    symbols [ defined "f" shn_xindex 0; defined "g" shn_xindex 1 ]
  in
  let indexes = Buffer.create 12 in
  List.iter (Buffer.add_int32_le indexes) [ 0l; 2l; 1l ];
  let o =
    read
      (write
         [
           section ".text" "ABCD";
           section ".data" "EF";
           symtab ~link:4 entries;
           strtab names;
           section ~kind:sht_symtab_shndx ~flags:0 ~link:3 ~align:4 ~entsize:4
             ".symtab_shndx" (Buffer.contents indexes);
         ])
  in
  equal ~msg:"a symbol's section index from the extended index table"
    (list symbol)
    [
      null_symbol;
      sym_entry "f" (Image { section = 2; offset = 4 });
      sym_entry "g" (Image { section = 1; offset = 1 });
    ]
    (syms o)

(* An object with more sections than a header field holds: the count, the names'
   index and a symbol's section index all take the extended forms. *)
let test_many_sections () =
  let filler = List.init (shn_loreserve + 8) (fun _ -> note "" "") in
  let indexes = Buffer.create 8 in
  let at = shn_loreserve + 9 in
  List.iter (Buffer.add_int32_le indexes) [ 0l; Int32.of_int at ];
  let entries, names = symbols [ defined "far" shn_xindex 1 ] in
  let sections =
    filler
    @ [
        section ".far" "ABCD";
        symtab ~link:(at + 2) entries;
        strtab names;
        section ~kind:sht_symtab_shndx ~flags:0 ~link:(at + 1) ~align:4
          ~entsize:4 ".symtab_shndx" (Buffer.contents indexes);
      ]
  in
  let o = read (write sections) in
  equal ~msg:"every section" int (at + 5) (Iarray.length o.sections);
  equal ~msg:"the names past the header's reach" string ".far"
    (Iarray.get o.sections at).name;
  equal ~msg:"the image" string "ABCD" (image o);
  equal ~msg:"a symbol in a section past the header's reach" (list symbol)
    [ null_symbol; sym_entry "far" (Image { section = at; offset = 1 }) ]
    (syms o)

(* The least CPU times of five reads each of [small] and [large], taken in turn
   so that load on the machine slows both alike. *)
let read_times small large =
  let once obj =
    let start = Sys.time () in
    ignore (Sys.opaque_identity (Elf.of_string obj));
    Sys.time () -. start
  in
  let s = ref infinity and l = ref infinity in
  for _ = 1 to 5 do
    s := Float.min !s (once small);
    l := Float.min !l (once large)
  done;
  (!s, !l)

(* Reading [make n] takes less than 128 times as long as reading
   [make (n / 16)]: 16 times if reading is linear, about 22 with its sorts, and
   256 if it is quadratic. Load raises the ratio: other processes evict the
   larger object from caches the smaller one stays in, which slowed each entry
   of the larger 2.5 times with a copy of this test on every core of an 8-core
   machine. The bound leaves twice that. A time [a n + b n^2] passes it only
   while [b n^2 < 14 a n]: at the counts below, a step repeated for each pair
   of sections takes longer than that. A ratio holds on a slow or instrumented
   build where a bound in seconds would not. *)
let linear_time make n =
  let small, large = read_times (make (n / 16)) (make n) in
  less
    ~msg:(strf "CPU time of %gs at n over %gs at n / 16" large small)
    float_exact ~than:128. (large /. small)

(* [count] sections, each with a symbol of its own at an extended index. *)
let many_symbols count =
  let filler = List.init count (fun _ -> note "" "") in
  let indexes = Buffer.create (4 * (count + 1)) in
  Buffer.add_int32_le indexes 0l;
  for i = 1 to count do
    Buffer.add_int32_le indexes (Int32.of_int i)
  done;
  let entries, names =
    symbols (List.init count (fun _ -> defined "" shn_xindex 0))
  in
  write
    (filler
    @ [
        symtab ~link:(count + 2) entries;
        strtab names;
        section ~kind:sht_symtab_shndx ~flags:0 ~link:(count + 1) ~align:4
          ~entsize:4 ".symtab_shndx" (Buffer.contents indexes);
      ])

(* Every section past the header's reach with a symbol of its own: reading takes
   time linear in the symbols. *)
let test_many_symbols () =
  let count = 70_000 in
  let o = read (many_symbols count) in
  equal ~msg:"the last symbol's section" symbol
    (sym_entry "" (Outside count))
    (Iarray.get o.symbols count);
  linear_time many_symbols count

(* [count] dynamic relocation sections without addends, each with its own symbol
   table, among [count] allocated sections the image lacks. *)
let many_relocation_sections count =
  let entries, names = symbols [ defined "f" 1 0x100 ] in
  let symtab_at = count + 2 and strtab_at = (3 * count) + 2 in
  write
    (section ~addr:0x100 ".text" "ABCD"
     :: List.init count (fun i ->
         section ~kind:sht_init_array ~addr:(0x1000 + (2 * i)) ".init_array" "x")
    @ List.init count (fun _ -> symtab ~link:strtab_at entries)
    @ List.init count (fun i ->
        rel_section ~link:(symtab_at + i) ~info:0 [ (0x102, 1, 1) ])
    @ [ strtab names ])

(* Reading takes time linear in the relocation sections. *)
let test_many_relocation_sections () =
  let count = 40_000 in
  let o = read (many_relocation_sections count) in
  equal ~msg:"every relocation" int count (List.length o.relocations);
  linear_time many_relocation_sections count

(* An object's image starts at its first section's address: reading allocates as
   much whatever the address, up to the largest. *)
let gen_address =
  Gen.(
    with_pp Format.pp_print_int
    @@ frequency
         [
           (1, int_range 1 0xffff);
           (1, map (fun k -> 1 lsl k) (int_range 0 61));
           (1, int_range 1 (max_int - 4));
           (1, constant (max_int - 4));
         ])

let at_address a = write [ section ~addr:a ".text" "ABCD" ]

let baseline =
  lazy
    (let obj = at_address 1 in
     snd (allocated (fun () -> Elf.of_string obj)))

let law_address a =
  let obj = at_address a in
  let r, bytes = allocated (fun () -> Elf.of_string obj) in
  let o = require_ok ~pp:Format.pp_print_string r in
  cover "past 1 GiB" (a > 1 lsl 30);
  equal ~msg:"the image's address" int a o.address;
  equal ~msg:"the image's size" int 4 o.size;
  equal ~msg:"bytes allocated" float_exact (Lazy.force baseline) bytes

(* Refusals: each object breaks one rule of an object that reads. *)

let objects_entries, objects_names =
  symbols [ defined "f" 1 0; defined "d" 2 4 ]

(* [.text] at 0 and [.data] at 8: an image of 16 bytes, with one relocation
   [(offset, symbol, kind, addend)] of section [target]. *)
let with_relocation ~target entry =
  write
    [
      section ~align:4 ".text" "ABCD";
      section ~align:8 ".data" "01234567";
      tbss ".tbss" 8;
      symtab ~link:5 objects_entries;
      strtab objects_names;
      rela_section ~link:4 ~info:target [ entry ];
    ]

let well_formed = with_relocation ~target:1 (1, 2, 1, 0)

(* The byte offset of entry [i], of [size] bytes, of section [s]. *)
let entry obj s i size =
  Int64.to_int
    (String.get_int64_le obj (shoff obj + (s * shdr_size) + sh_offset))
  + (i * size)

let symbol_entry = entry well_formed 4 1 24

let addressed ~text ~data =
  write [ section ~addr:text ".text" "ABCD"; section ~addr:data ".data" "EFGH" ]

let into_init_array =
  write
    [
      section ".text" "ABCD";
      section ~kind:sht_init_array ~flags:(shf_alloc lor shf_write) ~align:8
        ".init_array" (String.make 8 '\000');
      symtab ~link:4 objects_entries;
      strtab objects_names;
      rela_section ~link:3 ~info:2 [ (0, 1, 1, 0) ];
    ]

(* The largest power of two an [int] holds. *)
let max_align = 1 lsl 61

(* A symbol table whose 65 names are one name of 256 bytes, longer together than
   the object. *)
let shared_names =
  let table = Buffer.create 2048 in
  for _ = 0 to 64 do
    add_symbol table ~name:1 ~shndx:shn_undef ~value:0
  done;
  write
    [
      symtab ~link:2 (Buffer.contents table);
      strtab ("\000" ^ String.make 256 'n' ^ "\000");
    ]

let refusals =
  let len = String.length well_formed in
  let rela_entry = entry well_formed 6 0 24 in
  [
    ("a text", "not an object");
    ("the empty string", "");
    ("another magic number", patch well_formed 1 1 (Char.code 'X'));
    ("an unknown class", patch well_formed 4 1 3);
    ("a big-endian object", patch well_formed 5 1 2);
    ("a truncated header", String.sub well_formed 0 (ehdr_size - 1));
    ("section headers past the end", patch well_formed 40 8 (len - shdr_size));
    ("truncated section headers", String.sub well_formed 0 (len - 1));
    ( "section contents past the end",
      patch_section well_formed 2 sh_offset 8 (len - 4) );
    ( "a section size past the end",
      patch_section well_formed 4 sh_size 8 (2 * len) );
    ("a section name past its table", patch_section well_formed 1 sh_name 4 999);
    ("a symbol name past its table", patch well_formed symbol_entry 4 999);
    ( "a name that runs to its table's end",
      write
        [
          section ".text" "AB";
          symtab ~link:3 (fst (symbols [ defined "f" 1 0 ]));
          strtab "\000f";
        ] );
    ("section names in a section it lacks", patch well_formed 62 2 99);
    ( "symbol names in a section it lacks",
      patch_section well_formed 4 sh_link 4 99 );
    ( "relocation symbols in a section it lacks",
      patch_section well_formed 6 sh_link 4 99 );
    ("a relocation target it lacks", patch_section well_formed 6 sh_info 4 99);
    ("a symbol in a section it lacks", patch well_formed (symbol_entry + 6) 2 99);
    ( "a relocation's symbol the table lacks",
      patch well_formed (rela_entry + 12) 4 3 );
    ("an alignment of 3", patch_section well_formed 2 sh_addralign 8 3);
    ( "an alignment of 12 outside the image",
      patch_section well_formed 4 sh_addralign 8 12 );
    ("two sections overlapping in the image", addressed ~text:0x100 ~data:0x103);
    ( "two sections sharing bytes of the object",
      patch_section well_formed 2 sh_offset 8 ehdr_size );
    ("names longer than the object", shared_names);
    ( "an image longer than max_int",
      write [ section ~addr:0 ".a" "A"; section ~addr:(max_int - 1) ".b" "AB" ]
    );
    ( "an image aligned past max_int",
      write
        [
          section ~align:max_align ".a" "A";
          section ~align:max_align ".b" "B";
          section ~align:max_align ".c" "C";
        ] );
    ( "a section the image holds at a multiple of half its alignment",
      write [ section ~addr:0x102 ~align:4 ".text" "ABCD" ] );
    ( "a symbol value past the int range",
      patch well_formed (symbol_entry + 15) 1 0x40 );
    ( "an address past the int range",
      patch_section (addressed ~text:0x100 ~data:0x200) 2 sh_addr 8 (-16) );
    ( "a relocation past its section's end",
      with_relocation ~target:1 (5, 1, 1, 0) );
    ( "a relocation at the end of the image",
      with_relocation ~target:2 (8, 1, 1, 0) );
    ("a relocation into .tbss", with_relocation ~target:3 (0, 1, 1, 0));
    ("a relocation into .init_array", into_init_array);
  ]

let test_not_elf () =
  contains ~msg:"what a loader reports" ~sub:"not an ELF object"
    (refused "not an object")

let test_near_refusals () =
  ignore (read ~align:1 well_formed);
  ignore (read (addressed ~text:0x100 ~data:0x104));
  ignore (read (patch_section well_formed 2 sh_addralign 8 0));
  ignore (read (with_relocation ~target:1 (3, 1, 1, 0)));
  ignore (read (with_relocation ~target:2 (7, 1, 1, 0)));
  ignore (read (write [ section ~addr:0x104 ~align:4 ".text" "ABCD" ]));
  ignore
    (read
       (write
          [
            section ~addr:0x100 ".text" "A";
            { (tbss ~addr:0x103 ".tbss" 8) with align = 8 };
          ]));
  ignore (read (patch well_formed (symbol_entry + 15) 1 0xff));
  ignore
    (read
       (write
          [ section ~addr:0 ".a" "A"; section ~addr:(max_int - 2) ".b" "AB" ]))

(* [msg] says which cause the documentation lists an object breaks. *)
let test_messages () =
  let causes =
    [
      ("not ELF", "not an object");
      ("past the end", String.sub well_formed 0 (String.length well_formed - 1));
      ("a section it lacks", patch_section well_formed 4 sh_link 4 99);
      ("alignment", patch_section well_formed 2 sh_addralign 8 3);
      ("overlap", addressed ~text:0x100 ~data:0x103);
      ("shared bytes", patch_section well_formed 2 sh_offset 8 ehdr_size);
      ("names", shared_names);
      ( "too long",
        write
          [ section ~addr:0 ".a" "A"; section ~addr:(max_int - 1) ".b" "AB" ] );
      ("past its section", with_relocation ~target:1 (5, 1, 1, 0));
      ("outside the image", with_relocation ~target:3 (0, 1, 1, 0));
    ]
  in
  let messages = List.map (fun (c, obj) -> (c, refused obj)) causes in
  List.iteri
    (fun i (c, m) ->
      List.iteri
        (fun j (c', m') ->
          if i < j then not_equal ~msg:(c ^ " and " ^ c') string m m')
        messages)
    messages

let test_bad_align () =
  List.iter
    (fun align ->
      raises_match ~msg:(strf "align %d" align)
        (Exn.invalid_arg ?substring:None) (fun () ->
          Elf.of_string ~align well_formed))
    [ 0; -1; -4; 3; 6; 96; min_int; max_int ]

let test_largest_align () =
  let align = max_align in
  equal ~msg:"one section at 0" string "ABCD"
    (image (read ~align (write [ section ".text" "ABCD" ])));
  let o =
    read ~align (write [ section ".text" "ABCD"; section ".data" "EF" ])
  in
  equal ~msg:"the next at the alignment" (option int) (Some align)
    (section_named o ".data").offset;
  equal ~msg:"an image no memory holds" int (align + 2) o.size

(* The law: objects written from random sections *)

type part = Code | Note | Bss | Tbss

type case = {
  parts : sh list;
  syms : sym list;
  relocs : (bool * int * (int * int * int * int) list) list;
  align : int;
}

let pp_case ppf c =
  List.iteri
    (fun i (s : sh) ->
      Format.fprintf ppf
        "section %d: %S type=%d flags=%d addr=%#x size=%d align=%d %S@\n"
        (i + 1) s.name s.kind s.flags s.addr s.size s.align s.contents)
    c.parts;
  List.iter
    (fun s ->
      Format.fprintf ppf "symbol %S shndx=%#x value=%#x@\n" s.name s.shndx
        s.value)
    c.syms;
  List.iter
    (fun (explicit, t, es) ->
      List.iter
        (fun (o, s, k, a) ->
          Format.fprintf ppf
            "%s of %d: offset=%d symbol=%d kind=%d addend=%d@\n"
            (if explicit then "rela" else "rel")
            t o s k a)
        es)
    c.relocs;
  Format.fprintf ppf "align=%d" c.align

let names = [ ""; "a"; "b"; "k.kd" ]

(* Allocated sections, with bytes or without, but thread-local ones without. *)
let in_image (s : sh) =
  s.flags land shf_alloc <> 0
  && (s.kind = sht_progbits || (s.kind = sht_nobits && s.flags land shf_tls = 0))

let gen_part =
  let open Gen in
  let+ part = of_list [ Code; Code; Note; Bss; Tbss ]
  and+ contents = string_of ~size:(int_range 0 12) char
  and+ align = of_list [ 0; 1; 2; 4; 8; 64 ]
  and+ bss_size = int_range 0 16
  and+ gap = int_range 0 9
  and+ key = int_range 0 99 in
  let s =
    match part with
    | Code -> section ~align ".code" contents
    | Note -> note ".note" contents
    | Bss -> bss ".bss" bss_size
    | Tbss -> tbss ".tbss" bss_size
  in
  (s, gap, key)

(* An addressed object's sections, at increasing addresses from [start] with
   [gap] bytes before each, listed in the order of their [key]. *)
let at_addresses ~start parts =
  let _, placed =
    List.fold_left_map
      (fun at ((s : sh), gap, key) ->
        let a = max 1 s.align in
        let addr = (at + gap + a - 1) / a * a in
        (addr + s.size, (key, { s with addr })))
      start parts
  in
  List.map snd (List.stable_sort (fun (a, _) (b, _) -> Int.compare a b) placed)

(* A symbol of section [i] lies at its start, at its end, one past it, or in its
   middle. *)
let gen_sym (parts : sh array) =
  let open Gen in
  let+ name = of_list names
  and+ how = int_range 0 4
  and+ i = int_range 1 (Array.length parts)
  and+ value = int_range 0 0xffff
  and+ at = int_range 0 3 in
  let s = parts.(i - 1) in
  match how with
  | 0 -> undefined name
  | 1 -> defined name shn_abs value
  | 2 -> defined name shn_common 8
  | _ ->
      let rel = [| 0; s.size; s.size + 1; s.size / 2 |].(at) in
      defined name i (s.addr + rel)

(* Relocations of program sections, whose entries are offsets in their
   section. *)
let gen_relocs (parts : sh array) nsyms =
  let open Gen in
  let targets =
    List.filter
      (fun i -> parts.(i - 1).kind = sht_progbits && parts.(i - 1).size > 0)
      (List.init (Array.length parts) (fun i -> i + 1))
  in
  let gen_entry size =
    let+ offset = int_range 0 (size - 1)
    and+ sym = int_range 0 nsyms
    and+ kind = int_range 0 0xffff
    and+ addend = int_range (-64) 64 in
    (offset, sym, kind, addend)
  in
  if targets = [] then constant []
  else
    list ~size:(int_range 0 3)
      (let* t = of_list targets in
       let+ explicit = bool
       and+ es = list ~size:(int_range 1 3) (gen_entry parts.(t - 1).size) in
       let es =
         if explicit then es else List.map (fun (o, s, k, _) -> (o, s, k, 0)) es
       in
       (explicit, t, es))

let gen_case =
  let open Gen in
  with_pp pp_case
    (let* addressed = bool in
     let* start = int_range 0 9 in
     let* parts = list ~size:(int_range 1 5) gen_part in
     let parts =
       Array.of_list
         (if addressed then at_addresses ~start parts
          else List.map (fun (s, _, _) -> s) parts)
     in
     let* syms = list ~size:(int_range 0 6) (gen_sym parts) in
     let* relocs = gen_relocs parts (List.length syms) in
     let+ align = of_list [ 1; 2; 16; 128 ] in
     { parts = Array.to_list parts; syms; relocs; align })

(* The object of [c]: its parts, its symbol table and string table, then its
   relocation sections. *)
let object_of c =
  let n = List.length c.parts in
  let entries, names = symbols c.syms in
  let addr t = (List.nth c.parts (t - 1)).addr in
  let relocs =
    List.map
      (fun (explicit, t, es) ->
        let es = List.map (fun (o, s, k, a) -> (addr t + o, s, k, a)) es in
        if explicit then rela_section ~link:(n + 1) ~info:t es
        else
          rel_section ~link:(n + 1) ~info:t
            (List.map (fun (o, s, k, _) -> (o, s, k)) es))
      c.relocs
  in
  write (c.parts @ [ symtab ~link:(n + 2) entries; strtab names ] @ relocs)

(* The model: each part's image offset, as the documentation lays out the
   image. *)
let offsets c =
  let held = List.filter in_image c.parts in
  let addressed = List.exists (fun s -> s.addr <> 0) held in
  let round x a = (x + a - 1) / a * a in
  let start =
    let low = List.fold_left (fun m (s : sh) -> min m s.addr) max_int held in
    let align = List.fold_left (fun m (s : sh) -> max m s.align) 1 held in
    low / align * align
  in
  snd
    (List.fold_left_map
       (fun end_ (s : sh) ->
         if not (in_image s) then (end_, None)
         else if addressed then (end_, Some (s.addr - start))
         else
           let off = round end_ (max c.align (max 1 s.align)) in
           (off + s.size, Some off))
       0 c.parts)

(* The image's alignment: its sections' largest, and align when they follow the
   image's end. *)
let model_align c =
  let held = List.filter in_image c.parts in
  let largest = List.fold_left (fun m (s : sh) -> max m s.align) 1 held in
  if List.exists (fun (s : sh) -> s.addr <> 0) held then largest
  else max c.align largest

let model_image c =
  let placed =
    List.filter_map
      (fun ((s : sh), off) -> Option.map (fun off -> (s, off)) off)
      (List.combine c.parts (offsets c))
  in
  let length =
    List.fold_left (fun l ((s : sh), off) -> max l (off + s.size)) 0 placed
  in
  let b = Bytes.make length '\000' in
  List.iter
    (fun ((s : sh), off) ->
      Bytes.blit_string s.contents 0 b off (String.length s.contents))
    placed;
  Bytes.to_string b

(* The writer puts each section's bytes after the ELF header, in order. *)
let model_sections c : Elf.section list =
  let null : Elf.section =
    {
      name = "";
      kind = 0;
      flags = 0;
      offset = None;
      size = 0;
      at = 0;
      length = 0;
    }
  in
  let section (next, acc) (s : sh) offset =
    let at, length =
      if s.kind = sht_nobits then (0, 0) else (next, String.length s.contents)
    in
    let s : Elf.section =
      {
        name = s.name;
        kind = s.kind;
        flags = s.flags;
        offset;
        size = s.size;
        at;
        length;
      }
    in
    (next + length, s :: acc)
  in
  null
  :: List.rev
       (snd (List.fold_left2 section (ehdr_size, []) c.parts (offsets c)))

let model_symbols c =
  let parts = Array.of_list c.parts and offsets = Array.of_list (offsets c) in
  let place s : Elf.place =
    if s.shndx = shn_undef then Undefined
    else if s.shndx = shn_abs then Absolute s.value
    else if s.shndx >= shn_loreserve then Outside s.shndx
    else
      let p = parts.(s.shndx - 1) in
      let rel = s.value - p.addr in
      match offsets.(s.shndx - 1) with
      | Some off when 0 <= rel && rel <= p.size ->
          Image { section = s.shndx; offset = off + rel }
      | Some _ | None -> Outside s.shndx
  in
  null_symbol :: List.map (fun s -> sym_entry s.name (place s)) c.syms

let model_relocations c : Elf.relocation list =
  let symbols = Array.of_list (model_symbols c) in
  let offsets = Array.of_list (offsets c) in
  List.concat_map
    (fun (explicit, t, es) ->
      match offsets.(t - 1) with
      | None -> []
      | Some off ->
          List.map
            (fun (o, s, kind, a) : Elf.relocation ->
              let symbol = if s = 0 then no_symbol else symbols.(s) in
              let addend : Elf.addend =
                if explicit then Explicit a
                else
                  let sec = List.nth (model_sections c) t in
                  Implicit { at = sec.at + o; length = sec.length - o }
              in
              { offset = off + o; kind; addend; symbol })
            es)
    c.relocs

let model_symbol c name =
  List.find_map
    (fun (s : Elf.symbol) ->
      match s.place with
      | Image { offset; _ } when s.name = name && name <> "" -> Some offset
      | _ -> None)
    (model_symbols c)

let read_case c = read ~align:c.align (object_of c)

let law_tables c =
  let o = read_case c in
  let n = List.length c.parts in
  equal ~msg:"sections" (list elf_section) (model_sections c)
    (Iarray.to_list (Iarray.sub o.sections ~pos:0 ~len:(n + 1)));
  equal ~msg:"symbols" (list symbol) (model_symbols c) (syms o);
  equal ~msg:"relocations" (list relocation) (model_relocations c) o.relocations;
  invariants o

let law_image c =
  let offsets = offsets c in
  let held = List.filter_map Fun.id offsets in
  cover "addressed" (List.exists (fun s -> in_image s && s.addr <> 0) c.parts);
  cover "appended" (held <> [] && List.for_all (fun s -> s.addr = 0) c.parts);
  cover "a section without bytes in the image"
    (List.exists
       (fun s -> in_image s && s.kind = sht_nobits && s.size > 0)
       c.parts);
  cover "an empty section in the image"
    (List.exists (fun s -> in_image s && s.size = 0) c.parts);
  cover "a section past align"
    (List.exists (fun s -> in_image s && s.align > c.align) c.parts);
  cover "a gap" (String.contains (model_image c) '\000');
  cover "an image starting past address 0" ((read_case c).address > 0);
  let o = read_case c in
  equal ~msg:"image" string (model_image c) (image o);
  equal ~msg:"its alignment" int (model_align c) o.align

let law_lookup c =
  let o = read_case c in
  cover "a symbol at its section's end"
    (List.exists
       (fun (s : sym) ->
         s.shndx > 0 && s.shndx < shn_loreserve
         && s.value
            = (List.nth c.parts (s.shndx - 1)).addr
              + (List.nth c.parts (s.shndx - 1)).size)
       c.syms);
  cover "a name twice"
    (List.exists
       (fun n ->
         List.length (List.filter (fun (s : sym) -> s.name = n) c.syms) > 1)
       names);
  List.iter
    (fun name ->
      equal ~msg:name (option int) (model_symbol c name) (Elf.symbol o name))
    names

(* Corrupted objects *)

(* Real objects. fixtures/README.md says how each is made. *)

let fixture path =
  In_channel.with_open_bin
    (Filename.concat "fixtures" path)
    In_channel.input_all

let cubin_path = "simple_add_sm89.cubin"
let global_path = "global_sm89.cubin"
let hsaco_path = "amd_gfx1100.hsaco"
let amd_object_path = "amd_gfx1100.o"
let stripped_path = "amd_128_gfx1100.hsaco"
let host_path target = "host_" ^ target ^ ".o"

let corruptible =
  lazy
    [|
      fixture cubin_path;
      fixture hsaco_path;
      fixture amd_object_path;
      fixture (host_path "aarch64");
      well_formed;
      write ~extended:true
        [ section ~addr:0x100 ".text" "ABCD"; section ~addr:0x200 ".data" "EF" ];
    |]

(* [edits] write a byte in the header (region 0), in the section headers (1) or
   anywhere (2); [cut] truncates the object. *)
let corrupt (which, edits, cut) =
  let obj = (Lazy.force corruptible).(which) in
  let b = Bytes.of_string obj in
  let len = Bytes.length b in
  let headers = min (shoff obj) len in
  List.iter
    (fun (region, at, byte) ->
      let at =
        match region with
        | 0 -> at mod ehdr_size
        | 1 -> headers + (at mod max 1 (len - headers))
        | _ -> at mod len
      in
      Bytes.set_uint8 b at byte)
    edits;
  let obj = Bytes.to_string b in
  match cut with None -> obj | Some n -> String.sub obj 0 (n mod (len + 1))

let gen_corruption =
  let open Gen in
  let byte =
    frequency
      [
        (3, int_range 0 255);
        (2, of_list ~pp:Format.pp_print_int [ 0; 1; 0x7f; 0x80; 0xff ]);
      ]
  in
  triple (int_range 0 4)
    (list ~size:(int_range 1 4)
       (triple (int_range 0 2) (int_range 0 0xfffff) byte))
    (frequency
       [ (4, constant None); (1, map Option.some (int_range 0 0xfffff)) ])

let law_total corruption =
  let obj = corrupt corruption in
  match in_linear_memory obj with
  | Error _ -> cover "refused" true
  | Ok o ->
      cover "read" true;
      invariants o;
      Iarray.iter
        (fun (s : Elf.symbol) ->
          Option.iter
            (fun off -> less ~msg:s.name int ~than:(o.size + 1) off)
            (Elf.symbol o s.name))
        o.symbols

(* 32-bit objects *)

let ehdr32_size = 52
let shdr32_size = 40

(* A 32-bit object of [sections], with its names in a last [.shstrtab]. *)
let write32 ?(kind = 1) ?(machine = 3) sections =
  let names = Buffer.create 64 in
  Buffer.add_char names '\000';
  let name_of n =
    let at = Buffer.length names in
    Buffer.add_string names n;
    Buffer.add_char names '\000';
    at
  in
  let at = List.map (fun (s : sh) -> name_of s.name) sections in
  let at = at @ [ name_of ".shstrtab" ] in
  let sections =
    sections
    @ [ section ~kind:sht_strtab ~flags:0 ".shstrtab" (Buffer.contents names) ]
  in
  let body = Buffer.create 256 and headers = Buffer.create 256 in
  Buffer.add_string headers (String.make shdr32_size '\000');
  List.iter2
    (fun (s : sh) name ->
      let offset = ehdr32_size + Buffer.length body in
      if s.kind <> sht_nobits then Buffer.add_string body s.contents;
      List.iter
        (fun v -> Buffer.add_int32_le headers (Int32.of_int v))
        [
          name;
          s.kind;
          s.flags;
          s.addr;
          offset;
          s.size;
          s.link;
          s.info;
          s.align;
          s.entsize;
        ])
    sections at;
  let b = Buffer.create 1024 in
  Buffer.add_string b "\x7fELF\001\001\001";
  Buffer.add_string b (String.make 9 '\000');
  Buffer.add_uint16_le b kind;
  Buffer.add_uint16_le b machine;
  Buffer.add_int32_le b 1l;
  Buffer.add_int32_le b 0l;
  Buffer.add_int32_le b 0l;
  Buffer.add_int32_le b (Int32.of_int (ehdr32_size + Buffer.length body));
  Buffer.add_int32_le b 0l;
  List.iter (Buffer.add_uint16_le b)
    [
      ehdr32_size;
      0;
      0;
      shdr32_size;
      List.length sections + 1;
      List.length sections;
    ];
  Buffer.add_buffer b body;
  Buffer.add_buffer b headers;
  Buffer.contents b

(* A 32-bit symbol table's entries, after the null symbol, and its names. *)
let symbols32 syms =
  let names = Buffer.create 64 and table = Buffer.create 64 in
  Buffer.add_char names '\000';
  Buffer.add_string table (String.make 16 '\000');
  List.iter
    (fun s ->
      Buffer.add_int32_le table (Int32.of_int (Buffer.length names));
      Buffer.add_string names s.name;
      Buffer.add_char names '\000';
      Buffer.add_int32_le table (Int32.of_int s.value);
      Buffer.add_int32_le table 0l;
      Buffer.add_uint8 table 0x10;
      Buffer.add_uint8 table 0;
      Buffer.add_uint16_le table s.shndx)
    syms;
  (Buffer.contents table, Buffer.contents names)

(* 32-bit relocations: [r_info] holds the symbol above the type's 8 bits. *)
let relocs32 entries =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind, addend) ->
      Buffer.add_int32_le b (Int32.of_int offset);
      Buffer.add_int32_le b (Int32.of_int ((sym lsl 8) lor kind));
      Option.iter (fun a -> Buffer.add_int32_le b (Int32.of_int a)) addend)
    entries;
  Buffer.contents b

let test_elf32 () =
  let entries, names =
    symbols32 [ defined "f" 1 0; defined "d" 2 4; undefined "ext" ]
  in
  let o =
    read
      (write32
         [
           section ~align:4 ".text" "ABCD";
           section ~align:8 ".data" "01234567";
           section ~kind:sht_symtab ~flags:0 ~link:4 ~entsize:16 ".symtab"
             entries;
           strtab names;
           section ~kind:sht_rela ~flags:0 ~link:3 ~info:1 ~entsize:12
             ".rela.text"
             (relocs32 [ (0, 2, 5, Some (-4)) ]);
           section ~kind:sht_rel ~flags:0 ~link:3 ~info:1 ~entsize:8 ".rel.text"
             (relocs32 [ (2, 3, 6, None) ]);
         ])
  in
  invariants o;
  equal ~msg:"its class" int 32 o.bits;
  equal ~msg:"its type" int 1 o.kind;
  equal ~msg:"its machine" int 3 o.machine;
  equal ~msg:"text, then data at its alignment" string
    ("ABCD" ^ String.make 4 '\000' ^ "01234567")
    (image o);
  equal ~msg:"the symbols" (list symbol)
    [
      null_symbol;
      sym_entry "f" (Image { section = 1; offset = 0 });
      sym_entry "d" (Image { section = 2; offset = 12 });
      sym_entry "ext" Undefined;
    ]
    (syms o);
  equal ~msg:"the relocations" (list relocation)
    [
      {
        offset = 0;
        kind = 5;
        addend = Explicit (-4);
        symbol = sym_entry "d" (Image { section = 2; offset = 12 });
      };
      {
        offset = 2;
        kind = 6;
        addend = Implicit { at = (section_named o ".text").at + 2; length = 2 };
        symbol = sym_entry "ext" Undefined;
      };
    ]
    o.relocations

(* The shape of NVIDIA's Blackwell boot firmware: no type, four allocated
   sections with OS and processor flags, read by name. *)
let test_firmware32 () =
  let flags = shf_alloc lor 0x100 lor 0x1000_0000 in
  let parts =
    [
      ("hash", "H");
      ("signature", "SIG");
      ("publickey", "KEY");
      ("image", "IMAGE");
    ]
  in
  let o =
    read
      (write32 ~kind:0 ~machine:0
         (List.map (fun (n, c) -> section ~flags n c) parts))
  in
  invariants o;
  List.iter
    (fun (n, c) -> equal ~msg:n string c (contents o (section_named o n)))
    parts

(* Real objects *)

let test_cubin () =
  let o = read ~align:128 (fixture cubin_path) in
  invariants o;
  equal ~msg:"an executable" int 2 o.kind;
  equal ~msg:"for NVIDIA GPUs" int 190 o.machine;
  equal ~msg:"under the CUDA ABI" int 51 o.os_abi;
  equal ~msg:"its version" int 7 o.abi_version;
  equal ~msg:"for sm_89" int 0x590559 o.flags;
  equal ~msg:"debugging information stays out" (option int) None
    (section_named o ".debug_frame").offset;
  equal ~msg:"the constant bank first" (option int) (Some 0)
    (section_named o ".nv.constant0.simple_add").offset;
  equal ~msg:"its 380 bytes, then the code at 384" (option int) (Some 384)
    (section_named o ".text.simple_add").offset;
  equal ~msg:"the image ends with the code" int 896 o.size;
  equal ~msg:"the kernel's symbol" (option int) (Some 384)
    (Elf.symbol o "simple_add");
  equal ~msg:"symbols by index" symbol
    (sym_entry "simple_add" (Image { section = 11; offset = 384 }))
    (Iarray.get o.symbols 6);
  equal ~msg:"the debugging relocation is left out" (list relocation) []
    o.relocations

(* global_sm89.cubin: an uninitialised __device__ global, [scratch], in
   .nv.global (1 KiB, SHT_NOBITS), which .nv.constant4 relocates to. At an
   alignment of 128: .nv.constant4 at 0, the kernel's bank at 128, its code at
   512, then .nv.global at 1024. *)
let test_global () =
  let o = read ~align:128 (fixture global_path) in
  invariants o;
  equal ~msg:"the global's section takes 1 KiB of the image, no file byte"
    elf_section
    {
      name = ".nv.global";
      kind = sht_nobits;
      flags = shf_alloc lor shf_write;
      offset = Some 1024;
      size = 1024;
      at = 0;
      length = 0;
    }
    (section_named o ".nv.global");
  equal ~msg:"the image ends with it" int 2048 o.size;
  equal ~msg:"zeros in the image" string (String.make 1024 '\000')
    (String.sub (image o) 1024 1024);
  let scratch = sym_entry "scratch" (Image { section = 14; offset = 1024 }) in
  equal ~msg:"the global" symbol scratch (symbol_named o "scratch");
  equal ~msg:"the bank's relocation reaches it" (list relocation)
    [
      {
        offset = 0;
        kind = 2;
        addend =
          Implicit { at = (section_named o ".nv.constant4").at; length = 8 };
        symbol = scratch;
      };
    ]
    o.relocations

(* amd_gfx1100.hsaco: .rodata (64 bytes, aligned to 64) at 0x600 and .text
   (0x280 bytes, aligned to 256) at 0x1700; .dynamic at 0x2980 holds no bytes of
   the image, and .relro_padding and .bss after it are zeros in it. *)
let test_hsaco () =
  let o = read (fixture hsaco_path) in
  invariants o;
  equal ~msg:"a shared object" int 3 o.kind;
  equal ~msg:"for AMD GPUs" int 224 o.machine;
  equal ~msg:"under the HSA ABI" int 64 o.os_abi;
  equal ~msg:"code object version 5" int 3 o.abi_version;
  equal ~msg:"for gfx1100" int 0x41 o.flags;
  equal ~msg:".rodata's address, a multiple of .text's alignment of 256" int
    0x600 o.address;
  equal ~msg:"the image ends with .bss at 0x33f0" int 0x33f4 o.size;
  equal ~msg:"zeros between .rodata and .text" string
    (String.make (0x1100 - 0x40) '\000')
    (String.sub (image o) 0x40 (0x1100 - 0x40));
  equal ~msg:"the comment stays out" (option int) None
    (section_named o ".comment").offset;
  equal ~msg:"the kernel descriptor" (option int) (Some 0)
    (Elf.symbol o "add.kd");
  equal ~msg:"the kernel" (option int) (Some 0x1100) (Elf.symbol o "add");
  equal ~msg:"the symbol table over the dynamic one" int 11
    (Iarray.length o.symbols);
  equal ~msg:"a register count" symbol
    (sym_entry "add.num_vgpr" (Absolute 4))
    (symbol_named o "add.num_vgpr");
  equal ~msg:"a symbol in .dynamic" symbol
    (sym_entry "_DYNAMIC" (Outside 8))
    (symbol_named o "_DYNAMIC");
  equal ~msg:"a symbol in .bss" symbol
    (sym_entry "last" (Image { section = 10; offset = 0x33f0 }))
    (symbol_named o "last")

(* amd_128_gfx1100.hsaco: 128 kernels, linked without .symtab. Its image starts
   at .rodata's 0x18680 rounded down to .text's alignment of 256, and ends with
   .bss after .text at 0x1b700, 0xd2f4 bytes. *)
let test_stripped () =
  let o = read (fixture stripped_path) in
  invariants o;
  equal ~msg:"no symbol table" (option int) None
    (Iarray.find_index (fun (s : Elf.section) -> s.name = ".symtab") o.sections);
  equal ~msg:"the dynamic symbols" int 385 (Iarray.length o.symbols);
  equal ~msg:"a kernel descriptor" (option int) (Some 0x80)
    (Elf.symbol o "add0000.kd");
  equal ~msg:"its code" (option int) (Some 0x3100) (Elf.symbol o "add0000");
  equal ~msg:"the image" int 0xd2f4 o.size

(* amd_gfx1100.o: .text (0x280 bytes), .rodata at its alignment of 64, then
   .bss. The code's four relocations reach [last] in .bss; the descriptor's
   reaches the kernel. *)
let test_relocatable_amd () =
  let o = read (fixture amd_object_path) in
  invariants o;
  equal ~msg:"a relocatable object" int 1 o.kind;
  equal ~msg:".text, .rodata, then .bss" int 0x2c4 o.size;
  equal ~msg:"the descriptor in .rodata" (option int) (Some 0x280)
    (Elf.symbol o "add.kd");
  let last = sym_entry "last" (Image { section = 7; offset = 0x2c0 }) in
  let r offset kind addend symbol : Elf.relocation =
    { offset; kind; addend = Explicit addend; symbol }
  in
  equal ~msg:"the relocations" (list relocation)
    [
      r 0x38 10 4 last;
      r 0x40 11 0xc last;
      r 0x78 10 4 last;
      r 0x80 11 0xc last;
      r 0x290 5 0x10 (sym_entry "add" (Image { section = 2; offset = 0 }));
    ]
    o.relocations

(* Host objects *)

(* The call to [ext] is the relocation a loader fills with a slot that jumps to
   it: its kind, its addend and the undefined symbol by name. x86_64 puts
   [table] 0x80 bytes into .rodata.cst16, behind the kernels' constants. *)
let test_host (target, call, addend, table) =
  let o = read (fixture (host_path target)) in
  invariants o;
  let text = section_named o ".text" in
  equal ~msg:"the function" (option int) text.offset (Elf.symbol o "f");
  equal ~msg:"the table" (option int)
    (Option.map (( + ) table) (section_named o ".rodata.cst16").offset)
    (Elf.symbol o "table");
  let bss =
    require_some
      (Iarray.find_index (fun (s : Elf.section) -> s.name = ".bss") o.sections)
  in
  equal ~msg:"the counter is in .bss, in the image" symbol
    (sym_entry "counter"
       (Image
          {
            section = bss;
            offset = Option.get (Iarray.get o.sections bss).offset;
          }))
    (symbol_named o "counter");
  equal ~msg:"the empty .note.GNU-stack stays out" (option int) None
    (section_named o ".note.GNU-stack").offset;
  let calls =
    List.filter
      (fun (r : Elf.relocation) -> r.symbol.name = "ext")
      o.relocations
  in
  equal ~msg:"the call"
    (list (pair int elf_addend))
    [ (call, Explicit addend) ]
    (List.map (fun (r : Elf.relocation) -> (r.kind, r.addend)) calls);
  equal ~msg:"names an undefined symbol" (list symbol)
    [ sym_entry "ext" Undefined ]
    (List.map (fun (r : Elf.relocation) -> r.symbol) calls)

(* x86_64's unwind tables are allocated program data of their own type: the
   image holds them, and their relocations patch it. *)
let test_unwind () =
  let o = read (fixture "host_unwind_x86_64.o") in
  invariants o;
  let eh = section_named o ".eh_frame" in
  equal ~msg:"its type" int sht_x86_64_unwind eh.kind;
  let off = require_some ~msg:"the image holds .eh_frame" eh.offset in
  let into_eh (r : Elf.relocation) =
    r.offset >= off && r.offset < off + eh.size
  in
  equal ~msg:"its relocations, one per function" int 8
    (List.length (List.filter into_eh o.relocations))

(* Each test's limit, in seconds. *)
let timeout = 30.

let () =
  exit
  @@ run "rig_elf"
       [
         test ~timeout "an object's header fields" test_header;
         group ~timeout "layout"
           [
             test "sections without addresses follow the image's end in order"
               test_appended;
             test "each at a multiple of align and of its own alignment"
               test_align;
             test "the image ends where its last section ends" test_no_padding;
             test "sections with addresses go at them, gaps as zeros"
               test_addressed;
             test "a section without bytes is zeros in the image" test_bss;
             test "the image holds what the loader's memory holds" test_held;
             test "the code and data a loader's memory holds" test_allocated;
             test "every section by index" test_sections;
             test "an object without section names" test_no_names;
             prop ~count:300
               "the image holds each program section at its offset" gen_case
               law_image;
             prop ~count:300 "reading allocates as much at any address"
               gen_address law_address;
           ];
         group ~timeout "symbols"
           [
             test "each kind of place" test_places;
             test "an addressed object's symbol values are addresses"
               test_addressed_symbols;
             test "a lookup takes the first by index in the image" test_lookup;
             test "an object without a symbol table has none" test_no_symbols;
             test "the dynamic symbols when there is no symbol table"
               test_dynamic_symbols;
             prop ~count:300 "a lookup agrees with the table" gen_case
               law_lookup;
           ];
         group ~timeout "relocations"
           [
             test "kinds, addends and symbols, in order" test_relocations;
             test "each resolves through its own section's table" test_own_table;
             test "those of unloaded sections are left out"
               test_unloaded_relocations;
             test "a dynamic relocation patches an address in the image"
               test_dynamic_relocations;
             test "one without an addend has it in the object's bytes"
               test_implicit;
             test "one at a thread-local section's addresses"
               test_tbss_addresses;
           ];
         group ~timeout "extended numbering"
           [
             test "the section count in the null section" test_extended_header;
             test "a symbol's section in the extended index table" test_xindex;
             test "an object of more than 65,279 sections" test_many_sections;
             test "70,000 symbols at extended indexes, in linear time"
               test_many_symbols;
             test "40,000 relocation sections, in linear time"
               test_many_relocation_sections;
           ];
         prop ~timeout ~count:300 "an object reads back as written" gen_case
           law_tables;
         group ~timeout "refusals"
           [
             cases ~name:fst "a malformed object is an error" refusals
               (fun (_, obj) ->
                 ignore (require_error ~pp:pp_t (in_linear_memory obj)));
             test "a text is not an ELF object" test_not_elf;
             test "the neighbours of refusals read" test_near_refusals;
             test "each cause has its own message" test_messages;
             test "align must be a positive power of two" test_bad_align;
             test "the largest align" test_largest_align;
             prop ~count:2000 "a corrupted object reads or is refused"
               gen_corruption law_total;
           ];
         group ~timeout "32-bit objects"
           [
             test "sections, symbols and relocations" test_elf32;
             test "firmware read by section name" test_firmware32;
           ];
         group ~timeout "real objects"
           [
             test "an NVIDIA cubin" test_cubin;
             test "an NVIDIA cubin with an uninitialised global" test_global;
             test "an AMD code object" test_hsaco;
             test "an AMD code object without its symbol table" test_stripped;
             test "a relocatable AMD object" test_relocatable_amd;
             cases
               ~name:(fun (t, _, _, _) -> "a host object for " ^ t)
               "host objects"
               [ ("x86_64", 4, -4, 0x80); ("aarch64", 283, 0, 0) ]
               test_host;
             test "an x86_64 object's unwind tables" test_unwind;
           ];
       ]
