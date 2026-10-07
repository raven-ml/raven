(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ELF reader over objects a small writer builds, over real GPU and host
   objects, and over corrupted objects. *)

open Windtrap
module Elf = Device_elf

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
let shf_write = 0x1
let shf_alloc = 0x2
let shf_execinstr = 0x4
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
  let offsets =
    List.map
      (fun (s : sh) ->
        let at = ehdr_size + Buffer.length body in
        if s.kind <> sht_nobits then Buffer.add_string body s.contents;
        at)
      sections
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
  List.iter2
    (fun (s : sh) (name, offset) -> add_section_header b ~name s ~offset)
    sections
    (List.combine names offsets);
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
        "{ name = %S; kind = %d; flags = %#x; offset = %a; size = %d; contents \
         = %S }"
        s.name s.kind s.flags
        (Format.pp_print_option
           ~none:(fun ppf () -> Format.pp_print_string ppf "None")
           (fun ppf -> Format.fprintf ppf "Some %#x"))
        s.offset s.size s.contents)
    ~equal:( = )

let relocation =
  Testable.make
    ~pp:(fun ppf (r : Elf.relocation) ->
      Format.fprintf ppf "{ offset = %#x; kind = %d; addend = %d; symbol = %a }"
        r.offset r.kind r.addend pp_symbol r.symbol)
    ~equal:( = )

let pp_t ppf (o : Elf.t) =
  Format.fprintf ppf "<object: image of %d bytes, %d sections>"
    (String.length o.image) (Iarray.length o.sections)

let read ?align obj =
  require_ok ~pp:Format.pp_print_string (Elf.of_string ?align obj)

let refused ?align obj = require_error ~pp:pp_t (Elf.of_string ?align obj)

let section_named (o : Elf.t) name =
  require_some ~msg:name
    (Iarray.find_opt (fun (s : Elf.section) -> s.name = name) o.sections)

let symbol_named (o : Elf.t) name =
  require_some ~msg:name
    (Iarray.find_opt (fun (s : Elf.symbol) -> s.name = name) o.symbols)

let syms (o : Elf.t) = Iarray.to_list o.symbols
let sym_entry name place : Elf.symbol = { name; place }
let null_symbol = sym_entry "" Undefined
let no_symbol = sym_entry "" (Absolute 0)

(* The three equations of [Elf.t]'s documentation. *)
let invariants (o : Elf.t) =
  let length = String.length o.image in
  let count = Iarray.length o.sections in
  Iarray.iteri
    (fun i (s : Elf.section) ->
      match s.offset with
      | None -> ()
      | Some off ->
          let msg = Printf.sprintf "section %d %S" i s.name in
          at_least ~msg int ~than:0 off;
          at_least ~msg int ~than:0 s.size;
          at_most ~msg int ~than:length (off + s.size);
          equal ~msg string s.contents (String.sub o.image off s.size))
    o.sections;
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
      let msg = Format.asprintf "relocation at %#x" r.offset in
      at_least ~msg int ~than:0 r.offset;
      less ~msg int ~than:length r.offset;
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
    o.image;
  equal ~msg:"offsets"
    (list (option int))
    [ None; Some 0; None; Some 16; Some 24; None ]
    (List.map (fun (s : Elf.section) -> s.offset) (Iarray.to_list o.sections))

let test_align () =
  let obj = write [ section ".text" "ABC"; section ".data" "DEF" ] in
  equal ~msg:"each section at a multiple of align" string
    ("ABC" ^ String.make 125 '\000' ^ "DEF")
    (read ~align:128 obj).image;
  let obj = write [ section ".text" "ABC"; section ~align:64 ".data" "DEF" ] in
  equal ~msg:"the larger of align and the section's alignment" string
    ("ABC" ^ String.make 61 '\000' ^ "DEF")
    (read ~align:2 obj).image

let test_no_padding () =
  let o = read ~align:128 (write [ section ".a" "A"; section ".b" "BCD" ]) in
  equal ~msg:"the image ends where its last section ends" int 131
    (String.length o.image);
  let o = read (write [ section ".a" "A"; section ~align:64 ".empty" "" ]) in
  equal ~msg:"an empty last section ends the image at its offset" int 64
    (String.length o.image);
  let o = read (write [ note ".comment" "x" ]) in
  equal ~msg:"no section, no image" string "" o.image

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
  equal ~msg:"each at its address, zeros in the gaps, before the first too"
    string
    (String.make 8 '\000' ^ "ABCD" ^ String.make 4 '\000' ^ "RO")
    o.image;
  equal ~msg:"align has no effect" string o.image (read ~align:4096 obj).image;
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
  let o = read (write [ section ".text" "ABCD"; bss ".bss" 64 ]) in
  equal ~msg:"a section with no bytes is outside the image" elf_section
    {
      name = ".bss";
      kind = sht_nobits;
      flags = shf_alloc lor shf_write;
      offset = None;
      size = 64;
      contents = "";
    }
    (section_named o ".bss");
  equal ~msg:"and adds nothing to it" string "ABCD" o.image

let test_sections () =
  let o =
    read (write [ section ~flags:(shf_alloc lor shf_execinstr) ".text" "AB" ])
  in
  equal ~msg:"every section by index, from the null section" (list elf_section)
    [
      { name = ""; kind = 0; flags = 0; offset = None; size = 0; contents = "" };
      {
        name = ".text";
        kind = sht_progbits;
        flags = shf_alloc lor shf_execinstr;
        offset = Some 0;
        size = 2;
        contents = "AB";
      };
      {
        name = ".shstrtab";
        kind = sht_strtab;
        flags = 0;
        offset = None;
        size = 17;
        contents = "\000.text\000.shstrtab\000";
      };
    ]
    (Iarray.to_list o.sections)

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
           bss ".bss" 4;
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
  equal ~msg:"a symbol's value is its address" (list symbol)
    [
      null_symbol;
      sym_entry "k" (Image { section = 1; offset = 0x104 });
      sym_entry "k.kd" (Image { section = 2; offset = 0x200 });
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
           bss ".bss" 4;
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
    (option int) (Some 0x102)
    (Elf.symbol (read (write (text :: dynamic))) "k.kd");
  let static, names = symbols [ defined "k.kd" 1 0x101 ] in
  let o =
    read (write ((text :: dynamic) @ [ symtab ~link:5 static; strtab names ]))
  in
  equal ~msg:"the symbol table over the dynamic one" (option int) (Some 0x101)
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
  equal ~msg:"in the order of their sections, then of their entries"
    (list relocation)
    [
      { offset = 8; kind = 1; addend = -8; symbol = sym_entry "data" data };
      { offset = 2; kind = 4; addend = -4; symbol = sym_entry "ext" Undefined };
      { offset = 5; kind = 8; addend = 7; symbol = no_symbol };
      { offset = 14; kind = 9; addend = 0; symbol = sym_entry "ext" Undefined };
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
  equal ~msg:"one in a section patches its address" (list int) [ 0x102 ]
    (offsets 0x102);
  equal ~msg:"one in the zeros between sections too" (list int) [ 0x180 ]
    (offsets 0x180);
  equal ~msg:"and the image's last byte" (list int) [ 0x203 ] (offsets 0x203);
  ignore (refused (obj ~at:0x204))

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
  equal ~msg:"the image" string "ABCD" o.image;
  equal ~msg:"a symbol in a section past the header's reach" (list symbol)
    [ null_symbol; sym_entry "far" (Image { section = at; offset = 1 }) ]
    (syms o)

(* Every section past the header's reach with a symbol of its own, each at an
   extended index: reading takes time linear in the symbols. *)
let test_many_symbols () =
  let count = 70_000 in
  let filler = List.init count (fun _ -> note "" "") in
  let indexes = Buffer.create (4 * (count + 1)) in
  Buffer.add_int32_le indexes 0l;
  for i = 1 to count do
    Buffer.add_int32_le indexes (Int32.of_int i)
  done;
  let entries, names =
    symbols (List.init count (fun _ -> defined "" shn_xindex 0))
  in
  let obj =
    write
      (filler
      @ [
          symtab ~link:(count + 2) entries;
          strtab names;
          section ~kind:sht_symtab_shndx ~flags:0 ~link:(count + 1) ~align:4
            ~entsize:4 ".symtab_shndx" (Buffer.contents indexes);
        ])
  in
  let start = Sys.time () in
  let o = read obj in
  let seconds = Sys.time () -. start in
  equal ~msg:"the last symbol's section" symbol
    (sym_entry "" (Outside count))
    (Iarray.get o.symbols count);
  less ~msg:"CPU seconds" float_exact ~than:1.0 seconds

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
      bss ".bss" 8;
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

(* The longest image the reader lays out, 1 GiB. *)
let max_image = 1 lsl 30

(* An addressed object with [.bss] between [.text] and [.data], and a dynamic
   relocation at address [at]. *)
let dynamic_into ~at =
  write
    [
      section ~addr:0x100 ".text" "ABCD";
      bss ~addr:0x104 ".bss" 8;
      section ~addr:0x110 ".data" "EFGH";
      rela_section ~name:".rela.dyn" ~link:0 ~info:0 [ (at, 0, 1, 0) ];
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
    ("an image longer than 1 GiB", write [ section ~addr:max_image ".text" "A" ]);
    ( "an image aligned past 1 GiB",
      write [ section ".text" "A"; section ~align:(2 * max_image) ".data" "B" ]
    );
    ( "a section the image holds at a multiple of half its alignment",
      write [ section ~addr:0x102 ~align:4 ".text" "ABCD" ] );
    ( "a symbol value past the int range",
      patch well_formed (symbol_entry + 15) 1 0x40 );
    ( "a dynamic relocation into .bss between program sections",
      dynamic_into ~at:0x106 );
    ( "an address past the int range",
      patch_section (addressed ~text:0x100 ~data:0x200) 2 sh_addr 8 (-16) );
    ( "a relocation past its section's end",
      with_relocation ~target:1 (5, 1, 1, 0) );
    ( "a relocation at the end of the image",
      with_relocation ~target:2 (8, 1, 1, 0) );
    ("a relocation into .bss", with_relocation ~target:3 (0, 1, 1, 0));
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
            { (bss ~addr:0x103 ".bss" 8) with align = 8 };
          ]));
  ignore (read (patch well_formed (symbol_entry + 15) 1 0xff));
  ignore (read (dynamic_into ~at:0x10c))

(* [msg] says which cause the documentation lists an object breaks. *)
let test_messages () =
  let causes =
    [
      ("not ELF", "not an object");
      ("past the end", String.sub well_formed 0 (String.length well_formed - 1));
      ("a section it lacks", patch_section well_formed 4 sh_link 4 99);
      ("alignment", patch_section well_formed 2 sh_addralign 8 3);
      ("overlap", addressed ~text:0x100 ~data:0x103);
      ("too long", write [ section ~addr:max_image ".text" "A" ]);
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
      raises_match ~msg:(Printf.sprintf "align %d" align)
        (Exn.invalid_arg ?substring:None) (fun () ->
          Elf.of_string ~align well_formed))
    [ 0; -1; -4; 3; 6; 96; min_int; max_int ]

let test_largest_align () =
  let align = 1 lsl 61 in
  equal ~msg:"one section at 0" string "ABCD"
    (read ~align (write [ section ".text" "ABCD" ])).image;
  ignore
    (refused ~align (write [ section ".text" "ABCD"; section ".data" "EF" ]))

(* The law: objects written from random sections *)

type part = Code | Note | Bss

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
let in_image (s : sh) = s.kind = sht_progbits && s.flags land shf_alloc <> 0

let gen_part =
  let open Gen in
  let+ part = of_list [ Code; Code; Note; Bss ]
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
  let addressed = List.exists (fun s -> in_image s && s.addr <> 0) c.parts in
  let round x a = (x + a - 1) / a * a in
  snd
    (List.fold_left_map
       (fun end_ (s : sh) ->
         if not (in_image s) then (end_, None)
         else if addressed then (end_, Some s.addr)
         else
           let off = round end_ (max c.align (max 1 s.align)) in
           (off + s.size, Some off))
       0 c.parts)

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
    (fun ((s : sh), off) -> Bytes.blit_string s.contents 0 b off s.size)
    placed;
  Bytes.to_string b

let model_sections c : Elf.section list =
  let null : Elf.section =
    { name = ""; kind = 0; flags = 0; offset = None; size = 0; contents = "" }
  in
  null
  :: List.map2
       (fun (s : sh) offset : Elf.section ->
         let contents = if s.kind = sht_nobits then "" else s.contents in
         {
           name = s.name;
           kind = s.kind;
           flags = s.flags;
           offset;
           size = s.size;
           contents;
         })
       c.parts (offsets c)

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
    (fun (_, t, es) ->
      match offsets.(t - 1) with
      | None -> []
      | Some off ->
          List.map
            (fun (o, s, kind, addend) : Elf.relocation ->
              let symbol = if s = 0 then no_symbol else symbols.(s) in
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
  let image = List.filter_map Fun.id offsets in
  cover "addressed" (List.exists (fun s -> in_image s && s.addr <> 0) c.parts);
  cover "appended" (image <> [] && List.for_all (fun s -> s.addr = 0) c.parts);
  cover "an empty section in the image"
    (List.exists (fun s -> in_image s && s.size = 0) c.parts);
  cover "a section past align"
    (List.exists (fun s -> in_image s && s.align > c.align) c.parts);
  cover "a gap" (String.contains (model_image c) '\000');
  equal ~msg:"image" string (model_image c) (read_case c).image

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

let root = "../../../.."

let fixture path =
  In_channel.with_open_bin (Filename.concat root path) In_channel.input_all

let cubin_path = "packages/tolk/test/runtime/ops_nv/simple_add_sm89.cubin"
let amd = "packages/tolk/test/gen/runtime/ops_amd_fixtures/"
let hsaco_path = amd ^ "simple_add_gfx1100.hsaco"
let lds_path = amd ^ "lds_gfx1100.o"
let stripped_path = "packages/nx/lib/amd/kernels/gfx12-generic/unfold.8.co"

let corruptible =
  lazy
    [|
      fixture cubin_path;
      fixture hsaco_path;
      fixture lds_path;
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
  match Elf.of_string obj with
  | Error _ -> cover "refused" true
  | Ok o ->
      cover "read" true;
      invariants o;
      Iarray.iter
        (fun (s : Elf.symbol) ->
          Option.iter
            (fun off ->
              less ~msg:s.name int ~than:(String.length o.image + 1) off)
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
  equal ~msg:"its type" int 1 o.kind;
  equal ~msg:"its machine" int 3 o.machine;
  equal ~msg:"text, then data at its alignment" string
    ("ABCD" ^ String.make 4 '\000' ^ "01234567")
    o.image;
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
        addend = -4;
        symbol = sym_entry "d" (Image { section = 2; offset = 12 });
      };
      { offset = 2; kind = 6; addend = 0; symbol = sym_entry "ext" Undefined };
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
    (fun (n, c) -> equal ~msg:n string c (section_named o n).contents)
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
  equal ~msg:"the image ends with the code" int 896 (String.length o.image);
  equal ~msg:"the kernel's symbol" (option int) (Some 384)
    (Elf.symbol o "simple_add");
  equal ~msg:"symbols by index" symbol
    (sym_entry "simple_add" (Image { section = 11; offset = 384 }))
    (Iarray.get o.symbols 6);
  equal ~msg:"the debugging relocation is left out" (list relocation) []
    o.relocations

let test_hsaco () =
  let o = read (fixture hsaco_path) in
  invariants o;
  equal ~msg:"a shared object" int 3 o.kind;
  equal ~msg:"for AMD GPUs" int 224 o.machine;
  equal ~msg:"under the HSA ABI" int 64 o.os_abi;
  equal ~msg:"code object version 5" int 3 o.abi_version;
  equal ~msg:"for gfx1100" int 0x41 o.flags;
  equal ~msg:"the image ends with .text at 0x1600" int 0x1880
    (String.length o.image);
  equal ~msg:"zeros before .rodata" string (String.make 0x5c0 '\000')
    (String.sub o.image 0 0x5c0);
  equal ~msg:"and between .rodata and .text" string
    (String.make (0x1600 - 0x600) '\000')
    (String.sub o.image 0x600 (0x1600 - 0x600));
  equal ~msg:"the comment stays out" (option int) None
    (section_named o ".comment").offset;
  equal ~msg:"the kernel descriptor" (option int) (Some 0x5c0)
    (Elf.symbol o "simple_add.kd");
  equal ~msg:"the symbol table over the dynamic one" int 10
    (Iarray.length o.symbols);
  equal ~msg:"a register count" symbol
    (sym_entry "simple_add.num_vgpr" (Absolute 6))
    (symbol_named o "simple_add.num_vgpr");
  equal ~msg:"a symbol in .dynamic" symbol
    (sym_entry "_DYNAMIC" (Outside 8))
    (symbol_named o "_DYNAMIC");
  equal ~msg:"a symbol in .bss" symbol
    (sym_entry "__hip_cuid_3477dd1aa81fd581" (Outside 10))
    (symbol_named o "__hip_cuid_3477dd1aa81fd581")

let test_stripped () =
  let o = read (fixture stripped_path) in
  invariants o;
  equal ~msg:"the dynamic symbols" (list string)
    [ ""; "u"; "u.kd"; "__hip_cuid_5224b70de10a6e29" ]
    (List.map (fun (s : Elf.symbol) -> s.name) (syms o));
  equal ~msg:"the kernel descriptor" (option int) (Some 0x5c0)
    (Elf.symbol o "u.kd");
  equal ~msg:"the code" (option int) (Some 0x1600) (Elf.symbol o "u");
  equal ~msg:"the image" int 0x2e80 (String.length o.image)

let test_relocatable_amd () =
  let o = read (fixture lds_path) in
  invariants o;
  equal ~msg:".text, then .rodata at its alignment of 64" int 0x240
    (String.length o.image);
  equal ~msg:"the descriptor in .rodata" (option int) (Some 0x200)
    (Elf.symbol o "lds.kd");
  equal ~msg:"its relocation to the code" (list relocation)
    [
      {
        offset = 0x210;
        kind = 5;
        addend = 0x10;
        symbol = sym_entry "lds" (Image { section = 2; offset = 0 });
      };
    ]
    o.relocations

(* Host objects compiled at test time *)

let clang =
  lazy
    (Sys.command (Printf.sprintf "clang --version > %s 2>&1" Filename.null) = 0)

(* [src] compiled for [target] as the host loader takes it. *)
let compile ~target src =
  if not (Lazy.force clang) then skip ~reason:"no clang" ();
  let c = temp_file ~suffix:".c" () and o = temp_file ~suffix:".o" () in
  Out_channel.with_open_bin c (fun oc -> output_string oc src);
  let cmd =
    Printf.sprintf
      "clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib \
       -fno-ident --target=%s-none-unknown-elf %s -o %s"
      target (Filename.quote c) (Filename.quote o)
  in
  if Sys.command cmd <> 0 then failf "clang failed: %s" cmd;
  In_channel.with_open_bin o In_channel.input_all

let host_source =
  {|void ext(int);
static int counter;
static const int table[4] = {1, 2, 3, 4};
void f(int i) { ext(i); counter += table[i & 3]; }
|}

(* The call to [ext] is the relocation a loader fills with a slot that jumps to
   it: its kind, its addend and the undefined symbol by name. *)
let test_host (target, call, addend) =
  let o = read (compile ~target host_source) in
  invariants o;
  let text = section_named o ".text" in
  equal ~msg:"the function" (option int) text.offset (Elf.symbol o "f");
  equal ~msg:"the table" (option int) (section_named o ".rodata.cst16").offset
    (Elf.symbol o "table");
  let bss =
    require_some
      (Iarray.find_index (fun (s : Elf.section) -> s.name = ".bss") o.sections)
  in
  equal ~msg:"the counter is outside the image" symbol
    (sym_entry "counter" (Outside bss))
    (symbol_named o "counter");
  equal ~msg:"the empty .note.GNU-stack stays out" (option int) None
    (section_named o ".note.GNU-stack").offset;
  let calls =
    List.filter
      (fun (r : Elf.relocation) -> r.symbol.name = "ext")
      o.relocations
  in
  equal ~msg:"the call"
    (list (pair int int))
    [ (call, addend) ]
    (List.map (fun (r : Elf.relocation) -> (r.kind, r.addend)) calls);
  equal ~msg:"names an undefined symbol" (list symbol)
    [ sym_entry "ext" Undefined ]
    (List.map (fun (r : Elf.relocation) -> r.symbol) calls)

let () =
  exit
  @@ run "device_elf"
       [
         test "an object's header fields" test_header;
         group "layout"
           [
             test "sections without addresses follow the image's end in order"
               test_appended;
             test "each at a multiple of align and of its own alignment"
               test_align;
             test "the image ends where its last section ends" test_no_padding;
             test "sections with addresses go at them, gaps as zeros"
               test_addressed;
             test "a section without bytes stays out" test_bss;
             test "every section by index" test_sections;
             test "an object without section names" test_no_names;
             prop ~count:300
               "the image holds each program section at its offset" gen_case
               law_image;
           ];
         group "symbols"
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
         group "relocations"
           [
             test "kinds, addends and symbols, in order" test_relocations;
             test "each resolves through its own section's table" test_own_table;
             test "those of unloaded sections are left out"
               test_unloaded_relocations;
             test "a dynamic relocation patches an address in the image"
               test_dynamic_relocations;
           ];
         group "extended numbering"
           [
             test "the section count in the null section" test_extended_header;
             test "a symbol's section in the extended index table" test_xindex;
             test "an object of more than 65,279 sections" test_many_sections;
             test "70,000 symbols at extended indexes, in linear time"
               test_many_symbols;
           ];
         prop ~count:300 "an object reads back as written" gen_case law_tables;
         group "refusals"
           [
             cases ~name:fst "a malformed object is an error" refusals
               (fun (_, obj) -> ignore (refused obj));
             test "a text is not an ELF object" test_not_elf;
             test "the neighbours of refusals read" test_near_refusals;
             test "each cause has its own message" test_messages;
             test "align must be a positive power of two" test_bad_align;
             test "the largest align" test_largest_align;
             prop ~count:2000 "a corrupted object reads or is refused"
               gen_corruption law_total;
           ];
         group "32-bit objects"
           [
             test "sections, symbols and relocations" test_elf32;
             test "firmware read by section name" test_firmware32;
           ];
         group "real objects"
           [
             test "an NVIDIA cubin" test_cubin;
             test "an AMD code object" test_hsaco;
             test "a stripped AMD code object" test_stripped;
             test "a relocatable AMD object" test_relocatable_amd;
             cases
               ~name:(fun (t, _, _) -> "a host object for " ^ t)
               "host objects"
               [ ("x86_64", 4, -4); ("aarch64", 283, 0) ]
               test_host;
           ];
       ]
