(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ELF layout over objects the test builds: the image, symbols and relocations,
   and refusals. *)

open Windtrap
module Elf = Nx_device_elf

(* A 64-bit little-endian relocatable object with the given sections, after the
   null section: (name, type, address, contents, link, info, align, entry
   size). *)
let elf sections =
  let names = Buffer.create 64 in
  Buffer.add_char names '\000';
  let name_of s =
    let at = Buffer.length names in
    Buffer.add_string names s;
    Buffer.add_char names '\000';
    at
  in
  let sections = sections @ [ (".shstrtab", 3, 0, "", 0, 0, 1, 0) ] in
  let offs = List.map (fun (n, _, _, _, _, _, _, _) -> name_of n) sections in
  let shstrtab = Buffer.contents names in
  let sections =
    List.map
      (fun ((n, k, a, _, l, i, al, e) as s) ->
        if n = ".shstrtab" then (n, k, a, shstrtab, l, i, al, e) else s)
      sections
  in
  let body = Buffer.create 256 in
  Buffer.add_string body (String.make 64 '\000');
  let placed =
    List.map
      (fun (_, _, _, c, _, _, _, _) ->
        let at = Buffer.length body in
        Buffer.add_string body c;
        at)
      sections
  in
  while Buffer.length body mod 8 <> 0 do
    Buffer.add_char body '\000'
  done;
  let shoff = Buffer.length body in
  let hdr = Bytes.make 64 '\000' in
  Bytes.blit_string "\x7fELF\002\001\001" 0 hdr 0 7;
  Bytes.set_uint16_le hdr 16 1;
  Bytes.set_uint16_le hdr 18 183;
  Bytes.set_int64_le hdr 40 (Int64.of_int shoff);
  Bytes.set_uint16_le hdr 58 64;
  Bytes.set_uint16_le hdr 60 (List.length sections + 1);
  Bytes.set_uint16_le hdr 62 (List.length sections);
  Buffer.add_string body (String.make 64 '\000');
  List.iter
    (fun ((_, k, a, c, l, i, al, e), (name, at)) ->
      let h = Bytes.make 64 '\000' in
      Bytes.set_int32_le h 0 (Int32.of_int name);
      Bytes.set_int32_le h 4 (Int32.of_int k);
      Bytes.set_int64_le h 16 (Int64.of_int a);
      Bytes.set_int64_le h 24 (Int64.of_int at);
      Bytes.set_int64_le h 32 (Int64.of_int (String.length c));
      Bytes.set_int32_le h 40 (Int32.of_int l);
      Bytes.set_int32_le h 44 (Int32.of_int i);
      Bytes.set_int64_le h 48 (Int64.of_int al);
      Bytes.set_int64_le h 56 (Int64.of_int e);
      Buffer.add_bytes body h)
    (List.combine sections (List.combine offs placed));
  let s = Buffer.to_bytes body in
  Bytes.blit hdr 0 s 0 64;
  Bytes.to_string s

let sym name shndx value =
  let b = Bytes.make 24 '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int name);
  Bytes.set_uint16_le b 6 shndx;
  Bytes.set_int64_le b 8 (Int64.of_int value);
  Bytes.to_string b

let rela offset sym kind addend =
  let b = Bytes.make 24 '\000' in
  Bytes.set_int64_le b 0 (Int64.of_int offset);
  Bytes.set_int64_le b 8
    Int64.(logor (shift_left (of_int sym) 32) (of_int kind));
  Bytes.set_int64_le b 16 (Int64.of_int addend);
  Bytes.to_string b

let symbol =
  Testable.make
    ~pp:(fun ppf (s : Elf.symbol) ->
      match s.place with
      | Undefined -> Format.fprintf ppf "{ %S; undefined }" s.name
      | Defined { section; offset } ->
          Format.fprintf ppf "{ %S; section %d; offset %d }" s.name section
            offset)
    ~equal:( = )

let test_elf () =
  let strtab = "\000start\000obj\000ext\000" in
  let obj =
    elf
      [
        (".text", 1, 0, "ABCD", 0, 0, 4, 0);
        (".data", 1, 0, "01234567", 0, 0, 16, 0);
        ( ".symtab",
          2,
          0,
          sym 0 0 0 ^ sym 1 1 0 ^ sym 7 2 4 ^ sym 11 0 0,
          4,
          0,
          8,
          24 );
        (".strtab", 3, 0, strtab, 0, 0, 1, 0);
        (".rela.text", 4, 0, rela 0 2 5 3 ^ rela 2 3 6 (-4), 3, 1, 8, 24);
      ]
  in
  let o = Elf.load obj in
  equal ~msg:"the object's type" int 1 o.kind;
  equal ~msg:"its machine" int 183 o.machine;
  equal ~msg:"text, then data at its alignment" string
    ("ABCD" ^ String.make 12 '\000' ^ "01234567")
    o.image;
  equal ~msg:"the symbol table, by index" (array symbol)
    [|
      { name = ""; place = Undefined };
      { name = "start"; place = Defined { section = 1; offset = 0 } };
      { name = "obj"; place = Defined { section = 2; offset = 20 } };
      { name = "ext"; place = Undefined };
    |]
    o.symbols;
  equal ~msg:"symbols" (option int) (Some 20) (Elf.symbol o "obj");
  equal ~msg:"a symbol at 0" (option int) (Some 0) (Elf.symbol o "start");
  (match o.relocations with
  | [ r; e ] ->
      equal ~msg:"at" int 0 r.at;
      is_true ~msg:"target" (r.target = Elf.Offset 20);
      equal ~msg:"kind" int 5 r.kind;
      equal ~msg:"addend" int 3 r.addend;
      is_true ~msg:"an undefined symbol is named" (e.target = Elf.External "ext");
      equal ~msg:"its addend" int (-4) e.addend
  | _ -> fail "two relocations");
  is_none ~msg:"an undefined symbol has no offset" (Elf.symbol o "ext");
  let aligned = Elf.load ~align:128 obj in
  equal ~msg:"a forced alignment" (option int) (Some 132)
    (Elf.symbol aligned "obj");
  let fixed =
    elf
      [
        (".text", 1, 0x100, "ABCD", 0, 0, 4, 0);
        (".symtab", 2, 0, sym 0 0 0 ^ sym 1 1 0x102, 3, 0, 8, 24);
        (".strtab", 3, 0, "\000k.kd\000", 0, 0, 1, 0);
      ]
  in
  let o = Elf.load fixed in
  equal ~msg:"a section with an address goes at it" int 0x104
    (String.length o.image);
  equal ~msg:"its symbols are image offsets" (option int) (Some 0x102)
    (Elf.symbol o "k.kd");
  let bss =
    elf
      [
        (".text", 1, 0, "ABCD", 0, 0, 4, 0);
        (".bss", 8, 0, String.make 64 'x', 0, 0, 8, 0);
      ]
  in
  (match
     List.find_opt
       (fun (s : Elf.section) -> s.name = ".bss")
       (Elf.load bss).sections
   with
  | Some s ->
      equal ~msg:"a section with no bytes has a size" int 64 s.size;
      equal ~msg:"and no contents" string "" s.contents
  | None -> fail "no .bss");
  let failure = Exn.failure ~substring:"Nx_device_elf.load" in
  raises_match failure (fun () -> Elf.load "not an elf");
  raises_match failure (fun () -> Elf.load (String.sub obj 0 100))

let () =
  exit (run "nx.device.elf" [ test "layout, symbols, relocations" test_elf ])
