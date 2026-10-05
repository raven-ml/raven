(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cubins the test builds: their image, relocations, kernels and refusals. *)

open Windtrap
module Cubin = Nx_nv_cubin
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
  Bytes.set_uint16_le hdr 18 190;
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

(* An .nv.info attribute of the format that carries data: [data] after its
   header. *)
let attr param data =
  let h = Bytes.make 4 '\000' in
  Bytes.set_uint8 h 0 4;
  Bytes.set_uint8 h 1 param;
  Bytes.set_uint16_le h 2 (String.length data);
  Bytes.to_string h ^ data

(* An attribute of a format that holds a 16-bit value in place of data. *)
let value_attr param v =
  let h = Bytes.make 4 '\000' in
  Bytes.set_uint8 h 0 3;
  Bytes.set_uint8 h 1 param;
  Bytes.set_uint16_le h 2 v;
  Bytes.to_string h

(* A symbol index, then a 32-bit value. *)
let ii sym v =
  let b = Bytes.make 8 '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int sym);
  Bytes.set_int32_le b 4 (Int32.of_int v);
  Bytes.to_string b

(* A symbol index, then two 16-bit values. *)
let ihh sym a b =
  let s = Bytes.make 8 '\000' in
  Bytes.set_int32_le s 0 (Int32.of_int sym);
  Bytes.set_uint16_le s 4 a;
  Bytes.set_uint16_le s 6 b;
  Bytes.to_string s

let progbits = 1
let nobits = 8
let regcount = 0x2f
let min_stack_size = 0x12
let param_cbank = 0xa

(* A cubin of the kernel [k] and the sections [extra] after its own: code with a
   relocation of each type, bank 0, shared memory and attributes. *)
let cubin ?(text = String.make 0x40 '\000') ?(info = "") ?(extra = []) () =
  elf
    ([
       (".text.k", progbits, 0, text, 0, 0, 128, 0);
       (".nv.constant0.k", progbits, 0, String.make 0x20 '\000', 0, 0, 4, 0);
       (".nv.shared.k", nobits, 0, String.make 0x100 '\000', 0, 0, 16, 0);
       ( ".nv.info",
         progbits,
         0,
         attr regcount (ii 1 128)
         ^ value_attr 0x1b 0x0102
         ^ attr min_stack_size (ii 1 0x100)
         ^ info,
         0,
         0,
         4,
         0 );
       ( ".nv.info.k",
         progbits,
         0,
         attr param_cbank (ihh 2 0x1f0 0x20),
         0,
         0,
         4,
         0 );
       ( ".symtab",
         2,
         0,
         sym 0 0 0 ^ sym 1 1 0 ^ sym 3 2 0 ^ sym 6 0 0,
         7,
         0,
         8,
         24 );
       (".strtab", 3, 0, "\000k\000c0\000ext\000", 0, 0, 1, 0);
       ( ".rela.text.k",
         4,
         0,
         rela 0x10 1 0x2 0 ^ rela 0x20 2 0x38 8 ^ rela 0x30 2 0x39 8,
         6,
         1,
         8,
         24 );
     ]
    @ extra)

let load obj = require_ok ~pp:Format.pp_print_string (Cubin.of_string obj)
let kernel c name = require_some (Cubin.kernel c name)

(* A cubin of the kernels k and j, whose attributes in .nv.info name their
   functions by symbol: k's by name, j's by a nameless symbol in its code
   section. Each has its own bank 0; bank 3 is shared. *)
let two_kernels () =
  elf
    [
      (".text.k", progbits, 0, String.make 0x40 '\000', 0, 0, 128, 0);
      (".text.j", progbits, 0, String.make 0x80 '\000', 0, 0, 128, 0);
      (".nv.constant0.k", progbits, 0, String.make 0x20 '\000', 0, 0, 4, 0);
      (".nv.constant0.j", progbits, 0, String.make 0x30 '\000', 0, 0, 4, 0);
      (".nv.constant3", progbits, 0, String.make 8 '\000', 0, 0, 4, 0);
      ( ".nv.info",
        progbits,
        0,
        attr regcount (ii 1 32)
        ^ attr min_stack_size (ii 1 0x40)
        ^ attr regcount (ii 2 64)
        ^ attr min_stack_size (ii 2 0x80),
        0,
        0,
        4,
        0 );
      (".symtab", 2, 0, sym 0 0 0 ^ sym 1 1 0 ^ sym 0 2 0, 8, 0, 8, 24);
      (".strtab", 3, 0, "\000k\000", 0, 0, 1, 0);
    ]

let section (o : Elf.t) name =
  List.find (fun (s : Elf.section) -> s.name = name) o.sections

let bank =
  Testable.make
    ~pp:(fun ppf (b : Cubin.bank) ->
      Format.fprintf ppf "{ index = %d; offset = %d; bytes = %d }" b.index
        b.offset b.bytes)
    ~equal:( = )

(* Kernels *)

let test_kernel () =
  let obj = cubin () in
  let o = Elf.load ~align:128 obj in
  let k = kernel (load obj) "k" in
  equal ~msg:"code" int (section o ".text.k").offset k.code;
  equal ~msg:"code bytes" int 0x40 k.code_bytes;
  equal ~msg:"registers" int 128 k.registers;
  equal ~msg:"shared bytes" int 0x100 k.shared_bytes;
  equal ~msg:"stack bytes" int 0x100 k.stack_bytes;
  equal ~msg:"params offset" int 0x1f0 k.params_offset;
  equal ~msg:"banks" (list bank)
    [
      { index = 0; offset = (section o ".nv.constant0.k").offset; bytes = 0x20 };
    ]
    k.banks

let test_per_function () =
  let obj = two_kernels () in
  let o = Elf.load ~align:128 obj in
  let c = load obj in
  let k = kernel c "k" and j = kernel c "j" in
  equal ~msg:"k's registers" int 32 k.registers;
  equal ~msg:"k's stack" int 0x40 k.stack_bytes;
  equal ~msg:"j's registers" int 64 j.registers;
  equal ~msg:"j's stack" int 0x80 j.stack_bytes;
  let at name index bytes =
    { Cubin.index; offset = (section o name).offset; bytes }
  in
  equal ~msg:"k's banks" (list bank)
    [ at ".nv.constant0.k" 0 0x20; at ".nv.constant3" 3 8 ]
    k.banks;
  equal ~msg:"j's banks" (list bank)
    [ at ".nv.constant0.j" 0 0x30; at ".nv.constant3" 3 8 ]
    j.banks

let test_absent () =
  is_none ~msg:"a kernel the cubin lacks" (Cubin.kernel (load (cubin ())) "j")

let test_no_kernels () =
  let obj =
    elf [ (".nv.constant0", progbits, 0, String.make 0x20 '\000', 0, 0, 4, 0) ]
  in
  is_none ~msg:"a cubin without code" (Cubin.kernel (load obj) "any")

let test_truncated_attribute () =
  (* A regcount whose header claims 8 bytes the section lacks. *)
  let cut = String.sub (attr regcount (ii 1 64)) 0 6 in
  let k = kernel (load (cubin ~info:cut ())) "k" in
  equal ~msg:"the attributes before it" int 128 k.registers

let test_kernels () =
  equal (list string) [ "k"; "j" ] (Cubin.kernels (load (two_kernels ())))

(* Images *)

let round_up n a = (n + a - 1) / a * a

(* A cubin of code alone, whose image is that code. *)
let image_law n =
  let obj = elf [ (".text.k", progbits, 0, String.make n 'x', 0, 0, 128, 0) ] in
  let image = Cubin.image (load obj) in
  cover "code of a multiple of 4 KiB" (n mod 0x1000 = 0);
  equal ~msg:"length" int (round_up n 0x1000 + 0x1000) (String.length image);
  equal ~msg:"the code" string (String.make n 'x') (String.sub image 0 n);
  equal ~msg:"then zeros" string
    (String.make (String.length image - n) '\000')
    (String.sub image n (String.length image - n))

let sizes =
  Gen.frequency
    [
      (3, Gen.int_range 0 0x2400);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; 0xfff; 0x1000; 0x1001; 0x2000 ] );
    ]

(* Relocations *)

let relocate_law base =
  let obj = cubin () in
  let o = Elf.load ~align:128 obj in
  let c = load obj in
  let text = (section o ".text.k").offset in
  let c0 = (section o ".nv.constant0.k").offset in
  let expected = Bytes.of_string (Cubin.image c) in
  Bytes.set_int64_le expected (text + 0x10) (Int64.of_int (base + text));
  let v = base + c0 + 8 in
  Bytes.set_int32_le expected (text + 0x24) (Int32.of_int (v land 0xffff_ffff));
  Bytes.set_int32_le expected (text + 0x34)
    (Int32.of_int ((v lsr 32) land 0xffff_ffff));
  equal string (Bytes.to_string expected) (Cubin.relocate c ~base)

let bases =
  Gen.frequency
    [
      (3, Gen.map (fun n -> n * 0x1000) (Gen.int_range 0 (1 lsl 36)));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 0xffff_f000; 0x1_0000_0000; (1 lsl 48) - 0x1000 ] );
    ]

(* Refusals *)

let refused ~sub obj = contains ~sub (require_error (Cubin.of_string obj))

let test_refusals () =
  refused ~sub:"Nx_device_elf.load" "not a cubin";
  let reloc kind sym =
    cubin ~extra:[ (".rela.text.k", 4, 0, rela 0 sym kind 0, 6, 1, 8, 24) ] ()
  in
  refused ~sub:"unknown type 0x5" (reloc 5 1);
  refused ~sub:"undefined symbol ext" (reloc 0x2 3)

let () =
  exit
    (run "nx.nv.cubin"
       [
         group "kernels"
           [
             test "a kernel's code, registers, memory and banks" test_kernel;
             test "each kernel reads its own registers, stack and banks"
               test_per_function;
             test "a kernel the cubin lacks is none" test_absent;
             test "a cubin without code has no kernel" test_no_kernels;
             test "an attribute the section truncates is ignored"
               test_truncated_attribute;
             test "kernels lists the code sections in order" test_kernels;
           ];
         group "image"
           [
             prop "the sections, then zeros to 4 KiB and 4 KiB more" sizes
               image_law;
             prop "relocate writes each address at its offset" bases
               relocate_law;
           ];
         group "refusals"
           [
             test "no ELF, an unknown relocation, an undefined symbol"
               test_refusals;
           ];
       ])
