(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cubins: NVRTC's simple_add (test/fixtures/README.md), and cubins a small ELF
   writer builds, read against the ELF layout below them. Relocation types after
   NVIDIA's cuobjdump: R_CUDA_64 0x2, R_CUDA_ABS32_LO_32 0x38,
   R_CUDA_ABS32_HI_32 0x39. *)

open Windtrap
open Rig_nv_abi
module S = Rig_nv_abi_support

let pp_error = Format.pp_print_string

let pp_cubin ppf c =
  Format.fprintf ppf "<cubin: %d bytes, kernels %s>" (Cubin.size c)
    (String.concat ", " (Cubin.kernels c))

let read obj = require_ok ~pp:pp_error (Cubin.of_string obj)
let kernel = Testable.make ~pp:S.pp_kernel ~equal:( = )

(* An ELF writer: a 64-bit little-endian object of EM_CUDA *)

let em_cuda = 190

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

let progbits = 1
let symtab_kind = 2
let strtab_kind = 3
let rela_kind = 4
let rel_kind = 9
let nobits = 8
let nv_info_kind = 0x7000_0000
let alloc = 0x2
let exec = 0x4
let shn_undef = 0
let shn_abs = 0xfff1

let section ?(kind = progbits) ?(flags = alloc) ?(addr = 0) ?size ?(link = 0)
    ?(info = 0) ?(align = 1) ?(entsize = 0) name contents =
  let size = Option.value size ~default:(String.length contents) in
  { name; kind; flags; addr; contents; size; link; info; align; entsize }

let code ?addr ?(align = 128) name n =
  section ~flags:(alloc lor exec) ?addr ~align (".text." ^ name)
    (String.make n '\x5a')

let names_of names =
  let b = Buffer.create 256 in
  Buffer.add_char b '\000';
  let at n =
    if n = "" then 0
    else begin
      let at = Buffer.length b in
      Buffer.add_string b n;
      Buffer.add_char b '\000';
      at
    end
  in
  let offsets = List.map at names in
  (offsets, Buffer.contents b)

(* The object of [sections] at indexes 1, 2, ..., then its section names. *)
let write ?(machine = em_cuda) sections =
  let offsets, shstrtab =
    names_of (List.map (fun s -> s.name) sections @ [ ".shstrtab" ])
  in
  let sections =
    sections @ [ section ~kind:strtab_kind ~flags:0 ".shstrtab" shstrtab ]
  in
  let body = Buffer.create 4096 in
  let at =
    List.map
      (fun s ->
        let at = 64 + Buffer.length body in
        if s.kind <> nobits then Buffer.add_string body s.contents;
        at)
      sections
  in
  let count = List.length sections + 1 in
  let b = Buffer.create (64 + Buffer.length body + (64 * count)) in
  Buffer.add_string b "\x7fELF\002\001\001";
  Buffer.add_string b (String.make 9 '\000');
  Buffer.add_uint16_le b 2;
  Buffer.add_uint16_le b machine;
  Buffer.add_int32_le b 1l;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b (Int64.of_int (64 + Buffer.length body));
  Buffer.add_int32_le b 0l;
  List.iter (Buffer.add_uint16_le b) [ 64; 0; 0; 64; count; count - 1 ];
  Buffer.add_buffer b body;
  let header name s offset =
    Buffer.add_int32_le b (Int32.of_int name);
    Buffer.add_int32_le b (Int32.of_int s.kind);
    List.iter
      (fun v -> Buffer.add_int64_le b (Int64.of_int v))
      [ s.flags; s.addr; offset; s.size ];
    Buffer.add_int32_le b (Int32.of_int s.link);
    Buffer.add_int32_le b (Int32.of_int s.info);
    Buffer.add_int64_le b (Int64.of_int s.align);
    Buffer.add_int64_le b (Int64.of_int s.entsize)
  in
  header 0 (section ~kind:0 ~flags:0 "" "") 0;
  List.iter2
    (fun s (name, offset) -> header name s offset)
    sections (List.combine offsets at);
  Buffer.contents b

(* A symbol table's section and its names' section, linked to it at [index + 1]:
   each symbol is (name, section index, value), after the null symbol. *)
let symbols ~index syms =
  let offsets, strtab = names_of (List.map (fun (n, _, _) -> n) syms) in
  let b = Buffer.create 256 in
  let add name shndx value =
    Buffer.add_int32_le b (Int32.of_int name);
    Buffer.add_uint8 b 0x12;
    Buffer.add_uint8 b 0;
    Buffer.add_uint16_le b shndx;
    Buffer.add_int64_le b (Int64.of_int value);
    Buffer.add_int64_le b 0L
  in
  add 0 0 0;
  List.iter2 (fun (_, shndx, value) name -> add name shndx value) syms offsets;
  [
    section ~kind:symtab_kind ~flags:0 ~link:(index + 1) ~align:8 ~entsize:24
      ".symtab" (Buffer.contents b);
    section ~kind:strtab_kind ~flags:0 ".strtab" strtab;
  ]

(* A relocation section for the section [info], over the symbols at [link]: each
   relocation is (offset, symbol, type, addend). *)
let rela ~link ~info name relocs =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind, addend) ->
      Buffer.add_int64_le b (Int64.of_int offset);
      Buffer.add_int64_le b
        Int64.(logor (shift_left (of_int sym) 32) (of_int kind));
      Buffer.add_int64_le b (Int64.of_int addend))
    relocs;
  section ~kind:rela_kind ~flags:0 ~link ~info ~align:8 ~entsize:24 name
    (Buffer.contents b)

(* .nv.info attributes: a format byte, a parameter byte and a 16-bit size, then
   for format EIFMT_SVAL (4) that many bytes; other formats hold a 16-bit value
   in place of the size. simple_add_sm89.cubin's are of this form
   (fixtures/README.md). *)
let eiattr_param_cbank = 0x0a
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

let attribute fmt param data =
  let h = Bytes.create 4 in
  Bytes.set_uint8 h 0 fmt;
  Bytes.set_uint8 h 1 param;
  Bytes.set_uint16_le h 2 (String.length data);
  Bytes.to_string h ^ data

let u32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.of_int v)) vs;
  Bytes.to_string b

(* A symbol's attribute: its index, then a 32-bit value. *)
let of_symbol param sym v = attribute 4 param (u32s [ sym; v ])

(* EIATTR_PARAM_CBANK: a symbol's index, then the parameters' offset and size in
   16 bits each. *)
let param_cbank sym offset size =
  let d = Bytes.create 8 in
  Bytes.set_int32_le d 0 (Int32.of_int sym);
  Bytes.set_uint16_le d 4 offset;
  Bytes.set_uint16_le d 6 size;
  attribute 4 eiattr_param_cbank (Bytes.to_string d)

(* Attributes the reader passes over: a 16-bit value (EIFMT_HVAL 3) and data of
   a parameter it does not read (0x37, EIATTR_CUDA_API_VERSION). *)
let fillers = attribute 3 0x19 "" ^ attribute 4 0x37 (u32s [ 0x80 ])

let nv_info ?(name = ".nv.info") ?(info = 0) contents =
  section ~kind:nv_info_kind ~flags:0 ~info ~align:4 name contents

(* The ELF layout below a cubin *)

(* The sections the GPU's memory holds, as cubin.mli states them. *)
let held (s : Rig_elf.section) =
  Rig_elf.allocated ~machine:em_cuda s
  && not (String.starts_with ~prefix:".nv.shared." s.name)

let layout obj =
  require_ok ~pp:pp_error (Rig_elf.of_string ~align:128 ~held obj)

let page = 4096

(* simple_add, as llvm-readelf reads it *)

let simple_add () =
  read
    (In_channel.with_open_bin "../../fixtures/simple_add_sm89.cubin"
       In_channel.input_all)

let nvrtc =
  group ~timeout:10. "simple_add_sm89"
    [
      test "its one kernel is simple_add" (fun () ->
          equal (list string) [ "simple_add" ] (Cubin.kernels (simple_add ())));
      test "simple_add's code, registers, stack, parameters and bank 0"
        (fun () ->
          equal (option kernel)
            (Some
               {
                 Cubin.code = 0x180;
                 code_bytes = 0x200;
                 registers = 12;
                 shared_bytes = 0;
                 stack_bytes = 0;
                 params_offset = 0x160;
                 params_bytes = 0x1c;
                 banks = [ { index = 0; offset = 0; bytes = 0x17c } ];
               })
            (Cubin.kernel (simple_add ()) "simple_add"));
      test "its image of 0x380 bytes takes two pages" (fun () ->
          equal int (2 * page) (Cubin.size (simple_add ())));
      test "a relocation of its debugging information patches nothing"
        (fun () ->
          equal
            (list (pair int string))
            []
            (Cubin.patches (simple_add ()) ~base:0x7000_0000_0000));
      test "a kernel the cubin lacks is none" (fun () ->
          equal (option kernel) None (Cubin.kernel (simple_add ()) "simple"));
    ]

(* Relocations *)

let r_cuda_64 = 0x2
let r_cuda_abs32_lo_32 = 0x38
let r_cuda_abs32_hi_32 = 0x39

(* The bytes a relocation of [kind] at [offset] patches, as [(at, width)]. *)
let field kind offset = if kind = r_cuda_64 then (offset, 8) else (offset + 4, 4)

(* A REL section for the section [info]: the relocations of [rela] without their
   addends. *)
let rel ~link ~info name relocs =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind, _) ->
      Buffer.add_int64_le b (Int64.of_int offset);
      Buffer.add_int64_le b
        Int64.(logor (shift_left (of_int sym) 32) (of_int kind)))
    relocs;
  section ~kind:rel_kind ~flags:0 ~link ~info ~align:8 ~entsize:16 name
    (Buffer.contents b)

(* The image of [o], as Rig_elf.mli lays it out. *)
let elf_image (o : Rig_elf.t) =
  let b = Bytes.make o.size '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  Bytes.to_string b

(* What a relocation writes, as cubin.mli states it, over the ELF layout: the
   symbol's address base + offset + addend, modulo 2^64. A REL relocation's
   addend is the unsigned little-endian value of the bytes it patches in
   [image], the image of the ELF layout. *)
let expected_patch ~image ~base (r : Rig_elf.relocation) =
  match r.symbol.place with
  | Image { offset; _ } ->
      let at, width = field r.kind r.offset in
      let addend =
        match r.addend with
        | Explicit a -> Int64.of_int a
        | Implicit _ when at + width > String.length image -> 0L
        | Implicit _ when width = 8 -> String.get_int64_le image at
        | Implicit _ ->
            Int64.(
              logand (of_int32 (String.get_int32_le image at)) 0xffff_ffffL)
      in
      let address = Int64.(add (add (of_int base) (of_int offset)) addend) in
      let w = S.le64 address in
      if r.kind = r_cuda_64 then Some (at, w)
      else if r.kind = r_cuda_abs32_lo_32 then Some (at, String.sub w 0 4)
      else if r.kind = r_cuda_abs32_hi_32 then Some (at, String.sub w 4 4)
      else None
  | Undefined | Absolute _ | Outside _ -> None

(* Every patch of [o]'s relocations, in order. *)
let expected_patches ~base (o : Rig_elf.t) =
  let image = elf_image o in
  List.filter_map (expected_patch ~image ~base) o.relocations

(* A cubin of a 16-byte bank at 0 and 32 bytes of code at 128, the end of its
   image, with [relocs] in the code: the symbols are c at 8 in the bank, u
   undefined, a absolute, and s in shared memory, which the image does not
   hold. *)
let relocated relocs =
  write
    ([
       section ~align:4 ".nv.constant0.k" (String.make 16 '\001');
       code "k" 32;
       section ~kind:nobits ~flags:(alloc lor 1) ~size:16 ".nv.shared.k" "";
     ]
    @ symbols ~index:4
        [
          ("c", 1, 8); ("u", shn_undef, 0); ("a", shn_abs, 0x1234); ("s", 3, 0);
        ]
    @ [ rela ~link:4 ~info:2 ".rela.text.k" relocs ])

let relocations =
  group ~timeout:10. "relocations"
    [
      test "each type writes its part of the address, modulo 2^64" (fun () ->
          let base = 0x7000_0000_0000 in
          let address addend = S.le64 Int64.(add (of_int (base + 8)) addend) in
          let c =
            read
              (relocated
                 [
                   (0, 1, r_cuda_64, 0x10);
                   (8, 1, r_cuda_abs32_lo_32, 0x4);
                   (16, 1, r_cuda_abs32_hi_32, 0);
                   (24, 1, r_cuda_64, max_int);
                 ])
          in
          equal
            (list (pair int string))
            [
              (128, address 0x10L);
              (140, String.sub (address 4L) 0 4);
              (148, String.sub (address 0L) 4 4);
              (152, address (Int64.of_int max_int));
            ]
            (Cubin.patches c ~base));
      cases ~name:fst "a relocation the image cannot take is refused"
        [
          ("of another type", (0, 1, 0x1, 0));
          ("to an undefined symbol", (0, 2, r_cuda_64, 0));
          ("to an absolute symbol", (0, 3, r_cuda_64, 0));
          ("to a symbol outside the image", (0, 4, r_cuda_64, 0));
          ("past the end of the image", (28, 1, r_cuda_64, 0));
        ]
        (fun (_, r) ->
          ignore
            (require_error ~pp:pp_cubin (Cubin.of_string (relocated [ r ]))));
    ]

(* A cubin of 128 kernels, each scaling by a global the cubin relocates
   (many.cu), checked against the ELF layout. *)

let many () =
  In_channel.with_open_bin "fixtures/many_sm89.cubin" In_channel.input_all

let many_sm89 =
  group ~timeout:10. "many_sm89"
    [
      test "its kernels are its code sections, in order" (fun () ->
          let o = layout (many ()) in
          let code (s : Rig_elf.section) =
            match s.offset with
            | Some _ when String.starts_with ~prefix:".text." s.name ->
                Some (String.sub s.name 6 (String.length s.name - 6))
            | _ -> None
          in
          let names = List.filter_map code (Iarray.to_list o.sections) in
          equal ~msg:"count" int 128 (List.length names);
          equal (list string) names (Cubin.kernels (read (many ()))));
      test "its 128 relocations write each global's address" (fun () ->
          let obj = many () in
          let base = 0x7fff_0000_0000 in
          let expected = expected_patches ~base (layout obj) in
          equal ~msg:"count" int 128 (List.length expected);
          equal
            (list (pair int string))
            expected
            (Cubin.patches (read obj) ~base));
    ]

(* A kernel reading an uninitialised global, scale in .nv.global, and an
   initialised one, bias in .nv.global.init, each relocated into bank 4
   (globals.cu, fixtures/README.md). *)

let globals () =
  read
    (In_channel.with_open_bin "fixtures/globals_sm89.cubin" In_channel.input_all)

let globals_sm89 =
  group ~timeout:10. "globals_sm89"
    [
      test "an uninitialised global lies in the image, as zeros" (fun () ->
          let o = Cubin.elf (globals ()) in
          let s =
            require_some
              (Iarray.find_opt
                 (fun (s : Rig_elf.section) -> s.name = ".nv.global")
                 o.sections)
          in
          let at = require_some ~msg:"held" s.offset in
          equal ~msg:"bytes in the object" int 0 s.length;
          equal ~msg:"scale" (option int) (Some at) (Rig_elf.symbol o "scale");
          at_most ~msg:"its 4 bytes" int ~than:o.size (at + 4));
      test "a relocation to an uninitialised global writes its address"
        (fun () ->
          let c = globals () and base = 0x7fff_0000_0000 in
          let o = Cubin.elf c in
          let expected = expected_patches ~base o in
          equal ~msg:"count" int 2 (List.length expected);
          equal (list (pair int string)) expected (Cubin.patches c ~base));
    ]

(* The image *)

(* A cubin of a 16-byte bank at address 0 and 16 bytes of code at [at]: an image
   of [at + 16] bytes. *)
let spanning at =
  write
    [
      section ~align:1 ".nv.constant0.k" (String.make 16 '\001');
      code ~addr:at ~align:1 "k" 16;
    ]

let image =
  group ~timeout:10. "image"
    [
      cases ~name:fst
        "the image is the ELF image, then zeros to a page and a page more"
        [
          ("simple_add_sm89", simple_add);
          ("many_sm89", fun () -> read (many ()));
          ("globals_sm89", globals);
        ]
        (fun (_, c) ->
          let c = c () in
          equal int (S.round_up (Cubin.elf c).size page + page) (Cubin.size c));
      test "an ELF image of a whole page takes two" (fun () ->
          let c =
            read
              (write
                 [
                   section ~align:4 ".nv.constant0.k" (String.make 4 '\001');
                   code "k" (page - 128);
                 ])
          in
          equal (pair int int)
            (page, 2 * page)
            ((Cubin.elf c).size, Cubin.size c));
      test "a kernel's shared memory stays out of the image" (fun () ->
          let shared =
            section ~kind:nobits ~flags:(alloc lor 1) ~size:0x8000
              ".nv.shared.k" ""
          in
          let c = read (write [ code "k" 0x100; shared ]) in
          let k = require_some (Cubin.kernel c "k") in
          equal (pair int int) (0x8000, 2 * page) (k.shared_bytes, Cubin.size c));
      test "an uninitialised global's bytes count in the image" (fun () ->
          let global =
            section ~kind:nobits ~flags:(alloc lor 1) ~size:0x2000 ".nv.global"
              ""
          in
          let c = read (write [ code "k" 0x100; global ]) in
          equal int (4 * page) (Cubin.size c));
      test "a cubin without allocated sections takes a page" (fun () ->
          let c = read (write [ nv_info "" ]) in
          equal (pair int int) (0, page) ((Cubin.elf c).size, Cubin.size c));
      test "an image of 2^49 bytes reads" (fun () ->
          let c = read (spanning ((1 lsl 49) - page - 16)) in
          equal int (1 lsl 49) (Cubin.size c));
      test "an image past 2^49 bytes is refused" (fun () ->
          let obj = spanning ((1 lsl 49) - page - 15) in
          equal ~msg:"the ELF image" int
            ((1 lsl 49) - page + 1)
            (layout obj).size;
          ignore (require_error ~pp:pp_cubin (Cubin.of_string obj)));
      test "a field past the end of its section is refused" (fun () ->
          let obj =
            write
              ([
                 section ~align:4 ".nv.constant0.k" (String.make 4 '\001');
                 code "k" 16;
                 section ".nv.global.init" "GLOBALS!";
               ]
              @ symbols ~index:4 [ ("c", 1, 0) ]
              @ [ rel ~link:4 ~info:2 ".rel.text.k" [ (12, 1, r_cuda_64, 0) ] ]
              )
          in
          let o = layout obj in
          let code =
            require_some
              (Iarray.find_map
                 (fun (s : Rig_elf.section) ->
                   if s.name = ".text.k" then s.offset else None)
                 o.sections)
          in
          at_least ~msg:"the image holds the field" int ~than:(code + 20) o.size;
          contains ~sub:"past the end of its section"
            (require_error ~pp:pp_cubin (Cubin.of_string obj)));
      cases ~name:string_of_int "an object for another machine is refused"
        (* EM_X86_64, EM_AARCH64, EM_AMDGPU. *)
        [ 62; 183; 224 ] (fun machine ->
          contains ~sub:"EM_CUDA"
            (require_error ~pp:pp_cubin
               (Cubin.of_string (write ~machine [ code "k" 16 ]))));
      test "an object that is not ELF is refused" (fun () ->
          ignore (require_error ~pp:pp_cubin (Cubin.of_string "\x7fELF")));
      test "reading copies nothing: elf's file is the object" (fun () ->
          let obj = relocated [ (0, 1, r_cuda_64, 0) ] in
          equal string obj (Cubin.elf (read obj)).file);
    ]

(* Kernels *)

(* Kernels k1 and k10, one name a prefix of the other, with banks of the cubin
   and of a kernel, shared memory, parameters, a code section the image does not
   hold, and attributes naming their function by a section symbol, by a symbol
   in a kernel's code, by an undefined symbol of a kernel's name, and past the
   symbol table. *)
let two_kernels () =
  let info =
    String.concat ""
      [
        fillers;
        of_symbol eiattr_regcount 1 10;
        (* helper, in k1's code. *)
        of_symbol eiattr_regcount 5 11;
        (* A symbol named k10 in k1's code is k1's. *)
        of_symbol eiattr_min_stack_size 7 0x30;
        (* An undefined symbol named k10 is k10's. *)
        of_symbol eiattr_regcount 6 20;
        of_symbol eiattr_min_stack_size 99 0x50;
        of_symbol eiattr_min_stack_size 3 0x40;
      ]
  in
  let own =
    fillers ^ param_cbank 1 0x10 0x1c ^ of_symbol eiattr_regcount 1 99
  in
  write
    ([
       section ~align:4 ".nv.constant3" (String.make 16 '\002');
       code "k1" 64;
       section ~align:4 ".nv.constant3.k1" (String.make 32 '\002');
       section ~align:4 ".nv.constant0.k10" (String.make 0x40 '\002');
       code "k10" 32;
       section ~kind:nobits ~flags:(alloc lor 1) ~size:0x400 ".nv.shared.k1" "";
       nv_info info;
       nv_info ~name:".nv.info.k1" ~info:2 own;
       section ~flags:0 ".text.dead" (String.make 16 '\000');
     ]
    @ symbols ~index:10
        [
          ("", 2, 0);
          ("k1", 2, 0);
          ("", 5, 0);
          ("k10", 5, 0);
          ("helper", 2, 0x20);
          ("k10", shn_undef, 0);
          ("k10", 2, 0x10);
        ])

let kernels =
  group ~timeout:10. "kernels"
    [
      test "a kernel's record is what its sections and attributes say"
        (fun () ->
          let obj = two_kernels () in
          let o = layout obj and c = read obj in
          let at name =
            require_some ~msg:name
              (Iarray.find_map
                 (fun (s : Rig_elf.section) ->
                   if s.name = name then s.offset else None)
                 o.sections)
          in
          equal (list string) [ "k1"; "k10" ] (Cubin.kernels c);
          equal ~msg:"k1" (option kernel)
            (Some
               {
                 Cubin.code = at ".text.k1";
                 code_bytes = 64;
                 registers = 11;
                 shared_bytes = 0x400;
                 stack_bytes = 0x30;
                 params_offset = 0x10;
                 params_bytes = 0x1c;
                 banks =
                   [ { index = 3; offset = at ".nv.constant3.k1"; bytes = 32 } ];
               })
            (Cubin.kernel c "k1");
          equal ~msg:"k10" (option kernel)
            (Some
               {
                 Cubin.code = at ".text.k10";
                 code_bytes = 32;
                 registers = 20;
                 shared_bytes = 0;
                 stack_bytes = 0x40;
                 params_offset = 0;
                 params_bytes = 0;
                 banks =
                   [
                     { index = 3; offset = at ".nv.constant3"; bytes = 16 };
                     {
                       index = 0;
                       offset = at ".nv.constant0.k10";
                       bytes = 0x40;
                     };
                   ];
               })
            (Cubin.kernel c "k10");
          equal ~msg:"a code section outside the image" (option kernel) None
            (Cubin.kernel c "dead"));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "a truncated .nv.info attribute, or one the reader reads cut short, is \
         refused"
        [
          ("a header cut short", ".nv.info", fillers ^ "\x04\x2f", "truncated");
          ( "data cut short",
            ".nv.info",
            fillers ^ String.sub (of_symbol eiattr_regcount 1 99) 0 10,
            "truncated" );
          ( "a kernel's own, data cut short",
            ".nv.info.k",
            fillers ^ String.sub (param_cbank 1 0x160 0x1c) 0 9,
            "truncated" );
          ( "registers without their value",
            ".nv.info",
            fillers ^ attribute 4 eiattr_regcount (u32s [ 1 ]),
            "expected at least 8" );
          ( "a stack size without its value",
            ".nv.info",
            fillers ^ attribute 4 eiattr_min_stack_size (u32s [ 1 ]),
            "expected at least 8" );
          ( "a kernel's parameters without their offset",
            ".nv.info.k",
            fillers ^ attribute 4 eiattr_param_cbank (u32s [ 1 ]),
            "expected at least 8" );
        ]
        (fun (_, name, contents, sub) ->
          let obj =
            write
              ([ code "k" 16; nv_info ~name ~info:1 contents ]
              @ symbols ~index:3 [ ("k", 1, 0) ])
          in
          contains ~sub (require_error ~pp:pp_cubin (Cubin.of_string obj)));
      test "a cubin without code has no kernel" (fun () ->
          let c = read (write [ nv_info "" ]) in
          equal
            (pair (list string) (option kernel))
            ([], None)
            (Cubin.kernels c, Cubin.kernel c ""));
    ]

let () =
  exit
    (run "rig_nv_abi.cubin"
       [ nvrtc; many_sm89; relocations; globals_sm89; image; kernels ])
