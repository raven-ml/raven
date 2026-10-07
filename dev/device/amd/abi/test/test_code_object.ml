(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Code objects of this directory's fixtures (fixtures/README.md): a relocatable
   object whose descriptors reach their kernels through REL64 relocations, the
   object linked from it, and objects made by changing bytes of either.
   Descriptor fields at the offsets of LLVM's AMDHSAKernelDescriptor.h, machines
   and their processors after LLVM's ELF.h and AMDGPUUsage. *)

open Windtrap
open Device_amd_abi
module S = Device_amd_abi_support

let timeout = Device_amd_abi_support.timeout
let strf = Printf.sprintf

let fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

let relocatable () = fixture "kernels_gfx1030.o"
let linked () = fixture "kernels_gfx1030.hsaco"
let pp_error ppf e = Format.pp_print_string ppf e
let read obj = require_ok ~pp:pp_error (Code_object.of_string obj)

(* The image of [co], as the module's preamble writes it. *)
let image co =
  let o = Code_object.elf co in
  let b = Bytes.make (Code_object.size co) '\000' in
  let put (s : Device_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (off, p) = Bytes.blit_string p 0 b off (String.length p) in
  List.iter patch (Code_object.patches co);
  Bytes.to_string b

(* Changing an object *)

let patch obj at set =
  let b = Bytes.of_string obj in
  set b at;
  Bytes.to_string b

let set_u8 v b at = Bytes.set_uint8 b at v
let set_u32 v b at = Bytes.set_int32_le b at (Int32.of_int v)
let set_u64 v b at = Bytes.set_int64_le b at (Int64.of_int v)

(* The ELF header's EI_ABIVERSION, e_machine and e_flags; the code object
   versions of AMDGPUUsage's ELFABIVERSION_AMDGPU_HSA_V5 and V6. *)
let abi_version = 8
let machine = 18
let flags = 48
let v5 = 3
let v6 = 4

let section obj name =
  let o = require_ok ~pp:pp_error (Device_elf.of_string obj) in
  let named (s : Device_elf.section) = String.equal s.name name in
  require_some (Iarray.find_opt named o.sections)

(* The file offset of section [i]'s header and of entry [i] of a table of
   24-byte entries: symbols and RELA relocations. *)
let header obj i = Int64.to_int (String.get_int64_le obj 40) + (64 * i)
let entry obj table i = (section obj table).at + (24 * i)

(* A section header's sh_addr; a symbol's st_value; a relocation's r_offset and
   r_info. *)
let sh_addr = 16
let st_value = 8
let r_offset = 0
let r_info = 8

let index obj name =
  let o = require_ok ~pp:pp_error (Device_elf.of_string obj) in
  let rec go i =
    if (Iarray.get o.sections i).name = name then i else go (i + 1)
  in
  go 0

(* The fixtures' symbols and relocation types: R_AMDGPU_ABS32_LO and
   R_AMDGPU_REL64. *)
let sym_ext = 5
let abs32_lo = 1
let rel64 = 5
let info ~sym ~kind = (sym lsl 32) lor kind

(* Reading *)

let both = [ ("relocatable", relocatable); ("linked", linked) ]

(* A kernel's descriptor as the image holds it: the u32 fields of
   kernel_descriptor_t at 0, 4, 8, 44, 48 and 52, its i64 entry offset at 16,
   and its u16 code properties at 56: bit 0 enables the private segment buffer,
   bit 1 the dispatch pointer, bit 10 waves of 32 lanes. *)
let descriptor img at : Code_object.kernel =
  let u32 o =
    Int32.to_int (String.get_int32_le img (at + o)) land 0xffff_ffff
  in
  let props = String.get_uint16_le img (at + 56) in
  let bit n = props land (1 lsl n) <> 0 in
  {
    descriptor = at;
    entry = at + Int64.to_int (String.get_int64_le img (at + 16));
    group_segment = u32 0;
    private_segment = u32 4;
    kernarg_size = u32 8;
    rsrc1 = u32 48;
    rsrc2 = u32 52;
    rsrc3 = u32 44;
    wave32 = bit 10;
    dispatch_ptr = bit 1;
    private_segment_buffer = bit 0;
  }

let kernel =
  Testable.make
    ~pp:(fun ppf (k : Code_object.kernel) ->
      Format.fprintf ppf
        "{ descriptor = %d; entry = %d; group = %d; private = %d; kernarg = \
         %d; rsrc1 = 0x%x; rsrc2 = 0x%x; rsrc3 = 0x%x; wave32 = %b; \
         dispatch_ptr = %b; private_segment_buffer = %b }"
        k.descriptor k.entry k.group_segment k.private_segment k.kernarg_size
        k.rsrc1 k.rsrc2 k.rsrc3 k.wave32 k.dispatch_ptr k.private_segment_buffer)
    ~equal:( = )

let reading =
  group ~timeout "reading"
    [
      cases ~name:fst "an object keeps the bytes it was read from" both
        (fun (_, obj) ->
          let obj = obj () in
          equal string obj (Code_object.elf (read obj)).file);
      cases ~name:fst "the image is the ELF image in whole 32-bit words" both
        (fun (_, obj) ->
          let co = read (obj ()) in
          equal int
            (((Code_object.elf co).size + 3) / 4 * 4)
            (Code_object.size co));
      test "a linked image of 0x327b bytes takes 0x327c" (fun () ->
          let co = read (linked ()) in
          equal (pair int int) (0x327b, 0x327c)
            ((Code_object.elf co).size, Code_object.size co));
      cases ~name:fst "every patch lies in the image" both (fun (_, obj) ->
          let co = read (obj ()) in
          List.iter
            (fun (off, p) ->
              at_least int ~than:0 off;
              at_most int ~than:(Code_object.size co) (off + String.length p))
            (Code_object.patches co));
      test "a REL64 patch is its target's offset from the word, plus the addend"
        (fun () ->
          let co = read (relocatable ()) in
          let expected (r : Device_elf.relocation) =
            match r.symbol.place with
            | Image { offset; _ } ->
                let b = Bytes.create 8 in
                Bytes.set_int64_le b 0
                  (Int64.of_int (offset + r.addend - r.offset));
                (r.offset, Bytes.to_string b)
            | Undefined | Absolute _ | Outside _ -> fail "a fixture's symbol"
          in
          equal
            (list (pair int string))
            (List.map expected (Code_object.elf co).relocations)
            (Code_object.patches co);
          equal int 2 (List.length (Code_object.patches co)));
      test "a linked object has no patches" (fun () ->
          equal
            (list (pair int string))
            []
            (Code_object.patches (read (linked ()))));
      cases ~name:fst "kernels are the .kd symbols, in increasing order" both
        (fun (_, obj) ->
          equal (list string) [ "a"; "b" ] (Code_object.kernels (read (obj ()))));
      cases ~name:fst "a kernel is its descriptor in the image" both
        (fun (_, obj) ->
          let co = read (obj ()) in
          let img = image co in
          List.iter
            (fun name ->
              let k = require_some (Code_object.kernel co name) in
              equal ~msg:name kernel (descriptor img k.descriptor) k;
              equal ~msg:name (option int)
                (Device_elf.symbol (Code_object.elf co) (name ^ ".kd"))
                (Some k.descriptor))
            (Code_object.kernels co));
      cases ~name:fst "a kernel's entry is its symbol" both (fun (_, obj) ->
          let co = read (obj ()) in
          List.iter
            (fun name ->
              equal ~msg:name (option int)
                (Device_elf.symbol (Code_object.elf co) name)
                (Option.map
                   (fun (k : Code_object.kernel) -> k.entry)
                   (Code_object.kernel co name)))
            [ "a"; "b" ]);
      test "the descriptors' fields are the source's" (fun () ->
          let co = read (relocatable ()) in
          let fields name =
            let k = require_some (Code_object.kernel co name) in
            ( (k.group_segment, k.private_segment, k.kernarg_size),
              (k.wave32, k.dispatch_ptr, k.private_segment_buffer) )
          in
          let w = pair (triple int int int) (triple bool bool bool) in
          equal ~msg:"a" w ((256, 64, 24), (true, false, true)) (fields "a");
          equal ~msg:"b" w ((0, 0, 8), (false, true, false)) (fields "b"));
      cases ~name:(strf "%S") "a name that is no kernel's is none"
        [ ""; "c"; "a.kd"; "A"; "ext" ] (fun name ->
          is_none (Code_object.kernel (read (relocatable ())) name));
    ]

(* Processors *)

(* EF_AMDGPU_MACH values of LLVM's ELF.h, the generic version in the flags' top
   byte, and AMDGPUUsage's XNACK and SRAMECC settings of code object v4 on. *)
let generic ?(version = 1) mach = (version lsl 24) lor mach
let xnack_sramecc = 0xf00

let gpus =
  [
    (9, 0, 10);
    (9, 4, 2);
    (9, 5, 0);
    (10, 3, 0);
    (10, 3, 6);
    (11, 0, 0);
    (11, 0, 3);
    (11, 5, 1);
    (11, 5, 3);
    (12, 0, 0);
    (12, 0, 1);
  ]

let with_flags ?(abi = v6) f =
  patch (patch (relocatable ()) flags (set_u32 f)) abi_version (set_u8 abi)

let processors =
  cases
    ~name:(fun (name, f, _) -> strf "%s (0x%x)" name f)
    "the processor of a code object and the GPUs that run it"
    [
      ("gfx1030", 0x36, [ "gfx1030" ]);
      ("gfx90a", 0x3f, [ "gfx90a" ]);
      ("gfx1100", 0x41, [ "gfx1100" ]);
      ("gfx942", 0x4c, [ "gfx942" ]);
      ("gfx1201", 0x4e, [ "gfx1201" ]);
      ("gfx942", xnack_sramecc lor 0x4c, [ "gfx942" ]);
      ("gfx10-3-generic", generic 0x53, [ "gfx1030"; "gfx1036" ]);
      ( "gfx11-generic",
        generic 0x54,
        [ "gfx1100"; "gfx1103"; "gfx1151"; "gfx1153" ] );
      ("gfx12-generic", generic 0x59, [ "gfx1200"; "gfx1201" ]);
      ("gfx12-generic", generic ~version:2 0x59, [ "gfx1200"; "gfx1201" ]);
      ("gfx9-4-generic", generic 0x5f, [ "gfx942"; "gfx950" ]);
    ]
    (fun (name, f, runs) ->
      let co = read (with_flags f) in
      equal ~msg:"its processor" string name (Code_object.target co);
      equal ~msg:"the GPUs that run it" (list string) runs
        (List.filter_map
           (fun v ->
             let g = S.gpu v in
             if Code_object.runs_on co g then Some (Gpu.processor g) else None)
           gpus))

(* Refusals *)

let refused ?(sub = "") obj =
  let msg = require_error (Code_object.of_string obj) in
  contains ~sub msg

let refusals =
  group ~timeout "refusals"
    [
      test "bytes that are no ELF object" (fun () -> refused "not an object");
      test "an object for another machine" (fun () ->
          refused ~sub:"62"
            (patch (relocatable ()) machine (fun b at ->
                 Bytes.set_uint16_le b at 62)));
      cases ~name:(strf "0x%x") "a machine LLVM names no processor of"
        [ 0x0; 0x27; 0x60; 0xff ] (fun mach ->
          refused ~sub:(strf "0x%x" mach) (with_flags ~abi:v5 mach));
      test "a generic code object before version 6" (fun () ->
          refused ~sub:"gfx11-generic" (with_flags ~abi:v5 (generic 0x54)));
      test "a generic code object of generic version 0" (fun () ->
          refused ~sub:"generic version 0"
            (with_flags (generic ~version:0 0x54)));
      test "a relocation of another kind" (fun () ->
          let obj = relocatable () in
          refused ~sub:"R_AMDGPU_REL64"
            (patch obj
               (entry obj ".rela.rodata" 0 + r_info)
               (set_u64 (info ~sym:1 ~kind:abs32_lo))));
      test "a relocation to an undefined symbol" (fun () ->
          let obj = relocatable () in
          refused ~sub:"ext"
            (patch obj
               (entry obj ".rela.rodata" 0 + r_info)
               (set_u64 (info ~sym:sym_ext ~kind:rel64))));
      test "a relocation that names no symbol" (fun () ->
          let obj = relocatable () in
          refused
            (patch obj
               (entry obj ".rela.rodata" 0 + r_info)
               (set_u64 (info ~sym:0 ~kind:rel64))));
      (* .rodata ends the image, 128 bytes from 0x140: a word 4 bytes from its
         end passes it. *)
      test "a relocation that patches bytes past the image's end" (fun () ->
          let obj = relocatable () in
          refused
            (patch obj (entry obj ".rela.rodata" 1 + r_offset) (set_u64 0x7c)));
      (* a.kd 96 bytes into .rodata's 128: its 64 bytes pass the image's end. *)
      test "a descriptor past the image's end" (fun () ->
          let obj = relocatable () in
          refused ~sub:"a"
            (patch obj (entry obj ".symtab" 3 + st_value) (set_u64 0x60)));
      cases ~name:fst "an entry outside the image"
        [
          ("before it", fun k _ -> -k - 1);
          ("at its end", fun k size -> size - k);
          ("far past it", fun _ _ -> 1 lsl 40);
        ]
        (fun (_, offset) ->
          let obj = linked () in
          let co = read obj in
          let k = require_some (Code_object.kernel co "a") in
          let at =
            (section obj ".rodata").at + k.descriptor
            - Option.get (section obj ".rodata").offset
          in
          refused ~sub:"a"
            (patch obj (at + 16)
               (set_u64 (offset k.descriptor (Code_object.size co)))));
      test "an entry at the image's last byte is in it" (fun () ->
          let obj = linked () in
          let co = read obj in
          let k = require_some (Code_object.kernel co "a") in
          let rodata = section obj ".rodata" in
          let at = rodata.at + k.descriptor - Option.get rodata.offset in
          let co' =
            read
              (patch obj (at + 16)
                 (set_u64 (Code_object.size co - 1 - k.descriptor)))
          in
          equal (option int)
            (Some (Code_object.size co - 1))
            (Option.map
               (fun (k : Code_object.kernel) -> k.entry)
               (Code_object.kernel co' "a")));
      (* The image starts at 0x300 and ends with .data's 3 bytes. *)
      cases
        ~name:(fun (n, _) -> strf "an image of 2^48%+d bytes" n)
        "an image longer than 2^48 bytes is refused"
        [ (0, false); (1, true) ]
        (fun (extra, refuse) ->
          let obj = linked () in
          let address = (1 lsl 48) + 0x300 - 3 + extra in
          let obj =
            patch obj
              (header obj (index obj ".data") + sh_addr)
              (set_u64 address)
          in
          if refuse then refused ~sub:"2^48" obj
          else equal int (1 lsl 48) (Code_object.size (read obj)));
    ]

let () =
  exit (run "device_amd_abi.code_object" [ reading; processors; refusals ])
