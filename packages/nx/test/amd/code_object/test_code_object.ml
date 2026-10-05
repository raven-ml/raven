(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Code objects the test builds: relocation, padding, kernel names, the fields
   of a kernel descriptor at the offsets of LLVM's AMDHSAKernelDescriptor.h, the
   processor of the header's flags and the GPUs that run it, after LLVM's
   AMDGPUUsage, and refusals. *)

open Windtrap
module C = Nx_amd_code_object

(* A 64-bit little-endian relocatable object for [machine] with the header's ABI
   version [abi] and flags [flags], and the given sections, after the null
   section: (name, type, address, contents, link, info, align, entry size). *)
let elf ~machine ~abi ~flags sections =
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
  Bytes.set_uint8 hdr 8 abi;
  Bytes.set_uint16_le hdr 16 1;
  Bytes.set_uint16_le hdr 18 machine;
  Bytes.set_int32_le hdr 48 (Int32.of_int flags);
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

(* A kernel descriptor, laid out as AMDHSAKernelDescriptor.h lays out
   kernel_descriptor_t: 64 bytes. *)
let descriptor ~group ~private_ ~kernarg ~entry ~rsrc1 ~rsrc2 ~rsrc3 ~props =
  let b = Bytes.make 64 '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int group);
  Bytes.set_int32_le b 4 (Int32.of_int private_);
  Bytes.set_int32_le b 8 (Int32.of_int kernarg);
  Bytes.set_int64_le b 16 (Int64.of_int entry);
  Bytes.set_int32_le b 44 (Int32.of_int rsrc3);
  Bytes.set_int32_le b 48 (Int32.of_int rsrc1);
  Bytes.set_int32_le b 52 (Int32.of_int rsrc2);
  Bytes.set_uint16_le b 56 props;
  Bytes.to_string b

(* R_AMDGPU_REL64, and the kernel code properties of amd_hsa_kernel_code.h. *)
let rel64 = 5
let private_segment_buffer = 0x1
let dispatch_ptr = 0x2
let wave32 = 0x400

(* ELF.h's AMD GPU machine, code object version 6, and EF_AMDGPU_MACH values
   with the generic version in the flags' top byte. *)
let em_amdgpu = 224
let v5 = 3
let v6 = 4
let gfx1201 = 0x4e
let gfx12_generic = 0x59
let generic ?(version = 1) mach = (version lsl 24) lor mach

(* Laid out: [.rodata], two descriptors, at 0, of kernels whose code properties
   differ in each bit; [.data] at 128, a word that [reloc] patches to point to
   [k]; [.text], 6 bytes, at 256, where [k] starts: 262 bytes. Kernel [k]'s
   descriptor is the first, [a]'s the second, and [ext] is undefined. It is
   compiled for gfx1201 in a code object of version 6, unless [machine], [abi]
   and [flags] say otherwise. *)
let code_object ?(reloc = rela 0 2 rel64 4) ?(entry = 256)
    ?(machine = em_amdgpu) ?(abi = v6) ?(flags = gfx1201) () =
  let k =
    descriptor ~group:0x100 ~private_:0x40 ~kernarg:0x18 ~entry ~rsrc1:0x11
      ~rsrc2:0x22 ~rsrc3:0x3
      ~props:(private_segment_buffer lor wave32)
  and a =
    descriptor ~group:0 ~private_:0 ~kernarg:0 ~entry:(entry - 64) ~rsrc1:0
      ~rsrc2:0 ~rsrc3:0 ~props:dispatch_ptr
  in
  let strtab = "\000k.kd\000k\000a.kd\000ext\000" in
  elf ~machine ~abi ~flags
    [
      (".rodata", 1, 0, k ^ a, 0, 0, 64, 0);
      (".data", 1, 0, String.make 8 '\000', 0, 0, 8, 0);
      (".text", 1, 0, "\x01\x02\x03\x04\x05\x06", 0, 0, 256, 0);
      ( ".symtab",
        2,
        0,
        sym 0 0 0 ^ sym 1 1 0 ^ sym 6 3 0 ^ sym 8 1 64 ^ sym 13 0 0,
        5,
        0,
        8,
        24 );
      (".strtab", 3, 0, strtab, 0, 0, 1, 0);
      (".rela.data", 4, 0, reloc, 4, 2, 8, 24);
    ]

let load obj = require_ok ~msg:"loads" (C.of_string obj)

let test_image () =
  let img = C.image (load (code_object ())) in
  equal ~msg:"262 bytes padded to whole words" int 264 (String.length img);
  equal ~msg:"padded with zeros" string "\x05\x06\000\000"
    (String.sub img 260 4);
  equal ~msg:"a REL64 word holds its target's offset from it, plus the addend"
    int64
    (Int64.of_int (256 - 128 + 4))
    (String.get_int64_le img 128)

let test_kernels () =
  equal (list string) [ "a"; "k" ] (C.kernels (load (code_object ())))

let test_kernel () =
  let co = load (code_object ()) in
  let k = require_ok (C.kernel co "k") in
  equal ~msg:"descriptor" int 0 k.descriptor;
  equal ~msg:"entry" int 256 k.entry;
  equal ~msg:"group segment" int 0x100 k.group_segment;
  equal ~msg:"private segment" int 0x40 k.private_segment;
  equal ~msg:"kernarg size" int 0x18 k.kernarg_size;
  equal ~msg:"rsrc1" int 0x11 k.rsrc1;
  equal ~msg:"rsrc2" int 0x22 k.rsrc2;
  equal ~msg:"rsrc3" int 0x3 k.rsrc3;
  equal ~msg:"wave32" bool true k.wave32;
  equal ~msg:"dispatch_ptr" bool false k.dispatch_ptr;
  equal ~msg:"private_segment_buffer" bool true k.private_segment_buffer;
  let a = require_ok (C.kernel co "a") in
  equal ~msg:"a's descriptor" int 64 a.descriptor;
  equal ~msg:"a's entry, relative to its descriptor" int 256 a.entry;
  equal ~msg:"a's waves are 64 lanes" bool false a.wave32;
  equal ~msg:"a reads its dispatch packet" bool true a.dispatch_ptr;
  equal ~msg:"a reads no scratch descriptor" bool false a.private_segment_buffer

(* The GPUs each processor's code objects run on: the GPU itself, and every GPU
   a generic processor lists. *)
let gpus = [ "gfx1100"; "gfx1151"; "gfx1200"; "gfx1201"; "gfx942"; "gfx950" ]

let processors =
  cases
    ~name:(fun ((p, flags), _) -> Printf.sprintf "%s (flags 0x%x)" p flags)
    "the GPUs that run a code object"
    [
      (* (processor, flags), and the GPUs of [gpus] that run it *)
      (("gfx1201", gfx1201), [ "gfx1201" ]);
      (("gfx1100", 0x41), [ "gfx1100" ]);
      (("gfx942", 0x4c), [ "gfx942" ]);
      (("gfx12-generic", generic gfx12_generic), [ "gfx1200"; "gfx1201" ]);
      (("gfx11-generic", generic 0x54), [ "gfx1100"; "gfx1151" ]);
      (("gfx9-4-generic", generic 0x5f), [ "gfx942"; "gfx950" ]);
      ( ("gfx12-generic", generic ~version:2 gfx12_generic),
        [ "gfx1200"; "gfx1201" ] );
      (* XNACK and SRAMECC settings leave the processor as it is. *)
      (("gfx942", 0xf00 lor 0x4c), [ "gfx942" ]);
    ]
    (fun ((processor, flags), runs) ->
      let co = load (code_object ~flags ()) in
      equal ~msg:"its processor" string processor (C.target co);
      equal ~msg:"the GPUs that run it" (list string) runs
        (List.filter (C.runs_on co) gpus))

let errors =
  let co = load (code_object ()) in
  group "errors"
    [
      test "not an ELF object" (fun () ->
          ignore (require_error (C.of_string "not an elf")));
      test "an object for another machine" (fun () ->
          require_error (C.of_string (code_object ~machine:62 ()))
          |> starts_with ~affix:"not an AMD GPU code object (e_machine 62)");
      cases ~name:(Printf.sprintf "EF_AMDGPU_MACH 0x%x")
        "a processor LLVM does not name" [ 0x0; 0x1; 0x27; 0xff ] (fun mach ->
          require_error (C.of_string (code_object ~flags:mach ()))
          |> starts_with ~affix:"a code object for no AMD GPU");
      test "a generic code object before version 6" (fun () ->
          require_error
            (C.of_string
               (code_object ~abi:v5 ~flags:(generic gfx12_generic) ()))
          |> starts_with
               ~affix:"a gfx12-generic code object before code object version 6");
      test "a generic code object of version 0" (fun () ->
          require_error
            (C.of_string
               (code_object ~flags:(generic ~version:0 gfx12_generic) ()))
          |> starts_with
               ~affix:"a gfx12-generic code object of generic version 0");
      test "a GPU's code object before version 6" (fun () ->
          equal string "gfx1201"
            (C.target (load (code_object ~abi:v5 ~flags:gfx1201 ()))));
      test "a relocation of another kind" (fun () ->
          require_error (C.of_string (code_object ~reloc:(rela 0 2 1 0) ()))
          |> starts_with ~affix:"an unknown AMD GPU relocation 1");
      test "a relocation to an undefined symbol" (fun () ->
          require_error (C.of_string (code_object ~reloc:(rela 0 4 rel64 0) ()))
          |> starts_with ~affix:"the code object refers to an undefined");
      test "no such kernel" (fun () ->
          require_error (C.kernel co "ext")
          |> starts_with ~affix:"the code object has no kernel ext");
      test "code outside the image" (fun () ->
          ignore
            (require_error (C.kernel (load (code_object ~entry:264 ())) "k")));
    ]

let () =
  exit
    (run "nx.amd.code_object"
       [
         test "the image is relocated and padded" test_image;
         test "kernels are named by their descriptors" test_kernels;
         test "a descriptor's fields" test_kernel;
         processors;
         errors;
       ])
