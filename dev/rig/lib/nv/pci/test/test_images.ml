(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The firmware a GPU boots with, read from containers fixtures/firmware.py and
   fixtures/sections.py lay out as NVIDIA's firmware is (fixtures/README.md). *)

open Windtrap
module Images = Rig_nv_pci.Images

let file name =
  In_channel.with_open_bin
    (Filename.concat "fixtures" name)
    In_channel.input_all

let bytes (r : Images.range) = String.sub r.contents r.at r.length

(* As firmware.py lays the booter out: its data, 0x1000 bytes of code then 0x800
   of data, each byte its offset modulo 256; the first application from 0x100;
   the second of two 384-byte signatures, of 0xa2 bytes, written at 0x1010,
   0x10 into the data; engines 0x5 and ucode ID 9. *)
let test_booter () =
  let b = require_ok (Images.booter (file "booter.bin")) in
  equal int 0x1800 (String.length b.image);
  equal (pair int int) (0x100, 0xf00) b.code;
  equal (pair int int) (0x1000, 0x800) b.data;
  equal (triple int int int) (0x10, 0x5, 9) (b.pkc, b.engines, b.ucode);
  let patch = 0x1010 in
  String.iteri
    (fun i c ->
      let signed = i >= patch && i < patch + 384 in
      equal char (if signed then '\xa2' else Char.chr (i land 0xff)) c)
    b.image

let test_bootloader () =
  let b = require_ok (Images.bootloader (file "bootloader.bin")) in
  equal (pair int int) (0x400, 0x3000) (b.image.at, b.image.length);
  equal (triple int int int) (0x1000, 0x800, 0x100) (b.code, b.data, b.manifest)

let test_fmc () =
  let f = require_ok (Images.fmc (file "fmc.elf")) in
  List.iter
    (fun (r, c, n) -> equal string (String.make n c) (bytes r))
    [
      (f.hash, 'H', 48);
      (f.signature, 'S', 96);
      (f.public_key, 'P', 97);
      (f.fmc, 'I', 0x2000);
    ]

let test_gsp =
  cases "the GSP's image and its family's signature are sections"
    ~name:(fun (_, n) -> n)
    Rig_nv_pci.Chip.
      [
        (Ampere, ".fwsignature_ga10x");
        (Ada, ".fwsignature_ad10x");
        (Blackwell, ".fwsignature_gb20x");
      ]
    (fun (family, signature) ->
      let image, s = require_ok (Images.gsp family (file "sections.elf")) in
      equal string ".fwimage" (bytes image);
      equal string signature (bytes s))

(* Errors *)

(* A container cut anywhere before its last field is refused, never read past
   its end: the booter's data ends the file. *)
let test_cut =
  let s = file "booter.bin" in
  prop "a cut container is refused"
    (Gen.int_range 0 (String.length s - 1))
    (fun n ->
      contains ~sub:"points outside"
        (require_error (Images.booter (String.sub s 0 n))))

let test_no_signature () =
  let s = Bytes.of_string (file "booter.bin") in
  (* The signature count, the third word the header points to. *)
  let count_at = Int32.to_int (Bytes.get_int32_le s (0x18 + 0x18)) in
  Bytes.set_int32_le s count_at 0l;
  contains ~sub:"no signature"
    (require_error (Images.booter (Bytes.to_string s)))

let test_missing_section () =
  contains ~sub:"no section image"
    (require_error (Images.fmc (file "sections.elf")))

let test_no_elf () =
  contains ~sub:"no ELF object" (require_error (Images.gsp Ada "not an object"))

(* The fixtures' directory holds files of other contents than the pinned images,
   which a lookup skips. *)
let test_read_missing () =
  contains ~sub:"nvidia/ga102/gsp/gsp-570.144.bin"
    (require_error (Images.read Ada [ "fixtures" ]))

(* Pins *)

let test_names =
  cases "a family boots from its GSP image, bootloader and starter"
    ~name:(fun (n, _, _) -> n)
    Rig_nv_pci.Chip.
      [
        ("Ampere", Ampere, "nvidia/ga102/gsp/booter_load-570.144.bin");
        ("Ada", Ada, "nvidia/ad102/gsp/booter_load-570.144.bin");
        ("Blackwell", Blackwell, "nvidia/gb202/gsp/fmc-570.144.bin");
      ]
    (fun (_, family, starter) ->
      let names = Images.names family in
      equal int 3 (List.length names);
      equal string "nvidia/ga102/gsp/gsp-570.144.bin" (List.hd names);
      equal string starter (List.nth names 2);
      List.iter (fun n -> mem string n (List.map fst Images.pinned)) names)

let test_digests () =
  List.iter
    (fun (_, d) ->
      equal int 64 (String.length d);
      String.iter
        (fun c -> mem char c (List.init 16 (fun i -> "0123456789abcdef".[i])))
        d)
    Images.pinned

let test_origin () =
  starts_with ~affix:"https://" Images.origin;
  ends_with ~affix:"/" Images.origin

let () =
  exit
  @@ run "rig_nv_pci.images"
       [
         group ~timeout:10. "containers"
           [
             test "the booter's image is its data, signed" test_booter;
             test "the bootloader's offsets come from its descriptor"
               test_bootloader;
             test "the FMC's parts are its sections" test_fmc;
             test_gsp;
           ];
         group ~timeout:10. "errors"
           [
             test_cut;
             test "a booter with no signature is refused" test_no_signature;
             test "an object without a section is refused, naming it"
               test_missing_section;
             test "what is no ELF object is refused" test_no_elf;
             test "a missing image is refused, naming it" test_read_missing;
           ];
         group ~timeout:10. "pins"
           [
             test_names;
             test "every pin is a BLAKE2b-256 digest" test_digests;
             test "an image's URL is the origin and its path" test_origin;
           ];
       ]
