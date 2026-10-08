(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FWSEC from a VBIOS, on the images fixtures/vbios.py lays out as NVIDIA's RM
   reads them. *)

open Windtrap
module Vbios = Rig_nv_pci.Vbios

let rom name =
  In_channel.with_open_bin
    (Filename.concat "fixtures" name)
    In_channel.input_all

(* As vbios.py lays FWSEC out: 512 bytes of code then 1024 of data, its DMEM
   mapper at 0x80 of the data taking its command at 0x100, its signature checked
   at 0x200, the last of two signatures of 0x22 bytes. *)
let imem = 0x200
let dmem = 0x400
let mapper = 0x80
let command = 0x100
let pkc = 0x200
let frts = 0x7_ff00_0000
let u32 s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let test_fields () =
  let f = require_ok (Vbios.fwsec (rom "vbios.rom") ~frts) in
  equal int (imem + dmem) (String.length f.image);
  equal (list int)
    [ 0x1000; 0x2000; imem; 0x3000; dmem; pkc; 0x400; 3 ]
    [
      f.imem_pa;
      f.imem_va;
      f.imem_size;
      f.dmem_pa;
      f.dmem_size;
      f.pkc;
      f.engines;
      f.ucode;
    ]

(* The patch: the mapper's command is FRTS (0x15), its input buffer holds
   FWSECLIC_FRTS_CMD (version 1 and size of each part, the VBIOS read from the
   GPU's ROM (flags 2), the region in 4 KiB units, 1 MiB in 4 KiB units (0x100),
   in the GPU's memory (2)), and the last signature is in place; every other
   byte is the image's. *)
let test_patch () =
  let f = require_ok (Vbios.fwsec (rom "vbios.rom") ~frts) in
  let i = f.image in
  equal int 0x15 (u32 i (imem + mapper + 0x2c));
  let c = imem + command in
  equal (list int)
    [ 1; 24; 2; 1; 20; frts lsr 12; 0x100; 2 ]
    [
      u32 i c;
      u32 i (c + 4);
      u32 i (c + 0x14);
      u32 i (c + 0x18);
      u32 i (c + 0x1c);
      u32 i (c + 0x20);
      u32 i (c + 0x24);
      u32 i (c + 0x28);
    ];
  equal string (String.make 384 '\x22') (String.sub i (imem + pkc) 384);
  let original k = Char.chr (k land 0xff) in
  let patched k =
    (k >= imem + mapper + 0x2c && k < imem + mapper + 0x30)
    || (k >= c && k < c + 48)
    || (k >= imem + pkc && k < imem + pkc + 384)
    || (k >= imem + 0x10 && k < imem + 0x24)
    || (k >= imem + mapper + 8 && k < imem + mapper + 12)
  in
  String.iteri
    (fun k ch -> if not (patched k) then equal char (original k) ch)
    i

let test_refused =
  cases "a VBIOS without what FWSEC needs is refused, naming it"
    ~name:(fun (name, _) -> name)
    [
      ("vbios_no_bit.rom", "no BIT table");
      ("vbios_debug.rom", "no production FWSEC");
      ("vbios_no_mapper.rom", "no DMEM mapper");
    ]
    (fun (name, why) ->
      contains ~sub:why (require_error (Vbios.fwsec (rom name) ~frts)))

(* A ROM cut before the end of FWSEC's image, at byte 4396 (the extension image
   at 1536, the descriptor at 0x200 of it, 812 bytes, then 1536 of image), is
   refused, never read past its end; one cut after it is whole. *)
let needed = 1536 + 0x200 + 812 + imem + dmem

let test_cut =
  prop "a VBIOS cut before FWSEC's end is refused"
    (Gen.int_range 0 (String.length (rom "vbios.rom")))
    (fun n ->
      cover "cut before" (n < needed);
      cover "cut after" (n >= needed);
      let r = Vbios.fwsec (String.sub (rom "vbios.rom") 0 n) ~frts in
      equal bool (n >= needed) (Result.is_ok r))

let () =
  exit
  @@ run "rig_nv_pci.vbios"
       [
         group ~timeout:10. "fwsec"
           [
             test "FWSEC's layout comes from its descriptor" test_fields;
             test "the patch runs FRTS and carries the signature" test_patch;
             test_refused;
             test_cut;
           ];
       ]
