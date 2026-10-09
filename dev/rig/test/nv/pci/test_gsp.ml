(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GSP's RPC bodies and CPU sequences, against open-gpu-kernel-modules
   570.144: g_rpc-structures.h (rpc_gsp_rm_alloc_v03_00,
   rpc_gsp_rm_control_v03_00, rpc_set_page_directory_v1E_05,
   rpc_unloading_guest_driver_v1F_07, rpc_run_cpu_sequencer_v17_00), g_os_nvoc.h
   (PACKED_REGISTRY_TABLE) and rmgspseq.h (the opcodes). *)

open Windtrap
module Gsp = Rig_nv_pci.Gsp
module Falcon = Rig_nv_pci.Falcon

let u8 s o = Char.code s.[o]
let u32 s o = Int32.to_int (String.get_int32_le s o) land 0xffff_ffff
let u64 s o = Int64.to_int (String.get_int64_le s o)

(* rpc_gsp_rm_alloc_v03_00: hClient, hParent, hObject, hClass, status,
   paramsSize and flags as words, the parameters at 0x20. *)
let test_alloc () =
  let b = Gsp.rm_alloc ~client:0xc1000000 ~parent:7 ~obj:9 ~cls:0xc7c0 "abc" in
  equal int (0x20 + 3) (String.length b);
  equal (list int)
    [ 0xc1000000; 7; 9; 0xc7c0; 0; 3; 0 ]
    (List.init 7 (fun i -> u32 b (4 * i)));
  equal string "abc" (String.sub b 0x20 3)

(* rpc_gsp_rm_control_v03_00: hClient, hObject, cmd, status, paramsSize, flags,
   the parameters at 0x18. *)
let test_control () =
  let b = Gsp.rm_control ~client:1 ~obj:2 ~cmd:0x20800a0a "xy" in
  equal int (0x18 + 2) (String.length b);
  equal (list int)
    [ 1; 2; 0x20800a0a; 0; 2; 0 ]
    (List.init 6 (fun i -> u32 b (4 * i)));
  equal string "xy" (String.sub b 0x18 2)

let params = Gen.string_of ~size:(Gen.int_range 0 600) Gen.char

let test_answer =
  prop "an answer gives back the status and the parameters" params (fun p ->
      equal
        (result (pair int string) string)
        (Ok (0, p))
        (Gsp.rm_answer `Control (Gsp.rm_control ~client:1 ~obj:2 ~cmd:3 p));
      let a =
        Bytes.of_string (Gsp.rm_alloc ~client:1 ~parent:2 ~obj:3 ~cls:4 p)
      in
      Bytes.set_int32_le a 0x10 0x56l;
      equal
        (result (pair int string) string)
        (Ok (0x56, p))
        (Gsp.rm_answer `Alloc (Bytes.to_string a)))

let test_short_answer () =
  ignore (require_error (Gsp.rm_answer `Alloc "short"));
  let b = Bytes.of_string (Gsp.rm_control ~client:1 ~obj:2 ~cmd:3 "") in
  Bytes.set_int32_le b 0x10 64l;
  ignore (require_error (Gsp.rm_answer `Control (Bytes.to_string b)))

(* rpc_set_page_directory_v1E_05: hClient, hDevice, pasid, then at 0x10
   NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05: physAddress, numEntries at
   8, flags at 12 (APERTURE_VIDMEM, 0x8), hVASpace at 16, chId at 20,
   subDeviceId at 24, pasid at 28; no PASID is 0xffffffff. *)
let test_page_directory () =
  let b =
    Gsp.page_directory ~client:1 ~device:2 ~vaspace:3 ~root:0x20_0000 ~entries:4
  in
  equal int 48 (String.length b);
  equal (list int) [ 1; 2; 0xffffffff ] [ u32 b 0; u32 b 4; u32 b 8 ];
  equal int 0x20_0000 (u64 b 0x10);
  equal (list int)
    [ 4; 8; 3; 0; 1; 0xffffffff ]
    (List.map (fun o -> u32 b (0x10 + o)) [ 8; 12; 16; 20; 24; 28 ])

(* rpc_unloading_guest_driver_v1F_07: bInPMTransition, bGc6Entering, then
   newLevel at 4. *)
let test_unloading () =
  equal int 8 (String.length Gsp.unloading);
  equal (list int) [ 0; 0; 0x40 ]
    [ u8 Gsp.unloading 0; u8 Gsp.unloading 1; u32 Gsp.unloading 4 ]

(* PACKED_REGISTRY_TABLE: size and numEntries, then PACKED_REGISTRY_ENTRY of 16
   bytes each (nameOffset, type at 4, data at 8, length at 12), then the names,
   each ending in a zero byte; a DWORD entry's type is 1. *)
let test_registry =
  let key = Gen.string_of ~size:(Gen.int_range 1 20) (Gen.char_range 'A' 'z') in
  prop "the registry holds each key, its value and its name"
    Gen.(list ~size:(int_range 0 6) (pair key (int_range 0 0xffff_ffff)))
    (fun keys ->
      let t = Gsp.registry keys in
      equal int (String.length t) (u32 t 0);
      equal int (List.length keys) (u32 t 4);
      List.iteri
        (fun i (k, v) ->
          let e = 8 + (16 * i) in
          let name = u32 t e in
          equal string (k ^ "\000") (String.sub t name (String.length k + 1));
          equal (list int) [ 1; v; 4 ]
            [ u8 t (e + 4); u32 t (e + 8); u32 t (e + 12) ])
        keys)

(* CPU sequences *)

(* rpc_run_cpu_sequencer_v17_00: bufferSizeDWord, cmdIndex (the words used),
   regSaveArea of 8 words, then the commands from 0x28, each its opcode and its
   payload, a GSP_SEQ_BUF_PAYLOAD_ structure. Opcodes: REG_WRITE 0 (addr, val),
   REG_MODIFY 1 (addr, mask, val), REG_POLL 2 (addr, mask, val, timeout, error),
   DELAY_US 3 (val), REG_STORE 4 (addr, index), CORE_RESET 5, CORE_START 6,
   CORE_WAIT_FOR_HALT 7, CORE_RESUME 8. *)
let sequencer words =
  let n = List.length words in
  let b = Bytes.make (0x28 + (4 * n)) '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.set_int32_le b 4 (Int32.of_int n);
  List.iteri
    (fun i w -> Bytes.set_int32_le b (0x28 + (4 * i)) (Int32.of_int w))
    words;
  Bytes.to_string b

let pp_op ppf (op : Falcon.op) =
  match op with
  | Write (r, x) -> Format.fprintf ppf "write 0x%x 0x%x" r x
  | Modify (r, m, x) -> Format.fprintf ppf "modify 0x%x 0x%x 0x%x" r m x
  | Poll (_, r, m, Is x) -> Format.fprintf ppf "poll 0x%x 0x%x 0x%x" r m x
  | Delay us -> Format.fprintf ppf "delay %d" us
  | _ -> Format.fprintf ppf "other"

let show ops = List.map (Format.asprintf "%a" pp_op) ops

let test_sequence () =
  let ops =
    require_ok
      (Gsp.sequence ~libos:0
         (sequencer
            [
              0;
              0x1000;
              0x5;
              2;
              0x3000;
              0xff;
              0x10;
              1000;
              0;
              3;
              20;
              4;
              0x4000;
              2;
            ]))
  in
  equal (list string)
    [ "write 0x1000 0x5"; "poll 0x3000 0xff 0x10"; "delay 20" ]
    (show ops)

(* The RM's sequencer modifies a register as [(r & ~mask) | val] (kernel_gsp.c,
   GSP_SEQ_BUF_OPCODE_REG_MODIFY): the value's bits are set whether or not the
   mask covers them. *)
let test_sequence_modify =
  let word = Gen.int_range 0 0xffff_ffff in
  prop "a modify sets the register as the RM's sequencer does"
    Gen.(triple word word word)
    (fun (r, mask, v) ->
      match Gsp.sequence ~libos:0 (sequencer [ 1; 0x2000; mask; v ]) with
      | Ok [ Modify (0x2000, m, x) ] ->
          equal int (r land lnot mask lor v) (r land lnot m lor (x land m))
      | Ok ops -> failf "the modify ran as %s" (String.concat "; " (show ops))
      | Error why -> fail why)

let test_sequence_cores () =
  let one op =
    require_ok (Gsp.sequence ~libos:0x1234_5678_9000 (sequencer [ op ]))
  in
  equal (list string) (show (Falcon.start Falcon.gsp)) (show (one 6));
  equal (list string) (show (Falcon.wait_halt Falcon.gsp)) (show (one 7));
  let reset = one 5 in
  equal (list string)
    (show (Falcon.reset Falcon.gsp `Falcon))
    (show
       (List.filteri
          (fun i _ -> i < List.length (Falcon.reset Falcon.gsp `Falcon))
          reset));
  (* The resumption gives the GSP its libos arguments in its mailboxes,
     NV_PGSP_FALCON_MAILBOX0 and 1 (0x110040, 0x110044). *)
  let resume = show (one 8) in
  mem string "write 0x110040 0x56789000" resume;
  mem string "write 0x110044 0x1234" resume

let test_sequence_refused =
  cases "a sequence it cannot run is refused"
    ~name:(fun (n, _, _) -> n)
    [
      ("an unknown opcode", [ 9 ], "unknown opcode 9");
      ("a command cut short", [ 0; 0x1000 ], "ending inside");
    ]
    (fun (_, words, why) ->
      contains ~sub:why
        (require_error (Gsp.sequence ~libos:0 (sequencer words))))

(* The boot pool *)

module Tables = Rig_pci_support.Tables
module Page_table = Rig_pci.Page_table

let page = 0x1000

(* Page tables in a fake format whose boot pool [Gsp.boot_pool start] sizes. *)
let tables start =
  let boot = Gsp.boot_pool start in
  let s = Rig_pci.Space.create ~base:0 (1 lsl 40) in
  let t =
    Page_table.create
      (Tables.format (Tables.memory ()))
      s
      ~memory:(boot + (64 * Rig_pci_support.mib))
      ~boot ~tables:Main
      ~pages:[ (page, page) ]
  in
  Page_table.booted t;
  t

let booter n =
  `Booter
    {
      Rig_nv_pci.Images.image = String.make n 'b';
      code = (0, 0);
      data = (0, 0);
      pkc = 0;
      engines = 0;
      ucode = 0;
    }

let test_boot_pool =
  prop "the boot pool holds FWSEC and the booter once booting ended"
    Gen.(
      pair
        (int_range 1 Rig_nv_pci.Vbios.window)
        (int_range 1 (2 * Rig_pci_support.mib)))
    (fun (fwsec, n) ->
      let t = tables (booter n) in
      let palloc what n =
        not_equal ~msg:what (option int) None (Page_table.palloc ~boot:true t n)
      in
      palloc "FWSEC" fwsec;
      palloc "the booter" n)

let test_boot_pool_fmc () =
  equal int (2 * Rig_pci_support.mib)
    (Gsp.boot_pool
       (`Fmc
          (let r = { Rig_nv_pci.Images.contents = ""; at = 0; length = 0 } in
           {
             Rig_nv_pci.Images.fmc = r;
             hash = r;
             signature = r;
             public_key = r;
           })))

(* A failed boot *)

module Support = Rig_nv_pci_support

let range n =
  { Rig_nv_pci.Images.contents = String.make n 'x'; at = 0; length = n }

(* Firmware whose VBIOS holds no FWSEC: its boot fails once the GSP's memory is
   written and its first messages sent, before any falcon runs. *)
let firmware () =
  let start = booter 0x1000 in
  ( start,
    {
      Rig_nv_pci.Images.gsp = range 0x8000;
      signature = range 0x1000;
      bootloader = { image = range 0x1000; code = 0; data = 0; manifest = 0 };
      start;
    } )

let test_failed_boot () =
  let gpu = Support.gpu () in
  (* NV_PMC_BOOT_42 of an AD102. *)
  Rig_pci.Window.set32 gpu.regs 0xa00 ((0x19 lsl 24) lor (2 lsl 20));
  let fn = require_ok (Rig_pci.Function.take gpu.machine "0000:01:00.0") in
  let chip = require_ok (Rig_nv_pci.Chip.of_function fn) in
  let base = 64 lsl 30 in
  let space = Rig_pci.Space.create ~base (1 lsl 36) in
  require_ok (Rig_pci.Machine.reserve gpu.machine ~base (1 lsl 36));
  let start, fw = firmware () in
  let boot = Gsp.boot_pool start in
  let tables =
    Page_table.create
      (Tables.format (Tables.memory ()))
      space
      ~memory:(boot + (64 * Rig_pci_support.mib))
      ~boot ~tables:Main
      ~pages:[ (page, page) ]
  in
  Page_table.booted tables;
  Rig_pci.Function.set_bus_master fn true;
  let placement =
    {
      Gsp.chip;
      memory = 1 lsl 33;
      fn;
      tables;
      bar = Support.window (16 * Rig_pci_support.mib);
      space;
    }
  in
  ignore
    (require_error
       ~pp:(fun ppf _ -> Format.pp_print_string ppf "a GSP")
       (Gsp.boot placement fw));
  let events = List.rev !(gpu.events) in
  let allocs =
    List.length
      (List.filter (function Support.Alloc _ -> true | _ -> false) events)
  in
  let frees = List.filter (( = ) Support.Free) events in
  not_equal ~msg:"allocations" int 0 allocs;
  equal int ~msg:"frees" allocs (List.length frees);
  (* Every free comes after the bus mastering went off for the last time. *)
  let rec after_off off = function
    | [] -> ()
    | Support.Master on :: rest -> after_off (not on) rest
    | Free :: rest ->
        equal bool ~msg:"bus mastering off before a free" true off;
        after_off off rest
    | Alloc _ :: rest -> after_off off rest
  in
  after_off true events

let () =
  exit
  @@ run "rig_nv_pci.gsp"
       [
         group ~timeout:10. "rpcs"
           [
             test "an allocation" test_alloc;
             test "a control" test_control;
             test_answer;
             test "a short answer is refused" test_short_answer;
             test "a page directory" test_page_directory;
             test "unloading" test_unloading;
             test_registry;
           ];
         group ~timeout:10. "sequences"
           [
             test "register accesses" test_sequence;
             test_sequence_modify;
             test "the GSP's falcon" test_sequence_cores;
             test_sequence_refused;
           ];
         group ~timeout:10. "failures"
           [
             test "a failed boot gives its system memory back" test_failed_boot;
           ];
         group ~timeout:10. "boot pool"
           [
             test_boot_pool;
             test "the FMC's needs only the root table" test_boot_pool_fmc;
           ];
       ]
