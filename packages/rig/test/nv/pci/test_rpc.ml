(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GSP's records, calls and CPU sequences, against open-gpu-kernel-modules
   570.144: message_queue_cpu.c and the RPC headers, g_rpc-structures.h
   (rpc_gsp_rm_alloc_v03_00, rpc_gsp_rm_control_v03_00, rpc_free_v03_00,
   rpc_set_page_directory_v1E_05, rpc_unloading_guest_driver_v1F_07,
   rpc_run_cpu_sequencer_v17_00, rpc_rc_triggered_v17_02,
   rpc_os_error_log_v17_00), ctrl2080nvd.h (NV2080CtrlNocatJournalInsertRecord),
   g_os_nvoc.h (PACKED_REGISTRY_TABLE) and rmgspseq.h (the opcodes). *)

open Windtrap
open Rig_nv_pci

let u8 s o = Char.code s.[o]
let u32 s o = Int32.to_int (String.get_int32_le s o) land 0xffff_ffff
let u64 s o = Int64.to_int (String.get_int64_le s o)

(* rpc_global_enums.h's numbers. *)
let continuation = 0x47
let gsp_rm_control = 0x4c
let run_cpu_sequencer = 0x1002
let rc_triggered = 0x1004
let mmu_fault_queued = 0x1005
let os_error_log = 0x1006
let libos_print = 0x100c
let nocat_record = 0x1020

(* Records *)

(* An element is 4 KiB; its RPC header ends at byte 80, so a record of 16
   elements carries 16 * 4096 - 80 bytes of body. GSP_MSG_QUEUE_ELEMENT holds
   two buffers of 16 bytes, then its checksum at 0x20, its sequence number at
   0x24 and its element count at 0x28; its RPC header starts at 0x30, the
   signature at its byte 4. *)
let element = 4096
let record_max = (16 * element) - 80

(* The XOR of a string's 64-bit words, folded to 32 bits. *)
let fold s =
  let x = ref 0L in
  for i = 0 to (String.length s / 8) - 1 do
    x := Int64.logxor !x (String.get_int64_le s (8 * i))
  done;
  Int64.(to_int (logand (logxor !x (shift_right_logical !x 32)) 0xffff_ffffL))

let bodies =
  Gen.(
    one_of
      [
        string_of ~size:(int_range 0 64) char;
        map
          (fun n -> String.init n (fun i -> Char.chr (i land 0xff)))
          (int_range 0 ((2 * record_max) + 10));
      ])

let test_records =
  prop "records carry the body, the first with its function, each checksummed"
    Gen.(pair (int_range 0 0xffff) bodies)
    (fun (seq, body) ->
      let rs = Rpc.records ~seq gsp_rm_control body in
      cover "one record" (List.length rs = 1);
      cover "continued" (List.length rs > 1);
      equal int
        (Int.max 1 ((String.length body + record_max - 1) / record_max))
        (List.length rs);
      let ms =
        List.mapi
          (fun i r ->
            equal ~msg:"whole elements" int 0 (String.length r mod element);
            equal ~msg:"the checksum" int 0 (fold r);
            equal ~msg:"the sequence number" int (seq + i) (u32 r 0x24);
            let m, count = require_ok (Rpc.message r) in
            equal ~msg:"the element count" int (String.length r / element) count;
            equal int count (Rpc.elements r);
            at_most int ~than:record_max (String.length m.body);
            m)
          rs
      in
      equal string body (String.concat "" (List.map (fun m -> m.Rpc.body) ms));
      equal int gsp_rm_control (List.hd ms).fn;
      List.iter (fun m -> equal int continuation m.Rpc.fn) (List.tl ms))

(* A message reads back as the record that wrote it, its result the pending
   value the sender writes (NV_VGPU_MSG_RESULT_RPC_PENDING). *)
let test_message =
  prop "a message reads back its record"
    Gen.(pair (int_range 0 0x2000) (string_of ~size:(int_range 0 9000) char))
    (fun (fn, body) ->
      let r = List.hd (Rpc.records ~seq:7 fn body) in
      let m, _ = require_ok (Rpc.message r) in
      equal int fn m.fn;
      equal int 0xffff_ffff m.result;
      equal string body m.body)

(* The RPC header's signature is at byte 0x34 of the element. *)
let test_not_a_message () =
  let e = Bytes.of_string (List.hd (Rpc.records ~seq:0 gsp_rm_control "x")) in
  Bytes.set e 52 '\000';
  contains ~sub:"signature" (require_error (Rpc.message (Bytes.to_string e)));
  contains ~sub:"shorter" (require_error (Rpc.message "short"));
  let e = List.hd (Rpc.records ~seq:0 gsp_rm_control (String.make 5000 'x')) in
  contains ~sub:"longer" (require_error (Rpc.message (String.sub e 0 element)))

(* Faults *)

(* rpc_rc_triggered_v17_02: the channel at byte 4, the exception type at 16;
   nverror.h's ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT is 31. *)
let rc ~chid ~xid =
  let b = Bytes.make 48 '\000' in
  Bytes.set_int32_le b 4 (Int32.of_int chid);
  Bytes.set_int32_le b 16 (Int32.of_int xid);
  Bytes.to_string b

let event fn body =
  fst (require_ok (Rpc.message (List.hd (Rpc.records ~seq:0 fn body))))

let test_rc () =
  let why =
    require_some (Rpc.fault (event rc_triggered (rc ~chid:3 ~xid:31)))
  in
  contains ~sub:"channel 3" why;
  contains ~sub:"FIFO_ERROR_MMU_ERR_FLT" why;
  contains ~sub:"Xid 31" why

let test_rc_unknown () =
  let why =
    require_some (Rpc.fault (event rc_triggered (rc ~chid:1 ~xid:0xffff)))
  in
  contains ~sub:"an unknown error" why;
  is_some (Rpc.fault (event rc_triggered "short"))

let test_no_fault =
  cases "other messages carry no fault"
    ~name:(fun (n, _) -> n)
    [
      ("an error log", os_error_log);
      ("a CPU sequence", run_cpu_sequencer);
      ("a log print", libos_print);
      ("an answer", gsp_rm_control);
    ]
    (fun (_, fn) -> is_none (Rpc.fault (event fn (String.make 300 'x'))))

let test_mmu () = is_some (Rpc.fault (event mmu_fault_queued ""))

(* Reports *)

(* rpc_os_error_log_v17_00: the exception type at 0, the text at 0xc, 256
   bytes. *)
let error_log ~xid text =
  let b = Bytes.make 0x110 '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int xid);
  Bytes.blit_string text 0 b 0xc (Int.min 0x104 (String.length text));
  Bytes.to_string b

(* NV2080CtrlNocatJournalInsertRecord, of 1208 bytes, as GSP_POST_NOCAT_RECORD
   carries it: the source at 0x18, 65 bytes, then padding and subsystem to the
   error code at 0x60, 8 bytes. Those between are not zero here, so a source
   read past its field shows. *)
let journal ~source ~code =
  let b = Bytes.make 1208 '\000' in
  Bytes.fill b 0x59 7 'z';
  Bytes.blit_string source 0 b 0x18 (Int.min 65 (String.length source));
  Bytes.set_int64_le b 0x60 code;
  Bytes.to_string b

let report = option (pair string string)

let test_reports =
  cases "an error log and a journal record are reports of their subject"
    ~name:(fun (n, _, _) -> n)
    [
      ( "an error log's text, to its first zero byte",
        event os_error_log (error_log ~xid:13 "Graphics Exception\000junk"),
        Some ("Xid 13", "Xid 13: Graphics Exception") );
      ( "an error log's text, at most its 256 bytes",
        event os_error_log (error_log ~xid:31 (String.make 0x104 'a')),
        Some ("Xid 31", "Xid 31: " ^ String.make 0x100 'a') );
      ( "a journal record's source, to its first zero byte",
        event nocat_record
          (journal ~source:"FECS_A\000FECS_B\000FECS_C" ~code:3L),
        Some ("FECS_A", "FECS_A, error 0x3") );
      ( "a journal record's source, at most its 65 bytes",
        event nocat_record
          (journal ~source:(String.make 70 'b') ~code:0x41b_0143_05ccL),
        Some (String.make 65 'b', String.make 65 'b' ^ ", error 0x41b014305cc")
      );
      ( "a journal record's error code, all 64 bits",
        event nocat_record
          (journal ~source:"ASSERT" ~code:0x8000_0000_0000_0001L),
        Some ("ASSERT", "ASSERT, error 0x8000000000000001") );
    ]
    (fun (_, m, expected) -> equal report expected (Rpc.report m))

let test_short_reports =
  cases "a report shorter than its structure says so"
    ~name:(fun (n, _, _) -> n)
    [
      ("an error log", os_error_log, "abc");
      ( "an error log cut in its text",
        os_error_log,
        String.sub (error_log ~xid:13 "Graphics Exception") 0 0x20 );
      ("a journal record", nocat_record, "abc");
      ( "a journal record cut after its error code",
        nocat_record,
        String.sub (journal ~source:"FECS_A" ~code:3L) 0 0x68 );
    ]
    (fun (_, fn, body) ->
      let _, why = require_some (Rpc.report (event fn body)) in
      contains ~sub:"cut short" why)

let test_no_report =
  cases "other messages report nothing"
    ~name:(fun (n, _) -> n)
    [
      ("a stopped channel", rc_triggered);
      ("a queued MMU fault", mmu_fault_queued);
      ("a log print", libos_print);
      ("an answer", gsp_rm_control);
    ]
    (fun (_, fn) -> is_none (Rpc.report (event fn (String.make 1300 'x'))))

(* Calls *)

(* rpc_gsp_rm_alloc_v03_00: hClient, hParent, hObject, hClass, status,
   paramsSize and flags as words, the parameters at 0x20. *)
let test_alloc () =
  let b = Rpc.rm_alloc ~client:0xc1000000 ~parent:7 ~obj:9 ~cls:0xc7c0 "abc" in
  equal int (0x20 + 3) (String.length b);
  equal (list int)
    [ 0xc1000000; 7; 9; 0xc7c0; 0; 3; 0 ]
    (List.init 7 (fun i -> u32 b (4 * i)));
  equal string "abc" (String.sub b 0x20 3)

(* rpc_gsp_rm_control_v03_00: hClient, hObject, cmd, status, paramsSize, flags,
   the parameters at 0x18. *)
let test_control () =
  let b = Rpc.rm_control ~client:1 ~obj:2 ~cmd:0x20800a0a "xy" in
  equal int (0x18 + 2) (String.length b);
  equal (list int)
    [ 1; 2; 0x20800a0a; 0; 2; 0 ]
    (List.init 6 (fun i -> u32 b (4 * i)));
  equal string "xy" (String.sub b 0x18 2)

(* rpc_free_v03_00: NVOS00_PARAMETERS_v03_00's hRoot, hObjectParent,
   hObjectOld and status as words, the parent NV01_NULL_OBJECT, as
   rpcRmApiFree_GSP sends it. *)
let test_free () =
  let b = Rpc.free ~client:0xc1000000 ~obj:0xcf000004 in
  equal (list int)
    [ 0xc1000000; 0; 0xcf000004; 0 ]
    (List.init (String.length b / 4) (fun i -> u32 b (4 * i)))

let params = Gen.string_of ~size:(Gen.int_range 0 600) Gen.char

let test_answer =
  prop "an answer gives back the status and the parameters" params (fun p ->
      equal
        (result (pair int string) string)
        (Ok (0, p))
        (Rpc.rm_answer `Control (Rpc.rm_control ~client:1 ~obj:2 ~cmd:3 p));
      let a =
        Bytes.of_string (Rpc.rm_alloc ~client:1 ~parent:2 ~obj:3 ~cls:4 p)
      in
      Bytes.set_int32_le a 0x10 0x56l;
      equal
        (result (pair int string) string)
        (Ok (0x56, p))
        (Rpc.rm_answer `Alloc (Bytes.to_string a)))

let test_short_answer () =
  contains ~sub:"shorter" (require_error (Rpc.rm_answer `Alloc "short"));
  let b = Bytes.of_string (Rpc.rm_control ~client:1 ~obj:2 ~cmd:3 "") in
  Bytes.set_int32_le b 0x10 64l;
  contains ~sub:"64 bytes"
    (require_error (Rpc.rm_answer `Control (Bytes.to_string b)))

(* rpc_set_page_directory_v1E_05: hClient, hDevice, pasid, then at 0x10
   NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05: physAddress, numEntries at
   8, flags at 12 (APERTURE_VIDMEM, 0x8), hVASpace at 16, chId at 20,
   subDeviceId at 24, pasid at 28; no PASID is 0xffffffff. *)
let test_page_directory () =
  let b =
    Rpc.page_directory ~client:1 ~device:2 ~vaspace:3 ~root:0x20_0000 ~entries:4
  in
  equal int 48 (String.length b);
  equal (list int) [ 1; 2; 0xffffffff ] [ u32 b 0; u32 b 4; u32 b 8 ];
  equal int 0x20_0000 (u64 b 0x10);
  equal (list int)
    [ 4; 8; 3; 0; 1; 0xffffffff ]
    (List.map (fun o -> u32 b (0x10 + o)) [ 8; 12; 16; 20; 24; 28 ])

(* NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS (ctrl90f1.h): pageSize
   at 8, virtAddrLo at 0x10, virtAddrHi at 0x18, numLevelsToCopy at 0x20, then
   from 0x28 levels of 24 bytes: physAddress, size at 8, aperture at 16,
   pageShift at 20. CPU-RM hands over the tables of the 512 MiB from 4 GiB
   (SPLIT_VAS_SERVER_RM_MANAGED_VA_START and _SIZE), from the root down to the
   level whose entries map 2 MiB, each in the GPU's memory (GMMU_APERTURE_VIDEO,
   1) and of its size: 8 bytes an entry, 16 on that level
   (_gvaspacePopulatePDEentries). Version 2's root has 4 entries, version 3's
   2. *)
let test_reserved_pdes =
  cases "the GSP's range hands over its tables down to 2 MiB"
    ~name:(fun (n, _, _) -> n)
    [
      ( "version 2",
        Page_entry.V2,
        [ (47, 32); (38, 4096); (29, 4096); (21, 4096) ] );
      ( "version 3",
        Page_entry.V3,
        [ (56, 16); (47, 4096); (38, 4096); (29, 4096); (21, 4096) ] );
    ]
    (fun (_, v, levels) ->
      let tables = List.mapi (fun i _ -> 0x40_0000 + (i * 0x1000)) levels in
      let b = Rpc.reserved_pdes v tables in
      equal int 184 (String.length b);
      equal ~msg:"the range" (pair int int)
        (0x1_0000_0000, 0x2000_0000)
        Rpc.reserved;
      equal ~msg:"page size, range, levels" (list int)
        [ 0x20_0000; 0x1_0000_0000; 0x1_1fff_ffff; List.length levels ]
        [ u64 b 8; u64 b 0x10; u64 b 0x18; u32 b 0x20 ];
      List.iteri
        (fun i (shift, size) ->
          let l = 0x28 + (24 * i) in
          equal
            ~msg:(Printf.sprintf "level %d" i)
            (list int)
            [ List.nth tables i; size; 1; shift ]
            [ u64 b l; u64 b (l + 8); u32 b (l + 16); u8 b (l + 20) ])
        levels)

let test_reserved_depth () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rpc.reserved_pdes V2 [ 0; 0x1000; 0x2000 ]))

(* NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS (ctrl2080gpu.h): engineType, hClient,
   ChID, hChanClient at 0xc, hObject at 0x10, entryCount at 0x28, then from
   0x30 16 entries of 32 bytes: gpuPhysAddr, gpuVirtAddr at 8, size at 0x10,
   physAttr at 0x18, bufferId at 0x1c (16 bits), bInitialize at 0x1e,
   bNonmapped at 0x1f. Each entry decodes to [buffer; initialize; unmapped;
   physical address; size; attributes; virtual address]. *)
let promoted b =
  let entry i =
    let e = 0x30 + (0x20 * i) in
    [
      u32 b (e + 0x1c) land 0xffff;
      u8 b (e + 0x1e);
      u8 b (e + 0x1f);
      u64 b e;
      u64 b (e + 0x10);
      u32 b (e + 0x18);
      u64 b (e + 8);
    ]
  in
  List.init (u32 b 0x28) entry

(* What CPU-RM sends for a buffer (kgrctxPrepareInitializeCtxBuffer,
   kgrctxPreparePromoteCtxBuffer): one it initializes carries its physical
   address and size, in the GPU's memory and not cached by the GPU
   (NV2080_CTRL_GPU_INITIALIZE_CTX_APERTURE_VIDMEM and _GPU_CACHEABLE_NO,
   0x4), and is unmapped unless it also carries a virtual address. *)
let expected (p : Rpc.promotion) =
  let init, pa, size, attr =
    match p.phys with
    | Some (pa, size) -> (1, pa, size, 0x4)
    | None -> (0, 0, 0, 0)
  in
  let unmapped = if init = 1 && p.virt = None then 1 else 0 in
  [ p.buffer; init; unmapped; pa; size; attr; Option.value p.virt ~default:0 ]

let promotions =
  (* Pages up to the top of the GPU's 49-bit addresses, the last among them. *)
  let page = Gen.int_range 1 ((1 lsl 37) - 1) in
  let addr =
    Gen.map
      (fun p -> p * 0x1000)
      (Gen.one_of [ page; Gen.constant ((1 lsl 37) - 1) ])
  in
  let at =
    Gen.one_of
      [
        Gen.map (fun pa -> (Some pa, None)) (Gen.pair addr addr);
        Gen.map (fun va -> (None, Some va)) addr;
        Gen.map
          (fun (pa, va) -> (Some pa, Some va))
          (Gen.pair (Gen.pair addr addr) addr);
      ]
  in
  let promotion =
    Gen.map
      (fun (buffer, (phys, virt)) -> { Rpc.buffer; phys; virt })
      (Gen.pair (Gen.one_of [ Gen.int_range 0 12; Gen.constant 0xffff ]) at)
  in
  let size =
    Gen.one_of [ Gen.int_range 0 16; Gen.constant 0; Gen.constant 16 ]
  in
  Gen.list ~size promotion

let test_promote_ctx =
  prop "a promotion carries each buffer's addresses" promotions (fun l ->
      let has f = List.exists f l in
      cover "initialized and mapped"
        (has (fun (p : Rpc.promotion) -> p.phys <> None && p.virt <> None));
      cover "initialized, unmapped"
        (has (fun (p : Rpc.promotion) -> p.virt = None));
      cover "mapped only" (has (fun (p : Rpc.promotion) -> p.phys = None));
      cover "no buffer" (l = []);
      cover "16 buffers" (List.length l = 16);
      cover "the last id" (has (fun (p : Rpc.promotion) -> p.buffer = 0xffff));
      let b = Rpc.promote_ctx ~client:0xc100_0000 ~channel:0xcf00_0007 l in
      equal ~msg:"size" int 560 (String.length b);
      equal ~msg:"graphics engine, client, channel" (list int)
        [ 1; 0xc100_0000; 0xcf00_0007 ]
        [ u32 b 0; u32 b 0xc; u32 b 0x10 ];
      equal (list (list int)) (List.map expected l) (promoted b))

let test_promote_refused () =
  let p = { Rpc.buffer = 0; phys = None; virt = Some 0x1000 } in
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rpc.promote_ctx ~client:1 ~channel:2 (List.init 17 (fun _ -> p))));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rpc.promote_ctx ~client:1 ~channel:2 [ { p with virt = None } ]))

(* A compute object in an address space another owner keeps (UVM's, rig's), as
   NVIDIA's RM promotes its context: the object's constructor initializes the
   main context (0), PM (1) and patch (2) buffers before the object's RPC
   (kgrobjPromoteContext, kgrobjGetPromoteIds_FWCLIENT; alloc_free.c:869), the
   PM buffer allocated from the start in such a space
   (kgrctxShouldPreAllocPmBuffer_PF), and promotes no virtual address there
   (kgrctxPreparePromoteCtxBuffer). The space's owner then binds the address
   of each of the context's buffers, the main, the patch, the PM and every
   global buffer, in one call (nvGpuOpsBindChannelResources,
   kgrctxGetCtxBuffers). *)
let test_compute_context () =
  let main = { Rpc.pa = 0x10_0000; size = 0x6_0000; va = 0x10_0000_0000 }
  and pm = { Rpc.pa = 0x30_0000; size = 0x2_0000; va = 0x10_0050_0000 }
  and patch = { Rpc.pa = 0x20_0000; size = 0x1_0000; va = 0x10_0010_0000 } in
  let globals =
    [ (3, 0x10_0020_0000); (9, 0x10_0030_0000); (11, 0x10_0040_0000) ]
  in
  let first, second = Rpc.compute_context ~main ~pm ~patch ~globals in
  equal ~msg:"initialized before the object" (list (list int))
    [
      [ 0; 1; 1; 0x10_0000; 0x6_0000; 4; 0 ];
      [ 1; 1; 1; 0x30_0000; 0x2_0000; 4; 0 ];
      [ 2; 1; 1; 0x20_0000; 0x1_0000; 4; 0 ];
    ]
    (List.map expected first);
  equal ~msg:"bound after it" (list (list int))
    [
      [ 0; 0; 0; 0; 0; 0; 0x10_0000_0000 ];
      [ 2; 0; 0; 0; 0; 0; 0x10_0010_0000 ];
      [ 1; 0; 0; 0; 0; 0; 0x10_0050_0000 ];
      [ 3; 0; 0; 0; 0; 0; 0x10_0020_0000 ];
      [ 9; 0; 0; 0; 0; 0; 0x10_0030_0000 ];
      [ 11; 0; 0; 0; 0; 0; 0x10_0040_0000 ];
    ]
    (List.map expected second)

(* rpc_unloading_guest_driver_v1F_07: bInPMTransition, bGc6Entering, then
   newLevel at 4. *)
let test_unloading () =
  equal int 8 (String.length Rpc.unloading);
  equal (list int) [ 0; 0; 0x40 ]
    [ u8 Rpc.unloading 0; u8 Rpc.unloading 1; u32 Rpc.unloading 4 ]

(* PACKED_REGISTRY_TABLE: size and numEntries, then PACKED_REGISTRY_ENTRY of 16
   bytes each (nameOffset, type at 4, data at 8, length at 12), then the names,
   each ending in a zero byte; a DWORD entry's type is 1. *)
let test_registry =
  let key = Gen.string_of ~size:(Gen.int_range 1 20) (Gen.char_range 'A' 'z') in
  prop "the registry holds each key, its value and its name"
    Gen.(list ~size:(int_range 0 6) (pair key (int_range 0 0xffff_ffff)))
    (fun keys ->
      let t = Rpc.registry keys in
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

(* Channels *)

(* NVOS04_FLAGS (alloc_channel.h): the index of a channel's USERD within its
   page of 8 in bits 10:8, the index fixed in bit 11, the page in bits 20:12,
   the page fixed in bit 21. NVIDIA's RM gives the GSP the hardware id it chose
   with the page fixed and the index not, id / 8 and id mod 8
   (_kchannelSendChannelAllocRpc, kernel_channel.c). *)
let userd_bits = 0x3f_ff00
let ids = Gen.one_of [ Gen.int_range 0 4095; Gen.of_list [ 0; 7; 8; 4095 ] ]
let flag_words = Gen.int_range 0 0xffff_ffff

let test_channel_flags =
  prop "a channel's flags fix its id, the other flags kept"
    (Gen.pair flag_words ids) (fun (flags, id) ->
      cover "the first id" (id = 0);
      cover "the last of a page" (id = 7);
      cover "a second page" (id = 8);
      cover "the last id" (id = 4095);
      let f = Rpc.channel_flags flags id in
      equal ~msg:"the other flags" int
        (flags land lnot userd_bits)
        (f land lnot userd_bits);
      equal ~msg:"index, its page, the page fixed and the index not" (list int)
        [ id mod 8; id / 8; 1; 0 ]
        [ (f lsr 8) land 7; (f lsr 12) land 0x1ff; (f lsr 21) land 1;
          (f lsr 11) land 1 ];
      equal (option int) (Some id) (Rpc.channel_id f))

let test_channel_id =
  cases "flags that fix no page name no id"
    ~name:(fun (n, _, _) -> n)
    [
      ("no flags", 0, None);
      ("an index without its page", 0x0b00, None);
      ("the golden channel's", 0x20_0320, Some 3);
      ("a later page's", 0x20_1520, Some 13);
    ]
    (fun (_, flags, id) -> equal (option int) id (Rpc.channel_id flags))

let test_channel_range () =
  equal ~msg:"512 pages of 8" int 4096 Rpc.channel_ids;
  raises_match Exn.invalid_arg (fun () -> ignore (Rpc.channel_flags 0 (-1)));
  raises_match Exn.invalid_arg (fun () -> ignore (Rpc.channel_flags 0 4096))

(* Beside a GSP, NVIDIA's RM allocates a channel that is not the GSP's own from
   id 1 up, the lowest free: id 0 is kept for the GSP's
   (KFIFO_NUM_GSP_RESERVED_CHANNELS and kfifoChidMgrAllocChid, kernel_fifo.c,
   with eheap's growing-up allocation). *)
let test_free_channel_id =
  let ids = Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range 0 10) in
  let zeros = Gen.list ~size:(Gen.int_range 1 3) (Gen.constant 0) in
  let held = Gen.one_of [ Gen.constant []; zeros; ids ] in
  prop "a channel takes the lowest free id from 1" held (fun held ->
      cover "none held" (held = []);
      cover "id 1 held" (List.mem 1 held);
      cover "only id 0 held" (held <> [] && List.for_all (( = ) 0) held);
      let rec lowest id = if List.mem id held then lowest (id + 1) else id in
      equal (option int) (Some (lowest 1))
        (Rpc.free_channel_id (fun id -> List.mem id held)))

let test_no_channel_id () =
  equal (option int) None (Rpc.free_channel_id (fun id -> id < 4096));
  equal (option int) (Some 4095) (Rpc.free_channel_id (fun id -> id < 4095))

(* NV_CTRL_VF_DOORBELL (ga100 dev_ctrl.h), which NVIDIA's RM builds the token
   in on Ampere and Ada: the id in bits 11:0, the runlist in bits 22:16;
   NV_VIRTUAL_FUNCTION_DOORBELL (gb202 dev_vm.h) on Blackwell adds
   RUNLIST_DOORBELL, bit 30, enabled (kfifoGenerateWorkSubmitTokenHal_GA100,
   _GB202). *)
let test_token =
  cases "a work submit token is its channel's runlist and id"
    ~name:(fun (n, _, _, _, _) -> n)
    [
      ("Ada's", false, 2, 5, 0x2_0005);
      ("the last runlist and id", false, 127, 4095, 0x7f_0fff);
      ("the first", false, 0, 0, 0);
      ("Blackwell's", true, 2, 5, 0x4002_0005);
    ]
    (fun (_, runlist_doorbell, runlist, id, token) ->
      equal int token (Rpc.work_submit_token ~runlist_doorbell ~runlist id))

let test_token_range () =
  let token runlist id =
    ignore (Rpc.work_submit_token ~runlist_doorbell:false ~runlist id)
  in
  raises_match Exn.invalid_arg (fun () -> token 128 0);
  raises_match Exn.invalid_arg (fun () -> token 0 4096);
  raises_match Exn.invalid_arg (fun () -> token (-1) 0)

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

let rec pp_op ppf (o : Falcon.op) =
  let cond ppf : Falcon.cond -> unit = function
    | Is x -> Format.fprintf ppf "is 0x%x" x
    | Not x -> Format.fprintf ppf "not 0x%x" x
    | Differs r -> Format.fprintf ppf "differs from 0x%x" r
  in
  match o with
  | Write (r, x) -> Format.fprintf ppf "write 0x%x 0x%x" r x
  | Modify (r, m, x) -> Format.fprintf ppf "modify 0x%x 0x%x 0x%x" r m x
  | Copy (r, r') -> Format.fprintf ppf "copy 0x%x 0x%x" r r'
  | Poll (what, r, m, c) ->
      Format.fprintf ppf "poll %S 0x%x 0x%x %a" what r m cond c
  | Delay n -> Format.fprintf ppf "delay %d" n
  | When (r, m, c, ops) ->
      Format.fprintf ppf "when 0x%x 0x%x %a [%a]" r m cond c
        (Format.pp_print_list pp_op)
        ops
  | Expect (what, r, m, c) ->
      Format.fprintf ppf "expect %S 0x%x 0x%x %a" what r m cond c

let op = Testable.make ~pp:pp_op ~equal:( = )
let libos = 0x12_3456_7000
let sequence words = require_ok (Rpc.sequence ~libos (sequencer words))

(* The core's commands are the falcons' own sequences for the GSP's falcon: the
   falcon reset, then FBIF_CTL's ALLOW_PHYS_NO_CTX (bit 7 of 0x110624) set and
   DMACTL (0x11010c) cleared. *)
let core_reset =
  Falcon.reset Falcon.gsp `Falcon
  @ Falcon.[ Modify (0x110624, 0x80, 0x80); Write (0x11010c, 0) ]

let test_sequence =
  cases "each opcode is its register accesses, its payload read in order"
    ~name:(fun (n, _, _) -> n)
    [
      ("a write", [ 0; 0x1000; 0x5 ], [ Falcon.Write (0x1000, 0x5) ]);
      ( "a modify sets the value's bits outside its mask",
        [ 1; 0x2000; 0xf0; 0x13 ],
        [ Modify (0x2000, 0xf3, 0x13) ] );
      ( "a poll, its timeout and error left",
        [ 2; 0x3000; 0xff; 0x10; 1000; 0 ],
        [ Poll ("register 0x3000", 0x3000, 0xff, Is 0x10) ] );
      ("a delay", [ 3; 20 ], [ Delay 20 ]);
      ("a store", [ 4; 0x4000; 2 ], []);
      ("a core reset", [ 5 ], core_reset);
      ("a core start", [ 6 ], Falcon.start Falcon.gsp);
      ("a wait for the core to halt", [ 7 ], Falcon.wait_halt Falcon.gsp);
      ( "one after another",
        [ 0; 0x1000; 0x5; 3; 20; 6 ],
        Falcon.[ Write (0x1000, 0x5); Delay 20 ] @ Falcon.start Falcon.gsp );
      ("nothing", [], []);
    ]
    (fun (_, words, ops) -> equal (list op) ops (sequence words))

(* A resumption restarts the GSP's falcon as a RISC-V core with the libos
   arguments in MAILBOX0 and MAILBOX1 (0x110040 and 0x110044), low word first,
   then starts SEC2 (at 0x840000) and waits for it to hand the GSP over. *)
let test_resume () =
  let ops = sequence [ 8 ] in
  let reset = Falcon.reset Falcon.gsp `Riscv in
  let n = List.length reset in
  equal (list op) ~msg:"the reset" reset (List.filteri (fun i _ -> i < n) ops);
  equal (list op) ~msg:"the libos arguments"
    Falcon.[ Write (0x110040, 0x3456_7000); Write (0x110044, 0x12) ]
    (List.filteri (fun i _ -> i >= n && i < n + 2) ops);
  let start = Falcon.start Falcon.sec2 in
  equal (list op) ~msg:"SEC2 started" start
    (List.filteri (fun i _ -> i >= n + 2 && i < n + 2 + List.length start) ops);
  equal int ~msg:"then a poll and a check"
    (n + 2 + List.length start + 2)
    (List.length ops)

let test_sequence_refused =
  cases "a sequence it cannot run is refused"
    ~name:(fun (n, _, _) -> n)
    [
      ("an unknown opcode", sequencer [ 9 ], "unknown opcode 9");
      ("a command cut short", sequencer [ 0; 0x1000 ], "ending inside");
      ("a header cut short", String.make 0x20 '\000', "shorter");
      ( "more words than the message holds",
        String.sub (sequencer [ 3; 20 ]) 0 0x2c,
        "longer" );
    ]
    (fun (_, body, why) ->
      contains ~sub:why (require_error (Rpc.sequence ~libos body)))

let () =
  exit
  @@ run "rig_nv_pci.rpc"
       [
         group ~timeout:20. "records"
           [
             test_records;
             test_message;
             test "a GSP message is checked" test_not_a_message;
           ];
         group ~timeout:10. "faults"
           [
             test "a stopped channel is a fault naming it" test_rc;
             test "a stopped channel of an unknown error is a fault"
               test_rc_unknown;
             test "a queued MMU fault is a fault" test_mmu;
             test_no_fault;
           ];
         group ~timeout:10. "reports"
           [ test_reports; test_short_reports; test_no_report ];
         group ~timeout:10. "calls"
           [
             test "an allocation" test_alloc;
             test "a control" test_control;
             test "a free" test_free;
             test_answer;
             test "a short answer is refused" test_short_answer;
             test "a page directory" test_page_directory;
             test_reserved_pdes;
             test "tables of another depth raise" test_reserved_depth;
             test_promote_ctx;
             test "a promotion of 17 buffers or of an addressless one raises"
               test_promote_refused;
             test "a compute context is initialized, then bound"
               test_compute_context;
             test "unloading" test_unloading;
             test_registry;
           ];
         group ~timeout:10. "channels"
           [
             test_channel_flags;
             test_channel_id;
             test "an id the flags cannot hold raises" test_channel_range;
             test_free_channel_id;
             test "every id held leaves none" test_no_channel_id;
             test_token;
             test "a runlist or id the token cannot hold raises"
               test_token_range;
           ];
         group ~timeout:10. "sequences"
           [
             test_sequence;
             test "a resumption restarts the GSP with its arguments" test_resume;
             test_sequence_refused;
           ];
       ]
