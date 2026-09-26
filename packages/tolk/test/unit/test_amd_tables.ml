(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Load-bearing spot checks of the generated AMD hardware tables: 64-bit
   page-table flags, register-family version resolution, SMU interface
   selection, and the byte-accessor struct layouts. The broad net is the
   generator's own probe verification against the reference layouts. *)

open Windtrap
module Tables = Tolk_amd.Amd_tables
module Am = Tolk_amd.Amd_tables.Am_defs

let () =
  exit (run "Amd_tables"
    [
      group "page-table flags"
        [
          test "bit-63 flags survive as int64" (fun () ->
            equal int64 0x8000000000000000L Am.amdgpu_pte_is_pte;
            equal int64 0x8000000000000001L
              (Int64.logor Am.amdgpu_pde_pte_gfx12 Am.amdgpu_pte_valid));
          test "mtype encoders shift into the high word" (fun () ->
            equal int64 0xc0000000000000L (Am.amdgpu_pte_mtype_gfx12 0L 3);
            (* Encoding mtype 0 clears a previously set mtype field. *)
            equal int64 0L (Am.amdgpu_pte_mtype_nv10 0x7000000000000L 0);
            equal int64 0x100000000000280L
              (Am.amdgpu_pte_mtype_vg10
                 (Int64.logor Am.amdgpu_pte_tf (Am.amdgpu_pte_frag 5))
                 0));
        ];
      group "register families"
        [
          test "mmhub resolves its exact version" (fun () ->
            let ip =
              Tables.Ip.create ~name:"mmhub" ~version:(3, 0, 2)
                ~bases:[| 0x10000; 0x20000 |]
            in
            let r = Tables.Ip.reg ip "regMMVM_CONTEXT0_CNTL" in
            equal int 0x6c0 r.Tables.Reg.offset;
            equal int 0x106c0 r.Tables.Reg.addr;
            equal (pair int int) (1, 2)
              (List.assoc "page_table_depth"
                 (Array.to_list r.Tables.Reg.fields)));
          test "mp 13.0.6 falls back to the 13.0.0 family" (fun () ->
            let ip =
              Tables.Ip.create ~name:"mp" ~version:(13, 0, 6) ~bases:[| 0 |]
            in
            ignore (Tables.Ip.reg ip "regMP0_SMN_C2PMSG_81"));
          test "mp 11.0.0 carries the MP1 message registers" (fun () ->
            let ip =
              Tables.Ip.create ~name:"mp" ~version:(11, 0, 0) ~bases:[| 0 |]
            in
            let r = Tables.Ip.reg ip "mmMP1_SMN_C2PMSG_90" in
            equal int 0x29a r.Tables.Reg.offset);
        ];
      group "smu interface"
        [
          test "13.0.7 uses the 13.0.0 interface" (fun () ->
            let module S = (val Tables.smu ~version:(13, 0, 7)) in
            equal int 14 S.ppsmc_msg_setdriverdramaddrhigh);
          test "13.0.10 uses the consumer message and clock IDs" (fun () ->
            let module S = (val Tables.smu ~version:(13, 0, 10)) in
            equal int 14 S.ppsmc_msg_setdriverdramaddrhigh;
            equal int 6 S.ppsmc_msg_enableallsmufeatures;
            equal (option int) (Some 0x12) S.ppsmc_msg_transfertablesmu2dram;
            equal int 2 S.ppclk_uclk;
            equal (option int) (Some 0) S.ppclk_gfxclk);
          test "13.0.8 resolves to 13.0.6 and lacks a gfx clock" (fun () ->
            let module S = (val Tables.smu ~version:(13, 0, 8)) in
            equal int 13 S.ppsmc_msg_setdriverdramaddrhigh;
            equal (option int) None S.ppclk_gfxclk;
            equal (option int) (Some 3) S.ppsmc_msg_gfxdriverreset);
          test "13.0.15 resolves to the existing 13.0.12 interface" (fun () ->
            let module S = (val Tables.smu ~version:(13, 0, 15)) in
            equal (option int) (Some 3) S.ppsmc_msg_gfxdriverreset;
            equal (option int) (Some 9) S.ppsmc_msg_getmetricstable;
            equal (option int) None S.ppclk_gfxclk);
          test "14.0.3 resolves to 14.0.2" (fun () ->
            let module S = (val Tables.smu ~version:(14, 0, 3)) in
            equal (option int) None S.ppsmc_msg_mode1reset;
            equal int 0x32 S.ppsmc_msg_setpptlimit);
        ];
      group "struct layouts"
        [
          test "psp_fw_bin_desc reads at its position" (fun () ->
            let b = Bytes.make 0x20 '\x00' in
            Bytes.set_int32_le b 12 0xdeadbeefl;
            equal int 0xdeadbeef (Am.Psp_fw_bin_desc.offset_bytes b 4);
            equal int 0x10 Am.Psp_fw_bin_desc.sizeof);
          test "mqd setters write the register image area" (fun () ->
            let b = Bytes.make Am.V11_compute_mqd.sizeof '\x00' in
            Am.V11_compute_mqd.set_cp_hqd_pq_control b 0x12345678;
            (* cp_hqd_pq_control lives at dword 0x91 of the image. *)
            equal int 0x12345678
              (Int32.to_int (Bytes.get_int32_le b (0x91 * 4))));
          test "PSP13.0.15 uses the target firmware pin" (fun () ->
            equal string
              "3b28d53e75a88131155e3931378ac8434eca4880ada9211d3b4e8915b6289583"
              (List.assoc "psp_13_0_15_sos.bin" Tables.Fw_defs.hashes));
          test "firmware digests are indexed by file name" (fun () ->
            equal string
              "1dd1de8ecf5455ea4719c502b64b32ac18763d5601128c01b4a4a36211a122c2"
              (List.assoc "gc_11_0_0_mec.bin" Tables.Fw_defs.hashes));
        ];
    ])
