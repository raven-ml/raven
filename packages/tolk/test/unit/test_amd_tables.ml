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
  run "Amd_tables"
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
          test "13.0.8 resolves to 13.0.6 and lacks a gfx clock" (fun () ->
            let module S = (val Tables.smu ~version:(13, 0, 8)) in
            equal int 13 S.ppsmc_msg_setdriverdramaddrhigh;
            equal (option int) None S.ppclk_gfxclk;
            equal (option int) (Some 3) S.ppsmc_msg_gfxdriverreset);
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
          test "firmware digests are indexed by file name" (fun () ->
            equal string
              "801a09c9bf06188260db9b51ad8f978f15d84c72ca91b90643a2ef8af4074776"
              (List.assoc "gc_11_0_0_mec.bin" Tables.Fw_defs.hashes));
        ];
    ]
