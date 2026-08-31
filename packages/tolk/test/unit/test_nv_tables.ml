(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Load-bearing spot checks of the generated NVIDIA driver tables: known
   offsets and sizes per structure family, driver-version dispatch, QMD
   field lookups, the escape request-code math, and blob field round-trips.
   The broad net is the generator's own probe verification against the
   reference layouts. *)

open Windtrap
module Tables = Tolk_nv.Nv_tables
module Defs = Tolk_nv.Nv_tables.Defs
module Regs = Tolk_nv.Nv_tables.Reg_defs
module Gsp = Tolk_nv.Nv_tables.Gsp_defs
open Tolk_nv.Nv_tables.Versions

let reg family arch name =
  let arches = List.assoc family Regs.families in
  List.assoc name (List.assoc arch arches)

let () =
  run "Nv_tables"
    [
      group "constants"
        [
          test "class ids, escape numbers, control-page offset" (fun () ->
              (* Values from the reference driver headers. *)
              equal int 0x2b Defs.nv_esc_rm_alloc;
              equal int 0xc8 Defs.nv_esc_card_info;
              equal int 0xc9c0 Defs.ada_compute_a;
              equal int 0xcec0 Defs.blackwell_compute_b;
              equal int 0x8c Defs.ampere_a_control_gpfifo_gpput);
          test "method-argument bit ranges" (fun () ->
              equal (pair int int) (2, 0) Defs.nvc56f_sem_execute_operation;
              equal (pair int int) (24, 24)
                Defs.nvc56f_sem_execute_payload_size;
              equal (pair int int) (4, 3)
                Defs.nvc6b5_launch_dma_semaphore_type);
          test "mmu fault name tables" (fun () ->
              equal string "NV_PFAULT_FAULT_TYPE_PDE"
                (List.assoc 0 Defs.nv_pfault_fault_type);
              equal string "ATOMIC" (List.assoc 0xa Defs.nv_pfault_access_type));
        ];
      group "struct layouts"
        [
          test "rm alloc parameters" (fun () ->
              (* NVOS21_PARAMETERS *)
              equal int 0x20 Defs.Nvos21_parameters.sizeof;
              equal (pair int int) (8, 4) Defs.Nvos21_parameters.hobjectnew;
              equal (pair int int) (0x10, 8) Defs.Nvos21_parameters.pallocparms);
          test "memory allocation parameters" (fun () ->
              (* NV_MEMORY_ALLOCATION_PARAMS *)
              equal int 0x80 Defs.Nv_memory_allocation_params.sizeof;
              equal (pair int int) (0x18, 4)
                Defs.Nv_memory_allocation_params.attr;
              equal (pair int int) (0x40, 8)
                Defs.Nv_memory_allocation_params.size);
          test "uvm map external allocation" (fun () ->
              (* UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, UvmGpuMappingAttributes *)
              equal int 0x2430 Defs.Uvm_map_external_allocation_params.sizeof;
              equal int 0x18
                Defs.Uvm_map_external_allocation_params.pergpuattributes_offset;
              equal int 0x24
                Defs.Uvm_map_external_allocation_params
                .pergpuattributes_elem_size;
              equal (pair int int) (0x2418, 8)
                Defs.Uvm_map_external_allocation_params.gpuattributescount;
              equal (pair int int) (0x10, 4)
                Defs.Uvm_gpu_mapping_attributes.gpumappingtype);
          test "sm error states" (fun () ->
              (* NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS *)
              equal (pair int int) (0x12cc, 1)
                Defs.Nv83de_ctrl_debug_read_all_sm_error_states_params
                .mmufault_valid;
              equal int 0x30
                Defs.Nv83de_ctrl_debug_read_all_sm_error_states_params
                .smerrorstatearray_elem_size;
              equal (pair int int) (0x20, 8)
                Defs.Nv83de_sm_error_state_registers.hwwwarpesrpc64);
        ];
      group "version dispatch"
        [
          test "thresholds select the generation" (fun () ->
              (* NVOS46_PARAMETERS grew from 0x38 to 0x40 bytes at 580. *)
              equal int 0x38
                (Tables.defs_for_driver ~major:570).nvos46_parameters.sizeof;
              equal int 0x38
                (Tables.defs_for_driver ~major:579).nvos46_parameters.sizeof;
              equal int 0x40
                (Tables.defs_for_driver ~major:580).nvos46_parameters.sizeof;
              equal int 0x40
                (Tables.defs_for_driver ~major:609).nvos46_parameters.sizeof;
              equal (pair int int) (0x28, 8)
                (Tables.defs_for_driver ~major:570).nvos46_parameters.dmaoffset;
              equal (pair int int) (0x30, 8)
                (Tables.defs_for_driver ~major:610).nvos46_parameters.dmaoffset);
          test "610 drops the uvm free length" (fun () ->
              equal
                (option (pair int int))
                (Some (8, 8))
                (Tables.defs_for_driver ~major:570).uvm_free_params.length;
              equal
                (option (pair int int))
                None
                (Tables.defs_for_driver ~major:700).uvm_free_params.length;
              equal int 0x10
                (Tables.defs_for_driver ~major:610).uvm_free_params.sizeof);
          test "610 shifts the gpfifo allocation fields" (fun () ->
              equal (pair int int) (0x80, 4)
                (Tables.defs_for_driver ~major:570)
                  .nv_channelgpfifo_allocation_parameters
                  .enginetype;
              equal (pair int int) (0x88, 4)
                (Tables.defs_for_driver ~major:610)
                  .nv_channelgpfifo_allocation_parameters
                  .enginetype;
              equal (pair int int) (0x24, 4)
                (Tables.defs_for_driver ~major:610)
                  .nv_channelgpfifo_allocation_parameters
                  .huserdmemory);
          test "status codes are per generation" (fun () ->
              equal string "NV_ERR_NO_MEMORY"
                (List.assoc Defs.nv_err_no_memory
                   (Tables.defs_for_driver ~major:570).nv_status_codes);
              equal string "NV_ERR_NO_MEMORY"
                (List.assoc Defs.nv_err_no_memory
                   (Tables.defs_for_driver ~major:610).nv_status_codes));
        ];
      group "qmd fields"
        [
          test "v3 lookups" (fun () ->
              let v3 = Defs.nvc6c0_qmdv03_00_fields in
              equal (pair int int) (415, 384) (List.assoc "CTA_RASTER_WIDTH" v3);
              equal (pair int int) (799, 768)
                (List.assoc "RELEASE0_ADDRESS_LOWER" v3);
              (* Slot expansion: CONSTANT_BUFFER_ADDR_UPPER at slot 3. *)
              equal (pair int int) (1264, 1248)
                (List.assoc "CONSTANT_BUFFER_ADDR_UPPER_3" v3);
              equal int 181 (List.length v3));
          test "v5 lookups" (fun () ->
              let v5 = Defs.nvcec0_qmdv05_00_fields in
              equal (pair int int) (1279, 1248) (List.assoc "GRID_WIDTH" v5);
              equal (pair int int) (511, 480)
                (List.assoc "RELEASE_SEMAPHORE0_ADDR_LOWER" v5);
              equal (pair int int) (1375, 1344)
                (List.assoc "CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6_0" v5);
              equal int 296 (List.length v5));
        ];
      group "blobs"
        [
          test "escape request code" (fun () ->
              (* _IOWR('F', NV_ESC_RM_ALLOC, NVOS21_PARAMETERS) *)
              equal int 0xc020462b
                (Tables.escape_code ~nr:Defs.nv_esc_rm_alloc
                   ~size:Defs.Nvos21_parameters.sizeof));
          test "field round-trips" (fun () ->
              let b = Tables.create_blob 0x20 in
              Tables.set_field b (0, 8) 0x0123_4567_89ab_cdef;
              equal int 0x0123_4567_89ab_cdef (Tables.get_field b (0, 8));
              (* Little-endian: narrower reads see the low bytes. *)
              equal int 0xef (Tables.get_field b (0, 1));
              equal int 0xcdef (Tables.get_field b (0, 2));
              equal int 0x89ab_cdef (Tables.get_field b (0, 4));
              (* Truncation to the field width; -1 is an all-ones field. *)
              Tables.set_field b (8, 4) (-1);
              equal int 0xffff_ffff (Tables.get_field b (8, 4));
              equal int 0 (Tables.get_field b (0xc, 1));
              Tables.set_field b (0x10, 1) 0x1ff;
              equal int 0xff (Tables.get_field b (0x10, 1));
              (* A base addresses embedded structures and array elements. *)
              Tables.set_field ~base:4 b (0x10, 2) 0x1234;
              equal int 0x1234 (Tables.get_field b (0x14, 2)));
        ];
      group "register defs"
        [
          test "families cover the reference set" (fun () ->
              equal int 14 (List.length Regs.families);
              (* dev_falcon_v4 carries both the ampere and hopper layouts. *)
              equal
                (list string)
                [ "ga102"; "gh100" ]
                (List.map fst (List.assoc "dev_falcon_v4" Regs.families)));
          test "fixed register with a nonzero block base" (fun () ->
              (* NV_PMC_BOOT_0 lives at block base 0, offset 0. *)
              (match reg "nv_ref" "regs" "NV_PMC_BOOT_0" with
              | Regs.Reg { base = 0; off = Regs.Fixed 0; fields } ->
                  equal (pair int int) (24, 28)
                    (List.assoc "architecture_0" fields)
              | _ -> fail "NV_PMC_BOOT_0 is not a fixed register");
              (* NV_VIRTUAL_FUNCTION regs sit at block base 0xb80000. *)
              match
                reg "dev_vm" "tu102" "NV_VIRTUAL_FUNCTION_PRIV_L2_SYSMEM_INVALIDATE"
              with
              | Regs.Reg { base = 0xb80000; off = Regs.Fixed 0xf00; fields = [] }
                ->
                  ()
              | _ -> fail "unexpected NV_VIRTUAL_FUNCTION register");
          test "indexed register recovers base and stride" (fun () ->
              match reg "dev_falcon_v4" "ga102" "NV_PFALCON_FALCON_IMEMC" with
              | Regs.Reg
                  { base = 0; off = Regs.Indexed { base = 0x180; stride = 0x10 };
                    fields } ->
                  equal (pair int int) (2, 7) (List.assoc "offs" fields);
                  equal (pair int int) (28, 28) (List.assoc "secure" fields)
              | _ -> fail "NV_PFALCON_FALCON_IMEMC is not indexed");
          test "bitfield-only group and plain constant" (fun () ->
              (match reg "dev_mmu" "gh100" "NV_MMU_VER2_PTE" with
              | Regs.Group { fields } ->
                  equal (pair int int) (0, 0) (List.assoc "valid" fields);
                  equal (pair int int) (56, 63) (List.assoc "kind" fields)
              | _ -> fail "NV_MMU_VER2_PTE is not a group");
              match reg "nv_ref" "regs" "NV_PMC_BOOT_42_ARCHITECTURE_GA100" with
              | Regs.Const 0x17 -> ()
              | _ -> fail "unexpected architecture constant");
        ];
      group "gsp defs"
        [
          test "struct field offsets and sizes" (fun () ->
              equal int 0x100 Gsp.Gsp_fw_wpr_meta.sizeof;
              equal (pair int int) (0, 8) Gsp.Gsp_fw_wpr_meta.magic;
              equal (pair int int) (0x98, 8) Gsp.Gsp_fw_wpr_meta.frtsoffset;
              equal int 0x20 Gsp.Msgq_tx_header.sizeof;
              equal (pair int int) (0x1c, 4) Gsp.Msgq_tx_header.entryoff;
              (* A keyword field is renamed; a bitfield lowers to a byte pair. *)
              equal (pair int int) (0xc, 4) Gsp.Rpc_message_header.func;
              equal (pair int int) (0x30, 1) Gsp.Rpc_alloc_memory.ptedesc_idr;
              equal (pair int int) (0x32, 2)
                Gsp.Rpc_alloc_memory.ptedesc_length);
          test "rpc id and 64-bit magic" (fun () ->
              equal string "NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC"
                (List.assoc 0x67 Gsp.rpc_fns);
              equal string "NV_VGPU_MSG_EVENT_GSP_INIT_DONE"
                (List.assoc 0x1001 Gsp.rpc_events);
              equal int64 0xdc3aae21371a60b3L Gsp.gsp_fw_wpr_meta_magic);
          test "firmware digests by driver dir" (fun () ->
              let gsp = List.assoc "gsp-570.144.bin" Gsp.firmware in
              equal int 1 (List.length gsp);
              equal string
                "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"
                (List.assoc "ga102" gsp);
              let booter = List.assoc "booter_load-570.144.bin" Gsp.firmware in
              equal (list string) [ "ga102"; "ad102" ] (List.map fst booter);
              equal int 64
                (String.length (List.assoc "ga102" booter)));
        ];
    ]
