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
open Tolk_nv.Nv_tables.Versions

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
    ]
