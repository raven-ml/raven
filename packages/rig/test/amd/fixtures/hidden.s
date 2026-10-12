// Three kernels for gfx1201 and their metadata: every names each implicit
// argument code object version 5 defines, at the offsets of AMDGPUUsage's
// table, among 288 bytes of arguments, and bounds its workgroups at 128
// work-items; plain names an argument and nothing else; wide stores its
// second argument, a 32-bit word, at the address its first holds, among 8192
// bytes of arguments.

  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .amdhsa_code_object_version 5

  .text
  .globl every
  .p2align 8
  .type every,@function
every:
  s_endpgm

  .globl plain
  .p2align 8
  .type plain,@function
plain:
  s_endpgm

  .globl wide
  .p2align 8
  .type wide,@function
wide:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_load_b32 s4, s[0:1], 0x8
  s_wait_kmcnt 0x0
  v_mov_b32 v0, 0
  v_mov_b32 v1, s4
  global_store_b32 v0, v1, s[2:3]
  s_endpgm

  .rodata
  .p2align 6
  .amdhsa_kernel every
    .amdhsa_kernarg_size 288
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel plain
    .amdhsa_kernarg_size 8
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel wide
    .amdhsa_kernarg_size 8192
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.kernels:
  - .name: every
    .symbol: every.kd
    .kernarg_segment_size: 288
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 1
    .max_flat_workgroup_size: 128
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .offset: 32, .size: 4, .value_kind: hidden_block_count_x }
      - { .offset: 36, .size: 4, .value_kind: hidden_block_count_y }
      - { .offset: 40, .size: 4, .value_kind: hidden_block_count_z }
      - { .offset: 44, .size: 2, .value_kind: hidden_group_size_x }
      - { .offset: 46, .size: 2, .value_kind: hidden_group_size_y }
      - { .offset: 48, .size: 2, .value_kind: hidden_group_size_z }
      - { .offset: 50, .size: 2, .value_kind: hidden_remainder_x }
      - { .offset: 52, .size: 2, .value_kind: hidden_remainder_y }
      - { .offset: 54, .size: 2, .value_kind: hidden_remainder_z }
      - { .offset: 72, .size: 8, .value_kind: hidden_global_offset_x }
      - { .offset: 80, .size: 8, .value_kind: hidden_global_offset_y }
      - { .offset: 88, .size: 8, .value_kind: hidden_global_offset_z }
      - { .offset: 96, .size: 2, .value_kind: hidden_grid_dims }
      - { .offset: 104, .size: 8, .value_kind: hidden_printf_buffer }
      - { .offset: 112, .size: 8, .value_kind: hidden_none }
      - { .offset: 152, .size: 4, .value_kind: hidden_dynamic_lds_size }
  - .name: plain
    .symbol: plain.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 1
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
  - .name: wide
    .symbol: wide.kd
    .kernarg_segment_size: 8192
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 2
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .offset: 8, .size: 4, .value_kind: by_value }
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
  .end_amdgpu_metadata
