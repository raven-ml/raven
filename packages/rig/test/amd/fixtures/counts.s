// Three kernels for gfx1201, in waves of 32 work-items. valu runs 16 * k
// v_add_f32 a wave, k its first argument, a 32-bit word of at least 1, and no
// other VALU instruction: SQ_INSTS_VALU counts 16 * k a wave. Its four chains
// of additions are independent, so that waves issue one a cycle. hw_id2 stores
// its wave's HW_ID2 register, whose bits 27 to 24 are the wave's VMID, at the
// address its first argument holds. filter stores 1 | 1 << (vmid + 1), vmid its
// wave's VMID, there.

  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
  .amdhsa_code_object_version 5

  .text
  .globl valu
  .p2align 8
  .type valu,@function
valu:
  s_load_b32 s2, s[0:1], 0x0
  s_wait_kmcnt 0x0
.Lloop:
  .rept 4
  v_add_f32 v0, 1.0, v0
  v_add_f32 v1, 1.0, v1
  v_add_f32 v2, 1.0, v2
  v_add_f32 v3, 1.0, v3
  .endr
  s_sub_co_u32 s2, s2, 1
  s_cmp_lg_u32 s2, 0
  s_cbranch_scc1 .Lloop
  s_endpgm

  .globl hw_id2
  .p2align 8
  .type hw_id2,@function
hw_id2:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_getreg_b32 s4, hwreg(HW_REG_WAVE_HW_ID2)
  s_wait_kmcnt 0x0
  v_mov_b32 v0, 0
  v_mov_b32 v1, s4
  global_store_b32 v0, v1, s[2:3]
  s_endpgm

  .globl filter
  .p2align 8
  .type filter,@function
filter:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_getreg_b32 s4, hwreg(HW_REG_WAVE_HW_ID2)
  s_bfe_u32 s4, s4, 0x40018
  s_add_co_u32 s4, s4, 1
  s_lshl_b32 s4, 1, s4
  s_or_b32 s4, s4, 1
  s_wait_kmcnt 0x0
  v_mov_b32 v0, 0
  v_mov_b32 v1, s4
  global_store_b32 v0, v1, s[2:3]
  s_endpgm

  .rodata
  .p2align 6
  .amdhsa_kernel valu
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel hw_id2
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel filter
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.kernels:
  - .name: valu
    .symbol: valu.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0, .size: 4, .value_kind: by_value }
  - .name: hw_id2
    .symbol: hw_id2.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 2
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
  - .name: filter
    .symbol: filter.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 2
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
  .end_amdgpu_metadata
