// Two kernels for gfx1030 whose descriptors differ in every field a code
// object reads, three bytes of data that leave the linked image's end off a
// 32-bit word, and an undefined symbol.

  .amdgcn_target "amdgcn-amd-amdhsa--gfx1030"

  .text
  .globl a
  .p2align 8
  .type a,@function
a:
  s_endpgm

  .globl b
  .p2align 8
  .type b,@function
b:
  s_nop 0
  s_endpgm

  .data
  .byte 1, 2, 3

  .rodata
  .p2align 6
  .amdhsa_kernel a
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_private_segment_fixed_size 64
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel

  .p2align 6
  .amdhsa_kernel b
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_dispatch_ptr 1
    .amdhsa_wavefront_size32 0
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 16
  .end_amdhsa_kernel

  .globl ext
