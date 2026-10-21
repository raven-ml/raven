//===--- AMDHSAKernelDescriptor.h -----------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDHSA kernel descriptor definitions. For more information, visit
/// https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor
///
/// \warning
/// Any changes to this file should also be audited for corresponding changes
/// needed in both the assembler and disassembler, namely:
/// * AMDGPUAsmPrinter.{cpp,h}
/// * AMDGPUTargetStreamer.{cpp,h}
/// * AMDGPUDisassembler.{cpp,h}
//
//===----------------------------------------------------------------------===//

/* Excerpt of https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h. */

  KERNEL_CODE_PROPERTY(ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0, 1),
  KERNEL_CODE_PROPERTY(ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  KERNEL_CODE_PROPERTY(ENABLE_WAVEFRONT_SIZE32, 10, 1), // GFX10+
struct kernel_descriptor_t {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3; // GFX10+ and GFX90A+
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload;
  uint8_t reserved3[4];
};
  GROUP_SEGMENT_FIXED_SIZE_OFFSET = 0,
  PRIVATE_SEGMENT_FIXED_SIZE_OFFSET = 4,
  KERNARG_SIZE_OFFSET = 8,
  KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET = 16,
  COMPUTE_PGM_RSRC3_OFFSET = 44,
  COMPUTE_PGM_RSRC1_OFFSET = 48,
  COMPUTE_PGM_RSRC2_OFFSET = 52,
  KERNEL_CODE_PROPERTIES_OFFSET = 56,
