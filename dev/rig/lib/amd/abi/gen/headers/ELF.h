//===- llvm/BinaryFormat/ELF.h - ELF constants and structures ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header contains common, non-processor-specific data structures and
// constants for the ELF file format.
//
// The details of the ELF32 bits in this file are largely based on the Tool
// Interface Standard (TIS) Executable and Linking Format (ELF) Specification
// Version 1.2, May 1995. The ELF64 stuff is based on ELF-64 Object File Format
// Version 1.5, Draft 2, May 1998 as well as OpenBSD header files.
//
//===----------------------------------------------------------------------===//

/* Excerpt of https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/llvm/include/llvm/BinaryFormat/ELF.h. */

  EM_AMDGPU = 224,        // AMD GPU architecture
  ELFABIVERSION_AMDGPU_HSA_V6 = 4,
  EF_AMDGPU_MACH = 0x0ff,
  EF_AMDGPU_MACH_AMDGCN_GFX600          = 0x020,
  EF_AMDGPU_MACH_AMDGCN_GFX601          = 0x021,
  EF_AMDGPU_MACH_AMDGCN_GFX700          = 0x022,
  EF_AMDGPU_MACH_AMDGCN_GFX701          = 0x023,
  EF_AMDGPU_MACH_AMDGCN_GFX702          = 0x024,
  EF_AMDGPU_MACH_AMDGCN_GFX703          = 0x025,
  EF_AMDGPU_MACH_AMDGCN_GFX704          = 0x026,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X27   = 0x027,
  EF_AMDGPU_MACH_AMDGCN_GFX801          = 0x028,
  EF_AMDGPU_MACH_AMDGCN_GFX802          = 0x029,
  EF_AMDGPU_MACH_AMDGCN_GFX803          = 0x02a,
  EF_AMDGPU_MACH_AMDGCN_GFX810          = 0x02b,
  EF_AMDGPU_MACH_AMDGCN_GFX900          = 0x02c,
  EF_AMDGPU_MACH_AMDGCN_GFX902          = 0x02d,
  EF_AMDGPU_MACH_AMDGCN_GFX904          = 0x02e,
  EF_AMDGPU_MACH_AMDGCN_GFX906          = 0x02f,
  EF_AMDGPU_MACH_AMDGCN_GFX908          = 0x030,
  EF_AMDGPU_MACH_AMDGCN_GFX909          = 0x031,
  EF_AMDGPU_MACH_AMDGCN_GFX90C          = 0x032,
  EF_AMDGPU_MACH_AMDGCN_GFX1010         = 0x033,
  EF_AMDGPU_MACH_AMDGCN_GFX1011         = 0x034,
  EF_AMDGPU_MACH_AMDGCN_GFX1012         = 0x035,
  EF_AMDGPU_MACH_AMDGCN_GFX1030         = 0x036,
  EF_AMDGPU_MACH_AMDGCN_GFX1031         = 0x037,
  EF_AMDGPU_MACH_AMDGCN_GFX1032         = 0x038,
  EF_AMDGPU_MACH_AMDGCN_GFX1033         = 0x039,
  EF_AMDGPU_MACH_AMDGCN_GFX602          = 0x03a,
  EF_AMDGPU_MACH_AMDGCN_GFX705          = 0x03b,
  EF_AMDGPU_MACH_AMDGCN_GFX805          = 0x03c,
  EF_AMDGPU_MACH_AMDGCN_GFX1035         = 0x03d,
  EF_AMDGPU_MACH_AMDGCN_GFX1034         = 0x03e,
  EF_AMDGPU_MACH_AMDGCN_GFX90A          = 0x03f,
  EF_AMDGPU_MACH_AMDGCN_GFX940          = 0x040,
  EF_AMDGPU_MACH_AMDGCN_GFX1100         = 0x041,
  EF_AMDGPU_MACH_AMDGCN_GFX1013         = 0x042,
  EF_AMDGPU_MACH_AMDGCN_GFX1150         = 0x043,
  EF_AMDGPU_MACH_AMDGCN_GFX1103         = 0x044,
  EF_AMDGPU_MACH_AMDGCN_GFX1036         = 0x045,
  EF_AMDGPU_MACH_AMDGCN_GFX1101         = 0x046,
  EF_AMDGPU_MACH_AMDGCN_GFX1102         = 0x047,
  EF_AMDGPU_MACH_AMDGCN_GFX1200         = 0x048,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X49   = 0x049,
  EF_AMDGPU_MACH_AMDGCN_GFX1151         = 0x04a,
  EF_AMDGPU_MACH_AMDGCN_GFX941          = 0x04b,
  EF_AMDGPU_MACH_AMDGCN_GFX942          = 0x04c,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X4D   = 0x04d,
  EF_AMDGPU_MACH_AMDGCN_GFX1201         = 0x04e,
  EF_AMDGPU_MACH_AMDGCN_GFX950          = 0x04f,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X50   = 0x050,
  EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC    = 0x051,
  EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC = 0x052,
  EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC = 0x053,
  EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC   = 0x054,
  EF_AMDGPU_MACH_AMDGCN_GFX1152         = 0x055,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X56   = 0x056,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X57   = 0x057,
  EF_AMDGPU_MACH_AMDGCN_GFX1153         = 0x058,
  EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC   = 0x059,
  EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC  = 0x05f,
  EF_AMDGPU_MACH_AMDGCN_FIRST = EF_AMDGPU_MACH_AMDGCN_GFX600,
  EF_AMDGPU_MACH_AMDGCN_LAST = EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC,
  EF_AMDGPU_GENERIC_VERSION = 0xff000000,
  EF_AMDGPU_GENERIC_VERSION_OFFSET = 24,
