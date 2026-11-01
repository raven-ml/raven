////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

/* Excerpt of https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects/rocr-runtime/runtime/hsa-runtime/core/inc/sdma_registers.h. */

typedef struct SDMA_PKT_COPY_LINEAR_RECT_TAG {
  static const unsigned int pitch_bits = 19;
  static const unsigned int slice_bits = 28;
  static const unsigned int rect_xy_bits = 14;
  static const unsigned int rect_z_bits = 11;

  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved : 13;
      unsigned int element : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int src_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_3_DATA;
  } SRC_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int src_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int src_pitch : pitch_bits;
    };
    unsigned int DW_4_DATA;
  } SRC_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int src_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_5_DATA;
  } SRC_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int dst_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_8_DATA;
  } DST_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int dst_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int dst_pitch : pitch_bits;
    };
    unsigned int DW_9_DATA;
  } DST_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_10_DATA;
  } DST_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int rect_x : rect_xy_bits;
      unsigned int reserved_1 : 2;
      unsigned int rect_y : rect_xy_bits;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_11_DATA;
  } RECT_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int rect_z : rect_z_bits;
      unsigned int reserved_1 : 5;
      unsigned int dst_swap : 2;
      unsigned int reserved_2 : 6;
      unsigned int src_swap : 2;
      unsigned int reserved_3 : 6;
    };
    unsigned int DW_12_DATA;
  } RECT_PARAMETER_2_UNION;

} SDMA_PKT_COPY_LINEAR_RECT;
typedef struct SDMA_PKT_COPY_LINEAR_RECT_TAG_GFX12 {
  static const unsigned int pitch_bits   = 16;
  static const unsigned int slice_bits   = 32;
  static const unsigned int rect_xy_bits = 16;
  static const unsigned int rect_z_bits  = 14;

  union {
    struct {
      unsigned int op       :  8;
      unsigned int sub_op   :  8;
      unsigned int reserved : 13;
      unsigned int element  :  3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_offset_x : 16;
      unsigned int src_offset_y : 16;
    };
    unsigned int DW_3_DATA;
  } SRC_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int src_offset_z : 14;
      unsigned int reserved_1   : 2;
      unsigned int src_pitch    : pitch_bits;
    };
    unsigned int DW_4_DATA;
  } SRC_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int src_slice_pitch : slice_bits;
    };
    unsigned int DW_5_DATA;
  } SRC_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_offset_x : 16;
      unsigned int dst_offset_y : 16;
    };
    unsigned int DW_8_DATA;
  } DST_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int dst_offset_z : 14;
      unsigned int reserved_1   : 2;
      unsigned int dst_pitch    : pitch_bits;
    };
    unsigned int DW_9_DATA;
  } DST_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch : slice_bits;
    };
    unsigned int DW_10_DATA;
  } DST_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int rect_x : rect_xy_bits;
      unsigned int rect_y : rect_xy_bits;
      };
    unsigned int DW_11_DATA;
  } RECT_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int rect_z           : rect_z_bits;
      unsigned int reserved_1       : 6;
      unsigned int dst_cache_policy : 3;
      unsigned int reserved_2       : 5;
      unsigned int src_cache_policy : 3;
      unsigned int reserved_3       : 1;
    };
    unsigned int DW_12_DATA;
  } RECT_PARAMETER_2_UNION;

} SDMA_PKT_COPY_LINEAR_RECT_GFX12;
