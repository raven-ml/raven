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

/* Excerpt of https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects/rocr-runtime/runtime/hsa-runtime/core/inc/registers.h. */

SQ_RSRC_BUF                              = 0x00000000,
BUF_DATA_FORMAT_32                       = 0x00000004,
BUF_NUM_FORMAT_UINT                      = 0x00000004,
BUF_FORMAT_32_UINT                       = 0x00000014,
SQ_SEL_X                                 = 0x00000004,
SQ_SEL_Y                                 = 0x00000005,
SQ_SEL_Z                                 = 0x00000006,
SQ_SEL_W                                 = 0x00000007,
	union SQ_BUF_RSRC_WORD1 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
		unsigned int                 BASE_ADDRESS_HI : 16;
		unsigned int                          STRIDE : 14;
		unsigned int                   CACHE_SWIZZLE : 1;
		unsigned int                  SWIZZLE_ENABLE : 1;
#elif		defined(BIGENDIAN_CPU)
		unsigned int                  SWIZZLE_ENABLE : 1;
		unsigned int                   CACHE_SWIZZLE : 1;
		unsigned int                          STRIDE : 14;
		unsigned int                 BASE_ADDRESS_HI : 16;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};
        union SQ_BUF_RSRC_WORD1_GFX11 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int BASE_ADDRESS_HI : 16;
            unsigned int STRIDE : 14;
            unsigned int SWIZZLE_ENABLE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int SWIZZLE_ENABLE : 2;
            unsigned int STRIDE : 14;
            unsigned int BASE_ADDRESS_HI : 16;
#endif
          } bitfields, bits;
          unsigned int u32All;
          signed int i32All;
          float f32All;
        };
	union SQ_BUF_RSRC_WORD3 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
                unsigned int                       DST_SEL_X : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                    ELEMENT_SIZE : 2;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                     ATC__CI__VI : 1;
                unsigned int                     HASH_ENABLE : 1;
                unsigned int                            HEAP : 1;
                unsigned int                   MTYPE__CI__VI : 3;
                unsigned int                            TYPE : 2;
#elif		defined(BIGENDIAN_CPU)
                unsigned int                            TYPE : 2;
                unsigned int                   MTYPE__CI__VI : 3;
                unsigned int                            HEAP : 1;
                unsigned int                     HASH_ENABLE : 1;
                unsigned int                     ATC__CI__VI : 1;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                    ELEMENT_SIZE : 2;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_X : 3;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};
        union SQ_BUF_RSRC_WORD3_GFX11 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int DST_SEL_X : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_W : 3;
            unsigned int FORMAT : 6;
            unsigned int RESERVED1 : 3;
            unsigned int INDEX_STRIDE : 2;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int RESERVED2 : 4;
            unsigned int OOB_SELECT : 2;
            unsigned int TYPE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int TYPE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int RESERVED2 : 4;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int INDEX_STRIDE : 2;
            unsigned int RESERVED1 : 3;
            unsigned int FORMAT : 6;
            unsigned int DST_SEL_W : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_X : 3;
#endif
          } bitfields, bits;
        unsigned int	u32All;
	signed int	i32All;
	float	f32All;
        };
        union SQ_BUF_RSRC_WORD3_GFX12 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int DST_SEL_X : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_W : 3;
            unsigned int FORMAT : 6;
            unsigned int RESERVED1 : 3;
            unsigned int INDEX_STRIDE : 2;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int WRITE_COMPRESS_ENABLE : 1;
            unsigned int COMPRESSION_EN : 1;
            unsigned int COMPRESSION_ACCESS_MODE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int TYPE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int TYPE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int COMPRESSION_ACCESS_MODE : 2;
            unsigned int COMPRESSION_EN : 1;
            unsigned int WRITE_COMPRESS_ENABLE : 1;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int INDEX_STRIDE : 2;
            unsigned int RESERVED1 : 3;
            unsigned int FORMAT : 6;
            unsigned int DST_SEL_W : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_X : 3;
#endif
          } bitfields, bits;
        unsigned int	u32All;
	signed int	i32All;
	float	f32All;
        };
