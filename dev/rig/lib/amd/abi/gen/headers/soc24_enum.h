/*
 * Copyright 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

CS_PARTIAL_FLUSH                         = 0x00000007,
CACHE_FLUSH_AND_INV_TS_EVENT             = 0x00000014,
THREAD_TRACE_MARKER                      = 0x00000035,
THREAD_TRACE_FINISH                      = 0x00000037,
SQ_TT_TOKEN_MASK_SQDEC_BIT               = 0x00000001,
SQ_TT_TOKEN_MASK_SHDEC_BIT               = 0x00000002,
SQ_TT_TOKEN_MASK_GFXUDEC_BIT             = 0x00000004,
SQ_TT_TOKEN_MASK_COMP_BIT                = 0x00000008,
SQ_TT_TOKEN_MASK_CONTEXT_BIT             = 0x00000010,
SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT       = 0x00000000,
SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT        = 0x00000001,
