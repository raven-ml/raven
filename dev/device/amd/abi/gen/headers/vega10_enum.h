/*
 * Copyright (C) 2017  Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

SQ_THREAD_TRACE_TOKEN_MISC               = 0x00000000,
SQ_THREAD_TRACE_TOKEN_TIMESTAMP          = 0x00000001,
SQ_THREAD_TRACE_TOKEN_REG                = 0x00000002,
SQ_THREAD_TRACE_TOKEN_WAVE_START         = 0x00000003,
SQ_THREAD_TRACE_TOKEN_WAVE_ALLOC         = 0x00000004,
SQ_THREAD_TRACE_TOKEN_REG_CSPRIV         = 0x00000005,
SQ_THREAD_TRACE_TOKEN_WAVE_END           = 0x00000006,
SQ_THREAD_TRACE_TOKEN_EVENT              = 0x00000007,
SQ_THREAD_TRACE_TOKEN_EVENT_CS           = 0x00000008,
SQ_THREAD_TRACE_TOKEN_EVENT_GFX1         = 0x00000009,
SQ_THREAD_TRACE_TOKEN_INST               = 0x0000000a,
SQ_THREAD_TRACE_TOKEN_INST_PC            = 0x0000000b,
SQ_THREAD_TRACE_TOKEN_INST_USERDATA      = 0x0000000c,
SQ_THREAD_TRACE_TOKEN_ISSUE              = 0x0000000d,
SQ_THREAD_TRACE_TOKEN_PERF               = 0x0000000e,
SQ_THREAD_TRACE_TOKEN_REG_CS             = 0x0000000f,
CS_PARTIAL_FLUSH                         = 0x00000007,
CACHE_FLUSH_AND_INV_TS_EVENT             = 0x00000014,
THREAD_TRACE_MARKER                      = 0x00000035,
THREAD_TRACE_FINISH                      = 0x00000037,
