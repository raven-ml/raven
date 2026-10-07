/*
 * Copyright (C) 2019  Advanced Micro Devices, Inc.
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
 *
 */

#define SDMA_OP_COPY  1
#define SDMA_OP_FENCE  5
#define SDMA_OP_TRAP  6
#define SDMA_OP_POLL_REGMEM  8
#define SDMA_OP_TIMESTAMP  13
#define SDMA_SUBOP_TIMESTAMP_GET_GLOBAL  2
#define SDMA_SUBOP_COPY_LINEAR  0
#define SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift  8
#define SDMA_PKT_FENCE_HEADER_mtype_mask   0x00000007
#define SDMA_PKT_FENCE_HEADER_mtype_shift  16
#define SDMA_PKT_POLL_REGMEM_HEADER_func_mask   0x00000007
#define SDMA_PKT_POLL_REGMEM_HEADER_func_shift  28
#define SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask   0x00000001
#define SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift  31
#define SDMA_PKT_POLL_REGMEM_DW5_interval_mask   0x0000FFFF
#define SDMA_PKT_POLL_REGMEM_DW5_interval_shift  0
#define SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask   0x00000FFF
#define SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift  16
#define SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask   0x000000FF
#define SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift  8
