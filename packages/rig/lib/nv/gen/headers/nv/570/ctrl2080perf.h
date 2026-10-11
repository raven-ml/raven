/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define NV2080_CTRL_PERF_BOOST_FLAGS_CMD                1:0

#define NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX (0x00000002)

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA               4:4

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES         (0x00000001)

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY          6:6

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH     (0x00000001)

#define NV2080_CTRL_PERF_BOOST_DURATION_INFINITE      0xffffffff // If set this way, the boost will last until cleared.

#define NV2080_CTRL_CMD_PERF_BOOST                    (0x2080200a) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | NV2080_CTRL_PERF_BOOST_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_PERF_BOOST_PARAMS {
    NvU32 flags;
    NvU32 duration;
} NV2080_CTRL_PERF_BOOST_PARAMS;
