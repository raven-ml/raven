/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct NV0080_CTRL_GR_ROUTE_INFO {
    NvU32 flags;
    NV_DECLARE_ALIGNED(NvU64 route, 8);
} NV0080_CTRL_GR_ROUTE_INFO;

typedef NVXXXX_CTRL_XXX_INFO NV0080_CTRL_GR_INFO;

#define NV0080_CTRL_GR_INFO_INDEX_SM_VERSION                            (0x0000000C)

#define NV0080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM                      (0x0000000D)

#define NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS                       (0x00000014)

#define NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC                (0x00000017)

#define NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC                 (0x00000020)
