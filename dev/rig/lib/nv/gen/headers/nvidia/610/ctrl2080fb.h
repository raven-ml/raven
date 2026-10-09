/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef NVXXXX_CTRL_XXX_INFO NV2080_CTRL_FB_INFO;

#define NV2080_CTRL_FB_INFO_INDEX_BAR1_SIZE                         (0x00000005U)

#define NV2080_CTRL_FB_INFO_INDEX_HEAP_SIZE                         (0x00000009U)

#define NV2080_CTRL_FB_INFO_MAX_LIST_SIZE                           (0x00000080U)

#define NV2080_CTRL_CMD_FB_GET_INFO_V2 (0x20801303U) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | NV2080_CTRL_FB_GET_INFO_V2_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_FB_GET_INFO_V2_PARAMS {
    NvU32               fbInfoListSize;
    NV2080_CTRL_FB_INFO fbInfoList[NV2080_CTRL_FB_INFO_MAX_LIST_SIZE];
} NV2080_CTRL_FB_GET_INFO_V2_PARAMS;
