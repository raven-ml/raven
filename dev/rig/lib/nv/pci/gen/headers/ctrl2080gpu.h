/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY {
    NV_DECLARE_ALIGNED(NvU64 gpuPhysAddr, 8);
    NV_DECLARE_ALIGNED(NvU64 gpuVirtAddr, 8);
    NV_DECLARE_ALIGNED(NvU64 size, 8);
    NvU32 physAttr;
    NvU16 bufferId;
    NvU8  bInitialize;
    NvU8  bNonmapped;
} NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY;

#define NV2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES                        16U

#define NV2080_CTRL_CMD_GPU_PROMOTE_CTX                                    (0x2080012bU) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS {
    NvU32    engineType;
    NvHandle hClient;
    NvU32    ChID;
    NvHandle hChanClient;
    NvHandle hObject;
    NvHandle hVirtMemory;
    NV_DECLARE_ALIGNED(NvU64 virtAddress, 8);
    NV_DECLARE_ALIGNED(NvU64 size, 8);
    NvU32    entryCount;
    // C form: NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY promoteEntry[NV2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES];
    NV_DECLARE_ALIGNED(NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY promoteEntry[NV2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES], 8);
} NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS;
