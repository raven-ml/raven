/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/610.43.03/src/common/sdk/nvidia/inc/nvos.h. */

typedef struct
{
    NvHandle hObjectError;               // Error notifier for TSG
    NvHandle hObjectEccError;            // ECC Error notifier for TSG
    NvHandle hVASpace;                   // VA space handle for TSG
    NvU32    engineType;                 // Engine to which all channels in this TSG are associated with
    NvBool   bIsCallingContextVgpuPlugin;
} NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS;

typedef struct
{
    NvU32   index;
    NvV32   flags;
    NvU64   vaSize NV_ALIGN_BYTES(8);
    NvU64   vaStartInternal NV_ALIGN_BYTES(8);
    NvU64   vaLimitInternal NV_ALIGN_BYTES(8);
    NvU32   bigPageSize;
    NvU64   vaBase NV_ALIGN_BYTES(8);
    NvU32   pasid;
} NV_VASPACE_ALLOCATION_PARAMETERS;
