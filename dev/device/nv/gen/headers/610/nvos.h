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

#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR                0x00000000

#define NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1              3

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
    NvHandle hVASpace;
    NvU32    flags;
    NvU32    subctxId;
} NV_CTXSHARE_ALLOCATION_PARAMETERS;

#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT                      1:0

#define NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC                (0x00000001)
