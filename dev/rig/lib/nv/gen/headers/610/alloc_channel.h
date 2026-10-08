/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct NV_MEMORY_DESC_PARAMS {
    NV_DECLARE_ALIGNED(NvU64 base, 8);
    NV_DECLARE_ALIGNED(NvU64 size, 8);
    NvU32 addressSpace;
    NvU32 cacheAttrib;
} NV_MEMORY_DESC_PARAMS;

#define CC_CHAN_ALLOC_IV_SIZE_DWORD    3U

#define CC_CHAN_ALLOC_NONCE_SIZE_DWORD 8U

typedef struct NV_CHANNEL_ALLOC_PARAMS {

    NvHandle hObjectError; // error context DMA
    NvHandle hObjectBuffer; // no longer used
    NV_DECLARE_ALIGNED(NvU64 gpFifoOffset, 8);    // offset to beginning of GP FIFO
    NvU32    gpFifoEntries;    // number of GP FIFO entries

    NvU32    flags;


    NvHandle hContextShare; // context share handle

    // Pointer-based VASpace for the channel.
    NvHandle hVASpace;

    // Handle-based VASpace for the channel.
    NvHandle hHandleVASpace;

    // handle to UserD memory object for channel, ignored if hUserdMemory[0]=0
    NvHandle hUserdMemory[NV_MAX_SUBDEVICES];

    // offset to beginning of UserD within hUserdMemory[x]
    NV_DECLARE_ALIGNED(NvU64 userdOffset[NV_MAX_SUBDEVICES], 8);

    // engine type(NV2080_ENGINE_TYPE_*) with which this channel is associated
    NvU32    engineType;
    // Channel identifier that is unique for the duration of a RM session
    NvU32    cid;
    // One-hot encoded bitmask to match SET_SUBDEVICE_MASK methods
    NvU32    subDeviceId;
    NvHandle hObjectEccError; // ECC error context DMA

    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS instanceMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS userdMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS ramfcMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS mthdbufMem, 8);

    NvHandle hPhysChannelGroup;              // reserved
    NvU32    internalFlags;                 // reserved
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS errorNotifierMem, 8); // reserved
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS eccErrorNotifierMem, 8); // reserved
    NvU32    ProcessID;                 // reserved
    NvU32    SubProcessID;                 // reserved

    // IV used for CPU-side encryption / GPU-side decryption.
    NvU32    encryptIv[CC_CHAN_ALLOC_IV_SIZE_DWORD];          // reserved
    // IV used for CPU-side decryption / GPU-side encryption.
    NvU32    decryptIv[CC_CHAN_ALLOC_IV_SIZE_DWORD];          // reserved
    // Nonce used CPU-side signing / GPU-side signature verification.
    NvU32    hmacNonce[CC_CHAN_ALLOC_NONCE_SIZE_DWORD];       // reserved
    NvU32    tpcConfigID; // TPC Configuration Id as supported by DTD-PG Feature
} NV_CHANNEL_ALLOC_PARAMS;

typedef NV_CHANNEL_ALLOC_PARAMS NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS;
