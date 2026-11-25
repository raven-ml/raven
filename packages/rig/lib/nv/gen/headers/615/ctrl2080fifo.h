/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/615.71.09/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h. */

#define NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS         (0x2080110b) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_MESSAGE_ID" */

#define NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES (64)

typedef struct NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS {
    NvBool   bDisable;
    NvU32    numChannels;
    NvBool   bOnlyDisableScheduling;
    NvBool   bRewindGpPut;
    NV_DECLARE_ALIGNED(NvP64 pRunlistPreemptEvent, 8);
    // C form:  NvHandle hClientList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    NvHandle hClientList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
    // C form:  NvHandle hChannelList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    NvHandle hChannelList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
} NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS;

#define NV2080_CTRL_FIFO_DISABLE_CHANNEL_TRUE          (0x00000001)

#define NV2080_CTRL_FIFO_ONLY_DISABLE_SCHEDULING_TRUE  (0x00000001)
