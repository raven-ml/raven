/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct rpc_unloading_guest_driver_v1F_07
{
    NvBool     bInPMTransition;
    NvBool     bGc6Entering;
    NvU32      newLevel;
} rpc_unloading_guest_driver_v1F_07;

typedef struct rpc_set_page_directory_v1E_05
{
    NvHandle   hClient;
    NvHandle   hDevice;
    NvU32      pasid;
    NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05 params;
} rpc_set_page_directory_v1E_05;

typedef struct rpc_gsp_rm_alloc_v03_00
{
    NvHandle   hClient;
    NvHandle   hParent;
    NvHandle   hObject;
    NvU32      hClass;
    NvU32      status;
    NvU32      paramsSize;
    NvU32      flags;
    NvU8       reserved[4];
    NvU8       params[];
} rpc_gsp_rm_alloc_v03_00;

typedef struct rpc_gsp_rm_control_v03_00
{
    NvHandle   hClient;
    NvHandle   hObject;
    NvU32      cmd;
    NvU32      status;
    NvU32      paramsSize;
    NvU32      flags;
    NvU8       params[];
} rpc_gsp_rm_control_v03_00;

typedef struct rpc_run_cpu_sequencer_v17_00
{
    NvU32      bufferSizeDWord;
    NvU32      cmdIndex;
    NvU32      regSaveArea[8];
    NvU32      commandBuffer[];
} rpc_run_cpu_sequencer_v17_00;

typedef struct rpc_rc_triggered_v17_02
{
    NvU32      nv2080EngineType;
    NvU32      chid;
    NvU32      gfid;
    NvU32      exceptLevel;
    NvU32      exceptType;
    NvU32      scope;
    NvU16      partitionAttributionId;
    NvU32      mmuFaultAddrLo;
    NvU32      mmuFaultAddrHi;
    NvU32      mmuFaultType;
    NvBool     bCallbackNeeded;
    NvU32      rcJournalBufferSize;
    NvU8       rcJournalBuffer[];
} rpc_rc_triggered_v17_02;
