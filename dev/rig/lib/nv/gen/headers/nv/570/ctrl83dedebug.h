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

typedef struct NV83DE_SM_ERROR_STATE_REGISTERS {
    NvU32 hwwGlobalEsr;
    NvU32 hwwWarpEsr;
    NvU32 hwwWarpEsrPc;
    NvU32 hwwGlobalEsrReportMask;
    NvU32 hwwWarpEsrReportMask;
    NV_DECLARE_ALIGNED(NvU64 hwwEsrAddr, 8);
    NV_DECLARE_ALIGNED(NvU64 hwwWarpEsrPc64, 8);
    NvU32 hwwCgaEsr;
    NvU32 hwwCgaEsrReportMask;
} NV83DE_SM_ERROR_STATE_REGISTERS;

#define NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES (0x83de030c) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID" */

#define NV83DE_CTRL_DEBUG_MAX_SMS_PER_CALL             100

typedef struct NV83DE_MMU_FAULT_INFO {
    NvBool valid;
    NvU32  faultInfo;
} NV83DE_MMU_FAULT_INFO;

typedef struct NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS {
    NvHandle              hTargetChannel;
    NvU32                 numSMsToRead;
    NV_DECLARE_ALIGNED(NV83DE_SM_ERROR_STATE_REGISTERS smErrorStateArray[NV83DE_CTRL_DEBUG_MAX_SMS_PER_CALL], 8);
    NvU32                 mmuFaultInfo;       // Deprecated, use mmuFault field instead
    NV83DE_MMU_FAULT_INFO mmuFault;
    NvU32                 startingSM;
} NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS;

#define NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_MAX_ENTRIES 4

typedef struct NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY {
    NV_DECLARE_ALIGNED(NvU64 faultAddress, 8);
    NvU32 faultType;
    NvU32 accessType;
} NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY;

#define NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO (0x83de0328) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS_MESSAGE_ID" */

typedef struct NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS {
    NV_DECLARE_ALIGNED(NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY mmuFaultInfoList[NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_MAX_ENTRIES], 8);
    NvU32 count;
} NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS;
