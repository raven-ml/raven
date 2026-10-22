/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/570.144/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080nvd.h. */

#define NV2080_NOCAT_JOURNAL_MAX_DIAG_BUFFER     1024

#define NV2080_NOCAT_JOURNAL_MAX_STR_LEN         65

typedef struct NV2080CtrlNocatJournalInsertRecord {
    NvU32 flags;
    NV_DECLARE_ALIGNED(NvU64 timestamp, 8);
    NvU8  recType;
    NvU32 bugcheck;
    char  source[NV2080_NOCAT_JOURNAL_MAX_STR_LEN];
    NvU32 subsystem;
    NV_DECLARE_ALIGNED(NvU64 errorCode, 8);
    char  faultingEngine[NV2080_NOCAT_JOURNAL_MAX_STR_LEN];
    NvU32 tdrReason;
    NvU32 diagBufferLen;
    NvU8  diagBuffer[NV2080_NOCAT_JOURNAL_MAX_DIAG_BUFFER];
} NV2080CtrlNocatJournalInsertRecord;
