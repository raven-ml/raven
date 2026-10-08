/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define BIT_HEADER_SIGNATURE              0x00544942  // "BIT\0"

struct BIT_HEADER_V1_00
{
    bios_U016 Id;
    bios_U032 Signature;
    bios_U016 BCD_Version;
    bios_U008 HeaderSize;
    bios_U008 TokenSize;
    bios_U008 TokenEntries;
    bios_U008 HeaderChksum;
};

#define BIT_HEADER_V1_00_FMT "1w1d1w4b"

struct BIT_TOKEN_V1_00
{
    bios_U008 TokenId;
    bios_U008 DataVersion;
    bios_U016 DataSize;
    bios_U032 DataPtr;
};

#define BIT_TOKEN_V1_00_SIZE_6     6U

#define BIT_TOKEN_V1_00_SIZE_8     8U

#define BIT_TOKEN_V1_00_FMT_SIZE_6 "2b2w"

#define BIT_TOKEN_V1_00_FMT_SIZE_8 "2b1w1d"

#define BIT_TOKEN_FALCON_DATA       0x70

typedef struct
{
    bios_U032 FalconUcodeTablePtr;
} BIT_DATA_FALCON_DATA_V2;

#define BIT_DATA_FALCON_DATA_V2_4_FMT       "1d"

#define BIT_DATA_FALCON_DATA_V2_SIZE_4      4

typedef struct
{
    bios_U008 Version;
    bios_U008 HeaderSize;
    bios_U008 EntrySize;
    bios_U008 EntryCount;
    bios_U008 DescVersion;
    bios_U008 DescSize;
} FALCON_UCODE_TABLE_HDR_V1;

#define FALCON_UCODE_TABLE_HDR_V1_6_FMT     "6b"

typedef struct
{
    bios_U008 ApplicationID;
    bios_U008 TargetID;
    bios_U032 DescPtr;
} FALCON_UCODE_TABLE_ENTRY_V1;

#define FALCON_UCODE_TABLE_ENTRY_V1_6_FMT               "2b1d"

#define FALCON_UCODE_ENTRY_APPID_FWSEC_PROD             0x85

#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION                      15:8

#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3                   0x03

#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE                         31:16

typedef struct
{
    bios_U032 vDesc;
} FALCON_UCODE_DESC_HEADER;

#define FALCON_UCODE_DESC_HEADER_FORMAT   "1d"

typedef struct {
    FALCON_UCODE_DESC_HEADER Hdr;
    bios_U032 StoredSize;
    bios_U032 PKCDataOffset;
    bios_U032 InterfaceOffset;
    bios_U032 IMEMPhysBase;
    bios_U032 IMEMLoadSize;
    bios_U032 IMEMVirtBase;
    bios_U032 DMEMPhysBase;
    bios_U032 DMEMLoadSize;
    bios_U016 EngineIdMask;
    bios_U008 UcodeId;
    bios_U008 SignatureCount;
    bios_U016 SignatureVersions;
    bios_U016 Reserved;
} FALCON_UCODE_DESC_V3;

#define FALCON_UCODE_DESC_V3_SIZE_44    44

#define FALCON_UCODE_DESC_V3_44_FMT     "9d1w2b2w"

#define BCRT30_RSA3K_SIG_SIZE 384
