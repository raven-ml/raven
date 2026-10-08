/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR    0x18

#define PCI_DATA_STRUCT_SIGNATURE     0x52494350 // "PCIR" in dword format

#define PCI_DATA_STRUCT_SIGNATURE_NV  0x5344504E // "NPDS" in dword format

#define PCI_DATA_STRUCT_SIGNATURE_NV2 0x53494752 // "RGIS" in dword format

#define PCI_ROM_IMAGE_BLOCK_SIZE 512U

#define OFFSETOF_PCI_DATA_STRUCT_LEN        0xa

#define OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE  0x14

#define OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN  0x10

#define OFFSETOF_PCI_DATA_STRUCT_LAST_IMAGE 0x15

#define NV_PCI_DATA_EXT_SIG 0x4544504E // "NPDE" in dword format

#define NV_PCI_DATA_EXT_REV_10 0x100      // 1.0

#define NV_PCI_DATA_EXT_REV_11 0x101      // 1.1

#define OFFSETOF_PCI_DATA_EXT_STRUCT_SIG            0x0

#define OFFSETOF_PCI_DATA_EXT_STRUCT_LEN            0x6

#define OFFSETOF_PCI_DATA_EXT_STRUCT_REV            0x4

#define OFFSETOF_PCI_DATA_EXT_STRUCT_SUBIMAGE_LEN   0x8

#define OFFSETOF_PCI_DATA_EXT_STRUCT_LAST_IMAGE     0xa
