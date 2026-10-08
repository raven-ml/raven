/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES
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

#define NV_PMC_BOOT_0                                    0x00000000 /* R--4R */

#define NV_PMC_BOOT_0_MINOR_REVISION                            3:0 /* R--VF */

#define NV_PMC_BOOT_0_MAJOR_REVISION                            7:4 /* R--VF */

#define NV_PMC_BOOT_0_ARCHITECTURE_1                            8:8 /* R--VF */

#define NV_PMC_BOOT_0_IMPLEMENTATION                          23:20 /* R--VF */

#define NV_PMC_BOOT_0_ARCHITECTURE_0                          28:24 /* R--VF */

#define NV_PMC_BOOT_42                                   0x00000A00 /* R--4R */

#define NV_PMC_BOOT_42_MINOR_EXTENDED_REVISION                 11:8 /* R-XVF */

#define NV_PMC_BOOT_42_MINOR_REVISION                         15:12 /* R-XVF */

#define NV_PMC_BOOT_42_MAJOR_REVISION                         19:16 /* R-XVF */

#define NV_PMC_BOOT_42_IMPLEMENTATION                         23:20 /*       */

#define NV_PMC_BOOT_42_ARCHITECTURE                           29:24 /*       */

#define NV_PMC_BOOT_42_CHIP_ID                                29:20 /* R-XVF */

#define NV_PMC_BOOT_42_ARCHITECTURE_GA100                0x00000017 /*       */

#define NV_PMC_BOOT_42_ARCHITECTURE_AD100                0x00000019 /*       */

#define NV_PMC_BOOT_42_ARCHITECTURE_GB200                0x0000001B /*       */
