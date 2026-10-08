/*
 * SPDX-FileCopyrightText: Copyright (c) 2003-2022 NVIDIA CORPORATION & AFFILIATES
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

#define NV_PFALCON_FALCON_HWCFG2                                                                       0x000000f4     /* R--4R */

#define NV_PFALCON_FALCON_HWCFG2_SHA                                                                   0:0            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_BMEM                                                                  1:1            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_PKCBOOT                                                               2:2            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_DBGMODE                                                               3:3            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_KMEM                                                                  4:4            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_HSCODE_REVOCATION                                                     5:5            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_STRAP_FUN                                                             6:6            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_VHR                                                                   7:7            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_HS                                                                    8:8            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_SECUREBUS                                                             9:9            /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_RISCV                                                                 10:10          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_RISCV_PL3_DISABLE                                                     11:11          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING                                                         12:12          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_RISCV_BR_PRIV_LOCKDOWN                                                13:13          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_BOOT_FROM_HS                                                          14:14          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_RISCV_BR_ADPAIR                                                       15:15          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_SCP                                                                   16:16          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_GDMA                                                                  17:17          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_SE_LITE                                                               18:18          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_PRGN_RSVD_FUSE                                                        31:24          /* R-IVF */
