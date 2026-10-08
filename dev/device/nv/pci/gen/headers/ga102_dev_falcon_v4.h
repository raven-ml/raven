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

#define NV_PFALCON_FALCON_MAILBOX0                                                                     0x00000040     /* RW-4R */

#define NV_PFALCON_FALCON_MAILBOX1                                                                     0x00000044     /* RW-4R */

#define NV_PFALCON_FALCON_DMACTL                                                                       0x0000010c     /* RW-4R */

#define NV_PFALCON_FALCON_DMACTL_REQUIRE_CTX                                                           0:0            /* RWIVF */

#define NV_PFALCON_FALCON_DMACTL_DMEM_SCRUBBING                                                        1:1            /* R--VF */

#define NV_PFALCON_FALCON_DMACTL_IMEM_SCRUBBING                                                        2:2            /* R--VF */

#define NV_PFALCON_FALCON_DMATRFBASE                                                                   0x00000110     /* RW-4R */

#define NV_PFALCON_FALCON_DMATRFBASE_BASE                                                              31:0           /* RWIVF */

#define NV_PFALCON_FALCON_DMATRFMOFFS                                                                  0x00000114     /* RW-4R */

#define NV_PFALCON_FALCON_DMATRFMOFFS_OFFS                                                             23:0           /* RWIVF */

#define NV_PFALCON_FALCON_DMATRFCMD                                                                    0x00000118     /* RW-4R */

#define NV_PFALCON_FALCON_DMATRFCMD_FULL                                                               0:0            /* R-XVF */

#define NV_PFALCON_FALCON_DMATRFCMD_IDLE                                                               1:1            /* R-XVF */

#define NV_PFALCON_FALCON_DMATRFCMD_SEC                                                                3:2            /* RWXVF */

#define NV_PFALCON_FALCON_DMATRFCMD_IMEM                                                               4:4            /* RWXVF */

#define NV_PFALCON_FALCON_DMATRFCMD_WRITE                                                              5:5            /* RWXVF */

#define NV_PFALCON_FALCON_DMATRFCMD_SIZE                                                               10:8           /* RWXVF */

#define NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B                                                          0x00000006     /* RW--V */

#define NV_PFALCON_FALCON_DMATRFCMD_CTXDMA                                                             14:12          /* RWXVF */

#define NV_PFALCON_FALCON_DMATRFCMD_SET_DMTAG                                                          16:16          /* RWIVF */

#define NV_PFALCON_FALCON_DMATRFFBOFFS                                                                 0x0000011c     /* RW-4R */

#define NV_PFALCON_FALCON_DMATRFFBOFFS_OFFS                                                            31:0           /* RWIVF */

#define NV_PFALCON_FALCON_DMATRFBASE1                                                                  0x00000128     /* RW-4R */

#define NV_PFALCON_FALCON_DMATRFBASE1_BASE                                                             8:0            /* RWIVF */

#define NV_PFALCON_FALCON_HWCFG2                                                                       0x000000f4     /* R--4R */

#define NV_PFALCON_FALCON_HWCFG2_RISCV                                                                 10:10          /* R--VF */

#define NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING                                                         12:12          /* R--VF */

#define NV_PFALCON_FALCON_OS                                                                           0x00000080     /* RW-4R */

#define NV_PFALCON_FALCON_RM                                                                           0x00000084     /* RW-4R */

#define NV_PFALCON_FALCON_CPUCTL                                                                       0x00000100     /* RW-4R */

#define NV_PFALCON_FALCON_CPUCTL_STARTCPU                                                              1:1            /* -WXVF */

#define NV_PFALCON_FALCON_CPUCTL_HALTED                                                                4:4            /* R-XVF */

#define NV_PFALCON_FALCON_CPUCTL_ALIAS_EN                                                              6:6            /* RWIVF */

#define NV_PFALCON_FALCON_CPUCTL_ALIAS                                                                 0x00000130     /* -W-4R */

#define NV_PFALCON_FALCON_CPUCTL_ALIAS_STARTCPU                                                        1:1            /* -WXVF */

#define NV_PFALCON_FALCON_BOOTVEC                                                                      0x00000104     /* RW-4R */
