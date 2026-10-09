/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES
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

#define NV_PFSP_EMEMC(i)                                                                                 (0x008F2ac0+(i)*8) /* RW-4A */

#define NV_PFSP_EMEMC_OFFS                                                                               7:2            /* RWIVF */

#define NV_PFSP_EMEMC_BLK                                                                                15:8           /* RWIVF */

#define NV_PFSP_EMEMC_AINCW                                                                              24:24          /* RWIVF */

#define NV_PFSP_EMEMC_AINCR                                                                              25:25          /* RWIVF */

#define NV_PFSP_EMEMD(i)                                                                                 (0x008F2ac4+(i)*8) /* RW-4A */

#define NV_PFSP_EMEMD_DATA                                                                               31:0           /* RWXVF */

#define NV_PFSP_MSGQ_HEAD(i)                                                                             (0x008F2c80+(i)*8) /* RW-4A */

#define NV_PFSP_MSGQ_HEAD_VAL                                                                            31:0           /* RWIUF */

#define NV_PFSP_MSGQ_TAIL(i)                                                                             (0x008F2c84+(i)*8) /* RW-4A */

#define NV_PFSP_MSGQ_TAIL_VAL                                                                            31:0           /* RWIUF */

#define NV_PFSP_QUEUE_HEAD(i)                                                                            (0x008F2c00+(i)*8) /* RW-4A */

#define NV_PFSP_QUEUE_HEAD_ADDRESS                                                                       31:0           /* RWIVF */

#define NV_PFSP_QUEUE_TAIL(i)                                                                            (0x008F2c04+(i)*8) /* RW-4A */

#define NV_PFSP_QUEUE_TAIL_ADDRESS                                                                       31:0           /* RWIVF */
