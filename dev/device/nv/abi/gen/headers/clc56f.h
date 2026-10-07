/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define NVC56F_SET_OBJECT                                          (0x00000000)
#define NVC56F_NON_STALL_INTERRUPT                                 (0x00000020)
#define NVC56F_SEM_ADDR_LO                                         (0x0000005c)
#define NVC56F_SEM_ADDR_HI                                         (0x00000060)
#define NVC56F_SEM_PAYLOAD_LO                                      (0x00000064)
#define NVC56F_SEM_PAYLOAD_HI                                      (0x00000068)
#define NVC56F_SEM_EXECUTE                                         (0x0000006c)
#define NVC56F_SEM_EXECUTE_OPERATION                                       2:0
#define NVC56F_SEM_EXECUTE_OPERATION_RELEASE                        0x00000001
#define NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ                   0x00000003
#define NVC56F_SEM_EXECUTE_RELEASE_WFI                                   20:20
#define NVC56F_SEM_EXECUTE_RELEASE_WFI_EN                           0x00000001
#define NVC56F_SEM_EXECUTE_PAYLOAD_SIZE                                  24:24
#define NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT                       0x00000001
#define NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP                             25:25
#define NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN                     0x00000001
#define NVC56F_GP_ENTRY0_GET                                 31:2
#define NVC56F_GP_ENTRY1_GET_HI                               7:0
#define NVC56F_GP_ENTRY1_LEVEL                                9:9
#define NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE              0x00000001
#define NVC56F_GP_ENTRY1_LENGTH                             30:10
#define NVC56F_DMA_METHOD_ADDRESS                                  11:0
#define NVC56F_DMA_METHOD_SUBCHANNEL                               15:13
#define NVC56F_DMA_METHOD_COUNT                                    28:16
#define NVC56F_DMA_SEC_OP                                          31:29
#define NVC56F_DMA_SEC_OP_INC_METHOD                               (0x00000001)
