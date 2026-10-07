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

#define AMPERE_COMPUTE_B    0xC7C0
#define NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A                                                           0x02a0
#define NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_B                                                           0x02a4
#define NVC7C0_SEND_PCAS_A                                                                                 0x02b4
#define NVC7C0_SEND_SIGNALING_PCAS2_B                                                                      0x02c0
#define NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION                                                             3:0
#define NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE                                    0x00000009
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A                                                     0x02e4
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B                                                     0x02e8
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C                                                     0x02ec
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_A                                                                   0x0790
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_B                                                                   0x0794
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A                                                            0x07b0
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B                                                            0x07b4
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI                                                             0x1698
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION                                                    0:0
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE                                        0x00000001
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA                                                    4:4
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE                                        0x00000001
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT                                                     12:12
#define NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE                                           0x00000001
