/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR                                                0x00000F60   /* RW-4R */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_MAP                                                 31:10   /*       */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_BAR1_PENDING                                          0:0   /* R-IUF */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_BAR1_OUTSTANDING                                      1:1   /* R-IUF */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_MODE                                                  9:9   /* RWIUF */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_TARGET                                              11:10   /* RWIUF */

#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR_PTR                                                 31:12   /* RWIUF */
