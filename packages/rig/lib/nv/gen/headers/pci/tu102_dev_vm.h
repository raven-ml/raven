/*
 * SPDX-FileCopyrightText: Copyright (c) 2003-2023 NVIDIA CORPORATION & AFFILIATES
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

#define NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET                   0x00BBFFFF:0x00B80000 /* RW--D */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_APERTURE                             1:1 /* RWEVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_ADDR                                31:4 /* RWEVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB_ADDR                          19:0 /* RWEVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE                                   0x000030B0 /* RW-4R */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_VA                                   0:0 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_PDB                                  1:1 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_HUBTLB_ONLY                              2:2 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_REPLAY                                   5:3 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_SYS_MEMBAR                              6:6 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ACK                                     8:7 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CANCEL_CLIENT_ID                       14:9 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CANCEL_GPC_ID                         19:15 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CANCEL_CLIENT_TYPE                    20:20 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_USE_PASID                             21:21 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_USE_SIZE                              22:22 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PROP_FLUSH                            23:23 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CACHE_LEVEL                           26:24 /* RWXVF */

#define NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_TRIGGER                                31:31 /* -WEVF */
