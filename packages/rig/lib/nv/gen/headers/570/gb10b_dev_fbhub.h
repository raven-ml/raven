/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/570.144/src/common/inc/swref/published/blackwell/gb10b/dev_fbhub.h. */

#define NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_LO                                0x008a1d58 /* RW-4R */

#define NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_LO_ADR                                  31:0 /* RWIVF */

#define NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_HI                                0x008a1d5c /* RW-4R */

#define NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_HI_ADR                                  31:0 /* RWIVF */

#define NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_HI_ADR_MASK                       0x000fffff /* RW--V */
