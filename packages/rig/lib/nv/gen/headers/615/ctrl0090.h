/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/615.71.09/src/common/sdk/nvidia/inc/ctrl/ctrl0090.h. */

typedef enum NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE {
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE_NONE = 0,
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE_64B = 1,
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE_128B = 2,
} NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE;

typedef struct NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_FIELD {
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_SIZE size;
} NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_FIELD;

typedef struct NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_PARAMS {
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_FIELD l1;
    NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_FIELD t1;
} NV0090_CTRL_PROGRAM_VIDMEM_PROMOTE_PARAMS;

#define NV0090_CTRL_CMD_PROGRAM_VIDMEM_PROMOTE          (0x900107) /* finn: Evaluated from "(FINN_NV0090_KERNEL_GRAPHICS_CONTEXT_INTERFACE_ID << 8) | 0x7" */
