/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/570.144/src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h. */

#define NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES (0x83de030c) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID" */

#define NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO (0x83de0328) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS_MESSAGE_ID" */
