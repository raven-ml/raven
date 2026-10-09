/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef enum GSP_SEQ_BUF_OPCODE
{
    GSP_SEQ_BUF_OPCODE_REG_WRITE = 0,
    GSP_SEQ_BUF_OPCODE_REG_MODIFY,
    GSP_SEQ_BUF_OPCODE_REG_POLL,
    GSP_SEQ_BUF_OPCODE_DELAY_US,
    GSP_SEQ_BUF_OPCODE_REG_STORE,
    GSP_SEQ_BUF_OPCODE_CORE_RESET,
    GSP_SEQ_BUF_OPCODE_CORE_START,
    GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT,
    GSP_SEQ_BUF_OPCODE_CORE_RESUME,
} GSP_SEQ_BUF_OPCODE;

typedef struct
{
    NvU32 addr;
    NvU32 val;
} GSP_SEQ_BUF_PAYLOAD_REG_WRITE;

typedef struct
{
    NvU32 addr;
    NvU32 mask;
    NvU32 val;
} GSP_SEQ_BUF_PAYLOAD_REG_MODIFY;

typedef struct
{
    NvU32 addr;
    NvU32 mask;
    NvU32 val;
    NvU32 timeout;
    NvU32 error;
} GSP_SEQ_BUF_PAYLOAD_REG_POLL;

typedef struct
{
    NvU32 val;
} GSP_SEQ_BUF_PAYLOAD_DELAY_US;

typedef struct
{
    NvU32 addr;
    NvU32 index;
} GSP_SEQ_BUF_PAYLOAD_REG_STORE;
