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

typedef NV0080_CTRL_GR_ROUTE_INFO NV2080_CTRL_GR_ROUTE_INFO;

typedef NV0080_CTRL_GR_INFO NV2080_CTRL_GR_INFO;

#define NV2080_CTRL_GR_INFO_INDEX_SM_VERSION                            NV0080_CTRL_GR_INFO_INDEX_SM_VERSION

#define NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM                      NV0080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM

#define NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS                       NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS

#define NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC                NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC

#define NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC                 NV0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC

#define NV2080_CTRL_CMD_GR_GET_INFO                                     (0x20801201U) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | NV2080_CTRL_GR_GET_INFO_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_GR_GET_INFO_PARAMS {
    NvU32 grInfoListSize;
    NV_DECLARE_ALIGNED(NvP64 grInfoList, 8);
    NV_DECLARE_ALIGNED(NV2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} NV2080_CTRL_GR_GET_INFO_PARAMS;
