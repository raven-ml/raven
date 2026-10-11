/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef enum {
    GSP_DMA_TARGET_LOCAL_FB,
    GSP_DMA_TARGET_COHERENT_SYSTEM,
    GSP_DMA_TARGET_NONCOHERENT_SYSTEM,
    GSP_DMA_TARGET_COUNT
} GSP_DMA_TARGET;

typedef struct GSP_FMC_INIT_PARAMS
{
    // CC initialization "registry keys"
    NvU32 regkeys;
} GSP_FMC_INIT_PARAMS;

typedef struct GSP_ACR_BOOT_GSP_RM_PARAMS
{
    // Physical memory aperture through which gspRmDescPa is accessed
    GSP_DMA_TARGET target;
    // Size in bytes of the GSP-RM descriptor structure
    NvU32          gspRmDescSize;
    // Physical offset in the target aperture of the GSP-RM descriptor structure
    NvU64          gspRmDescOffset;
    // Physical offset in FB to set the start of the WPR containing GSP-RM
    NvU64          wprCarveoutOffset;
    // Size in bytes of the WPR containing GSP-RM
    NvU32          wprCarveoutSize;
    // Whether to boot GSP-RM or GSP-Proxy through ACR
    NvBool         bIsGspRmBoot;
} GSP_ACR_BOOT_GSP_RM_PARAMS;

typedef struct GSP_RM_PARAMS
{
    // Physical memory aperture through which bootArgsOffset is accessed
    GSP_DMA_TARGET target;
    // Physical offset in the memory aperture that will be passed to GSP-RM
    NvU64          bootArgsOffset;
} GSP_RM_PARAMS;

typedef struct GSP_SPDM_PARAMS
{
    // Physical Memory Aperture through which all addresses are accessed
    GSP_DMA_TARGET target;

    // Physical offset in the memory aperture where SPDM payload is stored
    NvU64 payloadBufferOffset;

    // Size of the above payload buffer
    NvU32 payloadBufferSize;
} GSP_SPDM_PARAMS;

typedef struct GSP_FMC_BOOT_PARAMS
{
    GSP_FMC_INIT_PARAMS         initParams;
    GSP_ACR_BOOT_GSP_RM_PARAMS  bootGspRmParams;
    GSP_RM_PARAMS               gspRmParams;
    GSP_SPDM_PARAMS             gspSpdmParams;
} GSP_FMC_BOOT_PARAMS;
