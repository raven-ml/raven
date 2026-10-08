/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct
{
  NvHandle  hRoot;
  NvHandle  hObjectParent;
  NvHandle  hObjectOld;
  NvV32     status;
} NVOS00_PARAMETERS;

#define NVOS02_FLAGS_PHYSICALITY                                   7:4

#define NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS                     (0x00000001)

#define NVOS02_FLAGS_COHERENCY                                     15:12

#define NVOS02_FLAGS_COHERENCY_CACHED                              (0x00000001)

#define NVOS02_FLAGS_MAPPING                                       31:30

#define NVOS02_FLAGS_MAPPING_NO_MAP                                (0x00000001)

typedef struct
{
    NvHandle    hRoot;
    NvHandle    hObjectParent;
    NvHandle    hObjectNew;
    NvV32       hClass;
    NvV32       flags;
    NvP64       pMemory NV_ALIGN_BYTES(8);
    NvU64       limit NV_ALIGN_BYTES(8);
    NvV32       status;
} NVOS02_PARAMETERS;

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvP64    pAllocParms NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS21_PARAMETERS;

#define NVOS32_TYPE_IMAGE                               0

#define NVOS32_TYPE_NOTIFIER                            13

#define NVOS32_ATTR_PAGE_SIZE                                24:23

#define NVOS32_ATTR_PAGE_SIZE_HUGE                      0x00000003

#define NVOS32_ATTR_LOCATION                                 26:25

#define NVOS32_ATTR_LOCATION_VIDMEM                     0x00000000

#define NVOS32_ATTR_LOCATION_PCI                        0x00000001

#define NVOS32_ATTR_PHYSICALITY                              28:27

#define NVOS32_ATTR_PHYSICALITY_CONTIGUOUS              0x00000002

#define NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS     0x00000003

#define NVOS32_ATTR2_ZBC                                       1:0

#define NVOS32_ATTR2_ZBC_PREFER_NO_ZBC                  0x00000001

#define NVOS32_ATTR2_GPU_CACHEABLE                             3:2

#define NVOS32_ATTR2_GPU_CACHEABLE_YES                  0x00000001

#define NVOS32_ATTR2_GPU_CACHEABLE_NO                   0x00000002

#define NVOS32_ATTR2_PAGE_SIZE_HUGE                           21:20

#define NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB                  0x00000001

#define NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT        0x00000001

#define NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE              0x00000100

#define NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED       0x00004000

#define NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED             0x00008000

#define NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM            0x00010000

typedef struct
{
    NvU32     owner;                        // [IN]  - memory owner ID
    NvU32     type;                         // [IN]  - surface type, see below TYPE* defines
    NvU32     flags;                        // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines

    NvU32     width;                        // [IN]  - width of surface in pixels
    NvU32     height;                       // [IN]  - height of surface in pixels
    NvS32     pitch;                        // [IN/OUT] - desired pitch AND returned actual pitch allocated

    NvU32     attr;                         // [IN/OUT] - surface attributes requested, and surface attributes allocated
    NvU32     attr2;                        // [IN/OUT] - surface attributes requested, and surface attributes allocated

    NvU32     format;                       // [IN/OUT] - format requested, and format allocated
    NvU32     comprCovg;                    // [IN/OUT] - compr covg requested, and allocated
    NvU32     zcullCovg;                    // [OUT] - zcull covg allocated

    NvU64     rangeLo   NV_ALIGN_BYTES(8);  // [IN]  - allocated memory will be limited to the range
    NvU64     rangeHi   NV_ALIGN_BYTES(8);  // [IN]  - from rangeBegin to rangeEnd, inclusive.

    NvU64     size      NV_ALIGN_BYTES(8);  // [IN/OUT]  - size of allocation - also returns the actual size allocated
    NvU64     alignment NV_ALIGN_BYTES(8);  // [IN]  - requested alignment - NVOS32_ALLOC_FLAGS_ALIGNMENT* must be on
    NvU64     offset    NV_ALIGN_BYTES(8);  // [IN/OUT]  - desired offset if NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
    NvU64     limit     NV_ALIGN_BYTES(8);  // [OUT] - returned surface limit
    NvP64     address   NV_ALIGN_BYTES(8);  // [OUT] - returned address

    NvU32     ctagOffset;                   // [IN] - comptag offset for this surface (see NVOS32_ALLOC_COMPTAG_OFFSET)
    NvHandle  hVASpace;                     // [IN]  - VASpace handle. Used when flag is VIRTUAL.

    NvU32     internalflags;                // [IN]  - internal flags to change allocation behaviors from internal paths

    NvU32     tag;                          // [IN] - memory tag used for debugging

    NvS32     numaNode;                     // [IN] - CPU NUMA node from which memory should be allocated
} NV_MEMORY_ALLOCATION_PARAMS;

#define NVOS33_FLAGS_CACHING_TYPE                                  25:23

#define NVOS33_FLAGS_CACHING_TYPE_CACHED                           0

#define NVOS33_FLAGS_CACHING_TYPE_UNCACHED                         1

#define NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED                    2

typedef struct
{
    NvHandle hClient;
    NvHandle hDevice;            // device or sub-device handle
    NvHandle hMemory;            // handle to memory object if provided -- NULL if not
    NvU64    offset NV_ALIGN_BYTES(8);
    NvU64    length NV_ALIGN_BYTES(8);
    NvP64    pLinearAddress NV_ALIGN_BYTES(8);     // pointer for returned address
    NvU32    status;
    NvU32    flags;
} NVOS33_PARAMETERS;

#define NVOS46_FLAGS_CACHE_SNOOP                                   4:4

#define NVOS46_FLAGS_CACHE_SNOOP_ENABLE                            (0x00000001)

#define NVOS46_FLAGS_PAGE_SIZE                                     11:8

#define NVOS46_FLAGS_PAGE_SIZE_4KB                                 (0x00000001)

#define NVOS46_FLAGS_DMA_OFFSET_FIXED                              15:15

#define NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE                         (0x00000001)

typedef struct
{
    NvHandle hClient;                // [IN] client handle
    NvHandle hDevice;                // [IN] device handle for mapping
    NvHandle hDma;                   // [IN] dma handle for mapping
    NvHandle hMemory;                // [IN] memory handle for mapping
    NvU64    offset NV_ALIGN_BYTES(8);     // [IN] offset of region
    NvU64    length NV_ALIGN_BYTES(8);     // [IN] limit of region
    NvV32    flags;                  // [IN] flags
    NvU64    dmaOffset NV_ALIGN_BYTES(8);  // [OUT] offset of mapping
                                           // [IN] if FLAGS_DMA_OFFSET_FIXED_TRUE
                                           //      *OR* hDma is NOT a CTXDMA handle
                                           //      (see NVOS46_FLAGS_DMA_OFFSET_FIXED)
    NvV32    status;                 // [OUT] status
} NVOS46_PARAMETERS;

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvP64    params NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

#define NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES     (0x00000000)

typedef struct
{
    NvU32   index;
    NvV32   flags;
    NvU64   vaSize NV_ALIGN_BYTES(8);
    NvU64   vaStartInternal NV_ALIGN_BYTES(8);
    NvU64   vaLimitInternal NV_ALIGN_BYTES(8);
    NvU32   bigPageSize;
    NvU64   vaBase NV_ALIGN_BYTES(8);
} NV_VASPACE_ALLOCATION_PARAMETERS;

#define NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED                   BIT(3)

#define NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING                  BIT(6)
