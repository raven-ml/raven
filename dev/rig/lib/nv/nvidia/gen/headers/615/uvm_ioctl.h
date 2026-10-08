/*******************************************************************************
    Copyright (c) 2013-2025 NVidia Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#   define UVM_IOCTL_BASE(i) i

#define UVM_REGISTER_GPU_VASPACE                                      UVM_IOCTL_BASE(25)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvS32           rmCtrlFd; // IN
    NvHandle        hClient;  // IN
    NvHandle        hVaSpace; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_REGISTER_GPU_VASPACE_PARAMS;

#define UVM_UNREGISTER_GPU_VASPACE                                    UVM_IOCTL_BASE(26)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_VASPACE_PARAMS;

#define UVM_REGISTER_CHANNEL                                          UVM_IOCTL_BASE(27)

typedef struct
{
    NvProcessorUuid gpuUuid;                     // IN
    NvS32           rmCtrlFd;                    // IN
    NvHandle        hClient;                     // IN
    NvHandle        hChannel;                    // IN
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_REGISTER_CHANNEL_PARAMS;

#define UVM_UNREGISTER_CHANNEL                                        UVM_IOCTL_BASE(28)

typedef struct
{
    NvHandle        hClient;  // IN
    NvHandle        hChannel; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_CHANNEL_PARAMS;

#define UVM_ENABLE_PEER_ACCESS                                       UVM_IOCTL_BASE(29)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_ENABLE_PEER_ACCESS_PARAMS;

#define UVM_MAP_EXTERNAL_ALLOCATION                                   UVM_IOCTL_BASE(33)

typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

#define UVM_FREE                                                      UVM_IOCTL_BASE(34)

typedef struct
{
    NvU64     base      NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_FREE_PARAMS;

#define UVM_REGISTER_GPU                                              UVM_IOCTL_BASE(37)

typedef struct
{
    NvProcessorUuid gpu_uuid;    // IN/OUT
    NvBool          numaEnabled; // OUT
    NvS32           numaNodeId;  // OUT
    NvS32           rmCtrlFd;    // IN
    NvHandle        hClient;     // IN
    NvHandle        hSmcPartRef; // IN
    NV_STATUS       rmStatus;    // OUT
} UVM_REGISTER_GPU_PARAMS;

#define UVM_UNREGISTER_GPU                                            UVM_IOCTL_BASE(38)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_PARAMS;

#define UVM_UNMAP_EXTERNAL                                            UVM_IOCTL_BASE(66)

typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_UNMAP_EXTERNAL_PARAMS;

#define UVM_CREATE_EXTERNAL_RANGE                                     UVM_IOCTL_BASE(73)

typedef struct
{
    NvU64                  base                             NV_ALIGN_BYTES(8); // IN
    NvU64                  length                           NV_ALIGN_BYTES(8); // IN
    NV_STATUS              rmStatus;                                           // OUT
} UVM_CREATE_EXTERNAL_RANGE_PARAMS;

#define UVM_MM_INITIALIZE                                             UVM_IOCTL_BASE(75)

typedef struct
{
    NvS32                   uvmFd;    // IN
    NV_STATUS               rmStatus; // OUT
} UVM_MM_INITIALIZE_PARAMS;
