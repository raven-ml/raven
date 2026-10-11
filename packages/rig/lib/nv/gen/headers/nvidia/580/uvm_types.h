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

#define UVM_PARENT_ID_MAX_SUB_PROCESSORS 8

#define UVM_MAX_GPUS          (NV_MAX_DEVICES * UVM_PARENT_ID_MAX_SUB_PROCESSORS)

typedef struct
{
    // UUID of the physical GPU if the GPU is not SMC capable or SMC enabled,
    // or the GPU instance UUID of the partition.
    NvProcessorUuid gpuUuid;
    NvU32           gpuMappingType;     // UvmGpuMappingType
    NvU32           gpuCachingType;     // UvmGpuCachingType
    NvU32           gpuFormatType;      // UvmGpuFormatType
    NvU32           gpuElementBits;     // UvmGpuFormatElementBits
    NvU32           gpuCompressionType; // UvmGpuCompressionType
} UvmGpuMappingAttributes;
