////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

typedef struct hsa_signal_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal. The value 0 is reserved.
   */
  uint64_t handle;
} hsa_signal_t;
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  HSA_FENCE_SCOPE_SYSTEM = 2
   HSA_PACKET_HEADER_TYPE = 0,
   HSA_PACKET_HEADER_BARRIER = 8,
   HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
   HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
   HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
typedef struct hsa_kernel_dispatch_packet_s {
  union {
    struct {
        /**
         * Packet header. Used to configure multiple packet parameters such as the
         * packet type. The parameters are described by ::hsa_packet_header_t.
         */
        uint16_t header;

        /**
         * Dispatch setup parameters. Used to configure kernel dispatch parameters
         * such as the number of dimensions in the grid. The parameters are described
         * by ::hsa_kernel_dispatch_packet_setup_t.
         */
        uint16_t setup;
    };
    uint32_t full_header;
  };

  /**
   * X dimension of work-group, in work-items. Must be greater than 0.
   */
  uint16_t workgroup_size_x;

  /**
   * Y dimension of work-group, in work-items. Must be greater than
   * 0. If the grid has 1 dimension, the only valid value is 1.
   */
  uint16_t workgroup_size_y;

  /**
   * Z dimension of work-group, in work-items. Must be greater than
   * 0. If the grid has 1 or 2 dimensions, the only valid value is 1.
   */
  uint16_t workgroup_size_z;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * X dimension of grid, in work-items. Must be greater than 0. Must
   * not be smaller than @a workgroup_size_x.
   */
  uint32_t grid_size_x;

  /**
   * Y dimension of grid, in work-items. Must be greater than 0. If the grid has
   * 1 dimension, the only valid value is 1. Must not be smaller than @a
   * workgroup_size_y.
   */
  uint32_t grid_size_y;

  /**
   * Z dimension of grid, in work-items. Must be greater than 0. If the grid has
   * 1 or 2 dimensions, the only valid value is 1. Must not be smaller than @a
   * workgroup_size_z.
   */
  uint32_t grid_size_z;

  /**
   * Size in bytes of private memory allocation request (per work-item).
   */
  uint32_t private_segment_size;

  /**
   * Size in bytes of group memory allocation request (per work-group). Must not
   * be less than the sum of the group memory used by the kernel (and the
   * functions it calls directly or indirectly) and the dynamically allocated
   * group segment variables.
   */
  uint32_t group_segment_size;

  /**
   * Opaque handle to a code object that includes an implementation-defined
   * executable code for the kernel.
   */
  uint64_t kernel_object;

#ifdef HSA_LARGE_MODEL
  void* kernarg_address;
#elif defined HSA_LITTLE_ENDIAN
  /**
   * Pointer to a buffer containing the kernel arguments. May be NULL.
   *
   * The buffer must be allocated using ::hsa_memory_allocate, and must not be
   * modified once the kernel dispatch packet is enqueued until the dispatch has
   * completed execution.
   */
  void* kernarg_address;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
#else
  uint32_t reserved1;
  void* kernarg_address;
#endif

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

} hsa_kernel_dispatch_packet_t;
