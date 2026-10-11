/* SPDX-License-Identifier: MIT
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * Copyright (c) 2026 The Raven authors.
 * Adapted from ROCm clr rocclr/device/blitcl.cpp at
 * 2007bc5482af4d67b402016d13a9c98b4170a881.
 * Changes: kernel parameters carry only addresses/length, unaligned fallback.
 */

kernel void copy_bytes(global uchar *dst, global const uchar *src, ulong bytes) {
  const constant uint *blocks =
      (const constant uint *)__builtin_amdgcn_implicitarg_ptr();
  ulong group = __builtin_amdgcn_workgroup_id_x();
  ulong threads = __builtin_amdgcn_workgroup_size_x();
  ulong id = group * threads + __builtin_amdgcn_workitem_id_x();
  ulong stride = (ulong)blocks[0] * threads;
  if ((((ulong)dst | (ulong)src) & 15) == 0) {
    global ulong2 *out = (global ulong2 *)dst;
    global const ulong2 *in = (global const ulong2 *)src;
    for (ulong i = id; i < bytes / 16; i += stride) out[i] = in[i];
    for (ulong i = bytes / 16 * 16 + id; i < bytes; i += stride) dst[i] = src[i];
  } else {
    for (ulong i = id; i < bytes; i += stride) dst[i] = src[i];
  }
}
