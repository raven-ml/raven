/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2026 The Raven authors.
 * Adapted from NVIDIA nvbandwidth kernels.cu at 82fc4e8c6afa0babb8687793678f615b3b8d793e.
 * Licensed under the Apache License, Version 2.0:
 * https://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an AS IS basis, without warranties or conditions.
 * Changes: arbitrary byte lengths, unaligned fallback, one copy per launch.
 */

extern "C" __global__ void copy_bytes(unsigned char *dst,
                                    const unsigned char *src,
                                    unsigned long long bytes) {
  const unsigned long long id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  if ((((unsigned long long)dst | (unsigned long long)src) & 15) == 0) {
    uint4 *out = (uint4 *)dst;
    const uint4 *in = (const uint4 *)src;
    for (unsigned long long i = id; i < bytes / 16; i += stride)
      __stcg(out + i, __ldcg(in + i));
    for (unsigned long long i = bytes / 16 * 16 + id; i < bytes; i += stride)
      dst[i] = src[i];
  } else {
    for (unsigned long long i = id; i < bytes; i += stride)
      dst[i] = src[i];
  }
}
