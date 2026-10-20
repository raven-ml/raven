/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2026 The Raven authors.
 * Adapted from NVIDIA nvbandwidth kernels.cu at 82fc4e8c6afa0babb8687793678f615b3b8d793e.
 * Licensed under the Apache License, Version 2.0:
 * https://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an AS IS basis, without warranties or conditions.
 * Changes: arbitrary byte lengths, unaligned fallback, one copy per launch,
 * and a fill of bytes with the same stores.
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

extern "C" __global__ void fill_bytes(unsigned char *dst,
                                    unsigned long long byte,
                                    unsigned long long bytes) {
  const unsigned long long id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  const unsigned char b = (unsigned char)byte;
  const unsigned int w = 0x01010101u * b;
  const uint4 v = make_uint4(w, w, w, w);
  unsigned long long head = (16 - ((unsigned long long)dst & 15)) & 15;
  if (head > bytes) head = bytes;
  const unsigned long long vectors = (bytes - head) / 16;
  uint4 *out = (uint4 *)(dst + head);
  for (unsigned long long i = id; i < head; i += stride)
    dst[i] = b;
  for (unsigned long long i = id; i < vectors; i += stride)
    __stcg(out + i, v);
  for (unsigned long long i = head + vectors * 16 + id; i < bytes; i += stride)
    dst[i] = b;
}
