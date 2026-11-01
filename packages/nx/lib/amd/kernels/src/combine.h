/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* How partial sums combine: one output tile's sum cut into partials, which
   [splits] workgroups compute, those workgroups the same in every way but
   the partial each takes. Each stores its partial; the last to take the
   tile's ticket adds them in order, +0 + p0 + p1 + ... + p(n-1), whichever
   order the workgroups ran in. A family's kernels call it after their own
   sums, before their epilogue. */

#ifndef NX_AMD_COMBINE_H
#define NX_AMD_COMBINE_H

#include "device.h"

/* Stores this work-item's N values [v] as partial [part] of a tile, the N *
   [threads] values each partial holds. */
template <typename T, int N>
DEVICE void put(const T (&v)[N], T *part, int threads) {
  const int tid = item();
#pragma unroll
  for (int f = 0; f < N; f++) part[(size_t)f * threads + tid] = v[f];
}

/* Whether the workgroup is the last of [splits] to take the tile's ticket,
   the word at [ticket], 0 before the first arrives. The release before the
   ticket makes the partials put before it visible at the GPU's scope; the
   acquire after it, the others' to the last workgroup. */
DEVICE bool last(uint32_t *ticket, int splits) {
  static SHARED int is_last;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (item() == 0)
    is_last = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT) ==
              (uint32_t)splits - 1;
  barrier();
  if (!is_last) return false;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return true;
}

/* [v], this work-item's values of the tile's sum: +0, then each of the
   [parts] partials from [part] on added in order. */
template <typename T, int N>
DEVICE void chain(T (&v)[N], const T *part, int64_t parts, int threads) {
  const int tid = item();
#pragma unroll
  for (int f = 0; f < N; f++) v[f] = T(0);
#pragma unroll 1
  for (int64_t r = 0; r < parts; r++) {
    const T *q = part + (size_t)r * N * threads + tid;
#pragma unroll
    for (int f = 0; f < N; f++) v[f] += q[(size_t)f * threads];
  }
}

#endif
