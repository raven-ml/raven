/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* How partial sums combine: one output tile's sum cut into [splits] ranges,
   each summed by its own workgroup, those workgroups the same in every way
   but their range. Each stores its partial; the last to take the tile's
   ticket adds the partials in range order, s0 + s1 + ... + s(n-1),
   whichever order the workgroups ran in. A family's kernels call it after
   their own sums, before their epilogue. */

#ifndef NX_AMD_COMBINE_H
#define NX_AMD_COMBINE_H

#include "device.h"

/* [v] is this work-item's N values of the partial of range [z] of a tile
   whose [splits] partials go to [part] (N * [threads] values each) and
   whose ticket is the word at [ticket], 0 before the first workgroup
   arrives. Returns true in the last workgroup, its [v] then the tile's
   sum, and false in the others, which stop there. The release before the
   ticket makes each partial visible at the GPU's scope; the acquire after
   it, the others' to the last workgroup. */
template <typename T, int N>
DEVICE bool combine(T (&v)[N], T *part, uint32_t *ticket, int splits, int z,
                    int threads) {
  static SHARED int last;
  const int tid = item();
  T *mine = part + (size_t)z * N * threads;
#pragma unroll
  for (int f = 0; f < N; f++) mine[(size_t)f * threads + tid] = v[f];
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid == 0)
    last = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL,
                                  __HIP_MEMORY_SCOPE_AGENT) ==
           (uint32_t)splits - 1;
  barrier();
  if (!last) return false;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#pragma unroll
  for (int f = 0; f < N; f++) v[f] = part[(size_t)f * threads + tid];
#pragma unroll 1
  for (int r = 1; r < splits; r++) {
    const T *q = part + (size_t)r * N * threads + tid;
#pragma unroll
    for (int f = 0; f < N; f++) v[f] += q[(size_t)f * threads];
  }
  return true;
}

#endif
