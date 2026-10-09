/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What device code of nx.amd's, and of its suite's harness, is written
   with: HIP compiled with no HIP header, so the attributes are spelled out,
   and the work-item's place read from the compiler's builtins. A kernel
   reads no implicit argument, so it knows its workgroup's size only as the
   constant it is compiled for. */

#ifndef NX_AMD_DEVICE_H
#define NX_AMD_DEVICE_H

#include <stdint.h>
#include <string.h>

#define KERNEL extern "C" __attribute__((global))
/* A kernel whose workgroups have [n] work-items. */
#define SIZED(n) __attribute__((amdgpu_flat_work_group_size(n, n)))
/* Every device function is inlined: a call keeps its frame in scratch
   memory. */
#define DEVICE inline __attribute__((device, always_inline))
#define SHARED __attribute__((shared))

typedef unsigned long long u64;
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

DEVICE uint32_t item(void) { return __builtin_amdgcn_workitem_id_x(); }
DEVICE uint32_t group(void) { return __builtin_amdgcn_workgroup_id_x(); }
DEVICE uint32_t group_y(void) { return __builtin_amdgcn_workgroup_id_y(); }
DEVICE uint32_t group_z(void) { return __builtin_amdgcn_workgroup_id_z(); }

/* The work-item's index in a grid of workgroups of [n] work-items. */
DEVICE u64 index_of(uint32_t n) { return (u64)group() * n + item(); }

/* Waits for the workgroup's work-items, each seeing the others' writes to
   the local data share before it. The fences order the local data share
   alone: a fence over global memory too would invalidate the compute
   unit's vector cache at every barrier. */
DEVICE void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* [v] of the lane [lane] ^ [d] of the wave of 32. */
template <typename T> DEVICE T xor_lane(T v, int d) {
  static_assert(sizeof(T) % 4 == 0, "whole 32-bit words");
  uint32_t w[sizeof(T) / 4];
  __builtin_memcpy(w, &v, sizeof v);
  const int to = (int)((item() % 32) ^ (uint32_t)d) * 4;
#pragma unroll
  for (unsigned i = 0; i < sizeof(T) / 4; i++)
    w[i] = (uint32_t)__builtin_amdgcn_ds_bpermute(to, (int)w[i]);
  __builtin_memcpy(&v, w, sizeof v);
  return v;
}

#endif
