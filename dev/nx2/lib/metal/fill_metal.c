/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Runs of launch records, and the rig.metal fill over them: per launch,
   its pipeline, its parameters as kernel buffer 0, and its threadgroups.
   Objective-C on macOS; elsewhere no Metal device opens and the fill is
   never called. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#include "nx_metal.h"

int nx_metal_add(nx_metal_records *r, uint32_t entry, const uint32_t groups[3],
                 const uint32_t threads[3], const void *params, uint32_t bytes,
                 uint32_t addrs, uint32_t scratch) {
  size_t need = r->len + sizeof(nx_metal_launch) + bytes;
  if (need > r->cap) {
    size_t cap = r->cap ? r->cap : 256;
    while (cap < need) cap *= 2;
    unsigned char *b = realloc(r->bytes, cap);
    if (b == NULL) return -1;
    r->bytes = b, r->cap = cap;
  }
  nx_metal_launch l = {entry, {groups[0], groups[1], groups[2]},
                       {threads[0], threads[1], threads[2]}, bytes, addrs,
                       scratch};
  memcpy(r->bytes + r->len, &l, sizeof l);
  memcpy(r->bytes + r->len + sizeof l, params, bytes);
  r->len = need;
  return 0;
}

void nx_metal_rebase(unsigned char *r, size_t len, uint64_t base) {
  for (size_t at = 0; at < len;) {
    nx_metal_launch l;
    memcpy(&l, r + at, sizeof l);
    for (uint32_t i = 0; i < l.addrs; i++)
      if (l.scratch >> i & 1) {
        uint64_t x;
        unsigned char *w = r + at + sizeof l + 8 * i;
        memcpy(&x, w, 8);
        x += base;
        memcpy(w, &x, 8);
      }
    at += sizeof l + l.bytes;
  }
}

/* Whether every launch of [run] names a kernel with a pipeline. */
static int resolved(const nx_metal_run *run) {
  const unsigned char *at = (const unsigned char *)(run + 1);
  const unsigned char *end = at + run->bytes;
  for (; at < end; at += sizeof(nx_metal_launch) +
                         ((const nx_metal_launch *)at)->bytes) {
    uint32_t k = ((const nx_metal_launch *)at)->entry;
    if (k >= run->count || run->pipelines[k] == 0) return 0;
  }
  return 1;
}

#ifdef __APPLE__

#import <Metal/Metal.h>

int nx_metal_fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  const nx_metal_run *run = arg;
  if (!resolved(run)) return 1;
  id<MTLComputeCommandEncoder> e = *(id *)queue;
  const unsigned char *at = (const unsigned char *)(run + 1);
  const unsigned char *end = at + run->bytes;
  while (at < end) {
    const nx_metal_launch *l = (const nx_metal_launch *)at;
    [e setComputePipelineState:(id)(uintptr_t)run->pipelines[l->entry]];
    if (l->bytes > 0) [e setBytes:l + 1 length:l->bytes atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(l->groups[0], l->groups[1],
                                        l->groups[2])
        threadsPerThreadgroup:MTLSizeMake(l->threads[0], l->threads[1],
                                          l->threads[2])];
    at += sizeof *l + l->bytes;
  }
  return 0;
}

#else

int nx_metal_fill(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  return resolved(arg) ? 0 : 1;
}

#endif
