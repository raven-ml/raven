/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>

#include "nx_amd.h"

#define NAME(name, ...) #name,
const char *const nx_amd_kernel_names[NX_AMD_KERNEL_COUNT] = {
    NX_AMD_KERNELS(NAME)};
#undef NAME

/* The argument segment hands out multiples of its addresses' alignment. */
#define SEGMENT_ALIGN 64

int nx_amd_add(nx_amd_records *r, uint32_t kernel, const uint32_t groups[3],
               const uint32_t threads[3], const void *params, uint32_t bytes,
               uint32_t addrs, uint32_t scratch) {
  if (bytes % 8 != 0 || (uint64_t)addrs * 8 > bytes ||
      (addrs < 32 && scratch >> addrs != 0))
    return -1;
  size_t need = r->len + sizeof(nx_amd_launch) + bytes;
  if (need > r->cap) {
    size_t cap = r->cap ? r->cap : 256;
    while (cap < need) cap *= 2;
    unsigned char *b = realloc(r->bytes, cap);
    if (b == NULL) return NX_OUT_OF_MEMORY;
    r->bytes = b, r->cap = cap;
  }
  nx_amd_launch l = {kernel, {groups[0], groups[1], groups[2]},
                     {threads[0], threads[1], threads[2]}, bytes, addrs,
                     scratch};
  memcpy(r->bytes + r->len, &l, sizeof l);
  memcpy(r->bytes + r->len + sizeof l, params, bytes);
  r->len = need;
  return 0;
}

void nx_amd_rebase(unsigned char *r, size_t len, uint64_t base) {
  for (size_t at = 0; at < len;) {
    nx_amd_launch l;
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

/* The dispatch of [l]'s kernel, or NULL if it has none the fill places or
   [l]'s parameters are not the bytes the kernel reads. */
static const nx_amd_dispatch *dispatch(const nx_amd_entries *e,
                                       const nx_amd_launch *l) {
  if (l->kernel >= e->count) return NULL;
  const nx_amd_dispatch *d = &e->kernels[l->kernel];
  return d->n <= NX_AMD_DISPATCH_WORDS && l->bytes == d->kernarg ? d : NULL;
}

int nx_amd_size(const nx_amd_run *run, uint64_t *words, uint64_t *bytes) {
  *words = 0, *bytes = 0;
  for (size_t at = 0; at < run->len;) {
    nx_amd_launch l;
    memcpy(&l, run->records + at, sizeof l);
    const nx_amd_dispatch *d = dispatch(run->entries, &l);
    if (d == NULL) return -1;
    *words += d->n;
    *bytes += (l.bytes + SEGMENT_ALIGN - 1) / SEGMENT_ALIGN * SEGMENT_ALIGN;
    at += sizeof l + l.bytes;
  }
  return 0;
}

int nx_amd_fill(void *queue, void *arg, uint64_t v) {
  const nx_amd_run *run = arg;
  const nx_amd_entries *e = run->entries;
  (void)v;
  for (size_t at = 0; at < run->len;) {
    nx_amd_launch l;
    memcpy(&l, run->records + at, sizeof l);
    const nx_amd_dispatch *d = dispatch(e, &l);
    if (d == NULL) return -1;
    uint64_t args = 0;
    if (l.bytes > 0) {
      void *host;
      int rc = e->segment(queue, l.bytes, &host, &args);
      if (rc != 0) return rc;
      memcpy(host, run->records + at + sizeof l, l.bytes);
    }
    uint32_t w[NX_AMD_DISPATCH_WORDS];
    memcpy(w, d->words, 4 * d->n);
    w[d->args] = (uint32_t)args, w[d->args + 1] = (uint32_t)(args >> 32);
    for (int k = 0; k < 3; k++)
      w[d->threads[k]] = l.threads[k], w[d->groups[k]] = l.groups[k];
    int rc = e->place(queue, w, d->n);
    if (rc != 0) return rc;
    at += sizeof l + l.bytes;
  }
  return 0;
}
