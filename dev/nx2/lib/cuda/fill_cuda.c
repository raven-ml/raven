/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>

#include "nx_cuda.h"

/* cuda.h's cuLaunchKernel extra keys. */
#define PARAM_BUFFER_POINTER ((void *)1)
#define PARAM_BUFFER_SIZE ((void *)2)
#define PARAM_END ((void *)0)

int nx_cuda_add(nx_cuda_records *r, uint32_t kernel, const uint32_t grid[3],
                const uint32_t block[3], uint32_t shared, const void *params,
                uint32_t bytes, uint32_t addrs, uint32_t scratch) {
  if (bytes % 8 != 0 || (uint64_t)addrs * 8 > bytes ||
      (addrs < 32 && scratch >> addrs != 0))
    return -1;
  size_t need = r->len + sizeof(nx_cuda_launch) + bytes;
  if (need > r->cap) {
    size_t cap = r->cap ? r->cap : 256;
    while (cap < need) cap *= 2;
    unsigned char *b = realloc(r->bytes, cap);
    if (b == NULL) return -2;
    r->bytes = b, r->cap = cap;
  }
  nx_cuda_launch l = {kernel, {grid[0], grid[1], grid[2]},
                      {block[0], block[1], block[2]}, shared, bytes, addrs,
                      scratch};
  memcpy(r->bytes + r->len, &l, sizeof l);
  memcpy(r->bytes + r->len + sizeof l, params, bytes);
  r->len = need;
  return 0;
}

int nx_cuda_fill(void *stream, void *arg, uint64_t v) {
  const nx_cuda_run *run = arg;
  const nx_cuda_entries *e = run->entries;
  (void)v;
  for (size_t at = 0; at < run->len;) {
    nx_cuda_launch l;
    memcpy(&l, run->records + at, sizeof l);
    if (l.kernel >= e->count) return 1;
    /* The parameters as one buffer, laid out as the kernel reads them. */
    size_t size = l.bytes;
    void *extra[] = {PARAM_BUFFER_POINTER,
                     (void *)(run->records + at + sizeof l), PARAM_BUFFER_SIZE,
                     &size, PARAM_END};
    int rc = e->launch(e->funcs[l.kernel], l.grid[0], l.grid[1], l.grid[2],
                       l.block[0], l.block[1], l.block[2], l.shared, stream,
                       NULL, l.bytes ? extra : NULL);
    if (rc != 0) return rc;
    at += sizeof l + l.bytes;
  }
  return 0;
}
