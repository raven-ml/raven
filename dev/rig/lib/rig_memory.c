/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Memory devices, whose memory is the host's and whose work runs in the
   submitting thread, and the readers of buffers rig.h declares.
   Nothing here blocks. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"
#include "rig_stubs.h"

/* Memory devices

   A memory device's state is its word, alone in a page, which other
   devices may map. */

struct memory_device {
  _Atomic uint64_t word;
};

static void *aligned(size_t align, size_t n) {
#ifdef _WIN32
  return _aligned_malloc(n, align);
#else
  void *p = NULL;
  return posix_memalign(&p, align, n) == 0 ? p : NULL;
#endif
}

static void aligned_free(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

/* A memory device's state, never freed: its word outlives it. */
value caml_rig_memory_new(value unit) {
  (void)unit;
  size_t page = rig_page_bytes();
  struct memory_device *m = aligned(page, page);
  if (m == NULL) caml_raise_out_of_memory();
  memset(m, 0, page);
  return caml_copy_nativeint((intnat)m);
}

/* [v_n] bytes of host memory, on a page from 64 KiB on, or 0. */
value caml_rig_memory_alloc(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *p = aligned(n >= (1 << 16) ? rig_page_bytes() : 64, n);
  return Val_long((intnat)p);
}

value caml_rig_memory_free(value v_p) {
  aligned_free((void *)Long_val(v_p));
  return Val_unit;
}

static int memory_room(void *self, const struct rig_part *parts, int n) {
  (void)self;
  for (int i = 0; i < n; i++)
    if (parts[i].words != NULL || parts[i].ring_units != 0 ||
        parts[i].segment_bytes != 0)
      return RIG_NEVER;
  return RIG_FITS;
}

/* Runs the parts in order, the fills with no queue context, then makes [v]
   observable. Its handles are host addresses. */
static int memory_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         const uint64_t *handles, int nhandles,
                         const char **failure) {
  struct memory_device *m = self;
  (void)waits;
  (void)nwaits;
  (void)handles;
  (void)nhandles;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->fill != NULL) {
      if (p->fill(NULL, p->arg, v) != 0) {
        *failure = "a fill failed";
        return RIG_FAILED;
      }
    } else if (p->copy_bytes != 0)
      memmove((char *)(uintptr_t)p->copy_dst + p->copy_dst_offset,
              (const char *)(uintptr_t)p->copy_src + p->copy_src_offset,
              (size_t)p->copy_bytes);
  }
  atomic_store_explicit(&m->word, v, memory_order_release);
  return RIG_OK;
}

value caml_rig_memory_room(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&memory_room);
}

value caml_rig_memory_submit(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&memory_submit);
}

value caml_rig_load64(value v_addr) {
  return Val_long((intnat)atomic_load_explicit(
      (_Atomic uint64_t *)Long_val(v_addr), memory_order_acquire));
}

/* Copies the string [v_s] to the host address [v_dst]. */
value caml_rig_blit_string(value v_s, value v_dst) {
  size_t n = caml_string_length(v_s);
  if (n > 0) memcpy((void *)Long_val(v_dst), String_val(v_s), n);
  return Val_unit;
}

/* Reading buffers

   A buffer is the record { mem; offset; length; generation }, its
   memory { dev; bytes; host; address; handle; claim; ... } and its claim
   { count; generation; why }, as the core's Def module lays them out. */

enum { BUFFER_MEM, BUFFER_OFFSET, BUFFER_LENGTH, BUFFER_GEN };
enum { MEMORY_DEV, MEMORY_BYTES, MEMORY_HOST, MEMORY_ADDRESS, MEMORY_HANDLE,
       MEMORY_CLAIM };
enum { CLAIM_COUNT, CLAIM_GEN, CLAIM_WHY };

void *rig_buffer_host(value b) {
  value mem = Field(b, BUFFER_MEM);
  intnat host = Long_val(Field(mem, MEMORY_HOST));
  if (host < 0) return NULL;
  return (char *)host + Long_val(Field(b, BUFFER_OFFSET));
}

size_t rig_buffer_bytes(value b) {
  return (size_t)Long_val(Field(b, BUFFER_LENGTH));
}

const char *rig_buffer_why(value b) {
  value claim = Field(Field(b, BUFFER_MEM), MEMORY_CLAIM);
  value g = atomic_load_explicit((_Atomic value *)&Field(claim, CLAIM_GEN),
                                 memory_order_acquire);
  if (Long_val(g) == Long_val(Field(b, BUFFER_GEN))) return NULL;
  return String_val(Field(claim, CLAIM_WHY));
}
