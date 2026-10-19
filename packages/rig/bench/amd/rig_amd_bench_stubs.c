/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Host memory for the AMD bench's rows: pages the GPU maps, and reads and
   writes of memory the host addresses. A failing call raises Failure with
   its step and errno. Every stub holds the runtime. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#define Addr_val(v) ((void *)Long_val(v))

/* Reads the [v_n] bytes at [v_p], 8 at a time: their sum. */
value rig_amd_bench_read(value v_p, value v_n) {
  const volatile uint64_t *p = Addr_val(v_p);
  size_t n = (size_t)Long_val(v_n) / 8;
  uint64_t s = 0;
  for (size_t i = 0; i < n; i++) s += p[i];
  return Val_long((intnat)s);
}

value rig_amd_bench_write(value v_p, value v_s) {
  memcpy(Addr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

#ifdef __linux__

#include <sys/mman.h>
#include <unistd.h>

static void fail(const char *step, int e) {
  char msg[128];
  snprintf(msg, sizeof msg, "%s: %s", step, strerror(e));
  caml_failwith(msg);
}

/* [v_n] bytes of page-aligned host memory, written once, never freed, in
   the host's base pages. Mapping memory for the GPU walks its pages: 256
   MiB took about 6 ms in 4 KiB pages and 1.4 to 3 ms in the huge pages the
   kernel had free for the process, so a row over huge pages read the
   machine's fragmentation. */
value rig_amd_bench_pages(value v_n) {
  void *p;
  size_t n = (size_t)Long_val(v_n);
  if (posix_memalign(&p, (size_t)sysconf(_SC_PAGESIZE), n))
    caml_raise_out_of_memory();
  if (madvise(p, n, MADV_NOHUGEPAGE)) fail("keeping huge pages out", errno);
  memset(p, 1, n);
  return Val_long((intnat)p);
}

#else

value rig_amd_bench_pages(value v_n) {
  (void)v_n;
  caml_failwith("the AMD bench runs on Linux");
}

#endif
