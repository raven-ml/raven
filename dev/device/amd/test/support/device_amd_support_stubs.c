/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The GPU lock and host memory, for the AMD suites. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

/* Takes the lock file at [path] for the life of the process: [false] if
   another process holds it. */
value device_amd_test_lock(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_false;
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return Val_false;
  if (flock(fd, LOCK_EX | LOCK_NB) == 0) return Val_true;
  close(fd);
  return Val_false;
#endif
}

value device_amd_test_read(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLreturn(caml_alloc_initialized_string(Long_val(v_n),
                                           (const char *)Nativeint_val(v_a)));
}

value device_amd_test_write(value v_a, value v_s) {
  memcpy((void *)Nativeint_val(v_a), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* A fill: words to place and segment bytes to take, through the
   capability's functions. */
struct fill {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  size_t n, bytes;
  uint32_t words[];
};

static int fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct fill *f = arg;
  int e = f->place(queue, f->words, f->n);
  if (e || f->bytes == 0) return e;
  void *host;
  uint64_t address;
  return f->segment(queue, f->bytes, &host, &address);
}

value device_amd_test_fill_entry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)fill);
}

value device_amd_test_fill_arg(value v_place, value v_segment, value v_ws,
                               value v_bytes) {
  size_t n = Wosize_val(v_ws);
  struct fill *f = malloc(sizeof *f + n * sizeof(uint32_t));
  if (f == NULL) caml_raise_out_of_memory();
  f->place = (int (*)(void *, const uint32_t *, size_t))Nativeint_val(v_place);
  f->segment =
      (int (*)(void *, size_t, void **, uint64_t *))Nativeint_val(v_segment);
  f->n = n;
  f->bytes = (size_t)Long_val(v_bytes);
  for (size_t i = 0; i < n; i++) f->words[i] = (uint32_t)Long_val(Field(v_ws, i));
  return caml_copy_nativeint((intnat)f);
}
