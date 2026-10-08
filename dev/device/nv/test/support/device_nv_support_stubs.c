/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The machine's GPU lock, host memory and the C room check. Every stub but
   the lock's holds the runtime: none blocks. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "device_nv.h"

#define Ptr_val(v) ((void *)Long_val(v))

/* The machine's GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap. */
value device_nv_test_lock(value v_path, value v_holder) {
#if defined(_WIN32)
  (void)v_path;
  (void)v_holder;
  return Val_int(ENOSYS);
#else
  /* The descriptor that holds the lock once taken. The suites take it from
     one domain. */
  static int held = -1;
  if (held >= 0) return Val_int(0);
  const char *path = String_val(v_path);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  /* O_EXCL: Linux refuses O_CREAT on another user's file in /tmp
     (fs.protected_regular). */
  if (fd < 0 && errno == ENOENT) {
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd < 0 && errno == EEXIST) fd = open(path, O_RDWR | O_CLOEXEC);
    else if (fd >= 0 && fchmod(fd, 0666) != 0) {
      int e = errno;
      close(fd);
      return Val_int(e);
    }
  }
  if (fd < 0) return Val_int(errno);
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) return Val_int(e);
    struct timespec nap = {0, 100 * 1000 * 1000};
    caml_release_runtime_system();
    nanosleep(&nap, NULL);
    caml_acquire_runtime_system();
    return Val_int(-1);
  }
  char note[1024] = "";
  snprintf(note, sizeof note, "%s, pid %ld\n", String_val(v_holder),
           (long)getpid());
  size_t len = strlen(note);
  if (ftruncate(fd, 0) != 0 || pwrite(fd, note, len, 0) != (ssize_t)len) {
    int e = errno;
    close(fd);
    return Val_int(e);
  }
  held = fd;
  return Val_int(0);
#endif
}

/* Host memory */

value device_nv_test_page_size(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

/* [v_n] zeroed writable bytes from a page. */
value device_nv_test_pages(value v_n) {
  size_t n = Long_val(v_n);
#if defined(_WIN32)
  void *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) caml_raise_out_of_memory();
#else
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  if (p == MAP_FAILED) caml_raise_out_of_memory();
#endif
  return Val_long((intnat)p);
}

value device_nv_test_free_pages(value v_p, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree(Ptr_val(v_p), 0, MEM_RELEASE);
#else
  munmap(Ptr_val(v_p), Long_val(v_n));
#endif
  return Val_unit;
}

value device_nv_test_get64(value v_p) {
  _Atomic uint64_t *p = Ptr_val(v_p);
  return Val_long((intnat)atomic_load_explicit(p, memory_order_acquire));
}

value device_nv_test_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = Ptr_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}

value device_nv_test_read(value v_p, value v_n) {
  CAMLparam2(v_p, v_n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(v_n));
  memcpy(Bytes_val(s), Ptr_val(v_p), Long_val(v_n));
  CAMLreturn(s);
}

value device_nv_test_write(value v_p, value v_s) {
  memcpy(Ptr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* Byte [i] of the pattern of [seed]. */
static uint8_t pattern_byte(uint64_t seed, uint64_t i) {
  uint64_t h = (i + (seed << 40)) * 0x9E3779B97F4A7C1ULL;
  return (uint8_t)((h ^ (h >> 29)) >> 17);
}

value device_nv_test_pattern(value v_p, value v_n, value v_seed) {
  uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++) p[i] = pattern_byte(seed, i);
  return Val_unit;
}

value device_nv_test_mismatch(value v_p, value v_n, value v_seed) {
  const uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++)
    if (p[i] != pattern_byte(seed, i)) return Val_long(i);
  return Val_long(-1);
}

/* The C edge */

static int no_fill(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 0;
}

/* The parts [v_parts], each an int array [| queue; kind; a; b; c; nafter;
   after...; words... |]: kind 0 is [a] ring words, 1 a copy of [c] bytes
   from the address [b] to the address [a], 2 a fill of [a] ring units and
   [b] segment bytes. In memory of their own, which the caller frees: the
   parts, then their after indices, then their words. */
static struct nx_part *parts_of(value v_parts) {
  int n = (int)Wosize_val(v_parts);
  size_t extra = 0;
  for (int i = 0; i < n; i++) extra += Wosize_val(Field(v_parts, i));
  struct nx_part *p =
      calloc(1, n * sizeof *p + extra * (sizeof(int) + sizeof(uint32_t)) + 1);
  if (p == NULL) caml_raise_out_of_memory();
  int *after = (int *)(p + n);
  uint32_t *words = (uint32_t *)(after + extra);
  for (int i = 0; i < n; i++) {
    value f = Field(v_parts, i);
    intnat kind = Long_val(Field(f, 1)), a = Long_val(Field(f, 2)),
           b = Long_val(Field(f, 3)), c = Long_val(Field(f, 4));
    int nafter = (int)Long_val(Field(f, 5));
    p[i].queue = (int)Long_val(Field(f, 0));
    p[i].nafter = nafter;
    p[i].after = after;
    for (int j = 0; j < nafter; j++) after[j] = (int)Long_val(Field(f, 6 + j));
    after += nafter;
    if (kind == 0) {
      p[i].n = (size_t)a;
      p[i].words = words;
      for (intnat j = 0; j < a; j++)
        words[j] = (uint32_t)Long_val(Field(f, 6 + nafter + j));
      words += a;
    } else if (kind == 1) {
      p[i].copy_dst = (uint64_t)a;
      p[i].copy_src = (uint64_t)b;
      p[i].copy_bytes = (uint64_t)c;
    } else {
      p[i].fill = no_fill;
      p[i].ring_units = (size_t)a;
      p[i].segment_bytes = (size_t)b;
    }
  }
  return p;
}

/* What device_nv_room answers for [v_parts]. */
value device_nv_test_room(value v_self, value v_parts) {
  struct nx_part *p = parts_of(v_parts);
  int r = device_nv_room((void *)Nativeint_val(v_self), p,
                         (int)Wosize_val(v_parts));
  free(p);
  return Val_int(r);
}

/* device_nv_submit of [v_parts] as the value [v_v], after the waits
   [v_waits], an address and a value each. */
value device_nv_test_submit(value v_self, value v_v, value v_waits,
                            value v_parts) {
  int nwaits = (int)(Wosize_val(v_waits) / 2);
  struct nx_wait *w = calloc(nwaits + 1, sizeof *w);
  if (w == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < nwaits; i++)
    w[i] = (struct nx_wait){
        .at = (uint64_t)Long_val(Field(v_waits, 2 * i)),
        .value = (uint64_t)Long_val(Field(v_waits, 2 * i + 1)),
        .kind = NX_WORD};
  struct nx_part *p = parts_of(v_parts);
  const char *failure = NULL;
  int r = device_nv_submit((void *)Nativeint_val(v_self),
                           (uint64_t)Long_val(v_v), w, nwaits, p,
                           (int)Wosize_val(v_parts), NULL, 0, &failure);
  free(p);
  free(w);
  return Val_int(r);
}
