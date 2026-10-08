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

/* The room check */

static int no_fill(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 0;
}

/* What device_nv_room answers for one part on queue [v_queue]: [v_words]
   zero words, a fill iff [v_fill], [v_units] ring units, [v_bytes] segment
   bytes, a copy of one byte between the handles 0 iff [v_copy], and the
   indices [v_after]. */
value device_nv_test_room(value v_self, value v_queue, value v_words,
                          value v_fill, value v_units, value v_bytes,
                          value v_copy, value v_after) {
  static const uint32_t words[8];
  int after[8];
  struct nx_part p;
  memset(&p, 0, sizeof p);
  p.queue = Int_val(v_queue);
  p.n = Long_val(v_words);
  if (p.n > 8) caml_invalid_argument("device_nv_test_room");
  p.words = p.n > 0 ? words : NULL;
  p.fill = Bool_val(v_fill) ? no_fill : NULL;
  p.ring_units = Long_val(v_units);
  p.segment_bytes = Long_val(v_bytes);
  p.copy_bytes = Bool_val(v_copy) ? 1 : 0;
  p.nafter = (int)Wosize_val(v_after);
  if (p.nafter > 8) caml_invalid_argument("device_nv_test_room");
  for (int i = 0; i < p.nafter; i++) after[i] = Int_val(Field(v_after, i));
  p.after = after;
  return Val_int(device_nv_room((void *)Nativeint_val(v_self), &p, 1));
}

value device_nv_test_room_byte(value *argv, int argn) {
  (void)argn;
  return device_nv_test_room(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6], argv[7]);
}
