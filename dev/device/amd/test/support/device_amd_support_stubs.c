/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The machine's GPU lock, host memory, fills and the C room, for the AMD
   suites. */

#define _GNU_SOURCE

#include <errno.h>
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

#include "device_amd.h"

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

/* The machine's GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap. */
value device_amd_test_lock(value v_path, value v_holder) {
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

/* [n] zeroed bytes of their own pages, which device_amd_test_free_pages
   gives back with the same [n]. The pages are mapped untouched, so a large
   area costs nothing until used. */
value device_amd_test_pages(value v_n) {
  size_t n = ((size_t)Long_val(v_n) + 4095) & ~(size_t)4095;
#if defined(_WIN32)
  void *p = VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == NULL) caml_raise_out_of_memory();
#else
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  if (p == MAP_FAILED) caml_raise_out_of_memory();
#endif
  return Val_long((intnat)p);
}

value device_amd_test_free_pages(value v_a, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree((void *)Long_val(v_a), 0, MEM_RELEASE);
#else
  size_t n = ((size_t)Long_val(v_n) + 4095) & ~(size_t)4095;
  munmap((void *)Long_val(v_a), n);
#endif
  return Val_unit;
}

value device_amd_test_read(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLreturn(caml_alloc_initialized_string(Long_val(v_n),
                                           (const char *)Long_val(v_a)));
}

value device_amd_test_write(value v_a, value v_s) {
  memcpy((void *)Long_val(v_a), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* A fill: words to place and segment bytes to take, through the
   capability's functions, then its own answer. */
struct fill {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  size_t n, bytes;
  int code;
  uint32_t words[];
};

static int fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct fill *f = arg;
  int e = f->place(queue, f->words, f->n);
  if (e) return e;
  if (f->bytes > 0) {
    void *host;
    uint64_t address;
    e = f->segment(queue, f->bytes, &host, &address);
    if (e) return e;
  }
  return f->code;
}

value device_amd_test_fill_entry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)fill);
}

value device_amd_test_fill_arg(value v_place, value v_segment, value v_ws,
                               value v_bytes, value v_code) {
  size_t n = Wosize_val(v_ws);
  struct fill *f = malloc(sizeof *f + n * sizeof(uint32_t));
  if (f == NULL) caml_raise_out_of_memory();
  f->place = (int (*)(void *, const uint32_t *, size_t))Nativeint_val(v_place);
  f->segment =
      (int (*)(void *, size_t, void **, uint64_t *))Nativeint_val(v_segment);
  f->n = n;
  f->bytes = (size_t)Long_val(v_bytes);
  f->code = Int_val(v_code);
  for (size_t i = 0; i < n; i++) f->words[i] = (uint32_t)Long_val(Field(v_ws, i));
  return caml_copy_nativeint((intnat)f);
}

/* What the room function at [v_entry] answers for one part on queue
   [v_queue] of [v_words] words (none at all if 0), a fill if [v_fill],
   [v_copy] bytes of copy if positive, running after part [v_after] if it
   is not negative. */
value device_amd_test_room(value v_entry, value v_self, value v_queue,
                           value v_words, value v_fill, value v_copy,
                           value v_after) {
  static const uint32_t words[64];
  int after = Int_val(v_after);
  struct nx_part p = {
      .queue = Int_val(v_queue),
      .words = Long_val(v_words) > 0 ? words : NULL,
      .n = (size_t)Long_val(v_words),
      .fill = Bool_val(v_fill) ? fill : NULL,
      .copy_bytes = (uint64_t)Long_val(v_copy),
      .after = &after,
      .nafter = after >= 0,
  };
  if (p.n > 64) caml_invalid_argument("device_amd_test_room");
  nx_room_fn *room = (nx_room_fn *)Nativeint_val(v_entry);
  return Val_int(room((void *)Nativeint_val(v_self), &p, 1));
}

value device_amd_test_room_byte(value *argv, int argn) {
  (void)argn;
  return device_amd_test_room(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6]);
}
