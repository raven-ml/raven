/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The driver alone: a device's room and submit entries called directly, with
   a fill that adds 1 to a word of its own, as rig calls them for a
   submission of no part or of that fill, naming the fill's word or the
   handles it was given, and the machine's GPU lock. A submit holds the
   runtime, even for a driver whose submit may block, except a turn's: it
   takes the floor's mutex as rig takes a device's turn, by try-lock and
   otherwise with the runtime released. The lock releases it for its nap.
   And a host kernel's claims on its operands, from C. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK turn;
static void turn_init(turn *t) { InitializeSRWLock(t); }
static int turn_try(turn *t) { return TryAcquireSRWLockExclusive(t); }
static void turn_lock(turn *t) { AcquireSRWLockExclusive(t); }
static void turn_unlock(turn *t) { ReleaseSRWLockExclusive(t); }
#else
#include <pthread.h>
typedef pthread_mutex_t turn;
static void turn_init(turn *t) { pthread_mutex_init(t, NULL); }
static int turn_try(turn *t) { return pthread_mutex_trylock(t) == 0; }
static void turn_lock(turn *t) { pthread_mutex_lock(t); }
static void turn_unlock(turn *t) { pthread_mutex_unlock(t); }
#endif

#include "rig.h"
#include "rig_edge.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

struct floor {
  void *self;
  rig_room_fn *room;
  rig_submit_fn *submit;
  uint64_t v;
  uint64_t word;
  uint64_t handle;
  uint64_t *handles; /* NULL: the fill's word, for each part */
  int nhandles;
  turn *turn;
  struct rig_part fill;
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

value rig_bench_floor_new(value v_self, value v_room, value v_submit,
                                  value v_fill) {
  struct floor *f = calloc(1, sizeof *f);
  if (f == NULL) caml_raise_out_of_memory();
  f->self = (void *)Nativeint_val(v_self);
  f->room = (rig_room_fn *)Nativeint_val(v_room);
  f->submit = (rig_submit_fn *)Nativeint_val(v_submit);
  f->fill.fill = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->fill.arg = &f->word;
  f->handle = (uint64_t)(uintptr_t)&f->word;
  f->turn = malloc(sizeof *f->turn);
  if (f->turn == NULL) caml_raise_out_of_memory();
  turn_init(f->turn);
  return caml_copy_nativeint((intnat)f);
}

/* Makes the floor's part a fill: the function [v_fill] of the argument at
   the host address [v_arg], declaring [v_units] ring units and [v_bytes]
   segment bytes. */
value rig_bench_floor_fill(value v_f, value v_fill, value v_arg, value v_units,
                           value v_bytes) {
  struct floor *f = Floor_val(v_f);
  f->fill = (struct rig_part){0};
  f->fill.fill = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->fill.arg = (void *)Long_val(v_arg);
  f->fill.ring_units = (size_t)Long_val(v_units);
  f->fill.segment_bytes = (size_t)Long_val(v_bytes);
  return Val_unit;
}

/* Makes the floor's part the [v_n] 32-bit words at the host address
   [v_words]. */
value rig_bench_floor_words(value v_f, value v_words, value v_n) {
  struct floor *f = Floor_val(v_f);
  f->fill = (struct rig_part){0};
  f->fill.words = (const uint32_t *)Long_val(v_words);
  f->fill.n = (size_t)Long_val(v_n);
  return Val_unit;
}

/* Makes the floor's part a copy, on the queue at index [v_queue], of the
   [v_n] bytes of the memory whose handle is [v_src] into the memory whose
   handle is [v_dst]. */
value rig_bench_floor_copy(value v_f, value v_queue, value v_dst, value v_src,
                           value v_n) {
  struct floor *f = Floor_val(v_f);
  f->fill = (struct rig_part){0};
  f->fill.queue = Int_val(v_queue);
  f->fill.copy_dst = (uint64_t)Nativeint_val(v_dst);
  f->fill.copy_src = (uint64_t)Nativeint_val(v_src);
  f->fill.copy_bytes = (uint64_t)Long_val(v_n);
  return Val_unit;
}

/* Makes [v_v] the value the floor's device received last. */
value rig_bench_floor_at(value v_f, value v_v) {
  Floor_val(v_f)->v = (uint64_t)Long_val(v_v);
  return Val_unit;
}

/* Makes [v_g] take [v_f]'s turn. */
value rig_bench_floor_share(value v_f, value v_g) {
  Floor_val(v_g)->turn = Floor_val(v_f)->turn;
  return Val_unit;
}

/* Makes each submit name the handles [v_handles]. */
value rig_bench_floor_handles(value v_f, value v_handles) {
  struct floor *f = Floor_val(v_f);
  int n = (int)Wosize_val(v_handles);
  uint64_t *h = malloc((size_t)(n == 0 ? 1 : n) * sizeof *h);
  if (h == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < n; i++) h[i] = (uint64_t)Nativeint_val(Field(v_handles, i));
  free(f->handles);
  f->handles = h;
  f->nhandles = n;
  return Val_unit;
}

static void submit(struct floor *f, int n) {
  const char *failure = NULL;
  const uint64_t *h = f->handles != NULL ? f->handles : &f->handle;
  int nh = f->handles != NULL ? f->nhandles : n;
  if (f->room(f->self, &f->fill, n) != RIG_FITS) abort();
  if (f->submit(f->self, ++f->v, NULL, 0, &f->fill, n, h, nh, &failure) !=
      RIG_OK)
    abort();
}

/* Hands the device the next value with [v_parts] parts, 0 or the fill, as
   one room check and one submit. */
value rig_bench_floor_submit(value v_f, value v_parts) {
  submit(Floor_val(v_f), Int_val(v_parts));
  return Val_unit;
}

/* As [rig_bench_floor_submit], under the floor's turn. It releases the
   runtime while it waits for the turn: [f] is malloc'd memory, which no
   collection frees, read before. */
value rig_bench_floor_turn_submit(value v_f, value v_parts) {
  struct floor *f = Floor_val(v_f);
  int released = 0;
  if (!turn_try(f->turn)) {
    caml_enter_blocking_section_no_pending();
    released = 1;
    turn_lock(f->turn);
  }
  submit(f, Int_val(v_parts));
  turn_unlock(f->turn);
  if (released) caml_leave_blocking_section();
  return Val_unit;
}

/* The machine's GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap, after which
   it reads no argument. */
value rig_bench_lock(value v_path, value v_holder) {
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

/* Drops the file [v_path]'s pages from the page cache, so that its next
   read comes from its medium: writes back its dirty pages, then asks the
   kernel to drop them (Linux's posix_fadvise DONTNEED; macOS's msync
   MS_INVALIDATE over a mapping of the file). Answers [0], or the errno of a
   failing call. Holds the runtime: it serves bench setup only. */
value rig_bench_evict(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_int(ENOSYS);
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return Val_int(errno);
  int e = 0;
  struct stat st;
  if (fsync(fd) != 0 || fstat(fd, &st) != 0) e = errno;
#if defined(__APPLE__)
  if (e == 0 && st.st_size > 0) {
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) e = errno;
    else {
      if (msync(p, (size_t)st.st_size, MS_INVALIDATE) != 0) e = errno;
      munmap(p, (size_t)st.st_size);
    }
  }
#else
  if (e == 0) e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
  close(fd);
  return Val_int(e);
#endif
}

/* Claims the memory of three buffers for reading, then releases each, as a
   host kernel with three operands does around its loop. */
value rig_bench_claim_3(value a, value b, value c) {
  enum rig_claim ca = rig_buffer_claim(a, RIG_READ);
  enum rig_claim cb = rig_buffer_claim(b, RIG_READ);
  enum rig_claim cc = rig_buffer_claim(c, RIG_READ);
  if (ca == RIG_CLAIMED) rig_buffer_release(a);
  if (cb == RIG_CLAIMED) rig_buffer_release(b);
  if (cc == RIG_CLAIMED) rig_buffer_release(c);
  return Val_unit;
}
