/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The driver alone: a device's room, submit and commit entries called
   directly, with a fill that adds 1 to a word of its own, as rig calls them
   for a submission of no part or of that fill, naming the fill's word or
   the handles it was given, each value committed as rig commits it. A
   submit holds the runtime, even for a driver whose submit may block,
   except a turn's: it takes the floor's mutex as rig takes a device's turn,
   by try-lock and otherwise with the runtime released. And a host kernel's
   claims on its operands, from C. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct floor {
  void *self;
  struct rig_driver driver;
  uint64_t v;
  uint64_t committed; /* the last value the driver reported or made committed */
  uint64_t word;
  uint64_t handle;
  uint64_t *handles; /* NULL: the fill's word, for each part */
  int nhandles;
  turn *turn;
  struct rig_part part;
  /* A launch's block: one group of one thread, its parameters 0. */
  _Alignas(16) uint8_t args[sizeof(struct rig_block) + RIG_PARAMS];
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

/* A floor over the driver's C state [v_edge] ([Rig.Driver.edge]), whose part
   is the fill [v_fill] of the floor's word. */
value rig_bench_floor_new(value v_edge, value v_fill) {
  struct floor *f = calloc(1, sizeof *f);
  if (f == NULL) caml_raise_out_of_memory();
  f->self = (void *)Nativeint_val(v_edge);
  f->driver = *rig_driver_of(f->self);
  f->part.kind = RIG_FILL;
  f->part.fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->part.fill.arg = &f->word;
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
  f->part = (struct rig_part){0};
  f->part.kind = RIG_FILL;
  f->part.fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->part.fill.arg = (void *)Long_val(v_arg);
  f->part.fill.ring_units = (size_t)Long_val(v_units);
  f->part.fill.segment_bytes = (size_t)Long_val(v_bytes);
  return Val_unit;
}

/* Makes the floor's part the [v_n] 32-bit words at the host address
   [v_words]. */
value rig_bench_floor_words(value v_f, value v_words, value v_n) {
  struct floor *f = Floor_val(v_f);
  f->part = (struct rig_part){0};
  f->part.kind = RIG_WORDS;
  f->part.words.at = (const uint32_t *)Long_val(v_words);
  f->part.words.n = (size_t)Long_val(v_n);
  return Val_unit;
}

/* Makes the floor's part a copy, on the queue at index [v_queue], of the
   [v_n] bytes of the memory whose handle is [v_src] into the memory whose
   handle is [v_dst]. */
value rig_bench_floor_copy(value v_f, value v_queue, value v_dst, value v_src,
                           value v_n) {
  struct floor *f = Floor_val(v_f);
  f->part = (struct rig_part){0};
  f->part.queue = Int_val(v_queue);
  f->part.kind = RIG_COPY;
  f->part.copy.dst = (uint64_t)Nativeint_val(v_dst);
  f->part.copy.src = (uint64_t)Nativeint_val(v_src);
  f->part.copy.bytes = (uint64_t)Long_val(v_n);
  return Val_unit;
}

/* Makes the floor's part a launch, on the queue at index 0, of the function
   whose entry is [v_code] and [v_launch], with [v_params] parameter bytes,
   over the floor's block. */
value rig_bench_floor_launch(value v_f, value v_code, value v_launch,
                             value v_params) {
  struct floor *f = Floor_val(v_f);
  struct rig_block *b = (struct rig_block *)f->args;
  for (int k = 0; k < 3; k++) b->groups[k] = b->threads[k] = 1;
  f->part = (struct rig_part){0};
  f->part.kind = RIG_LAUNCH;
  f->part.launch.code = (uint64_t)Long_val(v_code);
  f->part.launch.launch = (const void *)Nativeint_val(v_launch);
  f->part.launch.params = (uint32_t)Long_val(v_params);
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

static void encode(struct floor *f, int n) {
  const char *failure = NULL;
  const uint64_t *h = f->handles != NULL ? f->handles : &f->handle;
  int nh = f->handles != NULL ? f->nhandles : n;
  if (f->driver.room(f->self, &f->part, n, f->args) != RIG_FITS) abort();
  int r = f->driver.submit(f->self, ++f->v, NULL, 0, &f->part, n, f->args,
                           NULL, 0, h, nh, &failure);
  if (r == RIG_FAILED) abort();
  if (r == RIG_COMMITTED) f->committed = f->v;
}

/* Commits the values up to the last one handed over, unless the driver
   reported them committed, as rig does. */
static void commit(struct floor *f) {
  const char *failure = NULL;
  if (f->committed >= f->v) return;
  if (f->driver.commit(f->self, f->v, &failure) != RIG_OK) abort();
  f->committed = f->v;
}

static void submit(struct floor *f, int n) {
  encode(f, n);
  commit(f);
}

/* Hands the device the next value with [v_parts] parts, 0 or the fill, as
   one room check, one submit and its commit. */
value rig_bench_floor_submit(value v_f, value v_parts) {
  submit(Floor_val(v_f), Int_val(v_parts));
  return Val_unit;
}

/* As [rig_bench_floor_submit], without the commit. */
value rig_bench_floor_encode(value v_f, value v_parts) {
  encode(Floor_val(v_f), Int_val(v_parts));
  return Val_unit;
}

/* Commits the device's values up to the last one the floor handed over. */
value rig_bench_floor_commit(value v_f) {
  commit(Floor_val(v_f));
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

/* Whether a claim answer holds the claim. */
static int held(enum rig_claim c) {
  return c == RIG_CLAIMED || c == RIG_WAIT || c == RIG_LOST;
}

/* Claims the memory of three buffers for reading, then releases each, as a
   host kernel with three operands does around its loop. */
value rig_bench_claim_3(value a, value b, value c) {
  enum rig_claim ca = rig_buffer_claim(a, RIG_READ);
  enum rig_claim cb = rig_buffer_claim(b, RIG_READ);
  enum rig_claim cc = rig_buffer_claim(c, RIG_READ);
  if (held(ca)) rig_buffer_release(a);
  if (held(cb)) rig_buffer_release(b);
  if (held(cc)) rig_buffer_release(c);
  return Val_unit;
}
