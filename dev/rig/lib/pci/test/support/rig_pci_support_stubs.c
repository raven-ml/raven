/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the suites need from C: process memory to map, a far machine reached
   through a transport, the accesses of rig_pci.h, the monotonic clock,
   and the machine's GPU lock, which suites take in turn.

   A far machine holds [size] bytes at addresses [base, base + size) and
   nothing else. Its transport logs every access, fails on request, and
   fails on an access outside its bytes, so an access beyond a window whose
   bytes are the machine's is a failure the suite sees. Held, its accesses
   block until it is let go, as a link's round trip does. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_pci.h"

#include <errno.h>

/* Windows headers, which unixsupport.h brings, define [far]. */
#ifndef _WIN32
#include <caml/threads.h>
#include <caml/unixsupport.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

value rig_pci_test_memory(value n) {
  void *p = calloc(1, Long_val(n) + 1);
  if (p == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)p);
}

/* Far machines */

#define LOG 4096

struct access {
  int write;
  uint64_t address;
  size_t n;
};

struct far {
  struct rig_pci_transport transport; /* first: its address is the far's */
  uint64_t base;
  size_t size;
  uint8_t *bytes;
  int broken, outside, held, waiting;
  long left; /* accesses that succeed before it breaks, or -1 */
  size_t logged;
  struct access log[LOG];
};

/* The longest a held access waits before it fails. */
#define HOLD_MS 10000

/* Whether [f] was let go within HOLD_MS. */
static int let_go(struct far *f) {
  struct timespec ms = {0, 1000000};
  __atomic_store_n(&f->waiting, 1, __ATOMIC_RELEASE);
  for (int i = 0; i < HOLD_MS; i++) {
    if (!__atomic_load_n(&f->held, __ATOMIC_ACQUIRE)) return 1;
    nanosleep(&ms, NULL);
  }
  return 0;
}

static int far_access(struct far *f, int write, uint64_t a, void *p,
                      size_t n) {
  if (__atomic_load_n(&f->left, __ATOMIC_ACQUIRE) >= 0 &&
      __atomic_fetch_sub(&f->left, 1, __ATOMIC_ACQ_REL) <= 0)
    __atomic_store_n(&f->broken, 1, __ATOMIC_RELEASE);
  if (__atomic_load_n(&f->broken, __ATOMIC_ACQUIRE)) return -1;
  if (__atomic_load_n(&f->held, __ATOMIC_ACQUIRE) && !let_go(f)) return -1;
  if (a < f->base || a - f->base > f->size || n > f->size - (a - f->base)) {
    __atomic_store_n(&f->outside, 1, __ATOMIC_RELEASE);
    return -1;
  }
  size_t i = __atomic_fetch_add(&f->logged, 1, __ATOMIC_ACQ_REL);
  if (i < LOG) f->log[i] = (struct access){write, a, n};
  if (write)
    memcpy(f->bytes + (a - f->base), p, n);
  else
    memcpy(p, f->bytes + (a - f->base), n);
  return 0;
}

static int far_read(void *ctx, uint64_t a, void *dst, size_t n) {
  return far_access(ctx, 0, a, dst, n);
}

static int far_write(void *ctx, uint64_t a, const void *src, size_t n) {
  return far_access(ctx, 1, a, (void *)src, n);
}

static const char *far_failed(void *ctx) {
  struct far *f = ctx;
  if (__atomic_load_n(&f->broken, __ATOMIC_ACQUIRE))
    return "far: the link broke";
  if (__atomic_load_n(&f->outside, __ATOMIC_ACQUIRE))
    return "far: an access outside the machine's bytes";
  return NULL;
}

value rig_pci_test_far(value base, value size) {
  struct far *f = calloc(1, sizeof *f);
  uint8_t *bytes = calloc(1, Long_val(size) + 1);
  if (f == NULL || bytes == NULL) caml_raise_out_of_memory();
  f->transport = (struct rig_pci_transport){f, far_read, far_write,
                                                far_failed};
  f->base = Long_val(base);
  f->size = Long_val(size);
  f->bytes = bytes;
  f->left = -1;
  return Val_long((intnat)f);
}

/* Breaks [far] at its access [k] from now on, counting from 0: [k] accesses
   succeed. */
value rig_pci_test_far_break_at(value far, value k) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->left, Long_val(k), __ATOMIC_RELEASE);
  return Val_unit;
}

value rig_pci_test_far_break(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->broken, 1, __ATOMIC_RELEASE);
  return Val_unit;
}

value rig_pci_test_far_hold(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->waiting, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&f->held, 1, __ATOMIC_RELEASE);
  return Val_unit;
}

/* Whether an access waits on the hold. */
value rig_pci_test_far_waiting(value far) {
  struct far *f = (struct far *)Long_val(far);
  return Val_bool(__atomic_load_n(&f->waiting, __ATOMIC_ACQUIRE));
}

value rig_pci_test_far_let_go(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->held, 0, __ATOMIC_RELEASE);
  return Val_unit;
}

/* The accesses since the last call, oldest first, as (write, address, n). */
value rig_pci_test_far_log(value far) {
  CAMLparam1(far);
  CAMLlocal3(l, a, cell);
  struct far *f = (struct far *)Long_val(far);
  if (f->logged > LOG) caml_failwith("rig_pci_test_far_log: the log overflowed");
  l = Val_emptylist;
  for (size_t i = f->logged; i > 0; i--) {
    struct access *x = &f->log[i - 1];
    a = caml_alloc_tuple(3);
    Store_field(a, 0, Val_bool(x->write));
    Store_field(a, 1, Val_long(x->address));
    Store_field(a, 2, Val_long(x->n));
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = a;
    Field(cell, 1) = l;
    l = cell;
  }
  f->logged = 0;
  CAMLreturn(l);
}

/* The accesses of rig_pci.h */

static struct rig_pci_window window(value w) {
  struct rig_pci_window x;
  rig_pci_window_of(w, &x);
  return x;
}

/* (address, length, mapped pointer or 0, has a transport) */
value rig_pci_test_window_of(value w) {
  CAMLparam1(w);
  CAMLlocal1(r);
  struct rig_pci_window x = window(w);
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_long(x.address));
  Store_field(r, 1, Val_long(x.length));
  Store_field(r, 2, Val_long((intnat)x.mapped));
  Store_field(r, 3, Val_bool(x.transport != NULL));
  CAMLreturn(r);
}

value rig_pci_test_store32(value w, value off, value x) {
  struct rig_pci_window v = window(w);
  rig_pci_store32(&v, Long_val(off), (uint32_t)Long_val(x));
  return Val_unit;
}

value rig_pci_test_store64(value w, value off, value x) {
  struct rig_pci_window v = window(w);
  rig_pci_store64(&v, Long_val(off), Int64_val(x));
  return Val_unit;
}

value rig_pci_test_load32(value w, value off) {
  struct rig_pci_window v = window(w);
  return Val_long(rig_pci_load32(&v, Long_val(off)));
}

value rig_pci_test_load64(value w, value off) {
  struct rig_pci_window v = window(w);
  return caml_copy_int64(rig_pci_load64(&v, Long_val(off)));
}

value rig_pci_test_write(value w, value off, value s) {
  struct rig_pci_window v = window(w);
  rig_pci_write(&v, Long_val(off), String_val(s), caml_string_length(s));
  return Val_unit;
}

value rig_pci_test_flush(value w) {
  struct rig_pci_window v = window(w);
  rig_pci_flush(&v);
  return Val_unit;
}

value rig_pci_test_combines(value w) {
  struct rig_pci_window v = window(w);
  return Val_bool(v.combines);
}

value rig_pci_test_failed(value w) {
  CAMLparam1(w);
  struct rig_pci_window v = window(w);
  const char *why = rig_pci_failed(&v);
  if (why == NULL) CAMLreturn(Val_none);
  CAMLreturn(caml_alloc_some(caml_copy_string(why)));
}

/* The monotonic clock */

value rig_pci_test_now_ns(value unit) {
  (void)unit;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}

/* The GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap. */
value rig_pci_test_lock(value v_path, value v_holder) {
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
