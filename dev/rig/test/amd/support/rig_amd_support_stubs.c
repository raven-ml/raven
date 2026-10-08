/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The machine's GPU lock, host memory, fills and the C entries, for the
   AMD suite. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "rig_amd.h"

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
value rig_amd_test_lock(value v_path, value v_holder) {
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

/* [n] zeroed bytes of their own pages, which rig_amd_test_free_pages
   gives back with the same [n]. The pages are mapped untouched, so a large
   area costs nothing until used. */
value rig_amd_test_pages(value v_n) {
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

value rig_amd_test_free_pages(value v_a, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree((void *)Long_val(v_a), 0, MEM_RELEASE);
#else
  size_t n = ((size_t)Long_val(v_n) + 4095) & ~(size_t)4095;
  munmap((void *)Long_val(v_a), n);
#endif
  return Val_unit;
}

/* The monotonic clock, in nanoseconds. */
value rig_amd_test_now(value unit) {
  (void)unit;
#if defined(_WIN32)
  LARGE_INTEGER t, f;
  QueryPerformanceCounter(&t);
  QueryPerformanceFrequency(&f);
  return Val_long((intnat)((double)t.QuadPart * 1e9 / (double)f.QuadPart));
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
#endif
}

value rig_amd_test_read(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLreturn(caml_alloc_initialized_string(Long_val(v_n),
                                           (const char *)Long_val(v_a)));
}

value rig_amd_test_write(value v_a, value v_s) {
  memcpy((void *)Long_val(v_a), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* A fill: words to place, in two calls if [split] is inside them, and
   segment bytes to take, through the capability's functions, then its own
   answer. [address] is where its last call took the bytes. It lives in a
   bigarray, rig's buffer for the fill's argument. */
struct fill {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  size_t n, split, bytes;
  uint64_t address;
  int code;
  uint32_t words[];
};

static int fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct fill *f = arg;
  size_t first = f->split > 0 && f->split < f->n ? f->split : f->n;
  int e = f->place(queue, f->words, first);
  if (e == 0 && first < f->n)
    e = f->place(queue, f->words + first, f->n - first);
  if (e) return e;
  if (f->bytes > 0) {
    void *host;
    e = f->segment(queue, f->bytes, &host, &f->address);
    if (e) return e;
  }
  return f->code;
}

value rig_amd_test_fill_entry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)fill);
}

value rig_amd_test_fill_arg(value v_place, value v_segment, value v_ws,
                            value v_split, value v_bytes, value v_code) {
  CAMLparam3(v_place, v_segment, v_ws);
  CAMLlocal1(v_arg);
  size_t n = Wosize_val(v_ws);
  v_arg = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                             (intnat)(sizeof(struct fill) + n * sizeof(uint32_t)));
  struct fill *f = Caml_ba_data_val(v_arg);
  f->place = (int (*)(void *, const uint32_t *, size_t))Nativeint_val(v_place);
  f->segment =
      (int (*)(void *, size_t, void **, uint64_t *))Nativeint_val(v_segment);
  f->n = n;
  f->split = (size_t)Long_val(v_split);
  f->address = 0;
  f->bytes = (size_t)Long_val(v_bytes);
  f->code = Int_val(v_code);
  for (size_t i = 0; i < n; i++)
    f->words[i] = (uint32_t)Long_val(Field(v_ws, i));
  CAMLreturn(v_arg);
}

value rig_amd_test_fill_arg_byte(value *argv, int argn) {
  (void)argn;
  return rig_amd_test_fill_arg(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5]);
}

value rig_amd_test_fill_address(value v_arg) {
  return Val_long((intnat)((struct fill *)Caml_ba_data_val(v_arg))->address);
}

value rig_amd_test_data(value v_ba) {
  return Val_long((intnat)Caml_ba_data_val(v_ba));
}

/* The C entries. A part is the ints Rig_amd_support.Edge makes: the queue,
   the fill, its argument, ring units, segment bytes, the copy's
   destination, source and bytes, the counts of [after] indices and of
   words, then the indices, then the words. */

enum {
  part_queue,
  part_fill,
  part_arg,
  part_units,
  part_bytes,
  part_dst,
  part_src,
  part_copy,
  part_nafter,
  part_nwords,
  part_after
};

static intnat at(value a, int i) { return Long_val(Field(a, i)); }

/* The parts and waits in C, in one allocation the caller frees. */
static void *edge(value v_waits, value v_parts, struct rig_wait **wp,
                  struct rig_part **pp) {
  int nwaits = (int)(Wosize_val(v_waits) / 2);
  int nparts = (int)Wosize_val(v_parts);
  size_t nafter = 0, nwords = 0;
  for (int i = 0; i < nparts; i++) {
    value p = Field(v_parts, i);
    nafter += (size_t)at(p, part_nafter);
    nwords += (size_t)at(p, part_nwords);
  }
  size_t size = nwaits * sizeof(struct rig_wait) +
                nparts * sizeof(struct rig_part) + nafter * sizeof(int) +
                nwords * sizeof(uint32_t) + 1;
  char *mem = malloc(size);
  if (mem == NULL) caml_raise_out_of_memory();
  struct rig_wait *w = (struct rig_wait *)mem;
  struct rig_part *p = (struct rig_part *)(w + nwaits);
  int *after = (int *)(p + nparts);
  uint32_t *words = (uint32_t *)(after + nafter);
  for (int i = 0; i < nwaits; i++) {
    w[i].kind = RIG_WORD;
    w[i].at = (uint64_t)at(v_waits, 2 * i);
    w[i].value = (uint64_t)at(v_waits, 2 * i + 1);
  }
  for (int i = 0; i < nparts; i++) {
    value k = Field(v_parts, i);
    int na = (int)at(k, part_nafter), nw = (int)at(k, part_nwords);
    p[i] = (struct rig_part){
        .queue = (int)at(k, part_queue),
        .words = nw > 0 ? words : NULL,
        .n = (size_t)nw,
        .fill = (int (*)(void *, void *, uint64_t))at(k, part_fill),
        .arg = (void *)at(k, part_arg),
        .ring_units = (size_t)at(k, part_units),
        .segment_bytes = (size_t)at(k, part_bytes),
        .copy_dst = (uint64_t)at(k, part_dst),
        .copy_src = (uint64_t)at(k, part_src),
        .copy_bytes = (uint64_t)at(k, part_copy),
        .after = na > 0 ? after : NULL,
        .nafter = na};
    for (int j = 0; j < na; j++) *after++ = (int)at(k, part_after + j);
    for (int j = 0; j < nw; j++)
      *words++ = (uint32_t)at(k, part_after + na + j);
  }
  *wp = w;
  *pp = p;
  return mem;
}

/* What the room entry [v_entry] answers for [v_parts] on the device
   [v_self]. */
value rig_amd_test_room(value v_entry, value v_self, value v_parts) {
  struct rig_wait *w;
  struct rig_part *p;
  void *mem = edge(Atom(0), v_parts, &w, &p);
  rig_room_fn *room = (rig_room_fn *)Nativeint_val(v_entry);
  int r = room((void *)Nativeint_val(v_self), p, (int)Wosize_val(v_parts));
  free(mem);
  return Val_int(r);
}

/* What the submit entry [v_entry] answers for [v_parts] as the value [v_v]
   after the waits [v_waits] (address, value pairs): [None], or [Some why]
   for RIG_FAILED. */
value rig_amd_test_submit(value v_entry, value v_self, value v_v,
                          value v_waits, value v_parts) {
  CAMLparam5(v_entry, v_self, v_v, v_waits, v_parts);
  CAMLlocal1(v_why);
  struct rig_wait *w;
  struct rig_part *p;
  void *mem = edge(v_waits, v_parts, &w, &p);
  rig_submit_fn *submit = (rig_submit_fn *)Nativeint_val(v_entry);
  const char *failure = NULL;
  int r = submit((void *)Nativeint_val(v_self),
                 (uint64_t)Long_val(v_v), w, (int)(Wosize_val(v_waits) / 2), p,
                 (int)Wosize_val(v_parts), NULL, 0, &failure);
  free(mem);
  if (r != RIG_FAILED) CAMLreturn(Val_none);
  v_why = caml_copy_string(failure ? failure : "");
  CAMLreturn(caml_alloc_some(v_why));
}
