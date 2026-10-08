/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The CUDA suite's and bench's C: the machine's GPU lock, the harness's
   cubin, tables of entries, and a fill that runs a sequence of record
   runs. CUDA's functions are those the device's capability finds, bound
   once. Every stub holds the runtime: none blocks but the lock's nap. */

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

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "harness.h"
#include "nx_cuda.h"

#define STR_(x) #x
#define STR(x) STR_(x)

#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif

/* The harness's cubin */

__asm__(SECTION "\n"
        ".balign 16\n"
        ".globl " SYMBOL("nx_harness_cubin") "\n"
        SYMBOL("nx_harness_cubin") ":\n"
        ".incbin \"" STR(NX_HARNESS_CUBIN) "\"\n"
        ".globl " SYMBOL("nx_harness_cubin_end") "\n"
        SYMBOL("nx_harness_cubin_end") ":\n"
        ".text\n");

extern const char nx_harness_cubin[], nx_harness_cubin_end[];

value nx_cuda_support_cubin(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_harness_cubin_end - nx_harness_cubin,
                                       nx_harness_cubin);
}

value nx_cuda_support_kernels(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  static const char *const names[] = {
#define NAME(name) #name,
      NX_HARNESS_KERNELS(NAME)
#undef NAME
  };
  r = caml_alloc(NX_HARNESS_COUNT, 0);
  for (int i = 0; i < NX_HARNESS_COUNT; i++)
    Store_field(r, i, caml_copy_string(names[i]));
  CAMLreturn(r);
}

/* The floors' threads per block and floor_read's vectors per thread, as
   harness.h fixes them. */
value nx_cuda_support_floors(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(NX_COPY_THREADS));
  Store_field(r, 1, Val_int(NX_READ_THREADS));
  Store_field(r, 2, Val_int(NX_READ_VECS));
  CAMLreturn(r);
}

/* CUDA, as the capability finds it */

static nx_cuda_launch_fn launch_kernel;
static int (*get_attribute)(int *, int, int);

/* Binds cuLaunchKernel and cuDeviceGetAttribute, in this order. */
value nx_cuda_support_bind(value v_f) {
  launch_kernel = (nx_cuda_launch_fn)Nativeint_val(Field(v_f, 0));
  get_attribute = (void *)Nativeint_val(Field(v_f, 1));
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value nx_cuda_support_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}

/* The entries of the functions [v_funcs]. Never freed: a process makes one
   per image. */
value nx_cuda_support_entries(value v_funcs) {
  int n = (int)Wosize_val(v_funcs);
  nx_cuda_entries *e = malloc(sizeof *e + n * sizeof(void *));
  if (e == NULL) caml_raise_out_of_memory();
  e->launch = launch_kernel;
  e->count = (uint32_t)n;
  for (int i = 0; i < n; i++) e->funcs[i] = (void *)Long_val(Field(v_funcs, i));
  return caml_copy_nativeint((intnat)e);
}

/* Records */

/* The run nx_cuda_add appends the launches [v_ls] to, in order: each a
   pair of the kernel, grid, block, shared bytes, address count and scratch
   mask, an int array of eight, and the parameters. */
value nx_cuda_support_record(value v_ls) {
  CAMLparam1(v_ls);
  CAMLlocal1(r);
  nx_cuda_records rs = {NULL, 0, 0};
  int rc = 0;
  for (mlsize_t i = 0; i < Wosize_val(v_ls) && rc == 0; i++) {
    value l = Field(Field(v_ls, i), 0), ps = Field(Field(v_ls, i), 1);
#define L(i) ((uint32_t)Long_val(Field(l, i)))
    uint32_t grid[3] = {L(1), L(2), L(3)}, block[3] = {L(4), 1, 1};
    rc = nx_cuda_add(&rs, L(0), grid, block, L(5), String_val(ps),
                     (uint32_t)caml_string_length(ps), L(6), L(7));
#undef L
  }
  if (rc == 0)
    r = caml_alloc_initialized_string(rs.len, (const char *)rs.bytes);
  free(rs.bytes);
  if (rc == -1) caml_invalid_argument("nx_cuda_add refused a record");
  if (rc != 0) caml_raise_out_of_memory();
  CAMLreturn(r);
}

/* Fills */

static value bytes(size_t n) {
  value v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                               (intnat)n);
  memset(Caml_ba_data_val(v), 0, n);
  return v;
}

/* An nx_cuda_run of the records [v_records], launched with the entries
   [v_entries], the records copied after it. */
value nx_cuda_support_run(value v_entries, value v_records) {
  CAMLparam2(v_entries, v_records);
  CAMLlocal1(v);
  size_t len = caml_string_length(v_records);
  v = bytes(sizeof(nx_cuda_run) + len);
  nx_cuda_run *r = Caml_ba_data_val(v);
  unsigned char *records = (unsigned char *)(r + 1);
  memcpy(records, String_val(v_records), len);
  *r = (nx_cuda_run){(const nx_cuda_entries *)Nativeint_val(v_entries),
                     records, len};
  CAMLreturn(v);
}

/* A sequence of runs, each run [repeat] times. */
struct seq {
  int n;
  struct {
    const nx_cuda_run *run;
    uint32_t repeat;
  } s[];
};

static int seq_fill(void *stream, void *arg, uint64_t v) {
  const struct seq *q = arg;
  for (int i = 0; i < q->n; i++)
    for (uint32_t k = 0; k < q->s[i].repeat; k++) {
      int rc = nx_cuda_fill(stream, (void *)q->s[i].run, v);
      if (rc != 0) return rc;
    }
  return 0;
}

/* The sequence of the runs [v_runs], each repeated as [v_repeats] says.
   The caller keeps the runs alive while the sequence runs. */
value nx_cuda_support_seq(value v_runs, value v_repeats) {
  CAMLparam2(v_runs, v_repeats);
  CAMLlocal1(v);
  int n = (int)Wosize_val(v_runs);
  v = bytes(sizeof(struct seq) + n * sizeof(((struct seq *)0)->s[0]));
  struct seq *q = Caml_ba_data_val(v);
  q->n = n;
  for (int i = 0; i < n; i++) {
    q->s[i].run = Caml_ba_data_val(Field(v_runs, i));
    q->s[i].repeat = (uint32_t)Long_val(Field(v_repeats, i));
  }
  CAMLreturn(v);
}

value nx_cuda_support_seq_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)seq_fill);
}

/* The GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits, as rig's CUDA suite takes it. A missing file
   is made writable by every user of the machine. Once taken, the file
   names [v_holder] and the process's id. Answers [0] once the process
   holds the lock, [-1] after a nap of 100 ms if another process holds it,
   or the errno of a failing call. Releases the runtime for the nap. */
value nx_cuda_support_lock(value v_path, value v_holder) {
  CAMLparam2(v_path, v_holder);
#if defined(_WIN32)
  CAMLreturn(Val_int(ENOSYS));
#else
  static int held = -1;
  if (held >= 0) CAMLreturn(Val_int(0));
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
      CAMLreturn(Val_int(e));
    }
  }
  if (fd < 0) CAMLreturn(Val_int(errno));
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) CAMLreturn(Val_int(e));
    struct timespec nap = {0, 100 * 1000 * 1000};
    caml_release_runtime_system();
    nanosleep(&nap, NULL);
    caml_acquire_runtime_system();
    CAMLreturn(Val_int(-1));
  }
  char note[1024] = "";
  snprintf(note, sizeof note, "%s, pid %ld\n", String_val(v_holder),
           (long)getpid());
  size_t len = strlen(note);
  if (ftruncate(fd, 0) != 0 || pwrite(fd, note, len, 0) != (ssize_t)len) {
    int e = errno;
    close(fd);
    CAMLreturn(Val_int(e));
  }
  held = fd;
  CAMLreturn(Val_int(0));
#endif
}
