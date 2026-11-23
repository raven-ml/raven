/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the disk's suites ask of the system that OCaml's Unix does not give:
   limits on this process's open files and file sizes, dropping a file's
   cached pages and whether a file has an identity; and a compute job of the
   host's pool that runs until released. */

#define _GNU_SOURCE

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>

#include "rig_pool.h"

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <string.h>
#include <sys/mount.h>
#endif

/* rig.disk's hook for its suite (rig_disk_stubs.c), outside Windows. */
#ifndef _WIN32
extern void rig_disk_fault_runs(intnat run, int64_t wait_ns);
#endif

/* [fault_reads run wait] is rig_disk_fault_runs (run, wait in ns); nothing on
   Windows, which reads the runs of a copy one after the other. Keeps the
   runtime. */
value rig_disk_test_fault_reads(value v_run, value v_wait) {
#ifndef _WIN32
  rig_disk_fault_runs(Long_val(v_run), (int64_t)(Double_val(v_wait) * 1e9));
#else
  (void)v_run;
  (void)v_wait;
#endif
  return Val_unit;
}

/* [set_open_files n] sets this process's soft limit of open files to [n]: 0,
   the errno, or -1 on Windows, which has no such limit. Keeps the runtime:
   setrlimit does not block. */
value rig_disk_test_set_open_files(value v_n) {
#ifdef _WIN32
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_NOFILE, &l) == 0 ? 0 : errno);
#endif
}

/* [set_file_size n] sets this process's soft limit of a file's size to [n]
   bytes and ignores SIGXFSZ, so growing a file past it fails with EFBIG: 0,
   the errno, or -1 on Windows, which has no such limit. Keeps the runtime:
   setrlimit does not block. */
value rig_disk_test_set_file_size(value v_n) {
#ifdef _WIN32
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_FSIZE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR) return Val_int(errno);
  return Val_int(setrlimit(RLIMIT_FSIZE, &l) == 0 ? 0 : errno);
#endif
}

/* [drop_pages path] writes the file at [path]'s dirty pages to storage and asks
   the system to drop its cached pages, so that the next read reaches storage:
   0, or the errno. Linux only (posix_fadvise's POSIX_FADV_DONTNEED drops clean
   pages there); -1 elsewhere. Keeps the runtime: a test calls it alone. */
value rig_disk_test_drop_pages(value v_path) {
#if defined(__linux__)
  int fd = open(String_val(v_path), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return Val_int(errno);
  int code =
      fsync(fd) != 0 ? errno : posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  return Val_int(code);
#else
  (void)v_path;
  return Val_int(-1);
#endif
}

/* [identified path] is whether the system gives the file at [path] the
   identity rig_disk.mli names: a file handle (name_to_handle_at(2)) on
   Linux, an APFS or HFS+ volume on macOS, a file ID on Windows; none
   elsewhere. Keeps the runtime: a test calls it alone. */
value rig_disk_test_identified(value v_path) {
#if defined(__linux__)
  struct {
    struct file_handle fh;
    unsigned char bytes[MAX_HANDLE_SZ];
  } handle;
  int mount;
  handle.fh.handle_bytes = MAX_HANDLE_SZ;
  return Val_bool(name_to_handle_at(AT_FDCWD, String_val(v_path), &handle.fh,
                                    &mount, 0) == 0);
#elif defined(__APPLE__)
  struct statfs fs;
  if (statfs(String_val(v_path), &fs) != 0) return Val_false;
  return Val_bool(strcmp(fs.f_fstypename, "apfs") == 0 ||
                  strcmp(fs.f_fstypename, "hfs") == 0);
#elif defined(_WIN32)
  (void)v_path;
  return Val_true;
#else
  (void)v_path;
  return Val_false;
#endif
}

/* A held compute job */

static _Atomic int held_began, held_released;

static void held(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
  atomic_store(&held_began, 1);
  while (!atomic_load(&held_released)) {
  }
}

/* [hold_pool ()] runs a compute job of two chunks on two threads of the
   host's pool, whose every chunk lasts until [release_pool ()]. Releases the
   runtime. */
value rig_disk_test_hold_pool(value unit) {
  (void)unit;
  atomic_store(&held_began, 0);
  atomic_store(&held_released, 0);
  caml_enter_blocking_section();
  rig_pool_run(2, 2, 2, held, NULL);
  caml_leave_blocking_section();
  return Val_unit;
}

/* [pool_held ()] is whether a chunk of [hold_pool]'s job began. */
value rig_disk_test_pool_held(value unit) {
  (void)unit;
  return Val_bool(atomic_load(&held_began));
}

value rig_disk_test_release_pool(value unit) {
  (void)unit;
  atomic_store(&held_released, 1);
  return Val_unit;
}
