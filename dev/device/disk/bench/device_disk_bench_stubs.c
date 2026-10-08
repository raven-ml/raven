/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the disk bench: the system calls a copy, a borrow, an open and
   a barrier make, alone, on a descriptor the bench holds. A call that can
   block releases the runtime, as the disk's do, so floors of two domains run
   at once. A failure crosses as a negated code: errno, or GetLastError on
   Windows. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

static intnat open_path(const char *path, int writable) {
  HANDLE h =
      CreateFileA(path, writable ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  BY_HANDLE_FILE_INFORMATION info;
  if (h == INVALID_HANDLE_VALUE)
    return -(intnat)GetLastError();
  if (!GetFileInformationByHandle(h, &info)) {
    intnat code = -(intnat)GetLastError();
    CloseHandle(h);
    return code;
  }
  return (intnat)h;
}

static void close_handle(intnat h) { CloseHandle((HANDLE)h); }

static intnat transfer(int write, intnat h, intnat pos, char *buf, intnat n) {
  intnat done = 0;
  while (done < n) {
    OVERLAPPED ov;
    DWORD moved = 0, len = (DWORD)(n - done < (1 << 30) ? n - done : 1 << 30);
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(uint64_t)(pos + done);
    ov.OffsetHigh = (DWORD)((uint64_t)(pos + done) >> 32);
    BOOL ok = write ? WriteFile((HANDLE)h, buf + done, len, &moved, &ov)
                    : ReadFile((HANDLE)h, buf + done, len, &moved, &ov);
    if (!ok)
      return -(intnat)GetLastError();
    if (moved == 0)
      break;
    done += moved;
  }
  return done;
}

static intnat sync_handle(intnat h) {
  return FlushFileBuffers((HANDLE)h) ? 0 : -(intnat)GetLastError();
}

static intnat map_read(intnat h, intnat n, char *dst) {
  HANDLE m = CreateFileMappingW((HANDLE)h, NULL, PAGE_WRITECOPY, 0, 0, NULL);
  if (m == NULL)
    return -(intnat)GetLastError();
  char *p = MapViewOfFile(m, FILE_MAP_COPY, 0, 0, (SIZE_T)n);
  intnat code = p == NULL ? -(intnat)GetLastError() : 0;
  CloseHandle(m);
  if (p == NULL)
    return code;
  if (dst != NULL)
    memcpy(dst, p, (size_t)n);
  UnmapViewOfFile(p);
  return 0;
}

#else

static intnat open_path(const char *path, int writable) {
  int fd = open(path, (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC);
  struct stat st;
  if (fd < 0)
    return -(intnat)errno;
  if (fstat(fd, &st) != 0) {
    intnat code = -(intnat)errno;
    close(fd);
    return code;
  }
  return fd;
}

static void close_handle(intnat h) { close((int)h); }

static intnat transfer(int write, intnat h, intnat pos, char *buf, intnat n) {
  intnat done = 0;
  while (done < n) {
    size_t len = (size_t)(n - done);
    ssize_t moved = write ? pwrite((int)h, buf + done, len, (off_t)(pos + done))
                          : pread((int)h, buf + done, len, (off_t)(pos + done));
    if (moved < 0 && errno == EINTR)
      continue;
    if (moved < 0)
      return -(intnat)errno;
    if (moved == 0)
      break;
    done += moved;
  }
  return done;
}

/* The call the disk's barrier makes: on macOS, F_BARRIERFSYNC. */
static intnat sync_handle(intnat h) {
#if defined(__APPLE__)
  if (fcntl((int)h, F_BARRIERFSYNC) == 0)
    return 0;
#else
  if (fsync((int)h) == 0)
    return 0;
#endif
  return -(intnat)errno;
}

static intnat map_read(intnat h, intnat n, char *dst) {
  char *p =
      mmap(NULL, (size_t)n, PROT_READ | PROT_WRITE, MAP_PRIVATE, (int)h, 0);
  if (p == MAP_FAILED)
    return -(intnat)errno;
  if (dst != NULL)
    memcpy(dst, p, (size_t)n);
  munmap(p, (size_t)n);
  return 0;
}

#endif

/* [open_ path writable] is a descriptor of the regular file [path], checked
   with a stat as the disk checks it, or a negated code. Releases the
   runtime. */
value device_disk_bench_open(value v_path, value v_writable) {
  CAMLparam1(v_path);
  char *path = caml_stat_strdup(String_val(v_path));
  int writable = Bool_val(v_writable);
  caml_release_runtime_system();
  intnat h = open_path(path, writable);
  caml_acquire_runtime_system();
  caml_stat_free(path);
  CAMLreturn(Val_long(h));
}

/* [close h] closes the descriptor [h]. Releases the runtime. */
value device_disk_bench_close(value v_h) {
  caml_release_runtime_system();
  close_handle(Long_val(v_h));
  caml_acquire_runtime_system();
  return Val_unit;
}

/* [read h pos dst n] reads the [n] bytes of [h] from [pos] into host memory at
   [dst]: the bytes read, or a negated code. Releases the runtime. */
intnat device_disk_bench_read(intnat h, intnat pos, intnat dst, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(0, h, pos, (char *)dst, n);
  caml_acquire_runtime_system();
  return r;
}

value device_disk_bench_read_byte(value h, value pos, value dst, value n) {
  return Val_long(device_disk_bench_read(Long_val(h), Long_val(pos),
                                         Long_val(dst), Long_val(n)));
}

/* [write h pos src n] writes the [n] bytes at [src] into [h] from [pos]: the
   bytes written, or a negated code. Releases the runtime. */
intnat device_disk_bench_write(intnat h, intnat pos, intnat src, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(1, h, pos, (char *)src, n);
  caml_acquire_runtime_system();
  return r;
}

value device_disk_bench_write_byte(value h, value pos, value src, value n) {
  return Val_long(device_disk_bench_write(Long_val(h), Long_val(pos),
                                          Long_val(src), Long_val(n)));
}

/* [sync h] orders [h]'s written bytes as the disk's barrier does: 0, or a
   negated code. Releases the runtime. */
value device_disk_bench_sync(value v_h) {
  caml_release_runtime_system();
  intnat r = sync_handle(Long_val(v_h));
  caml_acquire_runtime_system();
  return Val_long(r);
}

/* [map h n dst] maps the [n > 0] bytes of [h] copy-on-write, copies them to
   host memory at [dst] unless [dst] is 0, and unmaps them: 0, or a negated
   code. Releases the runtime. */
value device_disk_bench_map(value v_h, value v_n, value v_dst) {
  intnat h = Long_val(v_h), n = Long_val(v_n);
  char *dst = (char *)Long_val(v_dst);
  caml_release_runtime_system();
  intnat r = map_read(h, n, dst);
  caml_acquire_runtime_system();
  return Val_long(r);
}

/* [evict h] drops [h]'s pages from the system's cache, so the next read
   reaches the storage device: 0, or a negated code. Only Linux evicts one
   file's pages without privileges; elsewhere it is [-1]. A page must be clean
   to be dropped: the caller syncs the file first. Releases the runtime. */
value device_disk_bench_evict(value v_h) {
#if defined(__linux__)
  int fd = (int)Long_val(v_h);
  caml_release_runtime_system();
  int code = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  caml_acquire_runtime_system();
  return Val_long(-(intnat)code);
#else
  (void)v_h;
  return Val_long(-1);
#endif
}

/* [evicts ()] is [true] iff [evict] drops pages here. */
value device_disk_bench_evicts(value unit) {
  (void)unit;
#if defined(__linux__)
  return Val_true;
#else
  return Val_false;
#endif
}
