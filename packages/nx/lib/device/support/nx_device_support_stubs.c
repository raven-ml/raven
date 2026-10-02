/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#if defined(_WIN32)
#define _CRT_RAND_S
#endif
#include <errno.h>
#include <stdatomic.h>
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

#ifdef _WIN32
#include <winsock2.h>
#include <mstcpip.h>
#include <windows.h>
#include <caml/unixsupport.h>
#else
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <caml/unixsupport.h>
#include <sys/random.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#endif

static void fail_errno(const char *what) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(errno));
  caml_failwith(msg);
}

#ifndef __linux__
static void fail_linux(const char *what) {
  char msg[256];
  snprintf(msg, sizeof msg, "%s needs Linux", what);
  caml_failwith(msg);
}
#endif

/* Mmio: every access is volatile and of exactly its width. */

intnat caml_nx_mmio_get8(intnat a) { return (intnat)(*(volatile uint8_t *)a); }
value caml_nx_mmio_get8_byte(value a) {
  return Val_long(caml_nx_mmio_get8(Nativeint_val(a)));
}

value caml_nx_mmio_set8(intnat a, intnat v) {
  *(volatile uint8_t *)a = (uint8_t)v;
  return Val_unit;
}
value caml_nx_mmio_set8_byte(value a, value v) {
  return caml_nx_mmio_set8(Nativeint_val(a), Long_val(v));
}

intnat caml_nx_mmio_get32(intnat a) {
  return (intnat)(*(volatile uint32_t *)a);
}
value caml_nx_mmio_get32_byte(value a) {
  return Val_long(caml_nx_mmio_get32(Nativeint_val(a)));
}

value caml_nx_mmio_set32(intnat a, intnat v) {
  *(volatile uint32_t *)a = (uint32_t)v;
  return Val_unit;
}
value caml_nx_mmio_set32_byte(value a, value v) {
  return caml_nx_mmio_set32(Nativeint_val(a), Long_val(v));
}

int64_t caml_nx_mmio_get64(intnat a) {
  return (int64_t)(*(volatile uint64_t *)a);
}
value caml_nx_mmio_get64_byte(value a) {
  return caml_copy_int64(caml_nx_mmio_get64(Nativeint_val(a)));
}

value caml_nx_mmio_set64(intnat a, int64_t v) {
  *(volatile uint64_t *)a = (uint64_t)v;
  return Val_unit;
}
value caml_nx_mmio_set64_byte(value a, value v) {
  return caml_nx_mmio_set64(Nativeint_val(a), Int64_val(v));
}

/* Bulk accesses go a 32-bit word at a time wherever the mapped side is
   aligned: device memory behind a BAR need not accept wider or narrower
   accesses the C library's copies would make. The process's side takes any
   alignment. */
static void read_words(uint8_t *dst, const volatile uint8_t *src, size_t n) {
  size_t i = 0;
  for (; i < n && ((uintptr_t)(src + i) & 3); i++) dst[i] = src[i];
  for (; i + 4 <= n; i += 4) {
    uint32_t w = *(const volatile uint32_t *)(src + i);
    memcpy(dst + i, &w, 4);
  }
  for (; i < n; i++) dst[i] = src[i];
}

static void write_words(volatile uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
  for (; i < n && ((uintptr_t)(dst + i) & 3); i++) dst[i] = src[i];
  for (; i + 4 <= n; i += 4) {
    uint32_t w;
    memcpy(&w, src + i, 4);
    *(volatile uint32_t *)(dst + i) = w;
  }
  for (; i < n; i++) dst[i] = src[i];
}

value caml_nx_mmio_read(value a, value n) {
  CAMLparam2(a, n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(n));
  read_words(Bytes_val(s), (const volatile uint8_t *)Nativeint_val(a),
             Long_val(n));
  CAMLreturn(s);
}

value caml_nx_mmio_write(value a, value s, value off, value n) {
  write_words((volatile uint8_t *)Nativeint_val(a),
              (const uint8_t *)String_val(s) + Long_val(off), Long_val(n));
  return Val_unit;
}

value caml_nx_mmio_fill(value a, value n, value c) {
  volatile uint8_t *p = (volatile uint8_t *)Nativeint_val(a);
  size_t len = Long_val(n), i = 0;
  uint8_t b = (uint8_t)Int_val(c);
  uint32_t w = b * 0x01010101u;
  while (i < len && ((uintptr_t)(p + i) & 3)) p[i++] = b;
  for (; i + 4 <= len; i += 4) *(volatile uint32_t *)(p + i) = w;
  for (; i < len; i++) p[i] = b;
  return Val_unit;
}

/* The bytes at an address as a bigarray that owns nothing. */
value caml_nx_mmio_bigarray(value a, value n) {
  return caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                            1, (void *)Nativeint_val(a), (intnat)Long_val(n));
}

/* On arm64 a fence orders memory only within the inner shareable domain; the
   stores to a BAR, write-combined or not, need the full-system barrier. */
value caml_nx_mmio_barrier(value unit) {
  (void)unit;
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
  return Val_unit;
}

/* Memory of the process */

value caml_nx_support_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  return Val_long(4096);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

#ifdef __linux__
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

/* Reserves [n] addresses at [base] with an inaccessible mapping. */
value caml_nx_sysmem_reserve(value base, value n) {
#ifdef __linux__
  void *want = (void *)Nativeint_val(base);
  void *p = mmap(want, Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                     MAP_FIXED_NOREPLACE,
                 -1, 0);
  if (p == MAP_FAILED) fail_errno("reserving the GPU's address range");
  if (p != want) {
    munmap(p, Long_val(n));
    caml_failwith("reserving the GPU's address range: the kernel placed it "
                  "elsewhere");
  }
  return Val_unit;
#else
  (void)base;
  (void)n;
  fail_linux("Reserving addresses for a GPU");
  return Val_unit;
#endif
}

/* Maps [n] bytes of shared and populated memory at [va], inside a
   reservation, or where the kernel chooses if [va] is 0, from a huge page if
   [huge], locked if [locked], and is their address. Populating takes time: the
   runtime is released. */
value caml_nx_sysmem_alloc(value va, value n, value huge, value locked) {
#ifdef __linux__
  void *at = (void *)Nativeint_val(va);
  int flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE |
              (Bool_val(locked) ? MAP_LOCKED : 0) | (at ? MAP_FIXED : 0) |
              (Bool_val(huge) ? MAP_HUGETLB : 0);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  void *p = mmap(at, len, PROT_READ | PROT_WRITE, flags, -1, 0);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  if (p == MAP_FAILED) fail_errno("allocating system memory");
  return caml_copy_nativeint((intnat)p);
#else
  (void)va;
  (void)n;
  (void)huge;
  (void)locked;
  fail_linux("Locked system memory");
  return Val_unit;
#endif
}

/* Unmaps [n] bytes at [va] that no reservation holds. */
value caml_nx_sysmem_unmap(value va, value n) {
#ifdef __linux__
  if (munmap((void *)Nativeint_val(va), Long_val(n)) != 0)
    fail_errno("releasing locked system memory");
  return Val_unit;
#else
  (void)va;
  (void)n;
  fail_linux("Locked system memory");
  return Val_unit;
#endif
}

/* Returns [n] bytes at [va] to their reservation. */
value caml_nx_sysmem_release(value va, value n) {
#ifdef __linux__
  void *p = mmap((void *)Nativeint_val(va), Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
  if (p == MAP_FAILED) fail_errno("releasing locked system memory");
  return Val_unit;
#else
  (void)va;
  (void)n;
  fail_linux("Locked system memory");
  return Val_unit;
#endif
}

/* Locking faults the memory in: the runtime is released. */
value caml_nx_sysmem_lock(value a, value n) {
#ifdef __linux__
  void *at = (void *)Nativeint_val(a);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  int r = mlock(at, len);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  if (r != 0)
    fail_errno("locking memory for a GPU (the locked-memory limit, "
               "ulimit -l, may be too low)");
  return Val_unit;
#else
  (void)a;
  (void)n;
  fail_linux("Locking memory");
  return Val_unit;
#endif
}

value caml_nx_sysmem_unlock(value a, value n) {
#ifdef __linux__
  if (munlock((void *)Nativeint_val(a), Long_val(n)) != 0)
    fail_errno("unlocking memory");
  return Val_unit;
#else
  (void)a;
  (void)n;
  fail_linux("Locking memory");
  return Val_unit;
#endif
}

/* The page-map entries of the [pages] pages from [va], as the kernel's
   [/proc/self/pagemap] gives them. The kernel walks the page tables for them:
   the runtime is released while it reads into a buffer of its own. */
value caml_nx_sysmem_pagemap(value va, value pages) {
  CAMLparam2(va, pages);
  CAMLlocal1(out);
#ifdef __linux__
  long page = sysconf(_SC_PAGESIZE);
  size_t n = Long_val(pages) * 8;
  off_t at = (off_t)((uintptr_t)Nativeint_val(va) / page) * 8;
  uint8_t *buf = malloc(n ? n : 1);
  if (buf == NULL) caml_raise_out_of_memory();
  const char *failed = NULL;
  int e = 0;
  caml_release_runtime_system();
  int fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    failed = "opening /proc/self/pagemap";
    e = errno;
  }
  for (size_t got = 0; !failed && got < n;) {
    ssize_t r = pread(fd, buf + got, n - got, at + got);
    if (r <= 0) {
      failed = "reading /proc/self/pagemap";
      e = r == 0 ? EIO : errno;
    } else
      got += r;
  }
  if (fd >= 0) close(fd);
  caml_acquire_runtime_system();
  if (failed) {
    free(buf);
    errno = e;
    fail_errno(failed);
  }
  out = caml_alloc_initialized_string(n, (const char *)buf);
  free(buf);
#else
  (void)pages;
  fail_linux("Reading physical addresses");
#endif
  CAMLreturn(out);
}

/* Files */

value caml_nx_file_open(value path, value write) {
  CAMLparam2(path, write);
#ifdef _WIN32
  caml_failwith("Opening device files needs a POSIX system");
  CAMLreturn(Val_unit);
#else
  int fd = open(String_val(path),
                (Bool_val(write) ? O_RDWR : O_RDONLY) | O_SYNC | O_CLOEXEC);
  if (fd < 0) {
    char msg[512];
    snprintf(msg, sizeof msg, "opening %s", String_val(path));
    fail_errno(msg);
  }
  CAMLreturn(Val_int(fd));
#endif
}

value caml_nx_file_close(value fd) {
#ifndef _WIN32
  close(Int_val(fd));
#endif
  return Val_unit;
}

value caml_nx_file_pread(value fd, value off, value n) {
  CAMLparam3(fd, off, n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(n));
#ifndef _WIN32
  ssize_t r = pread(Int_val(fd), Bytes_val(s), Long_val(n), Long_val(off));
  if (r != Long_val(n)) {
    if (r >= 0) errno = EIO;
    fail_errno("reading configuration space");
  }
#endif
  CAMLreturn(s);
}

value caml_nx_file_pwrite(value fd, value off, value s) {
#ifndef _WIN32
  ssize_t r = pwrite(Int_val(fd), String_val(s), caml_string_length(s),
                     Long_val(off));
  if (r != (ssize_t)caml_string_length(s)) {
    if (r >= 0) errno = EIO;
    fail_errno("writing configuration space");
  }
#else
  (void)fd;
  (void)off;
  (void)s;
#endif
  return Val_unit;
}

/* Maps [n] bytes of [fd] from [off], shared, not inherited by children. */
value caml_nx_file_map(value fd, value off, value n) {
#ifdef __linux__
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE, MAP_SHARED,
                 Int_val(fd), Long_val(off));
  if (p == MAP_FAILED) fail_errno("mapping a PCI BAR");
  madvise(p, Long_val(n), MADV_DONTFORK);
  return caml_copy_nativeint((intnat)p);
#else
  (void)fd;
  (void)off;
  (void)n;
  fail_linux("Mapping a PCI BAR");
  return Val_unit;
#endif
}

value caml_nx_file_unmap(value a, value n) {
#ifndef _WIN32
  munmap((void *)Nativeint_val(a), Long_val(n));
#else
  (void)a;
  (void)n;
#endif
  return Val_unit;
}

/* Takes an exclusive lock on the file at [path], creating it readable and
   writable by every user; [-1] if another process holds it. */
value caml_nx_file_lock(value path) {
  CAMLparam1(path);
#ifdef _WIN32
  caml_failwith("Locking a PCI function needs a POSIX system");
  CAMLreturn(Val_unit);
#else
  /* A link is never followed, and the mode of a file another process made is
     never changed: a lock file names nothing else. */
  int fd = open(String_val(path), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
                0644);
  if (fd < 0) fail_errno(String_val(path));
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    caml_failwith("the lock file is not a regular file");
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    close(fd);
    CAMLreturn(Val_int(-1));
  }
  CAMLreturn(Val_int(fd));
#endif
}

/* VFIO. The structures are encoded by the caller (vfio.ml); these stubs only
   open its files and make its requests, raising [Unix.Unix_error] with the
   errno so that the caller names what to change. */

value caml_nx_vfio_file(value path) {
  CAMLparam1(path);
#ifdef __linux__
  if (!caml_string_is_c_safe(path))
    caml_invalid_argument("a VFIO file name with a NUL byte");
  int fd = open(String_val(path), O_RDWR | O_CLOEXEC);
  if (fd < 0) caml_uerror("open", path);
  CAMLreturn(Val_int(fd));
#else
  (void)path;
  fail_linux("VFIO");
  CAMLreturn(Val_unit);
#endif
}

/* [ioctl(fd, request, arg)] for a request whose argument is a number. */
value caml_nx_vfio_ioctl(value fd, value request, value arg) {
#ifdef __linux__
  int r = ioctl(Int_val(fd), (unsigned long)Long_val(request),
                (unsigned long)Long_val(arg));
  if (r < 0) caml_uerror("ioctl", Nothing);
  return Val_int(r);
#else
  (void)fd;
  (void)request;
  (void)arg;
  fail_linux("VFIO");
  return Val_unit;
#endif
}

/* [ioctl(fd, request, b)] for a request whose argument is the structure [b],
   which the kernel may write back. Mapping memory pins it, which takes time:
   the runtime is released, over a copy of [b]. */
value caml_nx_vfio_ioctl_bytes(value fd, value request, value b) {
  CAMLparam3(fd, request, b);
#ifdef __linux__
  size_t n = caml_string_length(b);
  void *buf = malloc(n ? n : 1);
  if (buf == NULL) caml_raise_out_of_memory();
  memcpy(buf, Bytes_val(b), n);
  int f = Int_val(fd);
  unsigned long req = (unsigned long)Long_val(request);
  caml_release_runtime_system();
  int r = ioctl(f, req, buf);
  int e = errno;
  caml_acquire_runtime_system();
  memcpy(Bytes_val(b), buf, n);
  free(buf);
  if (r < 0) {
    errno = e;
    caml_uerror("ioctl", Nothing);
  }
  CAMLreturn(Val_int(r));
#else
  (void)fd;
  (void)request;
  (void)b;
  fail_linux("VFIO");
  CAMLreturn(Val_unit);
#endif
}

/* An eventfd an interrupt signals. */
value caml_nx_vfio_eventfd(value unit) {
  (void)unit;
#ifdef __linux__
  int fd = eventfd(0, EFD_CLOEXEC);
  if (fd < 0) caml_uerror("eventfd", Nothing);
  return Val_int(fd);
#else
  fail_linux("VFIO");
  return Val_unit;
#endif
}

/* Waits at most [ms] for the eventfd [efd], with the runtime released. */
value caml_nx_vfio_wait(value efd, value ms) {
#ifdef __linux__
  struct pollfd p = {.fd = Int_val(efd), .events = POLLIN};
  caml_release_runtime_system();
  int r = poll(&p, 1, Int_val(ms));
  uint64_t count;
  if (r > 0 && read(p.fd, &count, sizeof count) < 0) r = 0;
  caml_acquire_runtime_system();
  return Val_bool(r > 0);
#else
  (void)efd;
  (void)ms;
  return Val_false;
#endif
}

/* SHA-256 (FIPS 180-4) */

static const uint32_t sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha_block(uint32_t h[8], const uint8_t *p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
           (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
           g = h[6], k = h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) +
                  ((e & f) ^ (~e & g)) + sha_k[i] + w[i];
    uint32_t t2 =
        (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += k;
}

value caml_nx_sha256(value s) {
  CAMLparam1(s);
  CAMLlocal1(r);
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t n = caml_string_length(s), i = 0;
  const uint8_t *p = (const uint8_t *)String_val(s);
  for (; i + 64 <= n; i += 64) sha_block(h, p + i);
  uint8_t tail[128] = {0};
  size_t rest = n - i;
  memcpy(tail, p + i, rest);
  tail[rest] = 0x80;
  size_t len = rest + 9 <= 64 ? 64 : 128;
  uint64_t bits = (uint64_t)n * 8;
  for (int j = 0; j < 8; j++) tail[len - 1 - j] = (uint8_t)(bits >> (8 * j));
  sha_block(h, tail);
  if (len == 128) sha_block(h, tail + 64);
  r = caml_alloc_string(32);
  for (int j = 0; j < 8; j++)
    for (int b = 0; b < 4; b++)
      Bytes_val(r)[4 * j + b] = (uint8_t)(h[j] >> (24 - 8 * b));
  CAMLreturn(r);
}

/* Random bytes from the system's generator, for nonces. */
value caml_nx_random(value n) {
  CAMLparam1(n);
  CAMLlocal1(r);
  size_t len = Long_val(n);
  r = caml_alloc_string(len);
  unsigned char *p = Bytes_val(r);
#if defined(_WIN32)
  for (size_t i = 0; i < len; i++) {
    unsigned int w;
    if (rand_s(&w) != 0) caml_failwith("the system has no random bytes");
    p[i] = (unsigned char)w;
  }
#elif defined(__APPLE__)
  arc4random_buf(p, len);
#else
  for (size_t got = 0; got < len;) {
    size_t k = len - got < 256 ? len - got : 256;
    if (getentropy(p + got, k) != 0) fail_errno("reading random bytes");
    got += k;
  }
#endif
  CAMLreturn(r);
}

/* Keepalive probes on a connection after [idle] s without traffic, every
   [interval] s, [count] of them: a peer that answers none is gone, and so is
   one that acknowledges no data for as long. */
value caml_nx_keepalive(value fd, value idle, value interval, value count) {
#ifdef _WIN32
  struct tcp_keepalive k = {1, (ULONG)Int_val(idle) * 1000,
                            (ULONG)Int_val(interval) * 1000};
  DWORD n;
  if (WSAIoctl(Socket_val(fd), SIO_KEEPALIVE_VALS, &k, sizeof k, NULL,
               0, &n, NULL, NULL) != 0)
    caml_failwith("enabling keepalive");
#else
  int s = Int_val(fd), on = 1, i = Int_val(idle), v = Int_val(interval),
      c = Int_val(count);
  if (setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof on) != 0
#ifdef __APPLE__
      || setsockopt(s, IPPROTO_TCP, TCP_KEEPALIVE, &i, sizeof i) != 0
#else
      || setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &i, sizeof i) != 0
#endif
      || setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &v, sizeof v) != 0 ||
      setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &c, sizeof c) != 0)
    fail_errno("enabling keepalive");
#ifdef TCP_USER_TIMEOUT
  unsigned t = (unsigned)(i + v * c) * 1000;
  if (setsockopt(s, IPPROTO_TCP, TCP_USER_TIMEOUT, &t, sizeof t) != 0)
    fail_errno("bounding unacknowledged data");
#endif
#endif
  return Val_unit;
}

/* Memory of a host that a remote client uses: anonymous pages, [0] if the
   system has none. */
value caml_nx_host_alloc(value n) {
#ifdef _WIN32
  void *p = VirtualAlloc(NULL, Long_val(n), MEM_COMMIT | MEM_RESERVE,
                         PAGE_READWRITE);
  return caml_copy_nativeint((intnat)p);
#else
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return caml_copy_nativeint(p == MAP_FAILED ? 0 : (intnat)p);
#endif
}

value caml_nx_host_free(value a, value n) {
#ifdef _WIN32
  (void)n;
  VirtualFree((void *)Nativeint_val(a), 0, MEM_RELEASE);
#else
  munmap((void *)Nativeint_val(a), Long_val(n));
#endif
  return Val_unit;
}

/* Libraries the system may have, loaded when first needed. */

#ifndef _WIN32
static void *load_library(const char *const *names) {
  for (; *names; names++) {
    void *h = dlopen(*names, RTLD_NOW | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}
#endif

/* Decompression: zstd frames and xz streams, as distributions ship firmware.
   [None] if the library is missing; [Failure] if the data is corrupt. */

typedef unsigned long long (*zstd_size_t)(const void *, size_t);
typedef size_t (*zstd_decompress_t)(void *, size_t, const void *, size_t);
typedef unsigned (*zstd_is_error_t)(size_t);

static value some(value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, 0);
  Field(r, 0) = v;
  CAMLreturn(r);
}

value caml_nx_unzstd(value s) {
  CAMLparam1(s);
  CAMLlocal1(out);
#ifndef _WIN32
  static const char *const names[] = {"libzstd.so.1", "libzstd.so",
                                      "libzstd.1.dylib", "libzstd.dylib",
                                      NULL};
  void *h = load_library(names);
  if (!h) CAMLreturn(Val_none);
  zstd_size_t size = (zstd_size_t)dlsym(h, "ZSTD_getFrameContentSize");
  zstd_decompress_t dec = (zstd_decompress_t)dlsym(h, "ZSTD_decompress");
  zstd_is_error_t is_error = (zstd_is_error_t)dlsym(h, "ZSTD_isError");
  if (!size || !dec || !is_error) CAMLreturn(Val_none);
  unsigned long long n =
      size(String_val(s), caml_string_length(s));
  if (n >= (1ULL << 40)) caml_failwith("a corrupt or unsized zstd frame");
  out = caml_alloc_string(n);
  size_t r = dec(Bytes_val(out), n, String_val(s), caml_string_length(s));
  if (is_error(r) || r != n) caml_failwith("a corrupt zstd frame");
  CAMLreturn(some(out));
#else
  (void)s;
  CAMLreturn(Val_none);
#endif
}

typedef int (*lzma_decode_t)(uint64_t *, uint32_t, const void *,
                             const uint8_t *, size_t *, size_t, uint8_t *,
                             size_t *, size_t);

value caml_nx_unxz(value s) {
  CAMLparam1(s);
  CAMLlocal1(out);
#ifndef _WIN32
  static const char *const names[] = {"liblzma.so.5", "liblzma.so",
                                      "liblzma.5.dylib", "liblzma.dylib",
                                      NULL};
  void *h = load_library(names);
  if (!h) CAMLreturn(Val_none);
  lzma_decode_t dec = (lzma_decode_t)dlsym(h, "lzma_stream_buffer_decode");
  if (!dec) CAMLreturn(Val_none);
  size_t cap = caml_string_length(s) * 4 + 4096;
  for (;;) {
    uint8_t *buf = malloc(cap);
    if (!buf) caml_raise_out_of_memory();
    uint64_t limit = UINT64_MAX;
    size_t in_pos = 0, out_pos = 0;
    int r = dec(&limit, 0, NULL, (const uint8_t *)String_val(s), &in_pos,
                caml_string_length(s), buf, &out_pos, cap);
    if (r == 0 /* LZMA_OK */ || r == 1 /* LZMA_STREAM_END */) {
      out = caml_alloc_initialized_string(out_pos, (const char *)buf);
      free(buf);
      CAMLreturn(some(out));
    }
    free(buf);
    if (r != 10 /* LZMA_BUF_ERROR */) caml_failwith("a corrupt xz stream");
    cap *= 2;
  }
#else
  (void)s;
  CAMLreturn(Val_none);
#endif
}

/* HTTPS downloads through libcurl, with the runtime released. [Error why] if
   the library is missing or the transfer fails. */

typedef void *(*curl_init_t)(void);
typedef int (*curl_setopt_t)(void *, int, ...);
typedef int (*curl_perform_t)(void *);
typedef int (*curl_getinfo_t)(void *, int, ...);
typedef void (*curl_cleanup_t)(void *);
typedef const char *(*curl_strerror_t)(int);

struct sink {
  char *data;
  size_t len, cap;
};

static size_t sink_write(char *p, size_t size, size_t n, void *userdata) {
  struct sink *s = userdata;
  size_t k = size * n;
  if (s->len + k > s->cap) {
    size_t cap = s->cap ? s->cap : 1 << 20;
    while (cap < s->len + k) cap *= 2;
    char *d = realloc(s->data, cap);
    if (!d) return 0;
    s->data = d;
    s->cap = cap;
  }
  memcpy(s->data + s->len, p, k);
  s->len += k;
  return k;
}

static value result(int ok, value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, ok ? 0 : 1);
  Field(r, 0) = v;
  CAMLreturn(r);
}

value caml_nx_download(value url) {
  CAMLparam1(url);
  CAMLlocal1(v);
#ifndef _WIN32
  static const char *const names[] = {"libcurl.so.4", "libcurl.so",
                                      "libcurl-gnutls.so.4", "libcurl.4.dylib",
                                      "libcurl.dylib", NULL};
  void *h = load_library(names);
  if (!h)
    CAMLreturn(result(0, caml_copy_string("libcurl is not installed")));
  curl_init_t init = (curl_init_t)dlsym(h, "curl_easy_init");
  curl_setopt_t setopt = (curl_setopt_t)dlsym(h, "curl_easy_setopt");
  curl_perform_t perform = (curl_perform_t)dlsym(h, "curl_easy_perform");
  curl_getinfo_t getinfo = (curl_getinfo_t)dlsym(h, "curl_easy_getinfo");
  curl_cleanup_t cleanup = (curl_cleanup_t)dlsym(h, "curl_easy_cleanup");
  curl_strerror_t strerr = (curl_strerror_t)dlsym(h, "curl_easy_strerror");
  if (!init || !setopt || !perform || !getinfo || !cleanup || !strerr)
    CAMLreturn(result(0, caml_copy_string("libcurl lacks the easy interface")));
  char *u = strdup(String_val(url));
  if (!u) caml_raise_out_of_memory();
  struct sink s = {NULL, 0, 0};
  long status = 0;
  int rc;
  caml_release_runtime_system();
  void *c = init();
  if (!c) {
    rc = -1;
  } else {
    /* CURLOPT_URL, FOLLOWLOCATION, WRITEFUNCTION, WRITEDATA, FAILONERROR,
       CONNECTTIMEOUT, LOW_SPEED_LIMIT, LOW_SPEED_TIME,
       CURLINFO_RESPONSE_CODE */
    setopt(c, 10002, u);
    setopt(c, 52, 1L);
    setopt(c, 20011, sink_write);
    setopt(c, 10001, &s);
    setopt(c, 45, 1L);
    setopt(c, 78, 30L);
    setopt(c, 19, 1L);
    setopt(c, 20, 60L);
    rc = perform(c);
    getinfo(c, 0x200002, &status);
    cleanup(c);
  }
  caml_acquire_runtime_system();
  free(u);
  if (rc != 0) {
    free(s.data);
    char msg[512];
    snprintf(msg, sizeof msg, "%s (HTTP status %ld)",
             rc < 0 ? "libcurl failed to start" : strerr(rc), status);
    CAMLreturn(result(0, caml_copy_string(msg)));
  }
  v = caml_alloc_initialized_string(s.len, s.data ? s.data : "");
  free(s.data);
  CAMLreturn(result(1, v));
#else
  (void)url;
  CAMLreturn(result(0, caml_copy_string("downloads need a POSIX system")));
#endif
}
