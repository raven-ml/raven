/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
/* Windows has no mmap. The flag values only round-trip through the OCaml side
   back into caml_tolk_hcq_mmap, which honors anonymous private mappings and
   nothing else there. */
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_SHARED 1
#define MAP_PRIVATE 2
#define MAP_ANON 0x20
#define MAP_FIXED 0x10
#define MAP_NORESERVE 0x4000
#else
#include <sys/mman.h>
#include <unistd.h>
/* MAP_ANON is the BSD spelling of MAP_ANONYMOUS; older systems may only define
   one. */
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif
/* MAP_NORESERVE is advisory; systems without it accept plain reservations. */
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
#endif

static void raise_errno(const char *what) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
  caml_failwith(buf);
}

/* Files and mappings */

CAMLprim value caml_tolk_hcq_constants(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_tuple(10);
  Store_field(v, 0, Val_int(O_RDONLY));
  Store_field(v, 1, Val_int(O_RDWR));
  Store_field(v, 2, Val_int(PROT_NONE));
  Store_field(v, 3, Val_int(PROT_READ));
  Store_field(v, 4, Val_int(PROT_WRITE));
  Store_field(v, 5, Val_int(MAP_SHARED));
  Store_field(v, 6, Val_int(MAP_PRIVATE));
  Store_field(v, 7, Val_int(MAP_ANON));
  Store_field(v, 8, Val_int(MAP_FIXED));
  Store_field(v, 9, Val_int(MAP_NORESERVE));
  CAMLreturn(v);
}

#if defined(_WIN32)

/* Opening is plain file I/O; what Windows lacks is the driver node behind the
   path and mmap, so only anonymous memory is mapped, which is what the queue
   builders and their golden tests need. */

CAMLprim value caml_tolk_hcq_open(value v_path, value v_flags) {
  CAMLparam2(v_path, v_flags);
  int fd =
      _open(String_val(v_path), Int_val(v_flags) | _O_BINARY | _O_NOINHERIT);
  if (fd < 0) raise_errno(String_val(v_path));
  CAMLreturn(Val_int(fd));
}

CAMLprim value caml_tolk_hcq_close(value v_fd) {
  CAMLparam1(v_fd);
  if (_close(Int_val(v_fd)) != 0) raise_errno("close");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_hcq_mmap(value v_addr, value v_size, value v_prot,
                                  value v_flags, value v_fd, value v_offset) {
  CAMLparam5(v_addr, v_size, v_prot, v_flags, v_fd);
  CAMLxparam1(v_offset);
  (void)v_prot;
  if ((Int_val(v_flags) & MAP_ANON) == 0 || Int_val(v_fd) != -1)
    caml_failwith("tolk hcq: file mappings are unsupported on Windows");
  void *hint = (Int_val(v_flags) & MAP_FIXED) ? (void *)Nativeint_val(v_addr)
                                              : NULL;
  void *p = VirtualAlloc(hint, (SIZE_T)Long_val(v_size),
                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) caml_failwith("mmap: VirtualAlloc failed");
  CAMLreturn(caml_copy_nativeint((intnat)p));
}

CAMLprim value caml_tolk_hcq_munmap(value v_addr, value v_size) {
  CAMLparam2(v_addr, v_size);
  if (!VirtualFree((void *)Nativeint_val(v_addr), 0, MEM_RELEASE))
    caml_failwith("munmap: VirtualFree failed");
  CAMLreturn(Val_unit);
}

#else /* !_WIN32 */

CAMLprim value caml_tolk_hcq_open(value v_path, value v_flags) {
  CAMLparam2(v_path, v_flags);
  int fd = open(String_val(v_path), Int_val(v_flags) | O_CLOEXEC);
  if (fd < 0) raise_errno(String_val(v_path));
  CAMLreturn(Val_int(fd));
}

CAMLprim value caml_tolk_hcq_close(value v_fd) {
  CAMLparam1(v_fd);
  if (close(Int_val(v_fd)) != 0) raise_errno("close");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_hcq_mmap(value v_addr, value v_size, value v_prot,
                                  value v_flags, value v_fd, value v_offset) {
  CAMLparam5(v_addr, v_size, v_prot, v_flags, v_fd);
  CAMLxparam1(v_offset);
  void *p = mmap((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size),
                 Int_val(v_prot), Int_val(v_flags), Int_val(v_fd),
                 (off_t)Int64_val(v_offset));
  if (p == MAP_FAILED) raise_errno("mmap");
  CAMLreturn(caml_copy_nativeint((intnat)p));
}

CAMLprim value caml_tolk_hcq_munmap(value v_addr, value v_size) {
  CAMLparam2(v_addr, v_size);
  if (munmap((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size)) != 0)
    raise_errno("munmap");
  CAMLreturn(Val_unit);
}

#endif /* _WIN32 */

CAMLprim value caml_tolk_hcq_mmap_bc(value *argv, int argn) {
  (void)argn;
  return caml_tolk_hcq_mmap(argv[0], argv[1], argv[2], argv[3], argv[4],
                            argv[5]);
}

/* Volatile access to mapped device memory. The copies must keep the OCaml
   runtime lock held: the bytes value may move under the GC otherwise. */

CAMLprim value caml_tolk_hcq_read32(value v_addr) {
  CAMLparam1(v_addr);
  volatile uint32_t *p = (volatile uint32_t *)Nativeint_val(v_addr);
  CAMLreturn(caml_copy_int32(*p));
}

CAMLprim value caml_tolk_hcq_write32(value v_addr, value v_v) {
  volatile uint32_t *p = (volatile uint32_t *)Nativeint_val(v_addr);
  *p = (uint32_t)Int32_val(v_v);
  return Val_unit;
}

CAMLprim value caml_tolk_hcq_read64(value v_addr) {
  CAMLparam1(v_addr);
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  CAMLreturn(caml_copy_int64(*p));
}

CAMLprim value caml_tolk_hcq_write64(value v_addr, value v_v) {
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  *p = (uint64_t)Int64_val(v_v);
  return Val_unit;
}

/* Unboxed 64-bit read for polling loops; Val_long drops the top bit. */
CAMLprim value caml_tolk_hcq_read64_int(value v_addr) {
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  return Val_long((intnat)*p);
}

CAMLprim value caml_tolk_hcq_fence(value unit) {
  (void)unit;
  atomic_thread_fence(memory_order_seq_cst);
  return Val_unit;
}

static uint64_t monotonic_ms(void) {
#if defined(_WIN32)
  return GetTickCount64();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

CAMLprim value caml_tolk_hcq_monotonic_ms(value unit) {
  (void)unit;
  return Val_long((intnat)monotonic_ms());
}

/* State is [deadline_ms, failed]. One device serializes its submissions.
   These ordinary C functions run inside generated host code with the OCaml
   runtime released. Failures suppress publication and are reported by OCaml
   synchronization; never unwind through the generated frame. */
static uint64_t tolk_hcq_poll(volatile uint64_t *state,
                              volatile uint64_t *timeline, uint64_t target) {
  if (state[1]) return UINT64_MAX;
  uint64_t value = timeline[0];
  atomic_thread_fence(memory_order_acquire);
  if (value < target && monotonic_ms() >= state[0]) {
    state[1] = 1;
    return UINT64_MAX;
  }
  return value;
}

static uint32_t tolk_hcq_reserve(volatile uint64_t *state,
                                volatile uint64_t *read_ptr, uint64_t put,
                                uint64_t needed, uint64_t capacity) {
  if (capacity < 2 || (capacity & (capacity - 1)) || needed >= capacity) {
    state[1] = 2;
    return 0;
  }
  if (state[1]) return 0;
  /* Hardware may report only a ring-relative read position. Keep one slot
     unused so full and empty cannot share the same pointer difference. */
  while (((*read_ptr - put - 1) & (capacity - 1)) < needed) {
    if (monotonic_ms() >= state[0]) { state[1] = 1; return 0; }
  }
  atomic_thread_fence(memory_order_acquire);
  return 1;
}

static void tolk_hcq_publish(volatile uint64_t *state,
                             volatile uint64_t *write_ptr,
                             volatile uint64_t *doorbell, uint64_t next,
                             uint64_t flush_addr, uint64_t lag) {
  if (state[1] || *write_ptr == next) return;
  atomic_thread_fence(memory_order_seq_cst);
  *write_ptr = next;
  atomic_thread_fence(memory_order_seq_cst);
  if (flush_addr) *(volatile uint32_t *)(uintptr_t)flush_addr = 0;
  atomic_thread_fence(memory_order_seq_cst);
  *doorbell = next - lag;
}

/* progress is [host submitted sequence, GPU completed low dword]. The
   capacity bound keeps the low-dword distance unambiguous across rollover. */
static uint64_t nv_completed(volatile uint64_t *progress) {
  uint64_t submitted = progress[0];
  uint32_t completed = *(volatile uint32_t *)(progress + 1);
  atomic_thread_fence(memory_order_acquire);
  return submitted - (uint32_t)((uint32_t)submitted - completed);
}

static void tolk_hcq_wait_progress(volatile uint64_t *state,
                                   volatile uint64_t *progress, uint64_t target) {
  if (state[1]) return;
  while (nv_completed(progress) < target) {
    if (monotonic_ms() >= state[0]) { state[1] = 1; return; }
  }
}

CAMLprim value caml_tolk_hcq_wait_progress(value state, value progress, value target) {
  CAMLparam3(state, progress, target);
  volatile uint64_t *s = (volatile uint64_t *)(uintptr_t)Nativeint_val(state);
  volatile uint64_t *p = (volatile uint64_t *)(uintptr_t)Nativeint_val(progress);
  uint64_t t = (uint64_t)Int64_val(target);
  caml_release_runtime_system();
  tolk_hcq_wait_progress(s, p, t);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

/* Ampere exposes GPPut but no GPGet. Each stream therefore retires a
   software sequence after its engine has finished using command storage. */
/* A GPFIFO entry holds the stream's address, word-aligned below 2^40, in
   bits 2-39, the fetch flag in bit 41 and the word count in bits 42-62. An
   address outside that field would spill into the count, so it fails. */
static void tolk_hcq_gpfifo(volatile uint64_t *state, volatile uint64_t *ring,
                            volatile uint32_t *put, volatile uint32_t *doorbell,
                            uint64_t addr, uint64_t words, uint32_t token,
                            uint32_t capacity, volatile uint64_t *progress,
                            volatile uint64_t *retired) {
  if (state[1]) return;
  if (capacity < 2 || (capacity & (capacity - 1))) { state[1] = 2; return; }
  if ((addr & 3) || (addr >> 40) || (words >> 21)) { state[1] = 3; return; }
  uint64_t entry = addr | (words << 42) | (1ULL << 41);
  uint64_t next = progress[0] + 1;
  if (next >= capacity) tolk_hcq_wait_progress(state, progress, next - capacity + 1);
  if (state[1]) return;
  uint32_t p = *put;
  ring[p & (capacity - 1)] = entry;
  progress[0] = next;
  retired[0] = next;
  atomic_thread_fence(memory_order_seq_cst);
  *put = (p + 1) & (capacity - 1);
  atomic_thread_fence(memory_order_seq_cst);
  *doorbell = token;
}

CAMLprim value caml_tolk_hcq_submission_symbol(value name) {
  CAMLparam1(name);
  void *symbol;
  if (!strcmp(String_val(name), "tolk_hcq_poll")) symbol = (void *)tolk_hcq_poll;
  else if (!strcmp(String_val(name), "tolk_hcq_reserve")) symbol = (void *)tolk_hcq_reserve;
  else if (!strcmp(String_val(name), "tolk_hcq_publish")) symbol = (void *)tolk_hcq_publish;
  else if (!strcmp(String_val(name), "tolk_hcq_wait_progress")) symbol = (void *)tolk_hcq_wait_progress;
  else if (!strcmp(String_val(name), "tolk_hcq_gpfifo")) symbol = (void *)tolk_hcq_gpfifo;
  else caml_invalid_argument("unknown HCQ submission helper");
  CAMLreturn(caml_copy_nativeint((intnat)symbol));
}

CAMLprim value caml_tolk_hcq_memcpy_to_ptr(value v_dst, value v_src,
                                           value v_src_off, value v_len) {
  memcpy((void *)Nativeint_val(v_dst), Bytes_val(v_src) + Long_val(v_src_off),
         (size_t)Long_val(v_len));
  return Val_unit;
}

CAMLprim value caml_tolk_hcq_memcpy_from_ptr(value v_dst, value v_dst_off,
                                             value v_src, value v_len) {
  memcpy(Bytes_val(v_dst) + Long_val(v_dst_off), (void *)Nativeint_val(v_src),
         (size_t)Long_val(v_len));
  return Val_unit;
}
