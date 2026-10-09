/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The stores and loads a NIC shares with the process: the memory its rings
   live in, the doorbells, and the barriers that order them. The barriers are
   rdma-core's (util/udma_barrier.h): before the NIC reads what the host
   wrote, a store barrier for the NIC's view of host memory; after the host
   read a word the NIC wrote, a load barrier before the words behind it; and
   around a store to the NIC's registers, a barrier that flushes it.

   No stub allocates on the OCaml heap but [view], none releases the
   runtime: each is a handful of stores. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#define PTR(v) ((void *)Long_val(v))

#if defined(__x86_64__)
#define to_device() __asm__ volatile("" ::: "memory")
#define from_device() __asm__ volatile("lfence" ::: "memory")
#define flush_writes() __asm__ volatile("sfence" ::: "memory")
#elif defined(__aarch64__)
#define to_device() __asm__ volatile("dmb oshst" ::: "memory")
#define from_device() __asm__ volatile("dmb oshld" ::: "memory")
#define flush_writes() __asm__ volatile("dsb st" ::: "memory")
#else
#define to_device() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define from_device() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define flush_writes() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif

static uint32_t be32(uint32_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return __builtin_bswap32(v);
#else
  return v;
#endif
}

/* [v_n] bytes of zeroed memory at a page boundary, which a child process
   does not inherit, so that no copy on write moves a page the NIC reads:
   their address, or -errno. */
value caml_rig_mlx5_pages(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  if (p == MAP_FAILED) return Val_long(-errno);
#ifdef MADV_DONTFORK
  if (madvise(p, n, MADV_DONTFORK) != 0) {
    int e = errno;
    munmap(p, n);
    return Val_long(-e);
  }
#endif
  return Val_long((intnat)p);
}

value caml_rig_mlx5_free(value v_at, value v_n) {
  munmap(PTR(v_at), (size_t)Long_val(v_n));
  return Val_unit;
}

/* The [v_n] bytes at [v_at] as a bigarray that does not own them. */
value caml_rig_mlx5_view(value v_at, value v_n) {
  return caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                            1, PTR(v_at), (intnat)Long_val(v_n));
}

/* Rings a queue pair: the low 16 bits of the producer count [v_count] in
   its doorbell record at [v_record], after the entries; then the first 8
   bytes of the entry at [v_entry] in its doorbell register at [v_register],
   after the record, flushed. */
value caml_rig_mlx5_ring(value v_record, value v_count, value v_register,
                         value v_entry) {
  uint64_t first;
  to_device();
  *(volatile uint32_t *)PTR(v_record) =
      be32((uint32_t)Long_val(v_count) & 0xffff);
  flush_writes();
  memcpy(&first, PTR(v_entry), 8);
  *(volatile uint64_t *)PTR(v_register) = first;
  flush_writes();
  return Val_unit;
}

/* Stores the low 24 bits of a completion queue's consumer count [v_count] in
   its doorbell record at [v_record], after the reads of its completions. */
value caml_rig_mlx5_consumed(value v_record, value v_count) {
  to_device();
  *(volatile uint32_t *)PTR(v_record) =
      be32((uint32_t)Long_val(v_count) & 0xffffff);
  return Val_unit;
}

/* Arms a completion queue: the word [v_word] in its doorbell record's arm
   word at [v_record], then [v_word] and the queue's number [v_cq], both
   big-endian, in its doorbell register at [v_register], flushed. */
value caml_rig_mlx5_arm(value v_record, value v_register, value v_word,
                        value v_cq) {
  uint32_t words[2] = {be32((uint32_t)Long_val(v_word)),
                       be32((uint32_t)Long_val(v_cq))};
  uint64_t both;
  memcpy(&both, words, 8);
  *(volatile uint32_t *)PTR(v_record) = words[0];
  flush_writes();
  *(volatile uint64_t *)PTR(v_register) = both;
  flush_writes();
  return Val_unit;
}

/* Orders the loads after it after the loads before it, which read a word
   the NIC wrote. */
value caml_rig_mlx5_acquire(value unit) {
  (void)unit;
  from_device();
  return Val_unit;
}

/* A monotonic clock, in nanoseconds. */
value caml_rig_mlx5_now_ns(value unit) {
  struct timespec t;
  (void)unit;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}
