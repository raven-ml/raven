/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nx_device.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <pthread.h>
#endif

/* Release lists

   A resource a device must release once nothing reaches it has a token, a
   custom block that the values using the resource hold. The token's node
   keeps the resource's record, an OCaml value, as a generational global root.
   When the collector frees the token, in whichever domain sweeps it, its
   finaliser links the node onto its device's list, with no allocation and no
   lock. The device takes the list at its next operation and releases the
   records it holds. */

struct nx_release_node {
  struct nx_release_node *next;
  value record;
};

struct nx_release_list {
  _Atomic(struct nx_release_node *) head;
};

struct nx_token {
  struct nx_release_list *list;
  struct nx_release_node *node;
};

/* A new, empty release list, for a device: never freed. */
value caml_nx_device_release_list(value unit) {
  (void)unit;
  struct nx_release_list *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  atomic_init(&l->head, NULL);
  return caml_copy_nativeint((intnat)l);
}

static void token_finalize(value v) {
  struct nx_token *t = Data_custom_val(v);
  struct nx_release_node *n = t->node;
  struct nx_release_node *head =
      atomic_load_explicit(&t->list->head, memory_order_relaxed);
  do
    n->next = head;
  while (!atomic_compare_exchange_weak_explicit(
      &t->list->head, &head, n, memory_order_release, memory_order_relaxed));
}

static struct custom_operations token_ops = {
    "nx.device.token",          token_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A token that puts [v_record] on the list [v_list] once it is collected. The
   node is made here, so that the finaliser allocates nothing. */
value caml_nx_device_token(value v_list, value v_record) {
  CAMLparam2(v_list, v_record);
  CAMLlocal1(v);
  struct nx_release_node *n = malloc(sizeof *n);
  if (n == NULL) caml_raise_out_of_memory();
  v = caml_alloc_custom(&token_ops, sizeof(struct nx_token), 0, 1);
  n->next = NULL;
  n->record = v_record;
  caml_register_generational_global_root(&n->record);
  struct nx_token *t = Data_custom_val(v);
  t->list = (struct nx_release_list *)Nativeint_val(v_list);
  t->node = n;
  CAMLreturn(v);
}

/* The records the list [v_list] holds, which it no longer does. */
value caml_nx_device_released(value v_list) {
  CAMLparam1(v_list);
  CAMLlocal2(l, cell);
  struct nx_release_list *rl =
      (struct nx_release_list *)Nativeint_val(v_list);
  struct nx_release_node *n =
      atomic_exchange_explicit(&rl->head, NULL, memory_order_acquire);
  l = Val_emptylist;
  while (n != NULL) {
    cell = caml_alloc(2, 0);
    Store_field(cell, 0, n->record);
    Store_field(cell, 1, l);
    l = cell;
    caml_remove_generational_global_root(&n->record);
    struct nx_release_node *next = n->next;
    free(n);
    n = next;
  }
  CAMLreturn(l);
}

/* Copies at least this large release the runtime while they run. */
#define NX_DEVICE_BLOCKING_BYTES (1 << 16)

/* The bytes of the host's heap that live buffers hold. A buffer reserves its
   bytes against the host's budget and holds a token, a custom block whose
   finaliser returns them: the collector runs it where it frees the block,
   with no OCaml function to call. */
static _Atomic intnat heap_bytes;

intnat caml_nx_device_heap_bytes(value unit) {
  (void)unit;
  return atomic_load_explicit(&heap_bytes, memory_order_relaxed);
}

value caml_nx_device_heap_bytes_byte(value unit) {
  return Val_long(caml_nx_device_heap_bytes(unit));
}

static void heap_drop_all(void);
static intnat heap_kept(void);

/* A reservation counts the buffers the allocator keeps for reuse (see below)
   against the budget too, and gives them back when they are what stands in its
   way. */
value caml_nx_device_heap_reserve(intnat n, intnat budget) {
  intnat held = atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  do {
    if (n > budget - held - heap_kept()) {
      if (heap_kept() == 0 || n > budget - held) return Val_false;
      heap_drop_all();
    }
  } while (!atomic_compare_exchange_weak_explicit(
      &heap_bytes, &held, held + n, memory_order_relaxed,
      memory_order_relaxed));
  return Val_true;
}

value caml_nx_device_heap_reserve_byte(value n, value budget) {
  return caml_nx_device_heap_reserve(Long_val(n), Long_val(budget));
}

value caml_nx_device_heap_return(intnat n) {
  atomic_fetch_sub_explicit(&heap_bytes, n, memory_order_relaxed);
  return Val_unit;
}

value caml_nx_device_heap_return_byte(value n) {
  return caml_nx_device_heap_return(Long_val(n));
}

/* The bytes the collector has returned, ever, of buffers made before the last
   major cycle ended, and the number of major cycles ended. A token holds its
   bytes and the cycles ended when it was made. */
static _Atomic intnat heap_collected;
static _Atomic intnat heap_epoch;

static void heap_token_finalize(value v) {
  intnat n = ((intnat *)Data_custom_val(v))[0];
  intnat e = ((intnat *)Data_custom_val(v))[1];
  caml_nx_device_heap_return(n);
  if (e < atomic_load_explicit(&heap_epoch, memory_order_relaxed))
    atomic_fetch_add_explicit(&heap_collected, n, memory_order_relaxed);
}

static struct custom_operations heap_token_ops = {
    "nx.device.heap_token",     heap_token_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A token that returns [v_n] reserved bytes once it is collected. */
value caml_nx_device_heap_token(value v_n) {
  value v = caml_alloc_custom(&heap_token_ops, 2 * sizeof(intnat), 0, 1);
  ((intnat *)Data_custom_val(v))[0] = Long_val(v_n);
  ((intnat *)Data_custom_val(v))[1] =
      atomic_load_explicit(&heap_epoch, memory_order_relaxed);
  return v;
}

/* Pacing the collector

   The runtime reclaims a buffer's memory when it collects the bigarray that
   owns it, and paces its major cycles by the memory such blocks hold outside
   the heap: a cycle is due once that much memory has reached the major heap,
   allocated since the last one, as [caml_custom_get_max_major] gives. That is
   a share of the major heap, which bounds the garbage floating outside the heap
   by a share of the heap. A program whose memory is mostly buffers would then
   collect once per buffer as large as that share of its small heap, and mark
   the whole heap each time. The share is taken here of the program's whole
   memory instead: the major heap and the bytes buffers hold live, so that the
   garbage floating outside the heap stays the same share of what the program
   holds.

   The bytes held live are those held at the end of a major cycle that the next
   cycle did not collect. A cycle frees the garbage the previous cycle found, as
   it sweeps it, and the blocks it allocates survive it. Minor collections also
   free buffers that died young, made after the previous cycle ended, which it
   never held: only the buffers made before it ended count as collected. */

static _Atomic intnat heap_live;
static _Atomic intnat held_at_cycle;
static _Atomic intnat collected_at_cycle;

static void heap_trim(void);

/* Called at the end of each major cycle. */
value caml_nx_device_heap_cycle(value unit) {
  (void)unit;
  intnat held = atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  intnat collected =
      atomic_load_explicit(&heap_collected, memory_order_relaxed);
  intnat live =
      atomic_exchange_explicit(&held_at_cycle, held, memory_order_relaxed) -
      (collected - atomic_exchange_explicit(&collected_at_cycle, collected,
                                            memory_order_relaxed));
  atomic_store_explicit(&heap_live, live > 0 ? live : 0, memory_order_relaxed);
  atomic_fetch_add_explicit(&heap_epoch, 1, memory_order_relaxed);
  heap_trim();
  return Val_unit;
}

/* The memory a major cycle is due after: [caml_custom_get_max_major] is the
   major heap's bytes over 150, times the ratio; the same share of the bytes
   buffers hold live is added. */
static mlsize_t heap_cycle_bytes(void) {
  mlsize_t live =
      (mlsize_t)atomic_load_explicit(&heap_live, memory_order_relaxed);
  return caml_custom_get_max_major() +
         live / 150 *
             atomic_load_explicit(&caml_custom_major_ratio,
                                  memory_order_relaxed);
}

/* Reusing freed buffers

   A buffer's memory goes back to the C library when the collector frees it,
   and the library may return it to the system: glibc trims the top of its heap
   and unmaps large blocks, so the next buffer of that size faults its pages in
   again, and an eager operation, which makes a fresh output each time, runs at
   half the speed or less. Freed buffers are kept instead, each under its exact
   size, and the next allocation of that size takes the one freed last. They
   are kept while they hold no more than a major cycle's share of the program's
   memory, as [heap_cycle_bytes] gives it, the garbage a cycle may already leave
   floating, and at least [HEAP_CACHE_FLOOR]; past it the least recently freed
   go back to the library, and an allocation that fails gives them all back
   before trying again.

   A kept buffer holds its own links in its first bytes. A spin lock guards the
   list: the collector frees buffers on any domain, and holding the lock costs
   a few pointer writes. */

#define HEAP_CACHE_FLOOR ((size_t)32 << 20)

struct heap_entry {
  struct heap_entry *newer, *older;
  size_t n;
};

static atomic_flag heap_cache_lock = ATOMIC_FLAG_INIT;
static struct heap_entry *heap_newest, *heap_oldest;
static _Atomic size_t heap_cached;

static void heap_cache_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&heap_cache_lock,
                                           memory_order_acquire)) {
  }
}

static void heap_cache_release(void) {
  atomic_flag_clear_explicit(&heap_cache_lock, memory_order_release);
}

static void heap_unlink(struct heap_entry *e) {
  if (e->newer) e->newer->older = e->older;
  else heap_newest = e->older;
  if (e->older) e->older->newer = e->newer;
  else heap_oldest = e->newer;
  heap_cached -= e->n;
}

static mlsize_t heap_cycle_bytes(void);

/* The most the cache holds now. */
static size_t heap_cache_cap(void) {
  size_t cap = heap_cycle_bytes();
  return cap < HEAP_CACHE_FLOOR ? HEAP_CACHE_FLOOR : cap;
}

/* Unlinks the least recently kept buffers until the cache holds at most [cap]
   bytes, and is them, linked by [older]. The lock is held. */
static struct heap_entry *heap_over(size_t cap) {
  struct heap_entry *dropped = NULL;
  while (heap_cached > cap) {
    struct heap_entry *old = heap_oldest;
    heap_unlink(old);
    old->older = dropped;
    dropped = old;
  }
  return dropped;
}

static void heap_free_list(struct heap_entry *e) {
  while (e) {
    struct heap_entry *next = e->older;
    free(e);
    e = next;
  }
}

/* Keeps the [n] bytes at [data], or gives back the least recently kept. */
static void heap_keep(void *data, size_t n) {
  size_t cap = heap_cache_cap();
  struct heap_entry *e = data;
  heap_cache_acquire();
  e->n = n;
  e->older = heap_newest;
  e->newer = NULL;
  if (heap_newest) heap_newest->newer = e;
  else heap_oldest = e;
  heap_newest = e;
  heap_cached += n;
  struct heap_entry *dropped = heap_over(cap);
  heap_cache_release();
  heap_free_list(dropped);
}

/* Gives back the least recently kept buffers until the cache holds at most
   what it may now: run as a major cycle ends, so that a program that stops
   freeing buffers does not keep the share of a working set it dropped. */
static void heap_trim(void) {
  size_t cap = heap_cache_cap();
  heap_cache_acquire();
  struct heap_entry *dropped = heap_over(cap);
  heap_cache_release();
  heap_free_list(dropped);
}

/* The most recently kept buffer of exactly [n] bytes, or NULL. */
static void *heap_take(size_t n) {
  heap_cache_acquire();
  struct heap_entry *e = heap_newest;
  while (e && e->n != n) e = e->older;
  if (e) heap_unlink(e);
  heap_cache_release();
  return e;
}

static intnat heap_kept(void) {
  return (intnat)atomic_load_explicit(&heap_cached, memory_order_relaxed);
}

intnat caml_nx_device_heap_cached(value unit) {
  (void)unit;
  return heap_kept();
}

value caml_nx_device_heap_cached_byte(value unit) {
  (void)unit;
  return Val_long(heap_kept());
}

/* Gives every kept buffer back to the C library. */
static void heap_drop_all(void) {
  heap_cache_acquire();
  struct heap_entry *e = heap_newest;
  heap_newest = heap_oldest = NULL;
  heap_cached = 0;
  heap_cache_release();
  heap_free_list(e);
}

/* The runtime's operations of bigarrays. The runtime exports them but declares
   them only to itself (CAML_INTERNALS): a bigarray's block must be made here
   with [caml_alloc_custom], which takes the memory a cycle is due after, where
   [caml_ba_alloc] takes the runtime's. The block's operations are the
   runtime's with one change, its finaliser, which keeps the memory where the
   runtime's frees it: the block is a bigarray like any other, which the
   bigarray functions handle, as nx's tests of it check. A sub of it shares a
   proxy with it, as the runtime makes one, and whichever of them is collected
   last releases the memory: the block keeps it, a sub frees it. */
extern const struct custom_operations caml_ba_ops;

static void heap_finalize(value v) {
  struct caml_ba_array *b = Caml_ba_array_val(v);
  size_t n = (size_t)b->dim[0];
  if (b->proxy == NULL) heap_keep(b->data, n);
  else if (atomic_fetch_sub(&b->proxy->refcount, 1) == 1) {
    heap_keep(b->proxy->data, n);
    free(b->proxy);
  }
}

static struct custom_operations heap_ops;

/* Builds the operations, once, as the module initializes. */
value caml_nx_device_heap_init(value unit) {
  (void)unit;
  heap_ops = caml_ba_ops;
  heap_ops.finalize = heap_finalize;
  return Val_unit;
}

/* The [n] bytes at [data], from [malloc], as a [char] bigarray that keeps
   them once collected. */
static value heap_bigarray(void *data, size_t n) {
  value ba = caml_alloc_custom(&heap_ops,
                               SIZEOF_BA_ARRAY + sizeof(intnat), n,
                               heap_cycle_bytes());
  struct caml_ba_array *b = Caml_ba_array_val(ba);
  b->data = data;
  b->num_dims = 1;
  b->flags = CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED;
  b->proxy = NULL;
  b->dim[0] = (intnat)n;
  return ba;
}

/* [v_n] bytes of the heap, as a [char] bigarray that keeps them once
   collected. */
value caml_nx_device_heap_alloc(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *data = heap_take(n);
  if (data == NULL) data = malloc(n);
  if (data == NULL) {
    heap_drop_all();
    data = malloc(n);
  }
  if (data == NULL) caml_raise_out_of_memory();
  return heap_bigarray(data, n);
}

value caml_nx_device_heap_drop(value unit) {
  (void)unit;
  heap_drop_all();
  return Val_unit;
}

/* [v_n] bytes of the heap on a page, as a [char] bigarray that keeps them once
   collected, or [None] where the C library aligns nothing that [free]
   releases (Windows). A platform takes its buffers from this or from
   [caml_nx_device_heap_alloc], never both, so a kept buffer has the alignment
   of the ones it serves. */
value caml_nx_device_heap_aligned(value v_page, value v_n) {
  CAMLparam2(v_page, v_n);
#ifdef _WIN32
  (void)v_page;
  (void)v_n;
  CAMLreturn(Val_none);
#else
  size_t n = (size_t)Long_val(v_n), page = (size_t)Long_val(v_page);
  void *data = heap_take(n);
  if (data == NULL && posix_memalign(&data, page, n) != 0) {
    heap_drop_all();
    if (posix_memalign(&data, page, n) != 0) caml_raise_out_of_memory();
  }
  CAMLreturn(caml_alloc_some(heap_bigarray(data, n)));
#endif
}

/* The [v_n] bytes at [v_addr] as a [char] bigarray that owns nothing: memory a
   device holds, which the host addresses. */
value caml_nx_device_external_bytes(value v_addr, value v_n) {
  intnat dim = Long_val(v_n);
  return caml_ba_alloc(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1,
                       (void *)Nativeint_val(v_addr), &dim);
}

value caml_nx_device_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

intnat caml_nx_device_bigarray_address(value ba) {
  return (intnat)Caml_ba_data_val(ba);
}

value caml_nx_device_bigarray_address_byte(value ba) {
  return caml_copy_nativeint(caml_nx_device_bigarray_address(ba));
}

value caml_nx_device_memmove(intnat dst, intnat src, intnat n) {
  if (n >= NX_DEVICE_BLOCKING_BYTES) {
    caml_release_runtime_system();
    memmove((void *)dst, (const void *)src, (size_t)n);
    caml_acquire_runtime_system();
  } else {
    memmove((void *)dst, (const void *)src, (size_t)n);
  }
  return Val_unit;
}

value caml_nx_device_memmove_byte(value dst, value src, value n) {
  return caml_nx_device_memmove(Nativeint_val(dst), Nativeint_val(src),
                                Long_val(n));
}

extern value caml_ba_sub(value vb, value vofs, value vlen);

/* Gives the managed array [b] the proxy that its sub-arrays share, if it has
   none. The runtime makes it at the first sub without synchronization; here it
   is made at most once, whatever the domains that view [b] at once. Its first
   reference is [b]'s own, as the runtime counts it. */
static void ensure_proxy(struct caml_ba_array *b) {
  _Atomic(struct caml_ba_proxy *) *slot =
      (_Atomic(struct caml_ba_proxy *) *)&b->proxy;
  if ((b->flags & CAML_BA_MANAGED_MASK) == CAML_BA_EXTERNAL ||
      atomic_load_explicit(slot, memory_order_acquire) != NULL)
    return;
  struct caml_ba_proxy *proxy = malloc(sizeof *proxy);
  if (proxy == NULL) caml_raise_out_of_memory();
  atomic_store_explicit(&proxy->refcount, 1, memory_order_relaxed);
  proxy->data = b->data;
  proxy->size = b->flags & CAML_BA_MAPPED_FILE ? caml_ba_byte_size(b) : 0;
  struct caml_ba_proxy *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          slot, &none, proxy, memory_order_acq_rel, memory_order_acquire))
    free(proxy);
}

/* [v_len] elements of kind [v_kind] from byte [v_offset] of [v_src]. The
   header comes from [caml_ba_sub] over the whole of [v_src], so it joins
   [v_src]'s storage, which lives as long as any array over it. Its data,
   length and kind are rewritten, and flags the runtime does not define are
   cleared. */
value caml_nx_device_bigarray_view(value v_src, value v_kind, value v_offset,
                                   value v_len) {
  CAMLparam2(v_src, v_kind);
  CAMLlocal1(view);
  ensure_proxy(Caml_ba_array_val(v_src));
  view = caml_ba_sub(v_src, Val_long(0),
                     Val_long(Caml_ba_array_val(v_src)->dim[0]));
  struct caml_ba_array *b = Caml_ba_array_val(view);
  b->data = (char *)b->data + Long_val(v_offset);
  b->flags = (b->flags & (CAML_BA_LAYOUT_MASK | CAML_BA_MANAGED_MASK)) |
             Int_val(v_kind);
  b->dim[0] = Long_val(v_len);
  CAMLreturn(view);
}

/* The timeline's words are read and written by other threads and by devices:
   every access is atomic. */

int64_t caml_nx_device_load_u64(intnat addr) {
  return (int64_t)atomic_load_explicit((_Atomic uint64_t *)addr,
                                       memory_order_acquire);
}

value caml_nx_device_load_u64_byte(value addr) {
  return caml_copy_int64(caml_nx_device_load_u64(Nativeint_val(addr)));
}

value caml_nx_device_store_u64(intnat addr, int64_t v) {
  atomic_store_explicit((_Atomic uint64_t *)addr, (uint64_t)v,
                        memory_order_release);
  return Val_unit;
}

value caml_nx_device_store_u64_byte(value addr, value v) {
  return caml_nx_device_store_u64(Nativeint_val(addr), Int64_val(v));
}

static int64_t now_ms(void) {
#ifdef _WIN32
  return (int64_t)GetTickCount64();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

intnat caml_nx_device_now_ms(value unit) {
  (void)unit;
  return (intnat)now_ms();
}

value caml_nx_device_now_ms_byte(value unit) {
  return Val_long(caml_nx_device_now_ms(unit));
}

intnat caml_nx_device_now_ns(value unit) {
  (void)unit;
  return (intnat)nx_device_now_ns();
}

value caml_nx_device_now_ns_byte(value unit) {
  return Val_long(caml_nx_device_now_ns(unit));
}

static void yield(void) {
#ifdef _WIN32
  SwitchToThread();
#else
  sched_yield();
#endif
}

/* Waits until the word at [addr] reaches [v]. The timeout restarts whenever
   the word moves: a device that makes progress is not hung. */
intnat caml_nx_device_wait_u64(intnat addr, int64_t v, intnat timeout_ms) {
  _Atomic uint64_t *word = (_Atomic uint64_t *)addr;
  uint64_t target = (uint64_t)v;
  uint64_t seen = atomic_load_explicit(word, memory_order_acquire);
  if (seen >= target) return 1;
  int signaled = 0;
  caml_release_runtime_system();
  int64_t start = now_ms();
  for (;;) {
    uint64_t now = atomic_load_explicit(word, memory_order_acquire);
    if (now >= target) {
      signaled = 1;
      break;
    }
    if (now != seen) {
      seen = now;
      start = now_ms();
    } else if (now_ms() - start > timeout_ms) {
      break;
    }
    yield();
  }
  caml_acquire_runtime_system();
  return signaled;
}

value caml_nx_device_wait_u64_byte(value addr, value v, value timeout_ms) {
  return Val_long(caml_nx_device_wait_u64(Nativeint_val(addr), Int64_val(v),
                                          Long_val(timeout_ms)));
}

/* Host programs */

/* Code is never writable and executable at once. Elsewhere than arm64 macOS,
   it is mapped writable, filled, then made executable. arm64 macOS maps
   MAP_JIT memory, whose write protection each thread lifts for itself. */

static void code_fail(const char *what) {
  char msg[256];
#ifdef _WIN32
  snprintf(msg, sizeof msg, "Nx_device.Program.load: %s: error %lu", what,
           (unsigned long)GetLastError());
#else
  snprintf(msg, sizeof msg, "Nx_device.Program.load: %s: %s", what,
           strerror(errno));
#endif
  caml_failwith(msg);
}

value caml_nx_device_code_alloc(value v_size) {
  size_t size = (size_t)Long_val(v_size);
#ifdef _WIN32
  void *p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) code_fail("no executable memory");
#else
  int flags = MAP_PRIVATE | MAP_ANON;
  int prot = PROT_READ | PROT_WRITE;
#if defined(__APPLE__) && defined(__aarch64__)
  flags |= MAP_JIT;
  prot |= PROT_EXEC;
#endif
  void *p = mmap(NULL, size, prot, flags, -1, 0);
  if (p == MAP_FAILED) code_fail("no executable memory");
#endif
  return caml_copy_nativeint((intnat)p);
}

value caml_nx_device_code_install(value v_addr, value v_code) {
  char *p = (char *)Nativeint_val(v_addr);
  size_t n = caml_string_length(v_code);
#ifdef _WIN32
  DWORD old;
  memcpy(p, Bytes_val(v_code), n);
  if (!VirtualProtect(p, n, PAGE_EXECUTE_READ, &old))
    code_fail("cannot make the code executable");
  FlushInstructionCache(GetCurrentProcess(), p, n);
#else
#if defined(__APPLE__) && defined(__aarch64__)
  pthread_jit_write_protect_np(0);
  memcpy(p, Bytes_val(v_code), n);
  pthread_jit_write_protect_np(1);
#else
  memcpy(p, Bytes_val(v_code), n);
  if (mprotect(p, n, PROT_READ | PROT_EXEC) != 0)
    code_fail("cannot make the code executable");
#endif
  __builtin___clear_cache(p, p + n);
#endif
  return Val_unit;
}

value caml_nx_device_code_free(value v_addr, value v_size) {
  void *p = (void *)Nativeint_val(v_addr);
#ifdef _WIN32
  (void)v_size;
  VirtualFree(p, 0, MEM_RELEASE);
#else
  munmap(p, (size_t)Long_val(v_size));
#endif
  return Val_unit;
}

/* Compiler builtins. The compiler lowers a conversion the target cannot do
   inline, such as float32 to bfloat16 on x86_64 without AVX512-BF16, to a call
   into its runtime library, even where the source converts nothing: a 16-bit
   float merged across a branch is widened to float32 and rounded back. A
   loaded object has no such library, and a host's copy, if any, may take
   another calling convention: the object's is System V on x86_64, even on
   Windows. These are the host's copies, with the object's convention.

   A 16-bit float travels in the low bits of a floating-point register, where
   a float's bits start, so it is passed and returned as a float holding its
   bits. Rounding is to nearest, ties to even. A NaN keeps the high bits of its
   payload, and its lowest kept bit is set if a dropped bit was: it stays a
   NaN, and a widened value comes back with its bits. */

#if defined(_WIN32) && defined(__x86_64__)
#define NX_DEVICE_OBJECT_ABI __attribute__((sysv_abi))
#else
#define NX_DEVICE_OBJECT_ABI
#endif

static uint32_t float_bits(float x) {
  uint32_t b;
  memcpy(&b, &x, sizeof b);
  return b;
}

static float bits_float(uint32_t b) {
  float x;
  memcpy(&x, &b, sizeof x);
  return x;
}

static NX_DEVICE_OBJECT_ABI float truncsfbf2(float x) {
  uint32_t b = float_bits(x);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    b |= (b & 0xffffu) ? 0x10000u : 0;
  else
    b += 0x7fffu + ((b >> 16) & 1u);
  return bits_float(b >> 16);
}

static NX_DEVICE_OBJECT_ABI float truncsfhf2(float x) {
  uint32_t b = float_bits(x);
  uint32_t sign = (b >> 16) & 0x8000u, abs = b & 0x7fffffffu;
  uint32_t h;
  if (abs > 0x7f800000u) {
    h = 0x7c00u | ((abs >> 13) & 0x3ffu) | ((abs & 0x1fffu) ? 1u : 0);
  } else if (abs >= 0x477ff000u) {
    h = 0x7c00u;
  } else if (abs >= 0x38800000u) {
    uint32_t r = abs - 0x38000000u;
    h = (r + 0xfffu + ((r >> 13) & 1u)) >> 13;
  } else if (abs >= 0x33000000u) {
    uint32_t m = (abs & 0x7fffffu) | 0x800000u, shift = 126u - (abs >> 23);
    uint32_t rest = m & ((1u << shift) - 1u), tie = 1u << (shift - 1u);
    h = m >> shift;
    h += rest > tie || (rest == tie && (h & 1u));
  } else {
    h = 0;
  }
  return bits_float(sign | h);
}

static NX_DEVICE_OBJECT_ABI float extendhfsf2(float x) {
  uint32_t h = float_bits(x) & 0xffffu;
  uint32_t sign = (h & 0x8000u) << 16, m = h & 0x3ffu;
  int e = (h >> 10) & 0x1f;
  if (e == 0x1f) return bits_float(sign | 0x7f800000u | m << 13);
  if (e == 0 && m == 0) return bits_float(sign);
  if (e == 0) {
    for (e = 1; !(m & 0x400u); e--) m <<= 1;
    m &= 0x3ffu;
  }
  return bits_float(sign | (uint32_t)(e + 112) << 23 | m << 13);
}

static const struct {
  const char *name;
  NX_DEVICE_OBJECT_ABI float (*fn)(float);
} builtins[] = {
    {"__truncsfbf2", truncsfbf2},
    {"__truncsfhf2", truncsfhf2},
    {"__extendhfsf2", extendhfsf2},
};

/* The address of [name]: the host's copy of a compiler builtin, else the
   definition in the libraries the process loaded, which hold the C and math
   libraries, then in the compiler's runtime. [0] if none defines it. */
value caml_nx_device_symbol(value v_name) {
  const char *name = String_val(v_name);
  void *a = NULL;
  for (size_t i = 0; i < sizeof builtins / sizeof builtins[0]; i++)
    if (strcmp(name, builtins[i].name) == 0)
      return caml_copy_nativeint((intnat)builtins[i].fn);
#ifdef _WIN32
  HANDLE modules = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
  if (modules != INVALID_HANDLE_VALUE) {
    MODULEENTRY32 m;
    m.dwSize = sizeof m;
    for (BOOL more = Module32First(modules, &m); more && a == NULL;
         more = Module32Next(modules, &m))
      a = (void *)GetProcAddress(m.hModule, name);
    CloseHandle(modules);
  }
#else
  a = dlsym(RTLD_DEFAULT, name);
#ifdef __linux__
  /* Loads run with the host taken, one at a time. */
  static void *rt = NULL;
  if (a == NULL && rt == NULL) rt = dlopen("libgcc_s.so.1", RTLD_LAZY);
  if (a == NULL && rt != NULL) a = dlsym(rt, name);
#endif
#endif
  return caml_copy_nativeint((intnat)a);
}

/* Runs [f(buffers, values)] with the runtime released. The buffers' addresses
   and the values are read first, into memory the collector does not move: from
   Nx_device.Buffer.t values, or from (address, size) pairs when [addresses]. */
#define NX_DEVICE_CALL_WORDS 32

static value call(value v_entry, value v_buffers, value v_values,
                  int addresses) {
  CAMLparam3(v_entry, v_buffers, v_values);
  mlsize_t nb = Wosize_val(v_buffers), nv = Wosize_val(v_values);
  void *small_b[NX_DEVICE_CALL_WORDS];
  int64_t small_v[NX_DEVICE_CALL_WORDS];
  void **b = small_b;
  int64_t *v = small_v;
  if (nb > NX_DEVICE_CALL_WORDS) b = malloc(nb * sizeof *b);
  if (nv > NX_DEVICE_CALL_WORDS) v = malloc(nv * sizeof *v);
  if (b == NULL || v == NULL) {
    if (b != small_b) free(b);
    if (v != small_v) free(v);
    caml_raise_out_of_memory();
  }
  for (mlsize_t i = 0; i < nb; i++)
    b[i] = addresses
               ? (void *)Nativeint_val(Field(Field(v_buffers, i), 0))
               : nx_device_buffer_host(Field(v_buffers, i));
  for (mlsize_t i = 0; i < nv; i++) v[i] = (int64_t)Long_val(Field(v_values, i));
  void (*f)(void **, const int64_t *) =
      (void (*)(void **, const int64_t *))Nativeint_val(v_entry);
  caml_release_runtime_system();
  f(b, v);
  caml_acquire_runtime_system();
  if (b != small_b) free(b);
  if (v != small_v) free(v);
  CAMLreturn(Val_unit);
}

value caml_nx_device_call(value v_entry, value v_buffers, value v_values) {
  return call(v_entry, v_buffers, v_values, 0);
}

/* As [caml_nx_device_call], given each buffer as an (address, size) pair. */
value caml_nx_device_call_addresses(value v_entry, value v_buffers,
                                    value v_values) {
  return call(v_entry, v_buffers, v_values, 1);
}
