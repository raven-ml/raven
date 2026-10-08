/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's memory, the pace of the collector, and release lists. Only a
   copy of 64 KiB or more releases the runtime. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/domain_state.h>
#include <caml/fail.h>
#include <caml/gc_ctrl.h>
#include <caml/memory.h>
#include <caml/misc.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif

#include "device_core_stubs.h"

static intnat page_bytes(void) {
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return (intnat)info.dwPageSize;
#else
  return (intnat)sysconf(_SC_PAGESIZE);
#endif
}

value caml_device_core_page_size(value unit) {
  (void)unit;
  return Val_long(page_bytes());
}

/* The host clock: nanoseconds of the monotonic clock; on macOS, mach time,
   the time base of Metal's command buffer times. */
uint64_t dc_now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#elif defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

/* The host's instruction set, as its devices name architectures. */
value caml_device_core_arch(value unit) {
  (void)unit;
#if defined(__x86_64__) || defined(_M_X64)
  return caml_copy_string("x86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
  return caml_copy_string("arm64");
#else
  return caml_copy_string("");
#endif
}

value caml_device_core_now(value unit) {
  (void)unit;
  return Val_long((intnat)dc_now_ns());
}

/* Stores the host clock into the word at [word], for a device library's
   completion path: one aligned atomic store, no lock. */
void device_core_timestamp(void *word) {
  atomic_store_explicit((_Atomic uint64_t *)word, dc_now_ns(),
                        memory_order_release);
}

value caml_device_core_timestamp(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&device_core_timestamp);
}

/* Release lists

   A resource a device must release once nothing reaches it has a token, a
   custom block that the values using the resource hold. The token's node
   keeps the resource's release record, an OCaml value that does not reach
   the token, as a generational global root. When the collector frees the
   token, in whichever domain sweeps it, its finaliser links the node onto
   its device's list, with no allocation and no lock. The device takes the
   list at its next drain. */

struct release_node {
  struct release_node *next;
  value record;
};

struct release_list {
  _Atomic(struct release_node *) head;
};

struct token {
  struct release_list *list;
  struct release_node *node;
};

/* A new, empty release list, for a device: never freed. */
value caml_device_core_release_list(value unit) {
  (void)unit;
  struct release_list *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  atomic_init(&l->head, NULL);
  return Val_long((intnat)l);
}

static void token_finalize(value v) {
  struct token *t = Data_custom_val(v);
  struct release_node *n = t->node;
  struct release_node *head =
      atomic_load_explicit(&t->list->head, memory_order_relaxed);
  do
    n->next = head;
  while (!atomic_compare_exchange_weak_explicit(
      &t->list->head, &head, n, memory_order_release, memory_order_relaxed));
}

static struct custom_operations token_ops = {
    "device_core.token",        token_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The runtime's bound on the memory of a custom block allocated young. */
extern _Atomic uintnat caml_custom_minor_max_bsz;

/* The memory a major cycle is due after, for a token of [mem] bytes of a
   device with [room] bytes left in its budget: the share of the room the
   collector applies to its heap, so that the garbage floating in the
   device's memory stays a share of what the device could still allocate.
   Memory the host shares, of which [live] bytes are allocated ([live] < 0
   otherwise), is the program's own: there the share is at most that of the
   heap and those bytes, as for host buffers, since a share of a large room
   could leave gigabytes of garbage in the host's memory. It is at least a
   page, since a bound of 0 would mean the heap's, and at least the minor
   heap's bytes for a token the minor heap may hold, whose memory paces minor
   collections too: a nearly full device would otherwise run them once per few
   small buffers. */
static mlsize_t pace(mlsize_t mem, mlsize_t room, intnat live) {
  uintnat ratio =
      atomic_load_explicit(&caml_custom_major_ratio, memory_order_relaxed);
  mlsize_t max = room / 150 * ratio;
  if (live >= 0) {
    mlsize_t program = caml_custom_get_max_major() + (mlsize_t)live / 150 * ratio;
    if (program < max) max = program;
  }
  mlsize_t floor = (mlsize_t)page_bytes();
  if (max < floor) max = floor;
  if (mem <= atomic_load_explicit(&caml_custom_minor_max_bsz,
                                  memory_order_relaxed)) {
    mlsize_t minor = Bsize_wsize(Caml_state->minor_heap_wsz);
    if (max < minor) max = minor;
  }
  return max;
}

/* A token that puts [v_record] on the list [v_list] once it is collected,
   and paces the collector by the [v_mem] bytes it holds, out of [v_room] its
   device has left, with [v_live] for memory the host shares ([pace]). The
   node is made here, so that the finaliser allocates nothing. */
value caml_device_core_token(value v_list, value v_record, value v_mem,
                             value v_room, value v_live) {
  CAMLparam5(v_list, v_record, v_mem, v_room, v_live);
  CAMLlocal1(v);
  mlsize_t mem = (mlsize_t)Long_val(v_mem);
  struct release_node *n = malloc(sizeof *n);
  if (n == NULL) caml_raise_out_of_memory();
  v = caml_alloc_custom(&token_ops, sizeof(struct token), mem,
                        pace(mem, (mlsize_t)Long_val(v_room),
                             Long_val(v_live)));
  n->next = NULL;
  n->record = v_record;
  caml_register_generational_global_root(&n->record);
  struct token *t = Data_custom_val(v);
  t->list = (struct release_list *)Long_val(v_list);
  t->node = n;
  CAMLreturn(v);
}

/* The records the list [v_list] holds, which it no longer does. */
value caml_device_core_released(value v_list) {
  CAMLparam1(v_list);
  CAMLlocal2(l, cell);
  struct release_list *rl = (struct release_list *)Long_val(v_list);
  struct release_node *n =
      atomic_exchange_explicit(&rl->head, NULL, memory_order_acquire);
  l = Val_emptylist;
  while (n != NULL) {
    cell = caml_alloc(2, 0);
    Store_field(cell, 0, n->record);
    Store_field(cell, 1, l);
    l = cell;
    caml_remove_generational_global_root(&n->record);
    struct release_node *next = n->next;
    free(n);
    n = next;
  }
  CAMLreturn(l);
}

value caml_device_core_released_any(value v_list) {
  struct release_list *rl = (struct release_list *)Long_val(v_list);
  return Val_bool(atomic_load_explicit(&rl->head, memory_order_relaxed) !=
                  NULL);
}

/* The host's heap

   The bytes of the host's heap that live buffers hold. A buffer reserves its
   bytes against the host's budget and holds a token whose finaliser returns
   them. */
static _Atomic intnat heap_bytes;

static void heap_drop_all(void);
static intnat heap_kept(void);

/* A reservation counts the buffers kept for reuse (below) against the
   budget too, and gives them back when they are what stands in its way. */
value caml_device_core_heap_reserve(value v_n, value v_budget) {
  intnat n = Long_val(v_n), budget = Long_val(v_budget);
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

value caml_device_core_heap_return(value v_n) {
  atomic_fetch_sub_explicit(&heap_bytes, Long_val(v_n), memory_order_relaxed);
  return Val_unit;
}

/* The bytes the collector has returned, ever, of buffers made before the
   last major cycle ended, and the number of major cycles seen ended (see
   [slice_end]). A token holds its bytes and the cycles seen ended when it
   was made: a buffer made and dropped within one cycle was never held
   live. */
static _Atomic intnat heap_collected;
static _Atomic uintnat cycles_seen;

static void heap_token_finalize(value v) {
  intnat n = ((intnat *)Data_custom_val(v))[0];
  intnat e = ((intnat *)Data_custom_val(v))[1];
  atomic_fetch_sub_explicit(&heap_bytes, n, memory_order_relaxed);
  if ((uintnat)e < atomic_load_explicit(&cycles_seen, memory_order_relaxed))
    atomic_fetch_add_explicit(&heap_collected, n, memory_order_relaxed);
}

static struct custom_operations heap_token_ops = {
    "device_core.heap_token",   heap_token_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A token that returns [v_n] reserved bytes once it is collected. */
value caml_device_core_heap_token(value v_n) {
  value v = caml_alloc_custom(&heap_token_ops, 2 * sizeof(intnat), 0, 1);
  ((intnat *)Data_custom_val(v))[0] = Long_val(v_n);
  ((intnat *)Data_custom_val(v))[1] =
      (intnat)atomic_load_explicit(&cycles_seen, memory_order_relaxed);
  return v;
}

/* Pacing the collector

   The runtime paces major cycles by the memory custom blocks hold outside
   the heap: a cycle is due once that much has reached the major heap,
   allocated since the last one, as [caml_custom_get_max_major] gives. That
   is a share of the major heap, so a program whose memory is mostly buffers
   would collect once per buffer as large as that share of its small heap,
   and mark the whole heap each time. The share is taken here of the
   program's whole memory instead: the major heap and the bytes buffers hold
   live, so the garbage floating outside the heap stays the same share of
   what the program holds.

   The bytes held live are those held at the end of a major cycle that the
   next cycle did not collect. A cycle frees the garbage the previous cycle
   found, and the blocks it allocates survive it. Minor collections also free
   buffers that died young, made after the previous cycle ended, which it
   never held: only the buffers made before it ended count as collected. */

static _Atomic intnat heap_live;
static _Atomic intnat held_at_cycle;
static _Atomic intnat collected_at_cycle;

static void heap_trim(void);

/* Run once per major cycle, after it ends ([slice_end]). */
static void cycle_ended(void) {
  intnat held = atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  intnat collected =
      atomic_load_explicit(&heap_collected, memory_order_relaxed);
  intnat live =
      atomic_exchange_explicit(&held_at_cycle, held, memory_order_relaxed) -
      (collected - atomic_exchange_explicit(&collected_at_cycle, collected,
                                            memory_order_relaxed));
  atomic_store_explicit(&heap_live, live > 0 ? live : 0, memory_order_relaxed);
  heap_trim();
}

/* The end of each major slice, in whichever domain runs it, sees whether a
   major cycle ended since it last looked, and runs [cycle_ended] once if so.
   It runs C only, and the cache's lock it takes is never held across a poll
   point. */
/* The number of major cycles ended, which the runtime declares only to
   itself. */
extern uintnat caml_major_cycles_completed;

static caml_timing_hook previous_slice_end;

static void slice_end(void) {
  uintnat ended = caml_major_cycles_completed;
  uintnat seen = atomic_load_explicit(&cycles_seen, memory_order_relaxed);
  if (ended != seen &&
      atomic_compare_exchange_strong_explicit(&cycles_seen, &seen, ended,
                                              memory_order_relaxed,
                                              memory_order_relaxed))
    cycle_ended();
  if (previous_slice_end != NULL) previous_slice_end();
}

/* The memory a major cycle is due after: [caml_custom_get_max_major], the
   major heap's bytes over 150 times the ratio, plus the same share of the
   bytes buffers hold live. */
static mlsize_t heap_cycle_bytes(void) {
  mlsize_t live =
      (mlsize_t)atomic_load_explicit(&heap_live, memory_order_relaxed);
  return caml_custom_get_max_major() +
         live / 150 *
             atomic_load_explicit(&caml_custom_major_ratio,
                                  memory_order_relaxed);
}

value caml_device_core_heap_live(value unit) {
  (void)unit;
  return Val_long(atomic_load_explicit(&heap_live, memory_order_relaxed));
}

/* Reusing freed buffers

   A buffer's memory goes back to the C library when the collector frees it,
   and the library may return it to the system, so the next buffer of that
   size faults its pages in again. Freed buffers are kept instead, each under
   its exact size, and the next allocation of that size takes the one freed
   last. They are kept while they hold no more than a major cycle's share of
   the program's memory ([heap_cycle_bytes]), the garbage a cycle may leave
   floating anyway, and at least [HEAP_CACHE_FLOOR]; past it the least
   recently freed go back to the library, and an allocation that fails gives
   them all back before trying again. A kept buffer holds its links in its
   first bytes; a spin lock guards the list, held for a few pointer writes. */

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

static size_t heap_cache_cap(void) {
  size_t cap = heap_cycle_bytes();
  return cap < HEAP_CACHE_FLOOR ? HEAP_CACHE_FLOOR : cap;
}

/* Unlinks the least recently kept buffers until the cache holds at most
   [cap] bytes, and is them, linked by [older]. The lock is held. */
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

/* Gives back what the cache holds beyond what it may now: run as a major
   cycle ends, so a program that stops freeing buffers does not keep the
   share of a working set it dropped. */
static void heap_trim(void) {
  size_t cap = heap_cache_cap();
  heap_cache_acquire();
  struct heap_entry *dropped = heap_over(cap);
  heap_cache_release();
  heap_free_list(dropped);
}

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

value caml_device_core_heap_cached(value unit) {
  (void)unit;
  return Val_long(heap_kept());
}

static void heap_drop_all(void) {
  heap_cache_acquire();
  struct heap_entry *e = heap_newest;
  heap_newest = heap_oldest = NULL;
  heap_cached = 0;
  heap_cache_release();
  heap_free_list(e);
}

value caml_device_core_heap_drop(value unit) {
  (void)unit;
  heap_drop_all();
  return Val_unit;
}

/* The runtime's operations of bigarrays, which it exports but declares only
   to itself. A buffer's bigarray is made here with [caml_alloc_custom],
   which takes the memory a cycle is due after, and the runtime's operations
   with one change, a finaliser that keeps the memory where the runtime's
   frees it. A sub of it shares a proxy with it, and whichever is collected
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

/* Builds the operations and installs the slice hook, once, as the library
   initializes. */
value caml_device_core_heap_init(value unit) {
  (void)unit;
  heap_ops = caml_ba_ops;
  heap_ops.finalize = heap_finalize;
  previous_slice_end = atomic_exchange_explicit(
      &caml_major_slice_end_hook, slice_end, memory_order_relaxed);
  return Val_unit;
}

static value heap_bigarray(void *data, size_t n) {
  value ba = caml_alloc_custom(&heap_ops, SIZEOF_BA_ARRAY + sizeof(intnat),
                               n, heap_cycle_bytes());
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
value caml_device_core_heap_alloc(value v_n) {
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

/* [v_n] bytes of the heap on a page, or [None] where the C library aligns
   nothing that [free] releases (Windows). A platform takes its large buffers
   from this or from [caml_device_core_heap_alloc], never both, so a kept
   buffer has the alignment of the ones it serves. */
value caml_device_core_heap_aligned(value v_page, value v_n) {
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

/* Bigarrays over memory */

/* The [v_n] bytes at [v_addr] as a [char] bigarray that owns nothing:
   memory a device holds, which the host addresses. */
value caml_device_core_external_bytes(value v_addr, value v_n) {
  intnat dim = Long_val(v_n);
  return caml_ba_alloc(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1,
                       (void *)Long_val(v_addr), &dim);
}

value caml_device_core_bigarray_address(value ba) {
  return Val_long((intnat)Caml_ba_data_val(ba));
}

extern value caml_ba_sub(value vb, value vofs, value vlen);

/* Gives the managed array [b] the proxy its sub-arrays share, if it has
   none, at most once whatever the domains that view [b] at once. Its first
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

/* [v_len] elements of kind [v_kind] from byte [v_offset] of [v_src], over
   [v_src]'s storage, which lives as long as any array over it. */
value caml_device_core_bigarray_view(value v_src, value v_kind,
                                     value v_offset, value v_len) {
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

/* Copies [v_n] bytes between host addresses, releasing the runtime from
   64 KiB on. */
#define DC_BLOCKING_BYTES (1 << 16)

value caml_device_core_memmove(value v_dst, value v_src, value v_n) {
  void *dst = (void *)Long_val(v_dst);
  const void *src = (const void *)Long_val(v_src);
  size_t n = (size_t)Long_val(v_n);
  if (n >= DC_BLOCKING_BYTES) {
    caml_enter_blocking_section_no_pending();
    memmove(dst, src, n);
    caml_leave_blocking_section();
  } else
    memmove(dst, src, n);
  return Val_unit;
}
