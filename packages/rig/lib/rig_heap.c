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
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <stdio.h>
#endif

#include "rig_pool.h"
#include "rig_stubs.h"

size_t rig_page_bytes(void) {
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return (size_t)info.dwPageSize;
#else
  return (size_t)sysconf(_SC_PAGESIZE);
#endif
}

value caml_rig_page_size(value unit) {
  (void)unit;
  return Val_long((intnat)rig_page_bytes());
}

/* The host's instruction set, as its devices name architectures. */
value caml_rig_arch(value unit) {
  (void)unit;
#if defined(__x86_64__) || defined(_M_X64)
  return caml_copy_string("x86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
  return caml_copy_string("arm64");
#else
  return caml_copy_string("");
#endif
}

/* Release lists

   A resource a device must release once nothing reaches it has a token, a
   custom block that the values using the resource hold. The token's node
   keeps the resource's release record, an OCaml value that does not reach
   the token, as a generational global root. When the collector frees the
   token, in whichever domain sweeps it, its finaliser links the node onto
   its device's list, with no allocation and no lock. A drain routes the
   records one at a time from the list's hand, which it refills from the
   list, under the device's lock: a record leaves the hand in the section
   that stores it in its home, so an exception never drops one. A device's
   list lies in its C record, and counts the tokens made for it whose
   records its drains have not yet routed: the device keeps its slot, and so
   its record, while one remains. */

struct release_node {
  struct release_node *next;
  value record;
};

struct token {
  _Atomic(struct release_list *) list;
  struct release_node *node;
};

/* A new, empty release list of no device, never freed. */
value caml_rig_release_list(value unit) {
  (void)unit;
  struct release_list *l = calloc(1, sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)l);
}

/* Links the node of the token [t] onto its list, once: a token its list
   refused, never made or already linked has no list. The link takes the list
   in one exchange, so of two links on two domains one links the node. */
static void token_link(struct token *t) {
  struct release_list *l =
      atomic_exchange_explicit(&t->list, NULL, memory_order_relaxed);
  if (l == NULL) return;
  struct release_node *n = t->node;
  struct release_node *head =
      atomic_load_explicit(&l->head, memory_order_relaxed);
  do
    n->next = head;
  while (!atomic_compare_exchange_weak_explicit(
      &l->head, &head, n, memory_order_release, memory_order_relaxed));
}

static void token_finalize(value v) { token_link(Data_custom_val(v)); }

static struct custom_operations token_ops = {
    "rig.token",        token_finalize,
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
  mlsize_t floor = (mlsize_t)rig_page_bytes();
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
   device has left, with [v_live] for memory the host shares ([pace]); or
   [()] if the list is closed. The node is made here, so that the finaliser
   allocates nothing. */
value caml_rig_token(value v_list, value v_record, value v_mem,
                             value v_room, value v_live) {
  CAMLparam5(v_list, v_record, v_mem, v_room, v_live);
  CAMLlocal1(v);
  mlsize_t mem = (mlsize_t)Long_val(v_mem);
  struct release_list *l = (struct release_list *)Long_val(v_list);
  /* What may raise comes first, so a raise leaves the count as it was. */
  v = caml_alloc_custom(&token_ops, sizeof(struct token), mem,
                        pace(mem, (mlsize_t)Long_val(v_room),
                             Long_val(v_live)));
  struct token *t = Data_custom_val(v);
  atomic_init(&t->list, NULL);
  t->node = NULL;
  struct release_node *n = malloc(sizeof *n);
  if (n == NULL) caml_raise_out_of_memory();
  atomic_fetch_add(&l->count, 1);
  if (atomic_load(&l->closed)) {
    atomic_fetch_sub(&l->count, 1);
    free(n);
    CAMLreturn(Val_unit);
  }
  n->next = NULL;
  n->record = v_record;
  caml_register_generational_global_root(&n->record);
  t->node = n;
  atomic_store_explicit(&t->list, l, memory_order_relaxed);
  CAMLreturn(v);
}

/* Puts the record of the token [v_t] on its list now, as its collection
   would, which then does nothing; [()] is no token. Of several links of
   [v_t], on any domains, the first links it. */
value caml_rig_token_link(value v_t) {
  if (Is_block(v_t)) token_link(Data_custom_val(v_t));
  return Val_unit;
}

/* The node at the front of the hand of the list [v_list], taking the list
   into the hand when it is empty, or 0 for none. Under the device's lock. */
value caml_rig_release_front(value v_list) {
  struct release_list *rl = (struct release_list *)Long_val(v_list);
  struct release_node *n =
      atomic_load_explicit(&rl->hand, memory_order_relaxed);
  if (n == NULL) {
    n = atomic_exchange_explicit(&rl->head, NULL, memory_order_acquire);
    atomic_store_explicit(&rl->hand, n, memory_order_relaxed);
  }
  return Val_long((intnat)n);
}

/* The record of the node [v_n]. */
value caml_rig_node_record(value v_n) {
  return ((struct release_node *)Long_val(v_n))->record;
}

/* Frees the node at the front of the hand of the list [v_list], whose
   record was routed, and counts it routed. Under the device's lock. */
value caml_rig_release_pop(value v_list) {
  struct release_list *rl = (struct release_list *)Long_val(v_list);
  struct release_node *n =
      atomic_load_explicit(&rl->hand, memory_order_relaxed);
  atomic_store_explicit(&rl->hand, n->next, memory_order_relaxed);
  caml_remove_generational_global_root(&n->record);
  free(n);
  atomic_fetch_sub_explicit(&rl->count, 1, memory_order_release);
  return Val_unit;
}

value caml_rig_released_any(value v_list) {
  struct release_list *rl = (struct release_list *)Long_val(v_list);
  return Val_bool(atomic_load_explicit(&rl->head, memory_order_relaxed) !=
                      NULL ||
                  atomic_load_explicit(&rl->hand, memory_order_relaxed) !=
                      NULL);
}

/* The host's heap

   The bytes of the host's heap that live buffers hold. A buffer reserves its
   bytes against the host's budget, and the finaliser of its bigarray's block
   returns them. */
static _Atomic intnat heap_bytes;

/* The most bytes buffers held at once in the running major cycle, from what
   they held as it began, and in the cycle before it. */
static _Atomic intnat heap_peak, heap_peak_before;

static void heap_drop_all(void);
static intnat heap_kept(void);

/* A reservation counts the buffers kept for reuse (below) against the
   budget too, and gives them back when they are what stands in its way. */
value caml_rig_heap_reserve(value v_n, value v_budget) {
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
  intnat peak = atomic_load_explicit(&heap_peak, memory_order_relaxed);
  while (held + n > peak &&
         !atomic_compare_exchange_weak_explicit(&heap_peak, &peak, held + n,
                                                memory_order_relaxed,
                                                memory_order_relaxed)) {
  }
  return Val_true;
}

/* The bytes the collector has returned, ever, of buffers made before the
   last major cycle ended, and the number of major cycles seen ended (see
   [slice_end]). A buffer's block holds the cycles seen ended when it was
   made: a buffer made and dropped within one cycle was never held live. */
static _Atomic intnat heap_collected;
static _Atomic uintnat cycles_seen;

/* Pacing the collector

   The runtime paces major cycles by the memory custom blocks hold outside
   the heap: a block of [mem] bytes against a bound [max] adds [mem / max]
   of a cycle's work, at most a whole cycle, and that work is marking and
   sweeping the major heap (OCaml's runtime/memory.c, caml_adjust_gc_speed,
   and runtime/major_gc.c). Its own bound, [caml_custom_get_max_major], is
   [custom_major_ratio] / 150 of the heap's bytes, 29% by default, so that
   floating garbage stays a share of what the heap holds. A program whose
   memory is mostly buffers would then collect once per buffer as large as
   that share of its small heap, and mark the whole heap each time. The
   share is taken here of the program's whole memory instead: the major heap
   and the bytes buffers hold live, so the garbage floating outside the heap
   stays the same share of what the program holds. The live term matters
   where a program keeps its weights live and churns results: an Adam step
   of a 2048-wide MLP on the M1 Max ran 22 cycles over kaun's
   pointer-dense heap without it, against 8, and 7% slower.

   A program whose heap and live buffers are both small still has a share
   far below one result, and would run a cycle per result. A cycle is due
   instead after at least half the last-level cache ([heap_cycle_floor]):
   the results dropped over two cycles, which the cache below hands out
   again in turn, then fit in the last level. On fudge (Core Ultra 5 235),
   a dropped 4 MiB result took 97 us at a heap of 188k words and 46 with
   the floor; on the M1 Max, 255 and 181. Results larger than about a
   quarter of the last level still rotate past it: four 8 MiB results
   rotate through fudge's 24 MiB L3 where three did before.

   The bytes held live are those held at the end of a major cycle that the
   next cycle did not collect. A cycle frees the garbage the previous cycle
   found, and the blocks it allocates survive it. Minor collections also free
   buffers that died young, made after the previous cycle ended, which it
   never held: only the buffers made before it ended count as collected. */

static _Atomic intnat heap_live;
static _Atomic intnat held_at_cycle;
static _Atomic intnat collected_at_cycle;

static size_t heap_cache_cap(void);
static void heap_trim(size_t cap);

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
  heap_trim(heap_cache_cap());
  atomic_store_explicit(
      &heap_peak_before,
      atomic_exchange_explicit(&heap_peak, held, memory_order_relaxed),
      memory_order_relaxed);
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

/* Half the host's last-level cache, which [caml_rig_heap_init] reads once. */
static size_t heap_cycle_floor;

/* The bytes of the last-level cache the host reports, at least
   HEAP_LLC_LEAST. On Linux, the largest of CPU 0's caches
   (sysfs-devices-system-cpu, cache/indexN/size). On macOS, the L3
   (hw.l3cachesize) where there is one, as on Intel Macs, else the
   performance cores' L2 (hw.perflevel0.l2cachesize): 12 MiB on the M1
   Max, whose 48 MB system cache macOS does not report. 8 MiB is below the
   last level of every host rig is tested on, and stands for the hosts that
   report none, Windows included, and for those that report less than they
   have, as Apple's and Arm servers' system caches go unreported: a floor
   below the caches errs toward warm results. */
#define HEAP_LLC_LEAST ((size_t)8 << 20)

#if defined(__APPLE__)
/* The value of the sysctl [name], of 4 or 8 bytes, or 0. */
static size_t sysctl_bytes(const char *name) {
  union {
    int32_t i32;
    int64_t i64;
  } v = {0};
  size_t size = sizeof v;
  if (sysctlbyname(name, &v, &size, NULL, 0) != 0) return 0;
  if (size == sizeof v.i32) return v.i32 > 0 ? (size_t)v.i32 : 0;
  if (size == sizeof v.i64) return v.i64 > 0 ? (size_t)v.i64 : 0;
  return 0;
}
#endif

static size_t heap_llc_bytes(void) {
  size_t most = 0;
#if defined(__APPLE__)
  most = sysctl_bytes("hw.l3cachesize");
  if (most == 0) most = sysctl_bytes("hw.perflevel0.l2cachesize");
#elif defined(__linux__)
  for (int i = 0;; i++) {
    char name[64];
    snprintf(name, sizeof name,
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
    FILE *f = fopen(name, "r");
    if (f == NULL) break;
    unsigned long n = 0;
    char unit = 'K';
    int read = fscanf(f, "%lu%c", &n, &unit);
    fclose(f);
    if (read < 1) continue;
    size_t bytes = (size_t)n << (unit == 'M' ? 20 : unit == 'K' ? 10 : 0);
    if (bytes > most) most = bytes;
  }
#endif
  return most > HEAP_LLC_LEAST ? most : HEAP_LLC_LEAST;
}

/* The memory a major cycle is due after: [caml_custom_get_max_major], the
   major heap's bytes over 150 times the ratio, plus the same share of the
   bytes buffers hold live, and at least [heap_cycle_floor]. */
static mlsize_t heap_cycle_bytes(void) {
  mlsize_t live =
      (mlsize_t)atomic_load_explicit(&heap_live, memory_order_relaxed);
  mlsize_t share =
      caml_custom_get_max_major() +
      live / 150 *
          atomic_load_explicit(&caml_custom_major_ratio, memory_order_relaxed);
  return share > heap_cycle_floor ? share : heap_cycle_floor;
}

/* Reusing freed buffers

   A buffer's memory goes back to the C library when the collector frees it,
   and the library may return it to the system, so the next buffer of that
   size faults its pages in again. Freed buffers are kept instead, each under
   its exact size, and the next allocation of that size takes the one freed
   last. They are kept while they and the live buffers hold no more than the
   buffers alone held at once in the running major cycle and the one before
   ([heap_peak]), so the cache never raises the program's peak over them, or
   while they hold at most [HEAP_CACHE_FLOOR]. Past it the least recently
   freed go back to the library, as they do when a cycle ends, before the
   next one starts its peak; an allocation that fails gives them all back
   before trying again. A cycle's pace ([heap_cycle_bytes]) would be too low
   a bound: buffers made while a cycle runs survive it, so a cycle collects
   more than it was due after. A kept buffer holds its links in its
   first bytes: in the list of the cache, newest first, and in its size's
   bucket, so a take walks the buffers of its bucket only. A spin lock
   guards both, held for a few pointer writes and one bucket's walk. */

#define HEAP_CACHE_FLOOR ((size_t)32 << 20)
#define HEAP_BUCKETS 256

struct heap_entry {
  struct heap_entry *newer, *older; /* the cache, newest first */
  struct heap_entry *next, *prev;   /* its size's bucket, newest first */
  size_t n;
};

static atomic_flag heap_cache_lock = ATOMIC_FLAG_INIT;
static struct heap_entry *heap_newest, *heap_oldest;
static struct heap_entry *heap_bucket[HEAP_BUCKETS];
static _Atomic size_t heap_cached;

/* The bucket of buffers of [n] bytes: a multiplicative hash of their pages
   (buffers kept are 64 KiB or more). */
static struct heap_entry **bucket_of(size_t n) {
  return &heap_bucket[(uint64_t)(n / 4096) * UINT64_C(0x9E3779B97F4A7C15) >>
                      56];
}

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
  if (e->prev) e->prev->next = e->next;
  else *bucket_of(e->n) = e->next;
  if (e->next) e->next->prev = e->prev;
  heap_cached -= e->n;
}

static size_t heap_cache_cap(void) {
  intnat peak = atomic_load_explicit(&heap_peak, memory_order_relaxed);
  intnat before = atomic_load_explicit(&heap_peak_before, memory_order_relaxed);
  intnat room = (peak > before ? peak : before) -
                atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  return room < (intnat)HEAP_CACHE_FLOOR ? HEAP_CACHE_FLOOR : (size_t)room;
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
  struct heap_entry **b = bucket_of(n);
  e->next = *b;
  e->prev = NULL;
  if (*b) (*b)->prev = e;
  *b = e;
  heap_cached += n;
  struct heap_entry *dropped = heap_over(cap);
  heap_cache_release();
  heap_free_list(dropped);
}

/* Gives back what the cache holds beyond [cap], oldest first: run as a major
   cycle ends with what the cache may keep now, so a program that stops
   freeing buffers does not keep the share of a working set it dropped. */
static void heap_trim(size_t cap) {
  heap_cache_acquire();
  struct heap_entry *dropped = heap_over(cap);
  heap_cache_release();
  heap_free_list(dropped);
}

static void *heap_take(size_t n) {
  heap_cache_acquire();
  struct heap_entry *e = *bucket_of(n);
  while (e && e->n != n) e = e->next;
  if (e) heap_unlink(e);
  heap_cache_release();
  return e;
}

static intnat heap_kept(void) {
  return (intnat)atomic_load_explicit(&heap_cached, memory_order_relaxed);
}

static void heap_drop_all(void) {
  heap_cache_acquire();
  struct heap_entry *e = heap_newest;
  heap_newest = heap_oldest = NULL;
  memset(heap_bucket, 0, sizeof heap_bucket);
  heap_cached = 0;
  heap_cache_release();
  heap_free_list(e);
}

/* Gives back [v_n] bytes a reservation took ([caml_rig_heap_reserve]) for
   memory other than a buffer's bigarray, such as a device's pinned memory. */
value caml_rig_heap_release(value v_n) {
  atomic_fetch_sub_explicit(&heap_bytes, Long_val(v_n), memory_order_relaxed);
  return Val_unit;
}

/* The bytes host buffers hold reserved against the host's budget:
   Rig.allocated of the host. */
intnat rig_heap_held(void) {
  return atomic_load_explicit(&heap_bytes, memory_order_relaxed);
}

value caml_rig_heap_held(value unit) {
  (void)unit;
  return Val_long(rig_heap_held());
}

/* Gives back kept buffers until they fit in [v_budget] beside the bytes
   host buffers hold. */
value caml_rig_heap_trim(value v_budget) {
  intnat room = Long_val(v_budget) -
                atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  heap_trim(room > 0 ? (size_t)room : 0);
  return Val_unit;
}

value caml_rig_heap_drop(value unit) {
  (void)unit;
  heap_drop_all();
  return Val_unit;
}

/* The runtime's operations of bigarrays, which it exports but declares only
   to itself. A buffer's bigarray is one block made here with the runtime's
   operations but its finaliser. After the array's one dimension the block
   holds [struct heap_block]: where the bytes were allocated, which may lie
   before the array's first byte, and the cycles seen ended when it was
   made. The runtime gives every array it makes over another, a sub, a
   slice, a reshape, the other's operations and marks it a subarray. Such
   arrays share a proxy, made at the first ([ensure_proxy]), that names the
   allocation. The buffer's own array returns its reservation once
   collected, and the last array over the allocation, whichever it is, keeps
   or frees the bytes. */
extern const struct custom_operations caml_ba_ops;

struct heap_block {
  void *base;
  uintnat cycle;
};

#define HEAP_BLOCK_BYTES (SIZEOF_BA_ARRAY + sizeof(intnat) + \
                          sizeof(struct heap_block))

static struct custom_operations heap_ops;

static struct heap_block *heap_block(struct caml_ba_array *b) {
  return (struct heap_block *)&b->dim[1];
}

/* Buffers from this size start on a page, and the heap keeps them. */
static size_t heap_kept_from;

/* The page, which [caml_rig_heap_init] reads once. */
static size_t heap_page;

/* The alignment every host's [malloc] gives. */
#define MALLOC_ALIGN 16

/* The bytes allocated for a buffer of [n] bytes starting on a multiple of
   [align]: [align - MALLOC_ALIGN] more, to start on one, unless the C
   library aligns it, as [malloc] does to MALLOC_ALIGN and [posix_memalign]
   to a page (heap_malloc). Windows aligns nothing that [free] releases. */
static size_t heap_size(size_t n, size_t align) {
  if (align <= MALLOC_ALIGN) return n;
#ifndef _WIN32
  if (align >= heap_page) return n;
#endif
  return n + align - MALLOC_ALIGN;
}

/* The alignment of a buffer of [n] bytes: a page from [heap_kept_from], so
   devices can map it; RIG_LINE_BYTES from that line, so that pieces of
   whole lines threads write apart share none; malloc's below, since a
   buffer under a line holds no two such pieces, and aligning a scalar added
   10 ns to a 180 ns add on fudge (Core Ultra 5 235). */
static size_t heap_align(size_t n) {
  if (n >= heap_kept_from) return heap_page;
  return n >= RIG_LINE_BYTES ? RIG_LINE_BYTES : 1;
}

/* The bytes a buffer of [n] bytes takes of the C library's memory, which
   its reservation and the collector's pacing count. */
static size_t heap_footprint(size_t n) { return heap_size(n, heap_align(n)); }

value caml_rig_heap_footprint(value v_n) {
  return Val_long(heap_footprint((size_t)Long_val(v_n)));
}

/* The size by which [heap_finalize] keeps or frees a buffer of [n] bytes
   made by [caml_rig_heap_bytes]: the bytes allocated for one that starts on
   a page, which it keeps, else [n], below [heap_kept_from]. */
static size_t heap_allocated(size_t n) {
  return n >= heap_kept_from ? heap_size(n, heap_page) : n;
}

static void heap_finalize(value v) {
  struct caml_ba_array *b = Caml_ba_array_val(v);
  void *base = NULL;
  size_t size = 0;
  if (!(b->flags & CAML_BA_SUBARRAY)) {
    struct heap_block *h = heap_block(b);
    intnat n = b->dim[0], f = (intnat)heap_footprint((size_t)n);
    atomic_fetch_sub_explicit(&heap_bytes, f, memory_order_relaxed);
    if (h->cycle < atomic_load_explicit(&cycles_seen, memory_order_relaxed))
      atomic_fetch_add_explicit(&heap_collected, f, memory_order_relaxed);
    base = h->base;
    size = heap_allocated((size_t)n);
  }
  struct caml_ba_proxy *p = b->proxy;
  if (p != NULL) {
    if (atomic_fetch_sub(&p->refcount, 1) != 1) return;
    base = p->data;
    size = p->size;
    free(p);
  }
  if (size >= heap_kept_from) heap_keep(base, size);
  else free(base);
}

/* Builds the operations and installs the slice hook, once, as the library
   initializes, with the size from which buffers start on a page. */
value caml_rig_heap_init(value v_kept_from) {
  heap_kept_from = (size_t)Long_val(v_kept_from);
  heap_page = rig_page_bytes();
  heap_cycle_floor = heap_llc_bytes() / 2;
  heap_ops = caml_ba_ops;
  heap_ops.finalize = heap_finalize;
  previous_slice_end = atomic_exchange_explicit(
      &caml_major_slice_end_hook, slice_end, memory_order_relaxed);
  return Val_unit;
}

/* Buffers from this size pace major cycles by the program's memory
   ([heap_cycle_bytes]); smaller ones pace the collector as any bigarray
   does, so that dead small buffers bring minor collections. */
#define HEAP_PACED_FROM ((size_t)64 << 10)

/* The [n] bytes at [data], allocated from [base], as a [char] bigarray. */
static value heap_bigarray(void *base, void *data, size_t n) {
  size_t f = heap_footprint(n);
  value ba = n < HEAP_PACED_FROM
                 ? caml_alloc_custom_mem(&heap_ops, HEAP_BLOCK_BYTES, f)
                 : caml_alloc_custom(&heap_ops, HEAP_BLOCK_BYTES, f,
                                     heap_cycle_bytes());
  struct caml_ba_array *b = Caml_ba_array_val(ba);
  b->data = data;
  b->num_dims = 1;
  b->flags = CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED;
  b->proxy = NULL;
  b->dim[0] = (intnat)n;
  struct heap_block *h = heap_block(b);
  h->base = base;
  h->cycle = atomic_load_explicit(&cycles_seen, memory_order_relaxed);
  return ba;
}

/* [n] bytes of the C library starting on a multiple of [align], or NULL;
   [*base] is what [free] takes. A page comes from [posix_memalign] where
   the host has one, a smaller [align] from an allocation [align - 1] bytes
   larger ([heap_size]): on fudge (Core Ultra 5 235, glibc) [posix_memalign]
   of 8 bytes on 128 took 33 ns, [malloc] 5.5. */
static void *heap_malloc(size_t n, size_t align, void **base) {
  if (align <= 16) return *base = malloc(n);
#ifndef _WIN32
  if (align >= heap_page) {
    if (posix_memalign(base, align, n) != 0) return *base = NULL;
    return *base;
  }
#endif
  char *p = malloc(heap_size(n, align));
  *base = p;
  if (p == NULL) return NULL;
  return p + (align - (uintptr_t)p % align) % align;
}

/* [v_n] bytes of the heap starting on a multiple of [heap_align], as a
   [char] bigarray whose collection returns their reservation of
   [heap_footprint]; a kept buffer of the size if there is one. A failed
   allocation gives the kept buffers back and tries again once, then raises
   [Out_of_memory]. */
value caml_rig_heap_bytes(value v_n) {
  size_t n = (size_t)Long_val(v_n), align = heap_align(n);
  size_t size = heap_size(n, align);
  void *base = n >= heap_kept_from ? heap_take(size) : NULL;
  void *data;
  if (base != NULL)
    data = (char *)base + (align - (uintptr_t)base % align) % align;
  else {
    data = heap_malloc(n, align, &base);
    if (data == NULL) {
      heap_drop_all();
      data = heap_malloc(n, align, &base);
    }
    if (data == NULL) caml_raise_out_of_memory();
  }
  return heap_bigarray(base, data, n);
}

/* Bigarrays over memory */

/* Bigarrays over memory rig frees share a proxy whose first share
   is rig's own. The runtime frees a proxy, and the memory it names,
   only when its count falls to 0: rig's share keeps the runtime from
   ever freeing a device's or an io library's memory. The proxy names no
   memory, so the runtime's free of it would free nothing else. Rig
   gives up its share once the memory's buffers are collected and it is the
   last ([caml_rig_proxy_drop]), then frees the memory itself. */

value caml_rig_proxy_new(value unit) {
  (void)unit;
  struct caml_ba_proxy *p = malloc(sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  atomic_store_explicit(&p->refcount, 1, memory_order_relaxed);
  p->data = NULL;
  p->size = 0;
  return Val_long((intnat)p);
}

/* The [v_n] bytes at [v_addr] as a [char] bigarray sharing the proxy
   [v_p]. No allocation runs between the array's and its proxy's setting,
   so the runtime never finalises it without the proxy. */
value caml_rig_proxy_bytes(value v_p, value v_addr, value v_n) {
  struct caml_ba_proxy *p = (struct caml_ba_proxy *)Long_val(v_p);
  intnat dim = Long_val(v_n);
  /* [CAML_BA_SUBARRAY]: the bytes are another owner's, so the array
     paces no collection as memory of the heap. */
  value ba = caml_ba_alloc(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED |
                               CAML_BA_SUBARRAY,
                           1, (void *)Long_val(v_addr), &dim);
  atomic_fetch_add_explicit(&p->refcount, 1, memory_order_relaxed);
  Caml_ba_array_val(ba)->proxy = p;
  return ba;
}

/* Gives up rig's share of the proxy [v_p] if it is the last, and
   frees the proxy: [true] then, as no bigarray over its memory is left.
   The count falls from 1 to 0 in one step, after the runtime's last
   decrement, which a view finalised on another domain made. */
value caml_rig_proxy_drop(value v_p) {
  struct caml_ba_proxy *p = (struct caml_ba_proxy *)Long_val(v_p);
  uintnat last = 1;
  if (!atomic_compare_exchange_strong_explicit(&p->refcount, &last, 0,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return Val_false;
  free(p);
  return Val_true;
}

value caml_rig_bigarray_address(value ba) {
  return Val_long((intnat)Caml_ba_data_val(ba));
}

extern value caml_ba_sub(value vb, value vofs, value vlen);

/* Gives the managed array [v] the proxy its sub-arrays share, if it has
   none, at most once whatever the domains that view [v] at once. Its first
   reference is [v]'s own, as the runtime counts it. A heap buffer's proxy
   names its allocation, which the last array over it keeps or frees
   ([heap_finalize]). */
static void ensure_proxy(value v) {
  struct caml_ba_array *b = Caml_ba_array_val(v);
  _Atomic(struct caml_ba_proxy *) *slot =
      (_Atomic(struct caml_ba_proxy *) *)&b->proxy;
  if ((b->flags & CAML_BA_MANAGED_MASK) == CAML_BA_EXTERNAL ||
      atomic_load_explicit(slot, memory_order_acquire) != NULL)
    return;
  struct caml_ba_proxy *proxy = malloc(sizeof *proxy);
  if (proxy == NULL) caml_raise_out_of_memory();
  atomic_store_explicit(&proxy->refcount, 1, memory_order_relaxed);
  if (Custom_ops_val(v) == &heap_ops && !(b->flags & CAML_BA_SUBARRAY)) {
    proxy->data = heap_block(b)->base;
    /* the bytes [heap_finalize] keeps */
    proxy->size = heap_allocated((size_t)b->dim[0]);
  } else {
    proxy->data = b->data;
    proxy->size = b->flags & CAML_BA_MAPPED_FILE ? caml_ba_byte_size(b) : 0;
  }
  struct caml_ba_proxy *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          slot, &none, proxy, memory_order_acq_rel, memory_order_acquire))
    free(proxy);
}

/* [v_len] elements of kind [v_kind] from byte [v_offset] of [v_src], over
   [v_src]'s storage, which lives as long as any array over it. */
value caml_rig_bigarray_view(value v_src, value v_kind,
                                     value v_offset, value v_len) {
  CAMLparam2(v_src, v_kind);
  CAMLlocal1(view);
  ensure_proxy(v_src);
  view = caml_ba_sub(v_src, Val_long(0),
                     Val_long(Caml_ba_array_val(v_src)->dim[0]));
  struct caml_ba_array *b = Caml_ba_array_val(view);
  b->data = (char *)b->data + Long_val(v_offset);
  b->flags = (b->flags & (CAML_BA_LAYOUT_MASK | CAML_BA_MANAGED_MASK |
                          CAML_BA_SUBARRAY)) |
             Int_val(v_kind);
  b->dim[0] = Long_val(v_len);
  CAMLreturn(view);
}

/* Copies [v_n] bytes between host addresses, releasing the runtime from
   64 KiB on. */
#define BLOCKING_BYTES (1 << 16)

value caml_rig_memmove(value v_dst, value v_src, value v_n) {
  void *dst = (void *)Long_val(v_dst);
  const void *src = (const void *)Long_val(v_src);
  size_t n = (size_t)Long_val(v_n);
  if (n == 0) return Val_unit;
  if (n >= BLOCKING_BYTES) {
    caml_enter_blocking_section_no_pending();
    memmove(dst, src, n);
    caml_leave_blocking_section();
  } else
    memmove(dst, src, n);
  return Val_unit;
}
