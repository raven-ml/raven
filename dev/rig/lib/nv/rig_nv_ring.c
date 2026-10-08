/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The writer: rig_nv_room and rig_nv_submit.

   A submission of value v, on each channel it uses, is a run of ring
   entries. The driver's own words go into segments of its segment ring,
   each closed into an entry before the next entry that is not the
   driver's; a part's entries are copied as they are. On each channel it
   enters, the writer places first the words the channel owes (its engine
   binding at its first use; on COMPUTE a local memory and an invalidation
   when owed), then a wait for v-1 unless the channel itself released v-1,
   then the foreign waits; on COMPUTE the owed local memory and
   invalidation come after the waits, once every earlier value's work is
   done.

   Joins: a part that a later part on the other channel names in its
   [after], and a channel's last part when the other channel releases v,
   signal their channel's join word with the tag v << 16 | i + 1, i the
   part's index; the waiter acquires the tag. Tags rise on each join word,
   and the acquire compares 64 bits circularly, so ">=" is exact. The
   channel of the last part (COMPUTE when there is none) waits for the
   other channel's last part, then releases v into the timeline word: on
   COMPUTE a release that waits for the channel to be idle, on COPY a
   release after its copies completed.

   Order: the parts on one channel run in array order. COPY runs its copies
   one after another; on COMPUTE a part placed after launches the channel
   has not waited for starts with a wait for idle, as kernels a channel
   schedules run at once. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <string.h>

#include "rig_nv_stubs.h"

/* A join tag holds a part's index in 16 bits. */
#define MAX_PARTS 65535

/* The waits a submission may carry: rig_nv_room, which does not see
   them, keeps room for this many on each channel. */
#define MAX_WAITS 256

/* Orders the host's stores to memory before a store to a register:
   Linux's wmb(). */
static inline void store_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("sfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb st" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

/* Orders every access before it before every access after it: Linux's
   mb(), before a load that must follow the stores before it. */
static inline void full_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

static uint64_t tag(uint64_t v, int i) { return v << 16 | (uint64_t)(i + 1); }

static int other(int q) { return q == COMPUTE ? COPY : COMPUTE; }

/* Segments */

/* Closes the open segment of [c] into an entry, if it holds words. */
static void close_segment(const struct device *d, struct channel *c) {
  if (c->open_words == 0) return;
  uint64_t at = c->segments_gpu + (c->open & (c->size - 1));
  c->ring[c->put & (c->entries - 1)] =
      at + d->entry_base + c->open_words * d->entry_word;
  c->put++;
  c->open = c->written;
  c->open_words = 0;
}

static uint64_t fill(const struct hole *h, const uint64_t *values) {
  uint64_t x = values[h->slot];
  for (int i = 0; i < h->nops; i++)
    x = h->shift[i] ? x >> h->shift[i] : x + h->n[i];
  return x;
}

/* Writes template [k] with the values [a], [b], [n] into the open segment
   of [c]. A segment never wraps: one that would is closed, and the words go
   at the start of the ring. */
static void emit(const struct device *d, struct channel *c, int k, uint64_t a,
                 uint64_t b, uint64_t n) {
  const struct template *t = &d->t[k];
  uint64_t bytes = 4 * (uint64_t)t->nwords;
  uint64_t at = c->written & (c->size - 1);
  if (at + bytes > c->size) {
    close_segment(d, c);
    c->written += c->size - at;
    c->open = c->written;
    at = 0;
  }
  uint32_t *w = (uint32_t *)(c->segments + at);
  memcpy(w, t->words, bytes);
  const uint64_t values[3] = {a, b, n};
  for (int i = 0; i < t->nholes; i++) {
    const struct hole *h = &t->holes[i];
    uint64_t x = fill(h, values);
    w[h->at] = (uint32_t)x;
    if (h->wide) w[h->at + 1] = (uint32_t)(x >> 32);
  }
  c->written += bytes;
  c->open_words += (uint64_t)t->nwords;
}

/* Copies [n] words of ring entries, two words each, low first. */
static void put_entries(const struct device *d, struct channel *c,
                        const uint32_t *words, size_t n) {
  close_segment(d, c);
  for (size_t i = 0; i + 1 < n; i += 2)
    c->ring[c->put++ & (c->entries - 1)] =
        (uint64_t)words[i] | (uint64_t)words[i + 1] << 32;
}

/* Room */

/* Returns to [c] what the submissions up to [seen] used. */
static void reclaim(struct channel *c, uint64_t seen) {
  while (c->count > 0) {
    const struct mark *m = &c->marks[c->first];
    if (m->v > seen) return;
    c->freed = m->put;
    c->reclaimed = m->written;
    c->first = (c->first + 1) & (c->entries - 1);
    c->count--;
  }
}

static int is_copy(const struct rig_part *p) { return p->copy_bytes > 0; }

/* Whether [p]'s parts are ones the device runs. */
static int runs(const struct rig_part *p, int n) {
  if (n < 0 || n > MAX_PARTS) return 0;
  for (int i = 0; i < n; i++) {
    const struct rig_part *x = &p[i];
    if (x->queue != COMPUTE && x->queue != COPY) return 0;
    if (x->fill != NULL || x->ring_units != 0 || x->segment_bytes != 0)
      return 0;
    if (is_copy(x) && (x->queue != COPY || x->n != 0)) return 0;
    if (x->n % 2 != 0) return 0;
    for (int j = 0; j < x->nafter; j++)
      if (x->after[j] < 0 || x->after[j] >= i) return 0;
  }
  return 1;
}

/* Whether a part after part [i], on the other channel, runs after it. */
static int awaited(const struct rig_part *p, int n, int i) {
  for (int k = i + 1; k < n; k++) {
    if (p[k].queue == p[i].queue) continue;
    for (int j = 0; j < p[k].nafter; j++)
      if (p[k].after[j] == i) return 1;
  }
  return 0;
}

static uint64_t bytes_of(const struct device *d, int k) {
  return 4 * (uint64_t)d->t[k].nwords;
}

/* The most entries and segment bytes [p] takes of each channel, the words
   a channel may owe and MAX_WAITS waits counted, and a wrap of the segment
   ring that wastes a template's words and splits a segment. */
static void need(const struct device *d, const struct rig_part *p, int n,
                 uint64_t *entries, uint64_t *bytes) {
  int r = n > 0 ? p[n - 1].queue : COMPUTE;
  int used[CHANNELS] = {0, 0};
  used[r] = 1;
  for (int q = 0; q < CHANNELS; q++) entries[q] = bytes[q] = 0;
  for (int i = 0; i < n; i++) {
    int q = p[i].queue;
    used[q] = 1;
    entries[q] += 2 + p[i].n / 2;
    bytes[q] += (uint64_t)p[i].nafter * bytes_of(d, T_ACQUIRE);
    if (is_copy(&p[i]))
      bytes[q] += (p[i].copy_bytes + COPY_MAX - 1) / COPY_MAX *
                  bytes_of(d, T_COPY);
    bytes[q] += bytes_of(d, q == COMPUTE ? T_RELEASE : T_COPY_RELEASE);
    if (q == COMPUTE) bytes[q] += bytes_of(d, T_IDLE);
  }
  for (int q = 0; q < CHANNELS; q++) {
    if (!used[q]) continue;
    entries[q] += 3;
    bytes[q] += (2 + MAX_WAITS) * bytes_of(d, T_ACQUIRE) +
                bytes_of(d, q == COMPUTE ? T_SETUP : T_SETUP_COPY) +
                4 * TEMPLATE_WORDS;
    if (q == COMPUTE)
      bytes[q] += bytes_of(d, T_LOCAL) + bytes_of(d, T_INVALIDATE);
  }
  bytes[r] += bytes_of(d, r == COMPUTE ? T_RELEASE : T_COPY_RELEASE);
}

int rig_nv_room(void *self, const struct rig_part *p, int n) {
  struct device *d = self;
  if (!runs(p, n)) return RIG_NEVER;
  uint64_t entries[CHANNELS], bytes[CHANNELS];
  need(d, p, n, entries, bytes);
  for (int q = 0; q < CHANNELS; q++)
    if (entries[q] > d->ch[q].entries - 1 || bytes[q] > d->ch[q].size)
      return RIG_NEVER;
  uint64_t seen = atomic_load_explicit(d->word, memory_order_acquire);
  for (int q = 0; q < CHANNELS; q++) {
    struct channel *c = &d->ch[q];
    reclaim(c, seen);
    if (c->entries - 1 - (c->put - c->freed) < entries[q] ||
        c->size - (c->written - c->reclaimed) < bytes[q])
      return RIG_LATER;
  }
  return RIG_FITS;
}

/* Submitting */

/* Opens channel [q] for the work of [v], once per submission: its owed
   words, its wait for v-1 and the foreign waits. */
static void enter(struct device *d, int q, uint64_t v, int *used,
                  const struct rig_wait *waits, int nwaits) {
  if (used[q]) return;
  used[q] = 1;
  struct channel *c = &d->ch[q];
  c->open = c->written;
  c->open_words = 0;
  if (c->owes_setup) {
    emit(d, c, q == COMPUTE ? T_SETUP : T_SETUP_COPY, 0, 0, 0);
    c->owes_setup = 0;
  }
  if (c->released != v - 1) emit(d, c, T_ACQUIRE, d->word_gpu, v - 1, 0);
  for (int i = 0; i < nwaits; i++)
    emit(d, c, T_ACQUIRE, waits[i].at, waits[i].value, 0);
  if (q != COMPUTE) return;
  uint64_t local = atomic_load_explicit(&d->local, memory_order_acquire);
  if (local != 0 &&
      atomic_compare_exchange_strong(&d->local, &local, UINT64_C(0))) {
    uint64_t address = local & ((UINT64_C(1) << LOCAL_ADDRESS_BITS) - 1);
    uint64_t per_tpc = (local >> LOCAL_ADDRESS_BITS) << LOCAL_UNIT_SHIFT;
    emit(d, c, T_LOCAL, address, per_tpc, 0);
    atomic_store_explicit(&d->local_placed, v, memory_order_release);
  }
  if (atomic_exchange(&d->invalidate, 0))
    emit(d, c, T_INVALIDATE, 0, 0, 0);
}

/* Signals and releases on COMPUTE wait for idle first. */
static void signal(struct device *d, int q, uint64_t address, uint64_t value) {
  emit(d, &d->ch[q], q == COMPUTE ? T_RELEASE : T_COPY_RELEASE, address,
       value, 0);
}

static void place(struct device *d, const struct rig_part *p) {
  struct channel *c = &d->ch[p->queue];
  if (!is_copy(p)) {
    put_entries(d, c, p->words, p->n);
    return;
  }
  uint64_t dst = p->copy_dst + p->copy_dst_offset;
  uint64_t src = p->copy_src + p->copy_src_offset;
  for (uint64_t at = 0; at < p->copy_bytes; at += COPY_MAX) {
    uint64_t n = p->copy_bytes - at < COPY_MAX ? p->copy_bytes - at : COPY_MAX;
    emit(d, c, T_COPY, dst + at, src + at, n);
  }
}

int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                     int nwaits, const struct rig_part *p, int n,
                     const uint64_t *handles, int nhandles,
                     const char **failure) {
  (void)handles;
  (void)nhandles;
  (void)failure;
  struct device *d = self;
  int r = n > 0 ? p[n - 1].queue : COMPUTE;
  int used[CHANNELS] = {0, 0};
  int last[CHANNELS] = {-1, -1};
  /* Whether COMPUTE has launches it has not waited for. */
  int running = 0;
  for (int i = 0; i < n; i++) last[p[i].queue] = i;
  for (int i = 0; i < n; i++) {
    int q = p[i].queue;
    enter(d, q, v, used, waits, nwaits);
    for (int j = 0; j < p[i].nafter; j++) {
      int a = p[i].after[j];
      if (p[a].queue != q)
        emit(d, &d->ch[q], T_ACQUIRE, JOIN_GPU(d, other(q)), tag(v, a), 0);
    }
    if (q == COMPUTE && running) emit(d, &d->ch[q], T_IDLE, 0, 0, 0);
    place(d, &p[i]);
    if (q == COMPUTE) running = 1;
    if (awaited(p, n, i) || (i == last[q] && q != r)) {
      signal(d, q, JOIN_GPU(d, q), tag(v, i));
      if (q == COMPUTE) running = 0;
    }
  }
  enter(d, r, v, used, waits, nwaits);
  int o = other(r);
  if (used[o])
    emit(d, &d->ch[r], T_ACQUIRE, JOIN_GPU(d, o), tag(v, last[o]), 0);
  signal(d, r, d->word_gpu, v);
  d->ch[r].released = v;
  for (int q = 0; q < CHANNELS; q++) {
    if (!used[q]) continue;
    struct channel *c = &d->ch[q];
    close_segment(d, c);
    c->marks[(c->first + c->count++) & (c->entries - 1)] =
        (struct mark){v, c->put, c->written};
  }
  store_fence();
  for (int q = 0; q < CHANNELS; q++)
    if (used[q]) *d->ch[q].gp_put = (uint32_t)(d->ch[q].put & (d->ch[q].entries - 1));
  if (atomic_load_explicit(&d->bar_live, memory_order_acquire) > 0) {
    full_fence();
    (void)*d->bar;
  } else
    store_fence();
  for (int q = 0; q < CHANNELS; q++)
    if (used[q]) *d->doorbell = d->ch[q].token;
  d->last = v;
  return RIG_OK;
}
