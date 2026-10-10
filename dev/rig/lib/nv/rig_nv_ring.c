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
   channel of the last part (COMPUTE when there is none, or when v's low 32
   bits are 0) waits for the other channel's last part, then releases v
   into the timeline word: on
   COMPUTE a release that waits for the channel to be idle, on COPY a
   release after its copies completed.

   Order: the parts on one channel run in array order. COPY runs its copies
   one after another; on COMPUTE a part placed after launches the channel
   has not waited for starts with a wait for idle, as kernels a channel
   schedules run at once.

   Launches: a submission's launches get their descriptors and constant
   banks 0 in COMPUTE's launch ring, in part order, before its words; the
   ring is the GPU's memory where its BAR had room, which the GPU reads
   faster than host memory. A launch that directly follows a launch on
   COMPUTE, with no join between them, is chained: the descriptor before
   it schedules it once its own launch completed, so the writer places
   nothing for it. Any other launch is scheduled from the segment, after a
   wait for idle if launches run. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <string.h>

#include "rig_nv_stubs.h"

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

/* Writes template [k] with the values [a], [b], [n] into the open segment
   of [c]. A segment never wraps: one whose words would pass the ring's end,
   including one that ends exactly there, is closed first, and the words go
   at the start of the ring when they do not fit before its end. */
static void emit(const struct device *d, struct channel *c, int k, uint64_t a,
                 uint64_t b, uint64_t n) {
  const struct template *t = &d->t[k];
  uint64_t bytes = 4 * (uint64_t)t->nwords;
  uint64_t start = c->open & (c->size - 1);
  if (start + 4 * c->open_words + bytes > c->size) {
    close_segment(d, c);
    uint64_t end = c->written & (c->size - 1);
    if (end + bytes > c->size) c->written += c->size - end;
    c->open = c->written;
  }
  uint64_t at = c->written & (c->size - 1);
  const uint64_t values[3] = {a, b, n};
  rig_nv_fill(t, values, (uint32_t *)(c->segments + at));
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
    c->landed = m->launched;
    c->first = (c->first + 1) & (c->entries - 1);
    c->count--;
  }
}

static int is_copy(const struct rig_part *p) {
  return p->kind == RIG_COPY && p->copy.bytes > 0;
}

static const struct launch *launch_of(const struct rig_part *p) {
  return p->launch.launch;
}

static const struct rig_block *block_of(const struct rig_part *p,
                                        const uint8_t *args) {
  return (const struct rig_block *)(args + p->launch.block);
}

static uint64_t round_up(uint64_t n, uint64_t a) {
  return (n + a - 1) & ~(a - 1);
}

/* Whether the launch [p]'s block, in [args], is one its function runs. */
static int launchable(const struct rig_part *p, const uint8_t *args) {
  const struct launch *l = launch_of(p);
  const struct rig_block *b = block_of(p, args);
  uint64_t threads = 1;
  for (int i = 0; i < 3; i++) {
    if (b->groups[i] == 0 || b->groups[i] > l->max[i]) return 0;
    if (b->threads[i] == 0 || b->threads[i] > l->max[3 + i]) return 0;
    threads *= b->threads[i];
  }
  return threads <= l->max_threads &&
         round_up(b->shared, 128) <= l->max_shared;
}


/* The bytes of [p]'s constant bank 0: the driver's parameters, the
   kernel's and the bank the descriptor names, whichever ends last. */
static uint64_t bank_bytes(const struct rig_part *p) {
  const struct launch *l = launch_of(p);
  uint64_t n = (uint64_t)l->params_at + p->launch.params;
  if (n < l->bank0.nbytes) n = l->bank0.nbytes;
  return n < l->bank0_bytes ? l->bank0_bytes : n;
}

static uint64_t qmd_bytes(const struct rig_part *p) {
  return launch_of(p)->qmd[0][0].nbytes;
}

/* The words a part places, which only words do. */
static size_t words_of(const struct rig_part *p) {
  return p->kind == RIG_WORDS ? p->words.n : 0;
}

/* Whether [p]'s parts, whose launches' blocks lie in [args], are ones the
   device runs. */
static int runs(const struct rig_part *p, int n, const uint8_t *args) {
  if (n < 0 || n > MAX_PARTS) return 0;
  for (int i = 0; i < n; i++) {
    const struct rig_part *x = &p[i];
    if (x->queue != COMPUTE && x->queue != COPY) return 0;
    if (x->kind != RIG_WORDS && x->kind != RIG_COPY && x->kind != RIG_LAUNCH)
      return 0;
    if (is_copy(x) && x->queue != COPY) return 0;
    if (x->kind == RIG_LAUNCH &&
        (x->queue != COMPUTE || launch_of(x) == NULL || !launchable(x, args)))
      return 0;
    if (words_of(x) % 2 != 0) return 0;
    for (int j = 0; j < x->nafter; j++)
      if (x->after[j] < 0 || x->after[j] >= i) return 0;
  }
  return 1;
}

/* Marks in [d->awaited] the parts that a later part on the other channel
   runs after. */
static void mark_awaited(struct device *d, const struct rig_part *p, int n) {
  for (int i = 0; i < n; i++) d->awaited[i] = 0;
  for (int k = 0; k < n; k++)
    for (int j = 0; j < p[k].nafter; j++) {
      int a = p[k].after[j];
      if (p[a].queue != p[k].queue) d->awaited[a] = 1;
    }
}

static uint64_t bytes_of(const struct device *d, int k) {
  return 4 * (uint64_t)d->t[k].nwords;
}

/* The most entries and segment bytes [p] takes of each channel, the words
   a channel may owe and MAX_WAITS waits counted, and a wrap of the segment
   ring that wastes a template's words and splits a segment. */
/* The channel that releases [v]: the last part's, or COMPUTE when there is
   none or when [v]'s low 32 bits are 0. The copy engine may write a
   release's two 32-bit words one at a time, and the timeline word's high
   word changes only at those values, where the next value's release, on
   the other channel, may follow at once. COMPUTE writes its 64 bits at
   once. */
static int releaser(const struct rig_part *p, int n, uint64_t v) {
  if (n == 0 || (uint32_t)v == 0) return COMPUTE;
  return p[n - 1].queue;
}

static void need(const struct device *d, const struct rig_part *p, int n,
                 uint64_t *entries, uint64_t *bytes, uint64_t *launched) {
  uint64_t v = atomic_load_explicit(&d->last, memory_order_relaxed) + 1;
  int r = releaser(p, n, v);
  int used[CHANNELS] = {0, 0};
  used[r] = 1;
  for (int q = 0; q < CHANNELS; q++) entries[q] = bytes[q] = 0;
  /* The most a wrap past the launch ring's end wastes: the largest of the
     launches' descriptors and banks. */
  uint64_t wasted = 0;
  *launched = 0;
  for (int i = 0; i < n; i++) {
    int q = p[i].queue;
    used[q] = 1;
    entries[q] += 2 + words_of(&p[i]) / 2;
    bytes[q] += (uint64_t)p[i].nafter * bytes_of(d, T_ACQUIRE);
    if (is_copy(&p[i]))
      bytes[q] += (p[i].copy.bytes + COPY_MAX - 1) / COPY_MAX *
                  bytes_of(d, T_COPY);
    if (p[i].kind == RIG_LAUNCH) {
      uint64_t qmd = qmd_bytes(&p[i]) + QMD_ALIGN - 1;
      uint64_t bank = bank_bytes(&p[i]) + BANK_ALIGN - 1;
      bytes[q] += bytes_of(d, T_SCHEDULE);
      *launched += qmd + bank;
      if (qmd > wasted) wasted = qmd;
      if (bank > wasted) wasted = bank;
    }
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
  if (*launched > 0) *launched += wasted;
}

int rig_nv_room(void *self, const struct rig_part *p, int n,
                const uint8_t *args) {
  struct device *d = self;
  if (!runs(p, n, args)) return RIG_NEVER;
  uint64_t entries[CHANNELS], bytes[CHANNELS], launched;
  need(d, p, n, entries, bytes, &launched);
  for (int q = 0; q < CHANNELS; q++)
    if (entries[q] > d->ch[q].entries - 1 || bytes[q] > d->ch[q].size)
      return RIG_NEVER;
  const struct channel *compute = &d->ch[COMPUTE];
  if (launched > compute->launches_size) return RIG_NEVER;
  uint64_t seen = atomic_load_explicit(d->word, memory_order_acquire);
  for (int q = 0; q < CHANNELS; q++) {
    struct channel *c = &d->ch[q];
    reclaim(c, seen);
    if (c->entries - 1 - (c->put - c->freed) < entries[q] ||
        c->size - (c->written - c->reclaimed) < bytes[q])
      return RIG_LATER;
  }
  if (compute->launches_size - (compute->launched - compute->landed) <
      launched)
    return RIG_LATER;
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
  }
  if (atomic_exchange(&d->invalidate, 0))
    emit(d, c, T_INVALIDATE, 0, 0, 0);
}

/* Signals and releases on COMPUTE wait for idle first. */
static void signal(struct device *d, int q, uint64_t address, uint64_t value) {
  emit(d, &d->ch[q], q == COMPUTE ? T_RELEASE : T_COPY_RELEASE, address,
       value, 0);
}

/* Launches' memory */

/* Takes [bytes] of [c]'s launch ring at the count [*at], aligned to
   [align], a power of two that divides the ring's size: their count. The
   bytes never wrap: ones that would pass the ring's end start it. */
static uint64_t take(const struct channel *c, uint64_t *at, uint64_t bytes,
                     uint64_t align) {
  uint64_t size = c->launches_size;
  uint64_t x = round_up(*at, align);
  if ((x & (size - 1)) + bytes > size) x = (x | (size - 1)) + 1;
  *at = x + bytes;
  return x;
}

static uint64_t gpu_at(const struct channel *c, uint64_t x) {
  return c->launches_gpu + (x & (c->launches_size - 1));
}

static uint8_t *host_at(const struct channel *c, uint64_t x) {
  return c->launches + (x & (c->launches_size - 1));
}

/* [p]'s values: its block's sizes, its bank at [bank], the descriptor at
   [next] it chains, if not 0. */
static void values_of(const struct rig_part *p, const uint8_t *args,
                      uint64_t bank, uint64_t next, uint64_t *v) {
  const struct rig_block *b = block_of(p, args);
  for (int i = 0; i < 3; i++) {
    v[V_GRID_X + i] = b->groups[i];
    v[V_BLOCK_X + i] = b->threads[i];
  }
  v[V_BANK0] = bank;
  v[V_NEXT] = next;
  v[V_SHARED] = round_up(b->shared, 128);
}

/* Writes [p]'s constant bank 0 at [bank], a count of COMPUTE's launch
   ring: the driver's parameters, then the kernel's from [args], whose refs
   take their slots' addresses. Built on the stack, then copied, so that
   the ring's memory is only written. */
static void put_bank(struct device *d, const struct rig_part *p,
                     const uint8_t *args, const uint64_t *slots,
                     uint64_t bank) {
  struct channel *c = &d->ch[COMPUTE];
  const struct launch *l = launch_of(p);
  uint8_t w[STRUCTURE_BYTES + RIG_PARAMS];
  uint64_t v[VALUES];
  values_of(p, args, gpu_at(c, bank), 0, v);
  rig_nv_fill_structure(&l->bank0, v, w);
  rig_params(w + l->params_at, p, args, slots);
  uint64_t n = (uint64_t)l->params_at + p->launch.params;
  if (n < l->bank0.nbytes) n = l->bank0.nbytes;
  memcpy(host_at(c, bank), w, n);
}

/* Writes [p]'s descriptor at [qmd], with its bank at [bank], chaining the
   descriptor at [next] if not 0. */
static void put_qmd(struct device *d, const struct rig_part *p,
                    const uint8_t *args, uint64_t qmd, uint64_t bank,
                    uint64_t next) {
  struct channel *c = &d->ch[COMPUTE];
  const struct launch *l = launch_of(p);
  uint8_t w[STRUCTURE_BYTES];
  uint64_t v[VALUES];
  values_of(p, args, gpu_at(c, bank), next == 0 ? 0 : gpu_at(c, next), v);
  const struct structure *s = &l->qmd[v[V_SHARED] > 0][next != 0];
  rig_nv_fill_structure(s, v, w);
  memcpy(host_at(c, qmd), w, s->nbytes);
}

/* Whether part [b], on COMPUTE, is a launch chained to [a], the part before
   it there: a launch neither the other channel waits for nor followed by a
   wait for that channel. */
static int chained(const struct device *d, const struct rig_part *p, int a,
                   int b) {
  if (a < 0 || p[a].kind != RIG_LAUNCH || p[b].kind != RIG_LAUNCH ||
      d->awaited[a])
    return 0;
  for (int j = 0; j < p[b].nafter; j++)
    if (p[p[b].after[j]].queue != COMPUTE) return 0;
  return 1;
}

/* Takes the descriptor and bank of the launch [p] at the count [*at]. */
static void take_launch(const struct channel *c, const struct rig_part *p,
                        uint64_t *at, uint64_t *qmd, uint64_t *bank) {
  *qmd = take(c, at, qmd_bytes(p), QMD_ALIGN);
  *bank = take(c, at, bank_bytes(p), BANK_ALIGN);
}

/* Writes the descriptors and banks of the launches of [p] into COMPUTE's
   launch ring from its [launched] count, in part order. A descriptor is
   written once the part after it on COMPUTE is known, which it may
   chain. */
static void lay_launches(struct device *d, const struct rig_part *p, int n,
                         const uint8_t *args, const uint64_t *slots) {
  struct channel *c = &d->ch[COMPUTE];
  int before = -1, open = -1;
  uint64_t open_qmd = 0, open_bank = 0;
  for (int i = 0; i < n; i++) {
    if (p[i].queue != COMPUTE) continue;
    int chain = chained(d, p, before, i);
    before = i;
    uint64_t qmd = 0, bank = 0;
    if (p[i].kind == RIG_LAUNCH) {
      take_launch(c, &p[i], &c->launched, &qmd, &bank);
      put_bank(d, &p[i], args, slots, bank);
    }
    if (open >= 0)
      put_qmd(d, &p[open], args, open_qmd, open_bank, chain ? qmd : 0);
    open = p[i].kind == RIG_LAUNCH ? i : -1;
    open_qmd = qmd;
    open_bank = bank;
  }
  if (open >= 0) put_qmd(d, &p[open], args, open_qmd, open_bank, 0);
}

static void place(struct device *d, const struct rig_part *p) {
  struct channel *c = &d->ch[p->queue];
  if (p->kind == RIG_WORDS) {
    put_entries(d, c, p->words.at, p->words.n);
    return;
  }
  uint64_t dst = p->copy.dst + p->copy.dst_offset;
  uint64_t src = p->copy.src + p->copy.src_offset;
  for (uint64_t at = 0; at < p->copy.bytes; at += COPY_MAX) {
    uint64_t n = p->copy.bytes - at < COPY_MAX ? p->copy.bytes - at : COPY_MAX;
    emit(d, c, T_COPY, dst + at, src + at, n);
  }
}

int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                  int nwaits, const struct rig_part *p, int n,
                  const uint8_t *args, const uint64_t *slots, int nslots,
                  const uint64_t *handles, int nhandles,
                  uint64_t *times, const char **failure) {
  (void)times;
  (void)nslots;
  (void)handles;
  (void)nhandles;
  (void)failure;
  struct device *d = self;
  struct channel *compute = &d->ch[COMPUTE];
  int r = releaser(p, n, v);
  int used[CHANNELS] = {0, 0};
  int last[CHANNELS] = {-1, -1};
  /* Whether COMPUTE has launches it has not waited for. */
  int running = 0;
  for (int i = 0; i < n; i++) last[p[i].queue] = i;
  mark_awaited(d, p, n);
  /* The launches' memory, then the same counts again for their words. */
  uint64_t start = compute->launched, at = start;
  lay_launches(d, p, n, args, slots);
  int before = -1;
  for (int i = 0; i < n; i++) {
    int q = p[i].queue;
    enter(d, q, v, used, waits, nwaits);
    for (int j = 0; j < p[i].nafter; j++) {
      int a = p[i].after[j];
      if (p[a].queue != q)
        emit(d, &d->ch[q], T_ACQUIRE, JOIN_GPU(d, other(q)), tag(v, a), 0);
    }
    if (p[i].kind == RIG_LAUNCH) {
      uint64_t qmd, bank;
      take_launch(compute, &p[i], &at, &qmd, &bank);
      if (!chained(d, p, before, i)) {
        if (running) emit(d, compute, T_IDLE, 0, 0, 0);
        emit(d, compute, T_SCHEDULE, gpu_at(compute, qmd), 0, 0);
      }
    } else {
      if (q == COMPUTE && running) emit(d, &d->ch[q], T_IDLE, 0, 0, 0);
      place(d, &p[i]);
    }
    if (q == COMPUTE) {
      running = 1;
      before = i;
    }
    if (d->awaited[i] || (i == last[q] && q != r)) {
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
        (struct mark){v, c->put, c->written, c->launched};
  }
  store_fence();
  for (int q = 0; q < CHANNELS; q++)
    if (used[q])
      *d->ch[q].gp_put = (uint32_t)(d->ch[q].put & (d->ch[q].entries - 1));
  if (atomic_load_explicit(&d->bar_live, memory_order_acquire) > 0 ||
      (compute->bar && compute->launched != start)) {
    full_fence();
    (void)*d->bar;
  } else
    store_fence();
  for (int q = 0; q < CHANNELS; q++)
    if (used[q]) *d->doorbell = d->ch[q].token;
  atomic_store_explicit(&d->last, v, memory_order_release);
  return RIG_COMMITTED;
}
