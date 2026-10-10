/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The rings' one writer: rig_amd_room, rig_amd_submit, and the place
   and segment functions fills call. No function here calls the OCaml
   runtime or blocks.

   A submission of value v places, on each queue it uses: its own prefix (a
   wait for v-1 on the timeline word where the other queue released v-1), the
   compute queue's foreign waits, the parts in array order, the slot signals
   and waits that order parts of two queues, and on the last queue the
   release of v into the word. Each compute part follows a partial flush of
   the dispatches before it; the first, and any after a slot wait, also
   follows a cache acquire, and any other launch an invalidation of the
   caches above the L2, so that it reads what the dispatches before it
   wrote. A launch's parameters go in the segment. A submission without
   compute parts invalidates no cache: none of its work reads. It then
   flushes the host data path if the host wrote GPU memory through the BAR,
   and rings the doorbells. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "rig_amd_stubs.h"

/* The alignment of the bytes a fill takes from the segment. */
#define SEGMENT_ALIGN 64

/* A slot older than this is rewritten from the host before it is used. */
#define SLOT_STALE (UINT64_C(1) << 31)

/* AQL packets: 16 words, the first of which holds the header (hsa.h). */
#define AQL_WORDS 16
#define AQL_INVALID 1 /* HSA_PACKET_TYPE_INVALID */

#define LOW32(v) ((uint32_t)((v) & 0xffffffffu))

static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/* A copy of no bytes places nothing. */
static int is_copy(const struct rig_part *p) {
  return p->kind == RIG_COPY && p->copy.bytes > 0;
}

/* Templates */

static int words_of(const struct rig_amd *d, int t) {
  return d->templates[t].n;
}

/* Places [n] words on [q]: on a PM4 ring they wrap; on an SDMA ring they
   never do, and the ring's end is zeroed when they do not fit before it; on
   an AQL ring each packet's first word is stored last, with release order,
   since the queue may read a packet as soon as its header is valid. */
static void put_words(struct rig_amd_ring *q, const uint32_t *w, size_t n) {
  uint64_t mask = q->size - 1, at = q->put & mask;
  if (q->kind == RING_SDMA && at + n > q->size) {
    for (uint64_t i = at; i < q->size; i++) q->words[i] = 0;
    q->put += q->size - at;
    at = 0;
  }
  if (q->kind == RING_AQL)
    for (size_t p = 0; p < n; p += AQL_WORDS) {
      for (size_t i = 1; i < AQL_WORDS; i++)
        q->words[(at + p + i) & mask] = w[p + i];
      __atomic_store_n(&q->words[(at + p) & mask], w[p], __ATOMIC_RELEASE);
    }
  else
    for (size_t i = 0; i < n; i++) q->words[(at + i) & mask] = w[i];
  q->put += n;
}

/* Places template [t] on [q]. On an AQL ring the writer's PM4 words gather
   in the segment, from ib_at, until [flush] places them as one packet. */
static void emit(struct rig_amd *d, struct rig_amd_ring *q, int t,
                 uint64_t a0, uint64_t a1, uint64_t a2) {
  uint64_t args[3] = {a0, a1, a2};
  uint32_t w[RIG_AMD_TEMPLATE_WORDS];
  int n = rig_amd_fill(&d->templates[t], args, w);
  if (q->kind != RING_AQL) {
    put_words(q, w, (size_t)n);
    return;
  }
  struct rig_amd_segment *g = &d->segment;
  if (d->ib_n == 0) d->ib_at = g->put;
  memcpy(g->host + (d->ib_at + 4 * d->ib_n) % g->size, w, 4 * (size_t)n);
  d->ib_n += (size_t)n;
}

/* Places the PM4 words gathered for an AQL ring as one indirect buffer. */
static void flush(struct rig_amd *d, struct rig_amd_ring *q) {
  if (q->kind != RING_AQL || d->ib_n == 0) return;
  struct rig_amd_segment *g = &d->segment;
  uint64_t args[3] = {g->gpu + d->ib_at % g->size, d->ib_n, 0};
  uint32_t w[RIG_AMD_TEMPLATE_WORDS];
  int n = rig_amd_fill(&d->templates[A_IB], args, w);
  put_words(q, w, (size_t)n);
  g->put = d->ib_at + align_up(4 * d->ib_n, SEGMENT_ALIGN);
  d->ib_n = 0;
}

/* Room */

/* Takes back the room of the submissions the word shows reached. */
static uint64_t reclaim(struct rig_amd_marks *m, uint64_t free,
                        uint64_t word) {
  while (m->count > 0 && m->at[m->head].v <= word) {
    free = m->at[m->head].end;
    m->head = (m->head + 1) % RIG_AMD_MARKS;
    m->count--;
  }
  return free;
}

static void mark(struct rig_amd_marks *m, uint64_t v, uint64_t end) {
  m->at[(m->head + m->count) % RIG_AMD_MARKS] =
      (struct rig_amd_mark){v, end};
  m->count++;
}

static uint64_t copies(const struct rig_amd *d, uint64_t bytes) {
  return (bytes + d->max_copy - 1) / d->max_copy;
}

/* The bytes of launch [p]'s arguments: its parameters, and its function's
   arguments, implicit ones included. */
static uint64_t arguments(const struct rig_part *p) {
  const struct rig_amd_launch *l = p->launch.launch;
  return p->launch.params > l->kernarg ? p->launch.params : l->kernarg;
}

/* The words part [p] places itself, and the bytes it takes from the
   segment. */
static uint64_t part_words(const struct rig_amd *d, const struct rig_part *p,
                           uint64_t *bytes) {
  switch (p->kind) {
  case RIG_WORDS: return p->words.n;
  case RIG_FILL:
    *bytes += align_up(p->fill.segment_bytes, SEGMENT_ALIGN);
    return p->fill.ring_units;
  case RIG_LAUNCH: {
    const struct rig_amd_launch *l = p->launch.launch;
    *bytes += align_up(arguments(p), SEGMENT_ALIGN);
    return (uint64_t)l->n + (l->packet ? 0 : words_of(d, T_INVALIDATE));
  }
  default: return copies(d, p->copy.bytes) * words_of(d, S_COPY);
  }
}

/* The words a submission of [parts] places on each queue, at most, with
   RIG_AMD_WAITS waits, and the bytes it takes from the segment in one
   run: its fills' and launches', and on an AQL ring the writer's own PM4
   words, each indirect buffer of them aligned. */
static void need(const struct rig_amd *d, const struct rig_part *p, int n,
                 uint64_t *words, uint64_t *bytes) {
  int c = RIG_AMD_COMPUTE, s = RIG_AMD_COPY;
  uint64_t pm4 = words_of(d, T_WAIT) +
                 RIG_AMD_SCRATCH_WRITES * words_of(d, T_WRITE) +
                 RIG_AMD_WAITS * words_of(d, T_WAIT64) +
                 words_of(d, T_SIGNAL) + words_of(d, T_WAIT) +
                 words_of(d, T_RELEASE);
  uint64_t parts = 0, flushes = 2;
  words[s] = 3 * words_of(d, S_POLL) + words_of(d, S_FENCE) +
             words_of(d, S_TRAP);
  *bytes = SEGMENT_ALIGN;
  for (int i = 0; i < n; i++) {
    uint64_t w = part_words(d, &p[i], bytes);
    if (p[i].queue == c) {
      parts += w;
      pm4 += words_of(d, T_FLUSH) + words_of(d, T_ACQUIRE) +
             words_of(d, T_SIGNAL) +
             (uint64_t)p[i].nafter * words_of(d, T_WAIT);
      flushes++;
    } else
      words[s] += w + words_of(d, S_FENCE) +
                  (uint64_t)p[i].nafter * words_of(d, S_POLL);
  }
  if (d->rings[c].kind != RING_AQL) words[c] = pm4 + parts;
  else {
    words[c] = flushes * words_of(d, A_IB) + parts;
    *bytes += 4 * pm4 + flushes * SEGMENT_ALIGN;
  }
}

/* Whether launch [p], whose block is in [args], runs: on the compute ring,
   in the form its ring reads, over a grid and groups of no empty axis,
   within its function's threads per group and shared memory; for an AQL
   packet, whose grid counts work-items in 32 bits, over fewer than 2^32
   along each axis. */
static int launches(const struct rig_amd *d, const struct rig_part *p,
                    const uint8_t *args) {
  const struct rig_amd_launch *l = p->launch.launch;
  if (p->queue != RIG_AMD_COMPUTE || l == NULL || args == NULL ||
      l->packet != (d->rings[p->queue].kind == RING_AQL))
    return 0;
  const struct rig_block *b = (const void *)(args + p->launch.block);
  uint64_t threads = 1;
  for (int k = 0; k < 3; k++) {
    if (b->groups[k] == 0 || b->threads[k] == 0 ||
        b->threads[k] > l->max_threads)
      return 0;
    if (l->packet && (uint64_t)b->groups[k] * b->threads[k] > UINT32_MAX)
      return 0;
    threads *= b->threads[k];
  }
  return threads <= l->max_threads && b->shared <= l->max_shared;
}

/* Whether a part is one the device runs. A part's own declaration past its
   ring or the segment never fits, which also keeps a submission's sum of
   them far from wrapping. */
static int runs(const struct rig_amd *d, const struct rig_part *p, int i,
                const uint8_t *args) {
  if (p->queue != RIG_AMD_COMPUTE && p->queue != RIG_AMD_COPY) return 0;
  uint64_t size = d->rings[p->queue].size;
  switch (p->kind) {
  case RIG_WORDS:
    if (p->words.n >= size) return 0;
    if (d->rings[p->queue].kind == RING_AQL && p->words.n % AQL_WORDS)
      return 0;
    break;
  case RIG_FILL:
    if (p->fill.ring_units >= size ||
        p->fill.segment_bytes >= d->segment.size)
      return 0;
    break;
  case RIG_COPY:
    if (is_copy(p) && p->queue != RIG_AMD_COPY) return 0;
    break;
  case RIG_LAUNCH:
    if (!launches(d, p, args)) return 0;
    break;
  default:
    return 0;
  }
  for (int j = 0; j < p->nafter; j++)
    if (p->after[j] < 0 || p->after[j] >= i) return 0;
  return 1;
}

int rig_amd_room(void *self, const struct rig_part *parts, int n,
                 const uint8_t *args) {
  struct rig_amd *d = self;
  if (n > RIG_AMD_PARTS) return RIG_NEVER;
  for (int i = 0; i < n; i++)
    if (!runs(d, &parts[i], i, args)) return RIG_NEVER;
  uint64_t words[RIG_AMD_QUEUES], bytes;
  need(d, parts, n, words, &bytes);
  struct rig_amd_ring *c = &d->rings[RIG_AMD_COMPUTE];
  struct rig_amd_ring *s = &d->rings[RIG_AMD_COPY];
  struct rig_amd_segment *g = &d->segment;
  if (words[0] >= c->size || 2 * words[1] >= s->size || 2 * bytes > g->size)
    return RIG_NEVER;
  uint64_t word = atomic_load_explicit(d->word, memory_order_acquire);
  c->free = reclaim(&c->marks, c->free, word);
  s->free = reclaim(&s->marks, s->free, word);
  g->free = reclaim(&g->marks, g->free, word);
  int full = c->marks.count == RIG_AMD_MARKS ||
             s->marks.count == RIG_AMD_MARKS ||
             g->marks.count == RIG_AMD_MARKS;
  if (full || c->put + words[0] - c->free > c->size ||
      s->put + 2 * words[1] - s->free > s->size ||
      g->put + 2 * bytes - g->free > g->size)
    return RIG_LATER;
  return RIG_FITS;
}

/* Fills */

int rig_amd_place(void *queue, const uint32_t *words, size_t n) {
  struct rig_amd_writer *w = queue;
  if (n > w->ring_left) return 1;
  if (w->q->kind == RING_AQL && n % AQL_WORDS) return 3;
  put_words(w->q, words, n);
  w->ring_left -= n;
  return 0;
}

int rig_amd_segment(void *queue, size_t n, void **host, uint64_t *address) {
  struct rig_amd_writer *w = queue;
  struct rig_amd_segment *g = &w->d->segment;
  uint64_t take = align_up(n, SEGMENT_ALIGN);
  if (g->put + take > w->segment_end) return 2;
  uint64_t at = g->put % g->size;
  *host = g->host + at;
  *address = g->gpu + at;
  g->put += take;
  return 0;
}

/* Submitting */

/* The state of one submission while it is placed. */
struct submission {
  struct rig_amd *d;
  uint64_t v;
  int used[RIG_AMD_QUEUES];
  int last[RIG_AMD_QUEUES]; /* the queue's last part, or -1 */
  int nsignalled;
  int slot[RIG_AMD_SLOTS];
};

static uint64_t slot_gpu(const struct rig_amd *d, int i) {
  return d->slots_gpu + 8 * (uint64_t)i;
}

/* A new scratch: on an AQL queue, the pending writes to the queue's
   descriptor, placed once, by the queue, between submissions; on either,
   the address the launches from here on name. The one publisher holds the
   lock for a few stores, which the submit waits out: a kernel of this
   submission may need the scratch being published. */
static void scratch(struct submission *s, struct rig_amd_ring *r) {
  struct rig_amd *d = s->d;
  if (!atomic_load_explicit(&d->scratch_ready, memory_order_acquire)) return;
  int idle = 0;
  while (!atomic_compare_exchange_weak(&d->scratch_lock, &idle, 1)) idle = 0;
  if (atomic_load_explicit(&d->scratch_ready, memory_order_relaxed)) {
    for (int i = 0; i < d->scratch_n; i++)
      emit(d, r, T_WRITE, d->scratch_at[i], d->scratch_value[i], 0);
    d->scratch_gpu = d->scratch_next;
    atomic_store_explicit(&d->scratch_ready, 0, memory_order_relaxed);
    atomic_store_explicit(&d->scratch_taken, s->v, memory_order_release);
  }
  atomic_store_explicit(&d->scratch_lock, 0, memory_order_release);
}

/* Places the queue's prefix once per submission: its own wait for v-1
   where the other queue released it, and on compute the waits. */
static void enter(struct submission *s, int q, const struct rig_wait *waits,
                  int nwaits, int copy_parts) {
  struct rig_amd *d = s->d;
  struct rig_amd_ring *r = &d->rings[q];
  if (s->used[q]) return;
  s->used[q] = 1;
  if (q == RIG_AMD_COMPUTE) {
    if (r->released != s->v - 1) emit(d, r, T_WAIT, d->word_gpu, s->v - 1, 0);
    for (int i = 0; i < nwaits; i++)
      emit(d, r, T_WAIT64, waits[i].at, waits[i].value, 0);
    if (nwaits > 0 && copy_parts) {
      emit(d, r, T_SIGNAL, slot_gpu(d, RIG_AMD_SLOT_W), s->v, 0);
      s->slot[s->nsignalled++] = RIG_AMD_SLOT_W;
    }
    return;
  }
  if (r->released != s->v - 1) emit(d, r, S_POLL, d->word_gpu, s->v - 1, 0);
  if (nwaits > 0) emit(d, r, S_POLL, slot_gpu(d, RIG_AMD_SLOT_W), s->v, 0);
}

static void wait_slot(struct submission *s, int q, int j) {
  struct rig_amd *d = s->d;
  emit(d, &d->rings[q], q == RIG_AMD_COMPUTE ? T_WAIT : S_POLL,
       slot_gpu(d, j), s->v, 0);
}

/* Orders part [i] after what it follows on its queue: on compute, a partial
   flush of the dispatches before it (for the first part, those of v-1 where
   this queue released v-1; the own wait covers the other case), for the
   first part the pending scratch writes, which then change the descriptor
   while no dispatch runs, then its slot waits, then, for the first part or
   after a slot wait, the cache acquire, so that it reads what the host, the
   copy queue and other devices wrote. Answers whether it placed the
   acquire. */
static int prepare(struct submission *s, const struct rig_part *parts, int i,
                   int first) {
  struct rig_amd *d = s->d;
  int q = parts[i].queue;
  struct rig_amd_ring *r = &d->rings[q];
  int compute = q == RIG_AMD_COMPUTE, acquire = first;
  if (compute && (!first || r->released == s->v - 1))
    emit(d, r, T_FLUSH, 0, 0, 0);
  if (compute && first) scratch(s, r);
  for (int j = 0; j < parts[i].nafter; j++)
    if (parts[parts[i].after[j]].queue != q) {
      wait_slot(s, q, parts[i].after[j]);
      acquire = 1;
    }
  if (compute && acquire) emit(d, r, T_ACQUIRE, 0, 0, 0);
  return compute && acquire;
}

static void signal_slot(struct submission *s, int q, int i) {
  struct rig_amd *d = s->d;
  struct rig_amd_ring *r = &d->rings[q];
  emit(d, r, q == RIG_AMD_COMPUTE ? T_SIGNAL : S_FENCE, slot_gpu(d, i),
       s->v, 0);
  s->slot[s->nsignalled++] = i;
}

static void copy(struct submission *s, const struct rig_part *p) {
  struct rig_amd *d = s->d;
  struct rig_amd_ring *r = &d->rings[RIG_AMD_COPY];
  uint64_t dst = p->copy.dst + p->copy.dst_offset;
  uint64_t src = p->copy.src + p->copy.src_offset;
  for (uint64_t off = 0; off < p->copy.bytes; off += d->max_copy) {
    uint64_t n = p->copy.bytes - off;
    emit(d, r, S_COPY, dst + off, src + off, n < d->max_copy ? n : d->max_copy);
  }
}

/* Writes into [args] the implicit arguments [l] reads, of a launch of
   block [b]: its grid of whole groups starts at thread 0. */
static void implicit(const struct rig_amd_launch *l, const struct rig_block *b,
                     uint8_t *args) {
  const int32_t *h = l->hidden;
  uint16_t dims = 1, zero16 = 0;
  uint64_t zero64 = 0;
  for (int k = 0; k < 3; k++) {
    uint16_t t = (uint16_t)b->threads[k];
    if ((uint64_t)b->groups[k] * b->threads[k] > 1) dims = (uint16_t)(k + 1);
    if (h[H_BLOCKS + k] >= 0) memcpy(args + h[H_BLOCKS + k], &b->groups[k], 4);
    if (h[H_GROUP + k] >= 0) memcpy(args + h[H_GROUP + k], &t, 2);
    if (h[H_REMAINDER + k] >= 0) memcpy(args + h[H_REMAINDER + k], &zero16, 2);
    if (h[H_OFFSET + k] >= 0) memcpy(args + h[H_OFFSET + k], &zero64, 8);
  }
  if (h[H_DIMS] >= 0) memcpy(args + h[H_DIMS], &dims, 2);
  if (h[H_LDS] >= 0) memcpy(args + h[H_LDS], &b->shared, 4);
}

/* Places launch [p] on the compute ring: its arguments in the segment, its
   parameters with each ref's offset plus its slot's address, then the
   implicit arguments its function reads; then its dispatch, the LDS its
   groups take in their resources. A PM4 dispatch follows the invalidation,
   unless [acquired]; an AQL packet acquires at system scope and waits for
   the packets before it itself. */
static void launch(struct submission *s, const struct rig_part *p,
                   const uint8_t *args, const uint64_t *slots, int acquired) {
  struct rig_amd *d = s->d;
  struct rig_amd_ring *r = &d->rings[RIG_AMD_COMPUTE];
  struct rig_amd_segment *g = &d->segment;
  const struct rig_block *b = (const void *)(args + p->launch.block);
  uint64_t at = g->put % g->size;
  uint8_t *params = g->host + at;
  memcpy(params, b->params, p->launch.params);
  for (int i = 0; i < p->launch.nrefs; i++) {
    const struct rig_ref *f = &p->launch.refs[i];
    uint64_t x;
    memcpy(&x, b->params + f->at, 8);
    x += slots[f->slot];
    memcpy(params + f->at, &x, 8);
  }
  implicit(p->launch.launch, b, params);
  g->put += align_up(arguments(p), SEGMENT_ALIGN);
  if (!acquired && !((const struct rig_amd_launch *)p->launch.launch)->packet)
    emit(d, r, T_INVALIDATE, 0, 0, 0);
  uint64_t a[L_GRID] = {g->gpu + at, d->scratch_gpu};
  for (int k = 0; k < 3; k++) {
    a[L_THREADS + k] = b->threads[k];
    a[L_GROUPS + k] = b->groups[k];
  }
  uint32_t w[RIG_AMD_LAUNCH_WORDS];
  put_words(r, w, (size_t)rig_amd_dispatch(p->launch.launch, a, b->shared, w));
}

static void release(struct submission *s, int q) {
  struct rig_amd *d = s->d;
  struct rig_amd_ring *r = &d->rings[q];
  if (q == RIG_AMD_COMPUTE) {
    emit(d, r, T_RELEASE, d->word_gpu, s->v, 0);
    flush(d, r);
  } else {
    emit(d, r, S_FENCE, d->word_gpu, s->v, 0);
    emit(d, r, S_TRAP, 0, 0, 0);
  }
  r->released = s->v;
}

/* Hands the queues what was placed: the HDP flush the host's writes through
   the BAR need, read back so that it completed, then each queue's write
   position and doorbell. */
static void hand_over(struct submission *s) {
  struct rig_amd *d = s->d;
  for (int i = 0; i < RIG_AMD_HDPS; i++)
    if (atomic_load_explicit(&d->hdps[i].count, memory_order_acquire) > 0) {
      rig_amd_barrier();
      d->hdps[i].reg[0] = 0;
      (void)d->hdps[i].reg[0];
    }
  rig_amd_barrier();
  for (int q = 0; q < RIG_AMD_QUEUES; q++) {
    struct rig_amd_ring *r = &d->rings[q];
    if (!s->used[q]) continue;
    mark(&r->marks, s->v, r->put);
    uint64_t at = r->kind == RING_SDMA  ? 4 * r->put
                  : r->kind == RING_AQL ? r->put / AQL_WORDS
                                        : r->put;
    *r->write = at;
    rig_amd_barrier();
    *r->doorbell = r->kind == RING_AQL ? at - 1 : at;
  }
}

/* The slot of v mod RIG_AMD_SLOTS, if its last write is old, gets
   low32(v-1) from the host: no wait of the next 2^32 values compares with
   it, and no queued work writes it. */
static void refresh_slot(struct rig_amd *d, uint64_t v) {
  int i = (int)(v % RIG_AMD_SLOTS);
  if (v - d->slot_last[i] <= SLOT_STALE) return;
  d->slots[2 * i] = LOW32(v - 1);
  d->slot_last[i] = v - 1;
}

/* After a fill failed, or waits the queue cannot hold, with [failure_text]
   set: drops every word placed for v, places only v's release, after every
   earlier value, on compute, and answers the failure. On an AQL ring the
   dropped packets keep valid headers past the write position, which the
   queue may read: each gets the invalid header type. Bytes the fill took
   stay v's, and return once the word reaches v. Scratch writes v took are
   dropped too, yet count as placed: no work runs after v, so the scratch
   the descriptor still names is retired safely once the word reaches v. */
static int fail(struct submission *s, const char **failure) {
  struct rig_amd *d = s->d;
  for (int q = 0; q < RIG_AMD_QUEUES; q++) {
    struct rig_amd_ring *r = &d->rings[q];
    if (r->kind == RING_AQL)
      for (uint64_t p = r->start; p < r->put; p += AQL_WORDS)
        __atomic_store_n(&r->words[p & (r->size - 1)], AQL_INVALID,
                         __ATOMIC_RELEASE);
    r->put = r->start;
    s->used[q] = 0;
  }
  s->nsignalled = 0;
  d->ib_n = 0;
  enter(s, RIG_AMD_COMPUTE, NULL, 0, 0);
  release(s, RIG_AMD_COMPUTE);
  mark(&d->segment.marks, s->v, d->segment.put);
  hand_over(s);
  atomic_store_explicit(&d->last, s->v, memory_order_release);
  d->failure = d->failure_text;
  *failure = d->failure;
  return RIG_FAILED;
}

int rig_amd_submit(void *self, uint64_t v, const struct rig_wait *waits,
                   int nwaits, const struct rig_part *parts, int nparts,
                   const uint8_t *args, const uint64_t *slots, int nslots,
                   const uint64_t *handles, int nhandles,
                   const char **failure) {
  (void)nslots;
  (void)handles;
  (void)nhandles;
  struct rig_amd *d = self;
  if (d->failure) {
    *failure = d->failure;
    return RIG_FAILED;
  }
  struct submission s = {.d = d, .v = v, .last = {-1, -1}};
  uint8_t named[RIG_AMD_PARTS] = {0};
  int copy_parts = 0;
  for (int i = 0; i < nparts; i++) {
    s.last[parts[i].queue] = i;
    copy_parts |= parts[i].queue == RIG_AMD_COPY;
    for (int j = 0; j < parts[i].nafter; j++)
      if (parts[parts[i].after[j]].queue != parts[i].queue)
        named[parts[i].after[j]] = 1;
  }
  int r = nparts > 0 ? parts[nparts - 1].queue : RIG_AMD_COMPUTE;
  if (r == RIG_AMD_COPY && LOW32(v) == 0) r = RIG_AMD_COMPUTE;
  refresh_slot(d, v);
  for (int q = 0; q < RIG_AMD_QUEUES; q++)
    d->rings[q].start = d->rings[q].put;

  /* Room budgets RIG_AMD_WAITS waits, and a GPU without 64-bit waits has
     an empty wait template: either would hand over work that does not
     wait. */
  if (nwaits > RIG_AMD_WAITS) {
    snprintf(d->failure_text, sizeof d->failure_text,
             "%d waits; the device holds at most %d", nwaits, RIG_AMD_WAITS);
    return fail(&s, failure);
  }
  if (nwaits > 0 && d->templates[T_WAIT64].n == 0) {
    snprintf(d->failure_text, sizeof d->failure_text,
             "a wait on a word; the device's compute queue cannot wait");
    return fail(&s, failure);
  }

  /* The submission's segment bytes lie in one run, which never wraps. */
  uint64_t words[RIG_AMD_QUEUES], bytes;
  need(d, parts, nparts, words, &bytes);
  struct rig_amd_segment *g = &d->segment;
  if (g->put % g->size + bytes > g->size) g->put += g->size - g->put % g->size;
  g->start = g->put;

  struct rig_amd_ring *compute = &d->rings[RIG_AMD_COMPUTE];
  if (nwaits > 0) enter(&s, RIG_AMD_COMPUTE, waits, nwaits, copy_parts);
  int placed[RIG_AMD_QUEUES] = {0};
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    int q = p->queue;
    struct rig_amd_ring *ring = &d->rings[q];
    enter(&s, q, waits, nwaits, copy_parts);
    int acquired = prepare(&s, parts, i, !placed[q]++);
    flush(d, ring);
    if (p->kind == RIG_WORDS) put_words(ring, p->words.at, p->words.n);
    else if (p->kind == RIG_LAUNCH) launch(&s, p, args, slots, acquired);
    else if (p->kind == RIG_FILL) {
      struct rig_amd_writer w = {
          d, ring, p->fill.ring_units,
          g->put + align_up(p->fill.segment_bytes, SEGMENT_ALIGN)};
      int code = p->fill.fn(&w, p->fill.arg, v);
      if (code != 0) {
        snprintf(d->failure_text, sizeof d->failure_text,
                 "a fill on %s failed with %d",
                 q == RIG_AMD_COMPUTE ? "COMPUTE:0" : "COPY:0", code);
        return fail(&s, failure);
      }
    } else if (is_copy(p))
      copy(&s, p);
    if (named[i] || (s.last[q] == i && r != q)) signal_slot(&s, q, i);
  }
  flush(d, compute);
  enter(&s, r, waits, nwaits, copy_parts);
  int o = 1 - r;
  if (s.last[o] >= 0) wait_slot(&s, r, s.last[o]);
  release(&s, r);
  mark(&g->marks, v, g->put);
  hand_over(&s);
  for (int i = 0; i < s.nsignalled; i++) d->slot_last[s.slot[i]] = v;
  atomic_store_explicit(&d->last, v, memory_order_release);
  return RIG_COMMITTED;
}
