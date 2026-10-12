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

   Routes: a copy within this device's Device memory from 256 KiB runs the
   driver's copy kernel on COMPUTE, as a launch; every other part runs on
   its queue's channel. Each part's route is decided once, before any word
   is written.

   Joins: a part that a later part on the other channel runs after (one
   its [after] names, or its queue's previous part), and a channel's last
   part when the other channel releases v, signal their channel's join
   word with the tag v << 16 | i + 1, i the part's index; the waiter
   acquires the tag. Tags rise on each join word, and the acquire compares
   64 bits circularly, so ">=" is exact. The channel of the last part
   (COMPUTE when there is none, or when v's low 32 bits are 0) waits for
   the other channel's last part, then releases v into the timeline word:
   on COMPUTE a release that waits for the channel to be idle, on COPY a
   release after its copies completed.

   Order: the parts on one channel run in array order. COPY runs its copies
   one after another; on COMPUTE a part placed after launches the channel
   has not waited for starts with a wait for idle, as kernels a channel
   schedules run at once.

   Times: a timed value stamps, after its waits, each channel it enters,
   and, before its release, the channel that releases it (rig_nv_stubs.h's
   Times). A calibration writes a stamp between two readings of the host's
   clock, the closest pair of eight kept: at open, and in a settle that
   finds the device idle and the offset a second old. A stamp behind work
   would measure the work's queue, so a busy device keeps its offset.

   Launches: a submission's launches get their descriptors and constant
   banks 0 in COMPUTE's launch ring, in part order; the ring is the GPU's
   memory where its BAR had room, which the GPU reads faster than host
   memory. A launch that directly follows a launch on COMPUTE, with no join
   between them, is chained: the descriptor before it schedules it once
   its own launch completed, so the writer places nothing for it. Any
   other launch is scheduled from the segment, after a wait for idle if
   launches run. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "rig_nv_stubs.h"

/* The waits a submission may carry: rig_nv_room, which does not see
   them, keeps room for this many on each channel. */
#define MAX_WAITS 256

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
  uint64_t bytes = t->nbytes;
  uint64_t start = c->open & (c->size - 1);
  if (start + 4 * c->open_words + bytes > c->size) {
    close_segment(d, c);
    uint64_t end = c->written & (c->size - 1);
    if (end + bytes > c->size) c->written += c->size - end;
    c->open = c->written;
  }
  uint64_t at = c->written & (c->size - 1);
  const uint64_t values[3] = {a, b, n};
  rig_nv_fill(t, values, c->segments + at);
  c->written += bytes;
  c->open_words += bytes / 4;
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

/* Below this size, launching the kernel costs more than the copy saves. */
enum { COPY_KERNEL_BYTES = 256 * 1024 };

/* The copy kernel's parameters: destination, source and bytes. */
enum { COPY_PARAMS = 3 * sizeof(uint64_t) };

/* Whether the driver's copy kernel runs [p]: a copy within this device's
   Device memory, large enough to pay for its launch. */
static int kernel_copies(const struct device *d, const struct rig_part *p) {
  return p->kind == RIG_COPY && p->copy.device_memory &&
         p->copy.bytes >= COPY_KERNEL_BYTES && d->copy_launch != NULL;
}

/* Decides the routes of [p]'s parts into [r], the submission's scratch. The
   part before each on its queue joins it where it ran on the other channel:
   every edge still points forward in array order. The route depends only on
   the parts and on the device's facts, so a room check and its submission
   decide alike. */
static void route(const struct device *d, struct route *r,
                  const struct rig_part *p, int n) {
  int previous[CHANNELS] = {-1, -1};
  for (int i = 0; i < n; i++) {
    struct route *x = &r[i];
    int a = previous[p[i].queue];
    previous[p[i].queue] = i;
    x->kernel = (uint8_t)kernel_copies(d, &p[i]);
    x->channel = (uint8_t)(x->kernel ? COMPUTE : p[i].queue);
    x->awaited = 0;
    x->join = a >= 0 && r[a].channel != x->channel ? a : -1;
    if (x->join >= 0) r[a].awaited = 1;
    for (int j = 0; j < p[i].nafter; j++) {
      a = p[i].after[j];
      if (r[a].channel != x->channel) r[a].awaited = 1;
    }
  }
}

static const struct rig_block *block_of(const struct rig_part *p,
                                        const uint8_t *args) {
  return (const struct rig_block *)(args + p->launch.args.block);
}

static uint64_t round_up(uint64_t n, uint64_t a) {
  return (n + a - 1) & ~(a - 1);
}

/* Whether the launch [p]'s block, in [args], is one its function runs. */
static int launchable(const struct rig_part *p, const uint8_t *args) {
  const struct launch *l = p->launch.launch;
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

/* The bytes of [l]'s constant bank 0 with [params] bytes of parameters: the
   driver's parameters, the kernel's and the bank the descriptor names,
   whichever ends last. */
static uint64_t bank_bytes(const struct launch *l, uint64_t params) {
  uint64_t n = (uint64_t)l->params_at + params;
  if (n < l->bank0.nbytes) n = l->bank0.nbytes;
  return n < l->bank0_bytes ? l->bank0_bytes : n;
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
    if (x->kind == RIG_COPY && x->copy.bytes > 0 && x->queue != COPY)
      return 0;
    if (x->kind == RIG_LAUNCH &&
        (x->queue != COMPUTE || x->launch.launch == NULL ||
         !launchable(x, args)))
      return 0;
    if (words_of(x) % 2 != 0) return 0;
    for (int j = 0; j < x->nafter; j++)
      if (x->after[j] < 0 || x->after[j] >= i) return 0;
  }
  return 1;
}

static uint64_t bytes_of(const struct device *d, int k) {
  return d->t[k].nbytes;
}

/* The channel that releases [v]: the last part's, or COMPUTE when there is
   none or when [v]'s low 32 bits are 0. The copy engine may write a
   release's two 32-bit words one at a time, and the timeline word's high
   word changes only at those values, where the next value's release, on
   the other channel, may follow at once. COMPUTE writes its 64 bits at
   once. The writer assumes the copy engine's two-word release lands as one
   store, which NVIDIA's class headers do not state: were it two, the high
   word of a copy release of v-1 could land after COMPUTE's release of v and
   lower the word. Verified on sm_89 only. */
static int releaser(const struct route *routes, int n, uint64_t v) {
  if (n == 0 || (uint32_t)v == 0) return COMPUTE;
  return routes[n - 1].channel;
}

/* The most entries and segment bytes [p], routed by [routes], takes of
   each channel, the words a channel may owe and MAX_WAITS waits counted,
   and a wrap of the segment ring that wastes a template's words and splits
   a segment; and the most bytes its launches take of the launch ring. */
static void need(const struct device *d, const struct route *routes,
                 const struct rig_part *p, int n, uint64_t *entries,
                 uint64_t *bytes, uint64_t *launched) {
  uint64_t v = atomic_load_explicit(&d->last, memory_order_relaxed) + 1;
  int r = releaser(routes, n, v);
  int used[CHANNELS] = {0, 0};
  used[r] = 1;
  for (int q = 0; q < CHANNELS; q++) entries[q] = bytes[q] = 0;
  /* The most a wrap past the launch ring's end wastes: the largest of the
     launches' descriptors and banks. */
  uint64_t wasted = 0;
  *launched = 0;
  for (int i = 0; i < n; i++) {
    const struct route *x = &routes[i];
    int q = x->channel;
    used[q] = 1;
    entries[q] += 2 + words_of(&p[i]) / 2;
    bytes[q] += ((uint64_t)p[i].nafter + (x->join >= 0)) *
                bytes_of(d, T_ACQUIRE);
    const struct launch *l = NULL;
    uint64_t params = 0;
    if (p[i].kind == RIG_LAUNCH) {
      l = p[i].launch.launch;
      params = p[i].launch.args.params;
    } else if (x->kernel) {
      l = d->copy_launch;
      params = COPY_PARAMS;
    } else if (p[i].kind == RIG_COPY)
      bytes[q] += (p[i].copy.bytes + COPY_MAX - 1) / COPY_MAX *
                  bytes_of(d, T_COPY);
    if (l != NULL) {
      uint64_t qmd = l->qmd[0][0].nbytes + QMD_ALIGN - 1;
      uint64_t bank = bank_bytes(l, params) + BANK_ALIGN - 1;
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
    bytes[q] += bytes_of(d, q == COMPUTE ? T_STAMP : T_COPY_STAMP);
    bytes[q] += (2 + MAX_WAITS) * bytes_of(d, T_ACQUIRE) +
                bytes_of(d, q == COMPUTE ? T_SETUP : T_SETUP_COPY) +
                4 * TEMPLATE_WORDS;
    if (q == COMPUTE)
      bytes[q] += bytes_of(d, T_LOCAL) + bytes_of(d, T_INVALIDATE);
  }
  bytes[r] += bytes_of(d, r == COMPUTE ? T_RELEASE : T_COPY_RELEASE) +
              bytes_of(d, r == COMPUTE ? T_STAMP : T_COPY_STAMP);
  if (*launched > 0) *launched += wasted;
}

int rig_nv_room(void *self, const struct rig_part *p, int n,
                void *scratch, const uint8_t *args) {
  struct device *d = self;
  if (!runs(p, n, args)) return RIG_NEVER;
  route(d, scratch, p, n);
  uint64_t entries[CHANNELS], bytes[CHANNELS], launched;
  need(d, scratch, p, n, entries, bytes, &launched);
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

/* A launch on COMPUTE whose descriptor is written once the part after it
   there is known, which it may chain. */
struct pending {
  const struct launch *l; /* NULL for none */
  uint64_t qmd;
  uint64_t v[VALUES];
};

/* The state of one submission while it is written. */
struct submission {
  struct device *d;
  const struct route *routes; /* its parts' */
  uint64_t v;
  const struct rig_wait *waits;
  int nwaits;
  int used[CHANNELS];
  struct pending open;
  int running; /* whether COMPUTE has launches it has not waited for */
  uint64_t stamps; /* the GPU's address of its slot's records, or 0 */
};

/* Times */

/* Writes [v]'s times, unless written, from the records of its slot [x],
   whose work has completed: the earliest of its channels' starts and its
   end, less the offset. A record that does not hold [v] leaves both 0. */
static void convert(const struct device *d, struct timed *t,
                    volatile const struct stamps *x) {
  uint64_t *times = t->times, start = UINT64_MAX;
  if (times == NULL) return;
  t->times = NULL;
  for (int q = 0; q < CHANNELS; q++)
    if (x->start[q].value == t->v && x->start[q].stamp < start)
      start = x->start[q].stamp;
  if (start == UINT64_MAX || x->end.value != t->v) return;
  times[0] = start - (uint64_t)d->offset;
  times[1] = x->end.stamp - (uint64_t)d->offset;
}

/* The slot of [v]: the times of the earlier value that held it written,
   and, if [times], its records cleared and [v] with [times] kept. Answers
   the GPU's address of its records if [times], 0 otherwise. */
static uint64_t take_slot(struct device *d, uint64_t v, uint64_t *times) {
  uint64_t k = v & (d->slots - 1);
  struct timed *t = &d->timed[k];
  convert(d, t, &d->stamps[k]);
  if (times == NULL) return 0;
  memset((void *)&d->stamps[k], 0, sizeof d->stamps[k]);
  *t = (struct timed){v, times};
  return d->stamps_gpu + k * sizeof(struct stamps);
}

/* Places a stamp of [v] at [address] on channel [q]. */
static void stamp(struct device *d, int q, uint64_t address, uint64_t v) {
  emit(d, &d->ch[q], q == COMPUTE ? T_STAMP : T_COPY_STAMP, address, v, 0);
}

/* The age of an offset that a settle on an idle device measures again. */
#define RECALIBRATE_NS UINT64_C(1000000000)

/* Whether every value [d] was given is reached. */
static int idle(const struct device *d) {
  return atomic_load_explicit(d->word, memory_order_acquire) ==
         atomic_load_explicit(&d->last, memory_order_relaxed);
}

void rig_nv_settle(void *self, uint64_t v) {
  struct device *d = self;
  uint64_t k = v & (d->slots - 1);
  if (idle(d) && rig_now_ns() - d->calibrated >= RECALIBRATE_NS)
    rig_nv_calibrate(d);
  if (d->timed[k].v == v) convert(d, &d->timed[k], &d->stamps[k]);
}

/* The calibrations kept from. */
#define CALIBRATIONS 8

/* Writes a stamp of [i] into the calibration record on COMPUTE, after the
   channel's setup if owed, and waits for it: the host's clock just before
   the doorbell into [*before], just after the stamp shows into [*after].
   Answers 0 if no stamp shows within a second. The channel's reached
   submissions are reclaimed first, so that the words fit, and the words
   are marked as value 0's, which every room check reclaims. */
static int calibration(struct device *d, uint64_t i, uint64_t *before,
                       uint64_t *after) {
  struct channel *c = &d->ch[COMPUTE];
  reclaim(c, atomic_load_explicit(d->word, memory_order_acquire));
  c->open = c->written;
  c->open_words = 0;
  if (c->owes_setup) {
    emit(d, c, T_SETUP, 0, 0, 0);
    c->owes_setup = 0;
  }
  d->clock->stamp = 0;
  stamp(d, COMPUTE, d->clock_gpu, i);
  close_segment(d, c);
  c->marks[(c->first + c->count++) & (c->entries - 1)] =
      (struct mark){0, c->put, c->written, c->launched};
  store_fence();
  *c->gp_put = (uint32_t)(c->put & (c->entries - 1));
  rig_nv_flush(d, 0);
  *before = rig_now_ns();
  *d->doorbell = c->token;
  while (d->clock->value != i || d->clock->stamp == 0) {
    *after = rig_now_ns();
    if (*after - *before > 1000000000) return 0;
  }
  *after = rig_now_ns();
  return 1;
}

int rig_nv_calibrate(struct device *d) {
  uint64_t best = UINT64_MAX;
  d->calibrated = rig_now_ns();
  for (uint64_t i = 1; i <= CALIBRATIONS; i++) {
    uint64_t before, after;
    if (!calibration(d, i, &before, &after)) return 0;
    if (after - before >= best) continue;
    best = after - before;
    d->offset = (int64_t)(d->clock->stamp - (before + best / 2));
  }
  return 1;
}

/* Opens channel [q] for the work of [s], once per submission: its owed
   words, its wait for v-1 and the foreign waits. */
static void enter(struct submission *s, int q) {
  struct device *d = s->d;
  if (s->used[q]) return;
  s->used[q] = 1;
  struct channel *c = &d->ch[q];
  c->open = c->written;
  c->open_words = 0;
  if (c->owes_setup) {
    emit(d, c, q == COMPUTE ? T_SETUP : T_SETUP_COPY, 0, 0, 0);
    c->owes_setup = 0;
  }
  if (c->released != s->v - 1) emit(d, c, T_ACQUIRE, d->word_gpu, s->v - 1, 0);
  for (int i = 0; i < s->nwaits; i++)
    emit(d, c, T_ACQUIRE, s->waits[i].at, s->waits[i].value, 0);
  if (s->stamps != 0)
    stamp(d, q, s->stamps + (uint64_t)q * sizeof(struct record), s->v);
  if (q != COMPUTE) return;
  /* Only the OCaml side writes a pending memory, so an exchange after the
     load takes a pending one too: the newer, if it replaced it. */
  uint64_t local = atomic_load_explicit(&d->local, memory_order_acquire);
  if (local != 0 && !(local & LOCAL_TAKEN)) {
    local = atomic_exchange(&d->local, s->v << 1 | LOCAL_TAKEN);
    uint64_t address = local & ((UINT64_C(1) << LOCAL_ADDRESS_BITS) - 1);
    uint64_t per_tpc = (local >> LOCAL_ADDRESS_BITS) << LOCAL_UNIT_SHIFT;
    emit(d, c, T_LOCAL, address, per_tpc, 0);
  }
  if (atomic_exchange(&d->invalidate, 0))
    emit(d, c, T_INVALIDATE, 0, 0, 0);
}

/* Places the waits of part [i] for the parts on the other channel it runs
   after: its join, then those its [after] names. Answers whether it placed
   one. */
static int join(struct submission *s, const struct rig_part *p, int i) {
  struct device *d = s->d;
  const struct route *x = &s->routes[i];
  struct channel *c = &d->ch[x->channel];
  uint64_t at = JOIN_GPU(d, other(x->channel));
  int waits = x->join >= 0;
  if (waits) emit(d, c, T_ACQUIRE, at, tag(s->v, x->join), 0);
  for (int j = 0; j < p[i].nafter; j++) {
    int a = p[i].after[j];
    if (s->routes[a].channel == x->channel) continue;
    emit(d, c, T_ACQUIRE, at, tag(s->v, a), 0);
    waits = 1;
  }
  return waits;
}

/* Signals and releases on COMPUTE wait for idle first. */
static void signal_word(struct device *d, int q, uint64_t address,
                        uint64_t value) {
  emit(d, &d->ch[q], q == COMPUTE ? T_RELEASE : T_COPY_RELEASE, address,
       value, 0);
}

/* Launches' memory */

/* Takes [bytes] of [c]'s launch ring at its [launched] count, aligned to
   [align], a power of two that divides the ring's size: their count. The
   bytes never wrap: ones that would pass the ring's end start it. */
static uint64_t take(struct channel *c, uint64_t bytes, uint64_t align) {
  uint64_t size = c->launches_size;
  uint64_t x = round_up(c->launched, align);
  if ((x & (size - 1)) + bytes > size) x = (x | (size - 1)) + 1;
  c->launched = x + bytes;
  return x;
}

static uint64_t gpu_at(const struct channel *c, uint64_t x) {
  return c->launches_gpu + (x & (c->launches_size - 1));
}

static uint8_t *host_at(const struct channel *c, uint64_t x) {
  return c->launches + (x & (c->launches_size - 1));
}

/* A launch's values over [b], with its bank at [bank] and no descriptor
   chained. */
static void values_of(const struct rig_block *b, uint64_t bank, uint64_t *v) {
  for (int i = 0; i < 3; i++) {
    v[V_GRID_X + i] = b->groups[i];
    v[V_BLOCK_X + i] = b->threads[i];
  }
  v[V_SHARED] = round_up(b->shared, 128);
  v[V_BANK0] = bank;
  v[V_NEXT] = 0;
}

/* Writes [o]'s descriptor, chaining the descriptor at [next] if not 0, and
   leaves no launch open. Built on the stack, then copied, so that the
   ring's memory is only written. */
static void settle(const struct channel *c, struct pending *o, uint64_t next) {
  if (o->l == NULL) return;
  uint8_t w[STRUCTURE_BYTES];
  o->v[V_NEXT] = next == 0 ? 0 : gpu_at(c, next);
  const struct template *s = &o->l->qmd[o->v[V_SHARED] > 0][next != 0];
  rig_nv_fill(s, o->v, w);
  memcpy(host_at(c, o->qmd), w, s->nbytes);
  o->l = NULL;
}

/* Places a launch of [l] over [b] on COMPUTE, after [waited] says its part
   placed waits for the other channel: its descriptor and its constant bank
   0 in the launch ring, with the driver's parameters; then, unless the
   launch before it there chains it, a wait for idle if launches run, and
   its schedule. Answers where its [n] bytes of parameters go, which the
   caller writes. */
static uint8_t *dispatch(struct submission *s, const struct launch *l,
                         const struct rig_block *b, uint64_t n, int waited) {
  struct device *d = s->d;
  struct channel *c = &d->ch[COMPUTE];
  uint64_t qmd = take(c, l->qmd[0][0].nbytes, QMD_ALIGN);
  uint64_t bank = take(c, bank_bytes(l, n), BANK_ALIGN);
  struct pending next = {l, qmd, {0}};
  values_of(b, gpu_at(c, bank), next.v);
  uint8_t w[STRUCTURE_BYTES];
  rig_nv_fill(&l->bank0, next.v, w);
  memcpy(host_at(c, bank), w, l->bank0.nbytes);
  int chain = s->open.l != NULL && !waited;
  settle(c, &s->open, chain ? qmd : 0);
  if (!chain) {
    if (s->running) emit(d, c, T_IDLE, 0, 0, 0);
    emit(d, c, T_SCHEDULE, gpu_at(c, qmd), 0, 0);
  }
  s->open = next;
  return host_at(c, bank) + l->params_at;
}

static void launch(struct submission *s, const struct rig_part *p,
                   const uint8_t *args, const uint64_t *slots, int waited) {
  uint8_t *params = dispatch(s, p->launch.launch, block_of(p, args),
                             p->launch.args.params, waited);
  rig_params(params, &p->launch.args, args, slots);
}

/* Each thread copies vectors in a grid-stride loop; a bounded grid makes a
   large copy take a fixed amount of scheduling work. */
#define COPY_THREADS 256
#define COPY_GROUPS 256
#define COPY_VECTOR_BYTES 16

static void copy_compute(struct submission *s, const struct rig_part *p,
                         int waited) {
  uint64_t width = COPY_THREADS * COPY_VECTOR_BYTES;
  uint64_t groups = (p->copy.bytes + width - 1) / width;
  struct rig_block b = {
      .groups = {groups < COPY_GROUPS ? (uint32_t)groups : COPY_GROUPS, 1, 1},
      .threads = {COPY_THREADS, 1, 1}};
  uint64_t params[3] = {p->copy.dst + p->copy.dst_offset,
                        p->copy.src + p->copy.src_offset, p->copy.bytes};
  memcpy(dispatch(s, s->d->copy_launch, &b, sizeof params, waited), params,
         sizeof params);
}

/* Places the words or the copy engine's copies of [p] on [c]. */
static void place(struct device *d, struct channel *c,
                  const struct rig_part *p) {
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
                  void *scratch, const uint8_t *args,
                  const uint64_t *slots, int nslots, uint64_t *times,
                  const char **failure) {
  (void)nslots;
  (void)failure;
  struct device *d = self;
  struct channel *compute = &d->ch[COMPUTE];
  struct route *routes = scratch;
  struct submission s = {.d = d,
                         .routes = routes,
                         .v = v,
                         .waits = waits,
                         .nwaits = nwaits,
                         .stamps = take_slot(d, v, n > 0 ? times : NULL)};
  route(d, routes, p, n);
  int r = releaser(routes, n, v);
  int last[CHANNELS] = {-1, -1};
  for (int i = 0; i < n; i++) last[routes[i].channel] = i;
  uint64_t start = compute->launched;
  for (int i = 0; i < n; i++) {
    const struct route *x = &routes[i];
    int q = x->channel;
    enter(&s, q);
    int waited = join(&s, p, i);
    if (p[i].kind == RIG_LAUNCH) launch(&s, &p[i], args, slots, waited);
    else if (x->kernel) copy_compute(&s, &p[i], waited);
    else {
      if (q == COMPUTE) {
        settle(compute, &s.open, 0);
        if (s.running) emit(d, compute, T_IDLE, 0, 0, 0);
      }
      place(d, &d->ch[q], &p[i]);
    }
    if (q == COMPUTE) s.running = 1;
    if (x->awaited || (i == last[q] && q != r)) {
      signal_word(d, q, JOIN_GPU(d, q), tag(v, i));
      if (q == COMPUTE) {
        settle(compute, &s.open, 0);
        s.running = 0;
      }
    }
  }
  settle(compute, &s.open, 0);
  enter(&s, r);
  int o = other(r);
  if (s.used[o])
    emit(d, &d->ch[r], T_ACQUIRE, JOIN_GPU(d, o), tag(v, last[o]), 0);
  if (s.stamps != 0)
    stamp(d, r, s.stamps + offsetof(struct stamps, end), v);
  signal_word(d, r, d->word_gpu, v);
  d->ch[r].released = v;
  for (int q = 0; q < CHANNELS; q++) {
    if (!s.used[q]) continue;
    struct channel *c = &d->ch[q];
    close_segment(d, c);
    c->marks[(c->first + c->count++) & (c->entries - 1)] =
        (struct mark){v, c->put, c->written, c->launched};
  }
  store_fence();
  for (int q = 0; q < CHANNELS; q++)
    if (s.used[q])
      *d->ch[q].gp_put = (uint32_t)(d->ch[q].put & (d->ch[q].entries - 1));
  rig_nv_flush(d, compute->bar && compute->launched != start);
  for (int q = 0; q < CHANNELS; q++)
    if (s.used[q]) *d->doorbell = d->ch[q].token;
  atomic_store_explicit(&d->last, v, memory_order_release);
  return RIG_COMMITTED;
}
