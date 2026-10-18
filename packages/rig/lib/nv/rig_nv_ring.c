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

   Routes: a copy of one row within this device's Device memory from
   256 KiB runs the driver's copy kernel on COMPUTE, as a launch; every
   other part runs on its queue's channel. Each part's route is decided
   once, before any word is written.

   Joins: a part that a later part on the other channel runs after (one
   its [after] names, or its queue's previous part), and a channel's last
   part when the other channel releases v, signal their channel's join
   word with the channel's next join number, counted from JOINS_START at
   the device's open; the waiter acquires that number. A channel's numbers
   rise in the order its work runs, and the acquire compares 64 bits
   circularly, so ">=" is exact. The channel of the last part
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
   Times).

   Launches: a submission's launches get their descriptors and constant
   banks 0 in COMPUTE's launch ring, in part order; the ring is the GPU's
   memory where its BAR had room, which the GPU reads faster than host
   memory. A launch that directly follows a launch on COMPUTE, with no join
   between them, is chained: the descriptor before it schedules it once
   its own launch completed, so the writer places nothing for it. Any
   other launch is scheduled from the segment, after a wait for idle if
   launches run. */

#define _GNU_SOURCE

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "rig_nv_stubs.h"

/* The waits a submission may carry: rig_nv_room, which does not see
   them, keeps room for this many on each channel. */
#define MAX_WAITS 256

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

/* Writes template [k] with its [values] into the open segment of [c]. A
   segment never wraps: one whose words would pass the ring's end, including
   one that ends exactly there, is closed first, and the words go at the
   start of the ring when they do not fit before its end. A segment whose
   words would pass SEGMENT_BYTES_MAX is closed first too. */
static void emit_values(const struct device *d, struct channel *c, int k,
                        const uint64_t values[TEMPLATE_VALUES]) {
  const struct template *t = &d->t[k];
  uint64_t bytes = t->nbytes;
  if (4 * c->open_words + bytes > SEGMENT_BYTES_MAX) close_segment(d, c);
  uint64_t start = c->open & (c->size - 1);
  if (start + 4 * c->open_words + bytes > c->size) {
    close_segment(d, c);
    uint64_t end = c->written & (c->size - 1);
    if (end + bytes > c->size) c->written += c->size - end;
    c->open = c->written;
  }
  uint64_t at = c->written & (c->size - 1);
  rig_nv_fill(t, values, c->segments + at);
  c->written += bytes;
  c->open_words += bytes / 4;
}

/* [emit_values] of template [k] with the values [a], [b], [n]. */
static void emit(const struct device *d, struct channel *c, int k, uint64_t a,
                 uint64_t b, uint64_t n) {
  const uint64_t values[TEMPLATE_VALUES] = {a, b, n};
  emit_values(d, c, k, values);
}

/* Copies [n] words of ring entries, two words each, low first, after the
   open segment's entry. No words place nothing, so the segment stays
   open. */
static void put_entries(const struct device *d, struct channel *c,
                        const uint32_t *words, size_t n) {
  if (n == 0) return;
  close_segment(d, c);
  for (size_t i = 0; i + 1 < n; i += 2)
    c->ring[c->put++ & (c->entries - 1)] =
        (uint64_t)words[i] | (uint64_t)words[i + 1] << 32;
}

/* Room */

/* Returns to [c] what the submissions up to [seen] used. A submission
   before a growth used rings the growth replaced, of which nothing returns
   to [c]: the OCaml side frees them once [seen] passes them. */
static void reclaim(const struct device *d, struct channel *c,
                    uint64_t seen) {
  while (c->count > 0) {
    const struct mark *m = &c->marks[c->first];
    if (m->v > seen) return;
    c->freed = m->put;
    if (m->v >= c->segments_from) c->reclaimed = m->written;
    if (m->v >= c->launches_from) c->landed = m->launched;
    c->first = (c->first + 1) & (d->slots - 1);
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
  return p->kind == RIG_COPY && !p->copy.fill && p->copy.rows == 1 &&
         p->copy.device_memory && p->copy.width >= COPY_KERNEL_BYTES &&
         d->copy_launch != NULL;
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

/* The parameter bytes a launch of [l] given [params] places: those its
   kernel reads, so that the bank the descriptor names holds them, as CUDA
   leaves out a launch's bytes past its function's. */
static uint64_t placed(const struct launch *l, uint64_t params) {
  return params < l->params ? params : l->params;
}

/* The bytes of [l]'s constant bank 0 with [params] bytes of parameters: the
   driver's parameters, the kernel's it places and the bank the descriptor
   names, whichever ends last, at most the bank: entry checked that the
   kernel's parameters end within it. */
static uint64_t bank_bytes(const struct launch *l, uint64_t params) {
  uint64_t n = (uint64_t)l->params_at + placed(l, params);
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
  if (n < 0) return 0;
  for (int i = 0; i < n; i++) {
    const struct rig_part *x = &p[i];
    if (x->queue != COMPUTE && x->queue != COPY) return 0;
    if (x->kind != RIG_WORDS && x->kind != RIG_COPY && x->kind != RIG_LAUNCH)
      return 0;
    if (x->kind == RIG_COPY && x->copy.width > 0 && x->copy.rows > 0 &&
        x->queue != COPY)
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

static uint64_t ceil_div(uint64_t n, uint64_t k) { return (n + k - 1) / k; }

/* A fill of [n] bytes at [dst] as the copy engine runs it: [*head] bytes of
   one-byte components up to a multiple of 4, [*body] bytes of four-byte
   components, then the [*tail] bytes left, at most three. */
static void fill_split(uint64_t dst, uint64_t n, uint64_t *head,
                       uint64_t *body, uint64_t *tail) {
  uint64_t h = (4 - (dst & 3)) & 3;
  if (h > n) h = n;
  *head = h;
  *body = (n - h) & ~(uint64_t)3;
  *tail = n - h - *body;
}

/* Whether the copy part [p] goes as multi-line launches of up to LINES_MAX
   rows each: it has rows, each fits one launch, and its pitches fit the
   launch's fields. A run goes as single lines, as NVIDIA's UVM driver
   issues them. */
static int multi_line(const struct rig_part *p) {
  return p->copy.rows > 1 && p->copy.width <= COPY_MAX &&
         p->copy.src_pitch <= LINES_MAX && p->copy.dst_pitch <= LINES_MAX;
}

/* The bytes of the templates [place_copy] places for the copy part [p]. */
static uint64_t copy_bytes(const struct device *d, const struct rig_part *p) {
  uint64_t width = p->copy.width, rows = p->copy.rows;
  if (width == 0 || rows == 0) return 0;
  if (p->copy.fill) {
    uint64_t head, body, tail;
    fill_split(p->copy.dst + p->copy.dst_offset, width, &head, &body, &tail);
    return (uint64_t)((head > 0) + (tail > 0)) * bytes_of(d, T_FILL_BYTES) +
           ceil_div(body, COPY_MAX) * bytes_of(d, T_FILL);
  }
  if (multi_line(p)) return ceil_div(rows, LINES_MAX) * bytes_of(d, T_BOX);
  return rows * ceil_div(width, COPY_MAX) * bytes_of(d, T_COPY);
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

/* What a submission takes of the device's room: entries and segment bytes
   of each channel, and bytes of the launch ring. */
struct cost {
  uint64_t entries[CHANNELS], bytes[CHANNELS], launched;
};

/* The bytes of a release of the value on channel [q]. */
static uint64_t release_bytes(const struct device *d, int q) {
  return bytes_of(d, q == COMPUTE ? T_RELEASE : T_COPY_RELEASE);
}

/* The bytes of a stamp of the value's time on channel [q]. */
static uint64_t stamp_bytes(const struct device *d, int q) {
  return bytes_of(d, q == COMPUTE ? T_STAMP : T_COPY_STAMP);
}

/* The bytes that end the value on the channel [q] that releases it: its
   release and its last stamp. */
static uint64_t end_bytes(const struct device *d, int q) {
  return release_bytes(d, q) + stamp_bytes(d, q);
}

/* Adds what channel [q] takes when a submission uses it, besides its parts:
   the words a channel may owe, its first stamp, MAX_WAITS waits, and a wrap
   of the segment ring that wastes a template's words and splits a
   segment. */
static void channel_cost(const struct device *d, int q, struct cost *k) {
  k->entries[q] += 3;
  k->bytes[q] += stamp_bytes(d, q) +
                 (2 + MAX_WAITS) * bytes_of(d, T_ACQUIRE) +
                 bytes_of(d, q == COMPUTE ? T_SETUP : T_SETUP_COPY) +
                 4 * TEMPLATE_WORDS;
  if (q == COMPUTE)
    k->bytes[q] += bytes_of(d, T_LOCAL) + bytes_of(d, T_INVALIDATE);
}

/* Adds what a part on channel [q] takes that places [words] ring words,
   waits for [joins] parts of the other channel, schedules a launch of a
   [qmd]-byte descriptor and a [bank]-byte constant bank (both 0 for none),
   and places [copies] bytes of the copy engine's launches. Its ring words
   take an entry each two, after the entry that closes the segment before
   them; the driver's own words go into segments, whose closes
   [segment_closes] counts. */
static void part_cost(const struct device *d, int q, uint64_t words,
                      uint64_t joins, uint64_t qmd, uint64_t bank,
                      uint64_t copies, struct cost *k) {
  if (words > 0) k->entries[q] += 1 + words / 2;
  k->bytes[q] +=
      joins * bytes_of(d, T_ACQUIRE) + copies + release_bytes(d, q);
  if (qmd + bank > 0) {
    k->bytes[q] += bytes_of(d, T_SCHEDULE);
    k->launched += qmd + bank;
  }
  if (q == COMPUTE) k->bytes[q] += bytes_of(d, T_IDLE);
}

/* The entries that close segments of [bytes] of the driver's words when
   they reach SEGMENT_BYTES_MAX, beyond those [channel_cost] counts: a
   segment closes so after at least half of that, as a template takes far
   less. */
static uint64_t segment_closes(uint64_t bytes) {
  return bytes / (SEGMENT_BYTES_MAX / 2);
}

/* What [p], routed by [routes], takes, with the most a wrap past the launch
   ring's end wastes: the largest of its launches' descriptors and banks. */
static struct cost need(const struct device *d, const struct route *routes,
                        const struct rig_part *p, int n) {
  uint64_t v = atomic_load_explicit(&d->last, memory_order_relaxed) + 1;
  int r = releaser(routes, n, v);
  int used[CHANNELS] = {0, 0};
  used[r] = 1;
  struct cost k = {0};
  uint64_t wasted = 0;
  for (int i = 0; i < n; i++) {
    const struct route *x = &routes[i];
    int q = x->channel;
    used[q] = 1;
    const struct launch *l = NULL;
    uint64_t params = 0, qmd = 0, bank = 0, copies = 0;
    if (p[i].kind == RIG_LAUNCH) {
      l = p[i].launch.launch;
      params = p[i].launch.args.params;
    } else if (x->kernel) {
      l = d->copy_launch;
      params = COPY_PARAMS;
    } else if (p[i].kind == RIG_COPY)
      copies = copy_bytes(d, &p[i]);
    if (l != NULL) {
      qmd = l->qmd[0][0].nbytes + QMD_ALIGN - 1;
      bank = bank_bytes(l, params) + BANK_ALIGN - 1;
      if (qmd > wasted) wasted = qmd;
      if (bank > wasted) wasted = bank;
    }
    part_cost(d, q, words_of(&p[i]),
              (uint64_t)p[i].nafter + (x->join >= 0), qmd, bank, copies, &k);
  }
  for (int q = 0; q < CHANNELS; q++)
    if (used[q]) channel_cost(d, q, &k);
  k.bytes[r] += end_bytes(d, r);
  for (int q = 0; q < CHANNELS; q++)
    k.entries[q] += segment_closes(k.bytes[q]);
  if (k.launched > 0) k.launched += wasted;
  return k;
}

/* Adds [r] to [d]'s retired rings, which work after [until] never uses. */
static void retire(struct device *d, struct ring *r, uint64_t until) {
  r->until = until;
  r->next = atomic_load_explicit(&d->retired, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(
      &d->retired, &r->next, r, memory_order_release, memory_order_relaxed)) {
  }
}

/* Installs the rings a growth made that [d]'s rings are smaller than, for
   the work of the next value on. A ring one
   replaces is retired once the work before that value is done, as earlier
   work may still read it; a ring no larger than [d]'s is retired at once,
   as no work used it. */
static void install(struct device *d) {
  uint64_t v = atomic_load_explicit(&d->last, memory_order_relaxed) + 1;
  for (int i = 0; i <= CHANNELS; i++) {
    struct ring *g = atomic_exchange(&d->grown[i], NULL);
    if (g == NULL) continue;
    struct channel *c = &d->ch[i < CHANNELS ? i : COMPUTE];
    if (i < CHANNELS && g->size > c->size) {
      c->segments = g->host;
      c->segments_gpu = g->gpu;
      c->size = g->size;
      c->written = c->reclaimed = c->open = c->open_words = 0;
      c->segments_from = v;
    } else if (i == CHANNELS && g->size > c->launches_size) {
      c->launches = g->host;
      c->launches_gpu = g->gpu;
      c->launches_size = g->size;
      c->bar = g->bar;
      c->launched = c->landed = 0;
      c->launches_from = v;
    } else {
      retire(d, g, 0);
      continue;
    }
    struct ring *replaced = atomic_exchange(&d->current[i], g);
    if (replaced != NULL) retire(d, replaced, v - 1);
  }
}

/* Whether [d]'s rings hold what [k] takes, besides the work on them. */
static int holds(const struct device *d, const struct cost *k) {
  for (int q = 0; q < CHANNELS; q++)
    if (k->bytes[q] > d->ch[q].size) return 0;
  return k->launched <= d->ch[COMPUTE].launches_size;
}

/* Whether channels whose rings hold [compute] and [copy] entries hold the
   entries [k] takes, besides the work on them: a ring of [n] entries holds
   n - 1, since its put and get positions are equal only when it is empty. */
static int enters(uint64_t compute, uint64_t copy, const struct cost *k) {
  return k->entries[COMPUTE] < compute && k->entries[COPY] < copy;
}

static int entered(const struct device *d, const struct cost *k) {
  return enters(d->ch[COMPUTE].entries, d->ch[COPY].entries, k);
}

/* Installs the channels a growth made, while [d] drains or if their rings
   hold the entries [k] takes, once [d] reached every value submitted:
   RIG_FITS once installed, RIG_LATER until [d] reached them, and RIG_GROW
   if no channels a growth made hold them. Until then [d] drains: every
   room answers RIG_LATER, so that the device's work ends even while other
   threads submit. The channels installed before then hold no work, and the
   next submission on each new one places its setup words and, on COMPUTE,
   its local memory, which belong to a channel's context. */
static int switch_channels(struct device *d, const struct cost *k) {
  pthread_mutex_lock(&d->channels_mu);
  const struct channels *g = d->grown_channels;
  int r = RIG_GROW;
  if (g != NULL &&
      (d->draining ||
       enters(g->ch[COMPUTE].entries, g->ch[COPY].entries, k))) {
    uint64_t seen = atomic_load_explicit(d->word, memory_order_acquire);
    r = seen == atomic_load_explicit(&d->last, memory_order_relaxed)
            ? RIG_FITS
            : RIG_LATER;
    d->draining = r == RIG_LATER;
    for (int q = 0; r == RIG_FITS && q < CHANNELS; q++) {
      struct channel *c = &d->ch[q];
      reclaim(d, c, seen);
      c->ring = g->ch[q].ring;
      c->entries = g->ch[q].entries;
      c->gp_put = g->ch[q].gp_put;
      c->token = g->ch[q].token;
      c->notifier = g->ch[q].notifier;
      c->put = c->freed = 0;
      c->owes_setup = 1;
    }
    if (r == RIG_FITS) {
      d->ch[COMPUTE].owes_local = 1;
      d->installed = g->id;
      d->grown_channels = NULL;
    }
  }
  pthread_mutex_unlock(&d->channels_mu);
  return r;
}

_Thread_local struct asked rig_nv_asked;

/* Records, for the growth rig asks next on this thread, the bytes each of
   [d]'s rings lacks to hold [k], and the entries each channel's ring lacks.
   Each submission's growth makes what it needs, so another's growth,
   larger or short of memory, never stands in for it. */
static void ask(const struct device *d, const struct cost *k) {
  rig_nv_asked.d = d;
  for (int q = 0; q < CHANNELS; q++) {
    const struct channel *c = &d->ch[q];
    rig_nv_asked.bytes[q] = k->bytes[q] > c->size ? k->bytes[q] : 0;
    rig_nv_asked.entries[q] = k->entries[q] < c->entries ? 0 : k->entries[q];
  }
  rig_nv_asked.bytes[CHANNELS] =
      k->launched > d->ch[COMPUTE].launches_size ? k->launched : 0;
}

int rig_nv_room(void *self, const struct rig_part *p, int n,
                void *scratch, const uint8_t *args) {
  struct device *d = ((struct rig_nv_edge *)self)->d;
  if (!runs(p, n, args)) return RIG_NEVER;
  route(d, scratch, p, n);
  struct cost k = need(d, scratch, p, n);
  if (!enters(ENTRIES_MAX, ENTRIES_MAX, &k)) return RIG_NEVER;
  if (!holds(d, &k)) install(d);
  if ((d->draining || !entered(d, &k)) &&
      switch_channels(d, &k) == RIG_LATER)
    return RIG_LATER;
  if (!holds(d, &k) || !entered(d, &k)) {
    ask(d, &k);
    return RIG_GROW;
  }
  const struct channel *compute = &d->ch[COMPUTE];
  uint64_t seen = atomic_load_explicit(d->word, memory_order_acquire);
  /* Each value in flight holds a slot of the stamp ring and a mark of each
     channel it uses. */
  if (atomic_load_explicit(&d->last, memory_order_relaxed) + 1 - seen >
      d->slots)
    return RIG_LATER;
  for (int q = 0; q < CHANNELS; q++) {
    struct channel *c = &d->ch[q];
    reclaim(d, c, seen);
    if (c->entries - 1 - (c->put - c->freed) < k.entries[q] ||
        c->size - (c->written - c->reclaimed) < k.bytes[q])
      return RIG_LATER;
  }
  if (compute->launches_size - (compute->launched - compute->landed) <
      k.launched)
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
  uint64_t joins[CHANNELS];   /* each channel's last join before it */
  uint64_t v;
  const struct rig_wait *waits;
  int nwaits;
  int used[CHANNELS];
  struct pending open;
  int running; /* whether COMPUTE has launches it has not waited for */
  struct timed *timed; /* its slot, NULL unless it is timed */
};

/* Times */

/* The timer's registers in BAR 0: NV_PTIMER_TIME_0 and NV_PTIMER_TIME_1,
   its low and high 32 bits (open-gpu-kernel-modules' dev_timer.h). */
#define PTIMER_TIME_0 0x9400
#define PTIMER_TIME_1 0x9410

static uint32_t timer_word(const struct device *d, uint32_t at) {
  return *(volatile const uint32_t *)(d->bar + at);
}

/* The timer's registers are read high, low, high, as RM reads them, again
   while the high word moved, between two readings of the host's clock,
   whose middle is the host's time. */
int rig_nv_measure(const struct device *d, struct measure *m,
                   uint64_t *width) {
  if (d->source != NULL) {
    if (d->source->read(d->source, &m->gpu, &m->host, width) != 0)
      return MEASURE_FAILED;
    return *width <= READ_BOUND_NS ? MEASURE_OK : MEASURE_WIDE;
  }
  uint64_t before = rig_now_ns();
  uint32_t hi, lo;
  do {
    hi = timer_word(d, PTIMER_TIME_1);
    lo = timer_word(d, PTIMER_TIME_0);
  } while (timer_word(d, PTIMER_TIME_1) != hi);
  uint64_t after = rig_now_ns();
  m->gpu = (uint64_t)hi << 32 | lo;
  m->host = before + (after - before) / 2;
  *width = after - before;
  return *width <= READ_BOUND_NS ? MEASURE_OK : MEASURE_WIDE;
}

static uint64_t later(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* The host's time at the timer's [g], along the line through the first and
   the latest measures once they lie RATE_BASE_NS apart, at the timer's
   nominal rate of one nanosecond a tick from the latest before. */
static uint64_t host_ns(const struct device *d, uint64_t g) {
  const struct measure *a = &d->first, *b = &d->latest;
  int64_t dg = (int64_t)(g - b->gpu);
  if (b->host - a->host < RATE_BASE_NS) return b->host + (uint64_t)dg;
  double rate = (double)(int64_t)(b->host - a->host) /
                (double)(int64_t)(b->gpu - a->gpu);
  double dh = (double)dg * rate;
  return b->host + (uint64_t)(int64_t)(dh < 0 ? dh - 0.5 : dh + 0.5);
}

static const char unwritten[] = "a stamp of the GPU's work was never written";

/* Writes the times of slot [k]'s value, whose work has completed and which
   is the next to convert: the earliest of its channels' starts and its
   end, each no earlier than the end of the value converted before it,
   which completed first. Its times go, written or not; a stamp the work
   did not write is the device's failure. */
static void convert(struct device *d, uint64_t k) {
  struct timed *t = &d->timed[k];
  volatile const struct stamps *x = &d->stamps[k];
  uint64_t *times = t->times, start = UINT64_MAX;
  t->times = NULL;
  d->unconverted--;
  for (int q = 0; q < CHANNELS; q++) {
    if (!(t->entered >> q & 1)) continue;
    if (x->start[q].value != t->v) {
      d->failure = unwritten;
      return;
    }
    uint64_t at = host_ns(d, x->start[q].stamp);
    if (at < start) start = at;
  }
  if (x->end.value != t->v) {
    d->failure = unwritten;
    return;
  }
  start = later(start, d->last_end);
  uint64_t end = later(host_ns(d, x->end.stamp), start);
  d->last_end = end;
  times[0] = start;
  times[1] = end;
}

/* Writes, in value order, the times of every timed value up to [v] not
   yet written, whose work has completed with [v]'s. */
static void convert_upto(struct device *d, uint64_t v) {
  uint64_t u = d->converted + 1;
  if (v >= d->slots && u < v - d->slots + 1) u = v - d->slots + 1;
  for (; u <= v && d->unconverted > 0; u++) {
    uint64_t k = u & (d->slots - 1);
    if (d->timed[k].times != NULL && d->timed[k].v == u) convert(d, k);
  }
  d->converted = later(d->converted, v);
}

/* The slot of [v], once the earlier value that held it, whose work has
   completed, and every earlier timed value have their times written; and,
   if [times], its records cleared and [v] kept with [times]. Answers the
   slot's host side if [times], NULL otherwise. While no timed value waits
   for its times, every value before [v] counts as converted, so that a
   conversion never walks values nobody timed. */
static struct timed *take_slot(struct device *d, uint64_t v, uint64_t *times) {
  uint64_t k = v & (d->slots - 1);
  struct timed *t = &d->timed[k];
  if (t->times != NULL) convert_upto(d, t->v);
  if (d->unconverted == 0) d->converted = v - 1;
  if (times == NULL) return NULL;
  memset((void *)&d->stamps[k], 0, sizeof d->stamps[k]);
  *t = (struct timed){v, times, 0};
  d->unconverted++;
  return t;
}

/* The GPU's address of [t]'s records. */
static uint64_t records(const struct device *d, const struct timed *t) {
  return d->stamps_gpu + (uint64_t)(t - d->timed) * sizeof(struct stamps);
}

/* Places a stamp of [v] at [address] on channel [q]. */
static void stamp(struct device *d, int q, uint64_t address, uint64_t v) {
  emit(d, &d->ch[q], q == COMPUTE ? T_STAMP : T_COPY_STAMP, address, v, 0);
}

int rig_nv_settle(void *self, uint64_t v, const char **failure) {
  struct device *d = ((struct rig_nv_edge *)self)->d;
  struct measure m;
  if (d->source != NULL && rig_now_ns() - d->latest.host >= MEASURE_NS) {
    uint64_t width;
    int r = rig_nv_measure(d, &m, &width);
    if (r == MEASURE_FAILED) {
      *failure = "the GPU's timer could not be read";
      return RIG_FAILED;
    }
    if (r == MEASURE_OK) d->latest = m;
    if (r == MEASURE_OK && d->first_wide) {
      d->first = m;
      d->first_wide = 0;
    }
  }
  convert_upto(d, v);
  if (d->failure == NULL) return RIG_OK;
  *failure = d->failure;
  return RIG_FAILED;
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
  if (s->timed != NULL) {
    s->timed->entered |= (uint8_t)(1 << q);
    stamp(d, q, records(d, s->timed) + (uint64_t)q * sizeof(struct record),
          s->v);
  }
  if (q != COMPUTE) return;
  /* Only the OCaml side writes a pending memory, so an exchange after the
     load takes a pending one too: the newer, if it replaced it. */
  uint64_t local = atomic_load_explicit(&d->local, memory_order_acquire);
  if (local != 0 && !(local & LOCAL_TAKEN)) {
    d->placed_local = atomic_exchange(&d->local, s->v << 1 | LOCAL_TAKEN);
    c->owes_local = 1;
  }
  if (c->owes_local && d->placed_local != 0) {
    local = d->placed_local;
    uint64_t page = (local >> 1) & ((UINT64_C(1) << LOCAL_PAGE_BITS) - 1);
    uint64_t address = page << LOCAL_PAGE_SHIFT;
    uint64_t per_tpc = (local >> (1 + LOCAL_PAGE_BITS)) << LOCAL_UNIT_SHIFT;
    emit(d, c, T_LOCAL, address, per_tpc, 0);
  }
  c->owes_local = 0;
  if (atomic_exchange(&d->invalidate, 0))
    emit(d, c, T_INVALIDATE, 0, 0, 0);
}

/* The join number that part [a], placed, signalled on its channel. */
static uint64_t joined(const struct submission *s, int a) {
  const struct route *x = &s->routes[a];
  return s->joins[x->channel] + x->signal;
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
  if (waits) emit(d, c, T_ACQUIRE, at, joined(s, x->join), 0);
  for (int j = 0; j < p[i].nafter; j++) {
    int a = p[i].after[j];
    if (s->routes[a].channel == x->channel) continue;
    emit(d, c, T_ACQUIRE, at, joined(s, a), 0);
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
   0 in the launch ring, with the driver's parameters and zeros up to the
   kernel's; then, unless the launch before it there chains it, a wait for
   idle if launches run, and its schedule. Answers where its [n] bytes of
   parameters go, which the caller writes. */
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
  if (l->params_at > l->bank0.nbytes)
    memset(host_at(c, bank) + l->bank0.nbytes, 0,
           l->params_at - l->bank0.nbytes);
  int chain = s->open.l != NULL && !waited;
  settle(c, &s->open, chain ? qmd : 0);
  if (!chain) {
    if (s->running) emit(d, c, T_IDLE, 0, 0, 0);
    emit(d, c, T_SCHEDULE, gpu_at(c, qmd), 0, 0);
  }
  s->open = next;
  return host_at(c, bank) + l->params_at;
}

/* Every ref of a launch lies among the bytes its kernel reads, which rig
   checks against the entry's [params], so a launch places those alone. */
static void launch(struct submission *s, const struct rig_part *p,
                   const uint8_t *args, const uint64_t *slots, int waited) {
  const struct launch *l = p->launch.launch;
  struct rig_args a = p->launch.args;
  a.params = (uint32_t)placed(l, a.params);
  uint8_t *params = dispatch(s, l, block_of(p, args), a.params, waited);
  rig_params(params, &a, args, slots);
}

/* Each thread copies vectors in a grid-stride loop; a bounded grid makes a
   large copy take a fixed amount of scheduling work. */
#define COPY_THREADS 256
#define COPY_GROUPS 256
#define COPY_VECTOR_BYTES 16

static void copy_compute(struct submission *s, const struct rig_part *p,
                         int waited) {
  uint64_t width = COPY_THREADS * COPY_VECTOR_BYTES;
  uint64_t groups = (p->copy.width + width - 1) / width;
  struct rig_block b = {
      .groups = {groups < COPY_GROUPS ? (uint32_t)groups : COPY_GROUPS, 1, 1},
      .threads = {COPY_THREADS, 1, 1}};
  uint64_t params[3] = {p->copy.dst + p->copy.dst_offset,
                        p->copy.src + p->copy.src_offset, p->copy.width};
  memcpy(dispatch(s, s->d->copy_launch, &b, sizeof params, waited), params,
         sizeof params);
}

/* Places on [c] the copy engine's fill of [n] bytes of [byte] at [dst],
   as [fill_split] cuts it, the body in launches of at most COPY_MAX
   bytes. */
static void place_fill(const struct device *d, struct channel *c,
                       uint64_t dst, uint8_t byte, uint64_t n) {
  uint64_t head, body, tail;
  fill_split(dst, n, &head, &body, &tail);
  if (head > 0) emit(d, c, T_FILL_BYTES, dst, byte, head);
  for (uint64_t at = 0; at < body; at += COPY_MAX) {
    uint64_t k = body - at < COPY_MAX ? body - at : COPY_MAX;
    emit(d, c, T_FILL, dst + head + at, byte * UINT64_C(0x01010101), k / 4);
  }
  if (tail > 0) emit(d, c, T_FILL_BYTES, dst + head + body, byte, tail);
}

/* Places on [c] the copy engine's launches for the copy part [p]: a box as
   multi-line launches of up to LINES_MAX rows ([multi_line]), and a run,
   or a box whose row or pitch exceeds a launch's fields, as launches of one
   line of at most COPY_MAX bytes each. */
static void place_copy(const struct device *d, struct channel *c,
                       const struct rig_part *p) {
  uint64_t dst = p->copy.dst + p->copy.dst_offset;
  uint64_t width = p->copy.width, rows = p->copy.rows;
  if (width == 0 || rows == 0) return;
  if (p->copy.fill) {
    place_fill(d, c, dst, p->copy.byte, width);
    return;
  }
  uint64_t src = p->copy.src + p->copy.src_offset;
  uint64_t sp = p->copy.src_pitch, dp = p->copy.dst_pitch;
  if (multi_line(p)) {
    for (uint64_t r = 0; r < rows; r += LINES_MAX) {
      uint64_t k = rows - r < LINES_MAX ? rows - r : LINES_MAX;
      const uint64_t v[TEMPLATE_VALUES] = {dst + r * dp, src + r * sp, width,
                                           k, dp, sp};
      emit_values(d, c, T_BOX, v);
    }
    return;
  }
  for (uint64_t r = 0; r < rows; r++)
    for (uint64_t at = 0; at < width; at += COPY_MAX) {
      uint64_t n = width - at < COPY_MAX ? width - at : COPY_MAX;
      emit(d, c, T_COPY, dst + r * dp + at, src + r * sp + at, n);
    }
}

/* Places the words or the copy engine's launches of [p] on [c]. */
static void place(struct device *d, struct channel *c,
                  const struct rig_part *p) {
  if (p->kind == RIG_WORDS) {
    put_entries(d, c, p->words.at, p->words.n);
    return;
  }
  place_copy(d, c, p);
}

int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                  int nwaits, const struct rig_part *p, int n,
                  void *scratch, uint8_t *args,
                  const uint64_t *slots, int nslots, uint64_t *times,
                  const char **failure) {
  (void)nslots;
  (void)failure;
  struct device *d = ((struct rig_nv_edge *)self)->d;
  struct channel *compute = &d->ch[COMPUTE];
  struct route *routes = scratch;
  struct submission s = {.d = d,
                         .routes = routes,
                         .joins = {d->ch[COMPUTE].joins, d->ch[COPY].joins},
                         .v = v,
                         .waits = waits,
                         .nwaits = nwaits,
                         .timed = take_slot(d, v, n > 0 ? times : NULL)};
  route(d, routes, p, n);
  int r = releaser(routes, n, v);
  int last[CHANNELS] = {-1, -1};
  for (int i = 0; i < n; i++) last[routes[i].channel] = i;
  uint32_t signalled[CHANNELS] = {0, 0};
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
    int signal = x->awaited || (i == last[q] && q != r);
    routes[i].signal = signal ? ++signalled[q] : 0;
    if (signal) {
      signal_word(d, q, JOIN_GPU(d, q), joined(&s, i));
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
    emit(d, &d->ch[r], T_ACQUIRE, JOIN_GPU(d, o), joined(&s, last[o]), 0);
  if (s.timed != NULL)
    stamp(d, r, records(d, s.timed) + offsetof(struct stamps, end), v);
  signal_word(d, r, d->word_gpu, v);
  d->ch[r].released = v;
  for (int q = 0; q < CHANNELS; q++) {
    if (!s.used[q]) continue;
    struct channel *c = &d->ch[q];
    c->joins += signalled[q];
    close_segment(d, c);
    c->marks[(c->first + c->count++) & (d->slots - 1)] =
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
