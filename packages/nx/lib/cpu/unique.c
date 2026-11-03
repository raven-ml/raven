/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Numbering.

   A numbering hands each new key the next number as it meets the key's
   first row, so keys come numbered in order of first appearance as they
   are found. Each key found keeps its words, its hash, its first row and
   its count of rows (a group).

   Keys find their group in a hash table of linear probing, a slot a word:
   the group's number plus one below ID_BITS, 0 for an empty slot, and above
   it TAG bits of the key's hash, which turn most probes away before they
   read the group's words. The table doubles past half full. Keys of at
   most DIRECT_WORDS words whose columns' spans multiply to a small range
   find their group in a direct table instead, at the key's place in that
   range, as a mixed-radix number of the columns' offsets from their least.

   On several threads, each takes a block of consecutive rows and numbers
   its keys on its own, provisionally. The blocks' groups then merge into
   tables by the top bits of their hashes, a partition a thread: each
   partition takes its groups block by block, in each block's order, so the
   first block to hold a key gives its first row and its order among the
   partition's. A key's number is its first row's rank among every key's
   first rows, read from a bit per row set at each first row, and a pass
   over the ids maps the provisional numbers to the final ones. One thread
   needs neither the merge nor the pass.

   The hash is keyed by a seed drawn once a process from the system's
   random source (nx_cpu.ml), so that no keys chosen in advance share their
   hashes and make the probes long. The numbers follow first appearance
   alone, so the results are the same under any seed. */

#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

/* A slot's group number plus one takes its low ID_BITS bits, its key's
   hash the rest: numberings of fewer than 2^40 rows. */
#define ID_BITS 40
#define ID_MASK ((1ull << ID_BITS) - 1)

/* Fibonacci hashing's multiplier, 2^64 over the golden ratio; SplitMix64's
   finishing multiplier; wyhash's first prime, the multiplier of its
   folded products. */
#define GOLDEN 0x9E3779B97F4A7C15ull
#define FINISH 0xBF58476D1CE4E5B9ull
#define WYP0 0xA0761D6478BD642Full

/* The hash's keys: its first state, the word its finish mixes in, and
   SECRETS secrets its words take in turn. */
#define SECRETS 4
#define KEYS (2 + SECRETS)

/* A table's first slots, and a block's first room for groups. */
#define FIRST_LOG 10
#define FIRST_ROOM 256

/* Keys of at most DIRECT_WORDS words take a direct table where their
   columns' spans multiply to at most DIRECT_SLOTS values, and to at most
   twice a block's rows past DIRECT_FEW, so that filling the table costs
   less than the rows it serves. */
#define DIRECT_WORDS 4
#define DIRECT_SLOTS (1 << 20)
#define DIRECT_FEW 1024

/* A block's least rows: fewer would merge more groups than they number. */
#define BLOCK_ROWS 16384

/* What a row takes, in bytes one core copies in its time (cpu.h, Calls and
   jobs): its key's hash, a probe and its id, ROW_COST, and WORD_COST a word
   of key. On one thread, rows of one word took 59 bytes at 1000 rows of 100
   keys and 136 at 1000 keys (fudge under load 18, 2026-11-03). Past
   CACHED_ROWS rows, a row of its own key probes a table the caches do not
   hold, 1079 bytes at a million: each takes NX_CPU_LINE more, the bound of
   a call whose keys are as many as its rows. */
#define ROW_COST 256
#define WORD_COST 32
#define CACHED_ROWS 65536

typedef struct {
  int w, pre, log; /* words a key, partition bits, log2 of the slots */
  int64_t k, room;
  uint64_t *key;   /* each group's words */
  uint64_t *hash;  /* each group's hash */
  int64_t *first;  /* each group's first row */
  int64_t *count;  /* each group's rows */
  uint64_t *slot;  /* tag | number + 1, 0 empty */
  int32_t *direct; /* each value's number, -1 none */
} groups;

typedef struct {
  const uint64_t *x; /* the keys' first element */
  int64_t n, w, s0, s1;
  int64_t *ids;
  int blocks, pb, failed;
  int direct;
  uint64_t lo[DIRECT_WORDS], radix[DIRECT_WORDS];
  int64_t values;
  groups *g;        /* each block's, then each partition's */
  int64_t **order;  /* each block's groups by partition */
  int64_t **bucket; /* each block's partitions' starts in order */
  int64_t **prov;   /* each block's groups' numbers in their partition */
  uint64_t *bits;   /* a bit per row, set at each first row */
  int64_t *below;   /* the bits set before each word of bits */
  int64_t m;
  int64_t *first_out, *counts_out;
  uint64_t *span;   /* each block's least and greatest of each column */
  uint64_t key[KEYS]; /* the hash's keys (Hashes) */
} numbering;

/* Hashes */

/* The seed's keys (KEYS): key[0] the state a hash starts from, key[1]
   the word its finish mixes in, the rest the secrets of its words.

   Each word folds in as wyhash's: h = mum(h ^ k ^ s, WYP0), mum(a, b) the
   high and low halves of the 128-bit product a b xored. The fold is no
   bijection, so a difference in one word's bits does not pass through it
   as a fixed difference: the product's carries depend on every bit of
   h ^ k ^ s, which the secrets hide. A multiply and xor-shift step passed
   the difference of a word's top bit through unchanged for every input,
   whatever the seed, so rows of many words could cancel it word by word
   and share their hash under every seed. */
static uint64_t seed_key[KEYS];

static inline uint64_t mum(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && !defined(__clang__)
  uint64_t hi, lo = _umul128(a, b, &hi);
  return hi ^ lo;
#else
  unsigned __int128 p = (unsigned __int128)a * b;
  return (uint64_t)(p >> 64) ^ (uint64_t)p;
#endif
}

static inline uint64_t mix(const uint64_t *key, uint64_t h, int j,
                           uint64_t k) {
  return mum(h ^ k ^ key[2 + (j & (SECRETS - 1))], WYP0);
}

/* A bijection of the state: equal hashes are equal states. */
static inline uint64_t finish(uint64_t h, uint64_t end) {
  h = (h ^ end) * FINISH;
  return h ^ (h >> 31);
}

static inline __attribute__((always_inline)) uint64_t hash_of(
    const numbering *u, const uint64_t *row, int w) {
  uint64_t h = u->key[0];
  for (int j = 0; j < w; j++) h = mix(u->key, h, j, row[j]);
  return finish(h, u->key[1]);
}

/* Sets the seed: its keys are SplitMix64's next KEYS outputs from it. */
value nx_cpu_unique_seed(value v) {
  uint64_t x = (uint64_t)Int64_val(v);
  for (int i = 0; i < KEYS; i++) {
    x += GOLDEN;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * FINISH;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    __atomic_store_n(&seed_key[i], z ^ (z >> 31), __ATOMIC_RELAXED);
  }
  return Val_unit;
}

static void keys_now(uint64_t *key) {
  for (int i = 0; i < KEYS; i++)
    key[i] = __atomic_load_n(&seed_key[i], __ATOMIC_RELAXED);
}

/* The hash's state after the words of the OCaml int64 array [vw], under
   the seed: rows whose words leave one state hash alike. */
value nx_cpu_unique_state(value vw) {
  CAMLparam1(vw);
  uint64_t key[KEYS];
  keys_now(key);
  uint64_t h = key[0];
  for (mlsize_t j = 0; j < Wosize_val(vw); j++)
    h = mix(key, h, (int)j, (uint64_t)Int64_val(Field(vw, j)));
  CAMLreturn(caml_copy_int64((int64_t)h));
}

/* Groups */

static void drop(groups *g) {
  free(g->key);
  free(g->hash);
  free(g->first);
  free(g->count);
  free(g->slot);
  free(g->direct);
}

/* A table of [w]-word keys, partitioned by [pre] bits; [values] slots of a
   direct table where above 0. Answers -1 where memory runs out. */
static int start(groups *g, int w, int pre, int64_t values) {
  memset(g, 0, sizeof *g);
  g->w = w;
  g->pre = pre;
  g->log = FIRST_LOG;
  g->room = FIRST_ROOM;
  g->key = malloc((size_t)(g->room * w) * sizeof *g->key);
  g->hash = malloc((size_t)g->room * sizeof *g->hash);
  g->first = malloc((size_t)g->room * sizeof *g->first);
  g->count = malloc((size_t)g->room * sizeof *g->count);
  if (values > 0) {
    g->direct = malloc((size_t)values * sizeof *g->direct);
    if (g->direct != NULL)
      memset(g->direct, 0xFF, (size_t)values * sizeof *g->direct);
  } else
    g->slot = calloc((size_t)1 << g->log, sizeof *g->slot);
  int table = values > 0 ? g->direct != NULL : g->slot != NULL;
  int ok = g->key != NULL && g->hash != NULL && g->first != NULL &&
           g->count != NULL && table;
  return ok ? 0 : -1;
}

static inline uint64_t place_of(const groups *g, uint64_t h) {
  return (h << g->pre) >> (64 - g->log);
}

/* Doubles [g]'s slots and puts its groups back by their hashes. */
static int rehash(groups *g) {
  int log = g->log + 1;
  uint64_t *slot = calloc((size_t)1 << log, sizeof *slot);
  if (slot == NULL) return -1;
  free(g->slot);
  g->slot = slot;
  g->log = log;
  uint64_t mask = (1ull << log) - 1;
  for (int64_t q = 0; q < g->k; q++) {
    uint64_t h = g->hash[q], at = place_of(g, h);
    while (slot[at] != 0) at = (at + 1) & mask;
    slot[at] = h << ID_BITS | (uint64_t)(q + 1);
  }
  return 0;
}

static int widen(void **p, int64_t n, size_t size) {
  void *q = realloc(*p, (size_t)n * size);
  if (q == NULL) return -1;
  *p = q;
  return 0;
}

/* A new group of [row], hashing to [h], first at [first]: its number, or
   -1 where memory runs out. */
static __attribute__((noinline)) int64_t fresh(groups *g, uint64_t h,
                                               const uint64_t *row,
                                               int64_t first) {
  if (g->k == g->room) {
    int64_t room = 2 * g->room;
    if (widen((void **)&g->key, room * g->w, sizeof *g->key) ||
        widen((void **)&g->hash, room, sizeof *g->hash) ||
        widen((void **)&g->first, room, sizeof *g->first) ||
        widen((void **)&g->count, room, sizeof *g->count))
      return -1;
    g->room = room;
  }
  int64_t q = g->k++;
  memcpy(g->key + q * g->w, row, (size_t)g->w * sizeof *row);
  g->hash[q] = h;
  g->first[q] = first;
  g->count[q] = 0;
  return q;
}

/* A new group of [row] at the empty slot [at] of [g], hashing to [h]: its
   number, or -1 where memory runs out. */
static __attribute__((noinline)) int64_t added(groups *g, uint64_t at,
                                               uint64_t h, const uint64_t *row,
                                               int64_t first) {
  int64_t q = fresh(g, h, row, first);
  if (q < 0) return -1;
  g->slot[at] = h << ID_BITS | (uint64_t)(q + 1);
  if (2 * g->k > ((int64_t)1 << g->log) - 1 && rehash(g)) return -1;
  return q;
}

/* The number of [row]'s group, [h] its hash, made first at [first] where
   [g] has none; -1 where memory runs out. */
static inline __attribute__((always_inline)) int64_t find(groups *g,
                                                          uint64_t h,
                                                          const uint64_t *row,
                                                          int w,
                                                          int64_t first) {
  uint64_t mask = (1ull << g->log) - 1, tag = h << ID_BITS;
  uint64_t at = place_of(g, h);
  for (;; at = (at + 1) & mask) {
    uint64_t s = g->slot[at];
    if (s == 0) break;
    if ((s & ~ID_MASK) != tag) continue;
    int64_t q = (int64_t)(s & ID_MASK) - 1;
    const uint64_t *k = g->key + q * w;
    int j = 0;
    while (j < w && k[j] == row[j]) j++;
    if (j == w) return q;
  }
  return added(g, at, h, row, first);
}

/* Blocks */

/* The rows of block [b], of [u]'s blocks. */
static void rows_of(const numbering *u, int64_t b, int64_t *lo, int64_t *hi) {
  *lo = u->n * b / u->blocks;
  *hi = u->n * (b + 1) / u->blocks;
}

/* Row [i]'s words: in place where they lie side by side, else copied into
   [buf]. */
static inline const uint64_t *row_at(const numbering *u, int64_t i,
                                     uint64_t *buf) {
  const uint64_t *r = u->x + i * u->s0;
  if (u->s1 == 1 || u->w == 1) return r;
  for (int64_t j = 0; j < u->w; j++) buf[j] = r[j * u->s1];
  return buf;
}

/* Numbers rows [lo, hi) into [g] by its direct table, keys of [w] words,
   writing each row's number at ids. A group's hash is made only where
   blocks merge. The loop keeps what it reads in locals: its stores of ids
   and counts could otherwise alias them. */
static inline __attribute__((always_inline)) int direct_rows(
    const numbering *u, groups *g, int64_t lo, int64_t hi, int w,
    uint64_t *buf) {
  const uint64_t *x = u->x;
  int64_t s0 = u->s0, s1 = u->s1, *ids = u->ids, *count = g->count;
  int32_t *direct = g->direct;
  uint64_t least[DIRECT_WORDS], radix[DIRECT_WORDS];
  for (int j = 0; j < w; j++) {
    least[j] = u->lo[j];
    radix[j] = u->radix[j];
  }
  int merged = u->blocks > 1;
  for (int64_t i = lo; i < hi; i++) {
    const uint64_t *r = x + i * s0;
    uint64_t at = (r[0] - least[0]) * radix[0];
    for (int j = 1; j < w; j++) at += (r[j * s1] - least[j]) * radix[j];
    int64_t q = direct[at];
    if (q < 0) {
      const uint64_t *row = row_at(u, i, buf);
      q = fresh(g, merged ? hash_of(u, row, w) : 0, row, i);
      if (q < 0) return -1;
      direct[at] = (int32_t)q;
      count = g->count;
    }
    ids[i] = q;
    count[q]++;
  }
  return 0;
}

/* Numbers rows [lo, hi) into [g] by its hash table, keys of [w] words, as
   find does, its table in locals that a new group refreshes. */
static inline __attribute__((always_inline)) int hashed_rows(
    const numbering *u, groups *g, int64_t lo, int64_t hi, int w,
    uint64_t *buf) {
  int64_t *ids = u->ids, *count = g->count;
  const uint64_t *slot = g->slot, *key = g->key;
  uint64_t mask = (1ull << g->log) - 1;
  int shift = 64 - g->log;
  for (int64_t i = lo; i < hi; i++) {
    const uint64_t *row = row_at(u, i, buf);
    uint64_t h = hash_of(u, row, w), tag = h << ID_BITS, at = h >> shift;
    int64_t q = -1;
    for (;; at = (at + 1) & mask) {
      uint64_t e = slot[at];
      if (e == 0) break;
      if ((e & ~ID_MASK) != tag) continue;
      const uint64_t *k = key + ((int64_t)(e & ID_MASK) - 1) * w;
      int j = 0;
      while (j < w && k[j] == row[j]) j++;
      if (j == w) {
        q = (int64_t)(e & ID_MASK) - 1;
        break;
      }
    }
    if (q < 0) {
      q = added(g, at, h, row, i);
      if (q < 0) return -1;
      count = g->count;
      slot = g->slot;
      key = g->key;
      mask = (1ull << g->log) - 1;
      shift = 64 - g->log;
    }
    ids[i] = q;
    count[q]++;
  }
  return 0;
}

static int number_rows(const numbering *u, groups *g, int64_t lo, int64_t hi) {
  uint64_t *buf = malloc((size_t)u->w * sizeof *buf);
  if (buf == NULL) return -1;
  int e, w = (int)u->w;
  if (u->direct && w == 1) e = direct_rows(u, g, lo, hi, 1, buf);
  else if (u->direct && w == 2) e = direct_rows(u, g, lo, hi, 2, buf);
  else if (u->direct) e = direct_rows(u, g, lo, hi, w, buf);
  else if (w == 1) e = hashed_rows(u, g, lo, hi, 1, buf);
  else if (w == 2) e = hashed_rows(u, g, lo, hi, 2, buf);
  else e = hashed_rows(u, g, lo, hi, w, buf);
  free(buf);
  return e;
}

static inline int partition_of(const numbering *u, uint64_t h) {
  return u->pb == 0 ? 0 : (int)(h >> (64 - u->pb));
}

/* Block [b]'s groups, ordered by partition and in each by number: a count
   per partition, then a stable placement. */
static int bucketed(numbering *u, int64_t b) {
  const groups *g = &u->g[b];
  int parts = 1 << u->pb;
  int64_t *from = calloc((size_t)parts + 1, sizeof *from);
  int64_t *order = malloc((size_t)(g->k > 0 ? g->k : 1) * sizeof *order);
  int64_t *prov = malloc((size_t)(g->k > 0 ? g->k : 1) * sizeof *prov);
  u->bucket[b] = from;
  u->order[b] = order;
  u->prov[b] = prov;
  if (from == NULL || order == NULL || prov == NULL) return -1;
  for (int64_t q = 0; q < g->k; q++) from[partition_of(u, g->hash[q]) + 1]++;
  for (int p = 0; p < parts; p++) from[p + 1] += from[p];
  int64_t *next = malloc((size_t)parts * sizeof *next);
  if (next == NULL) return -1;
  memcpy(next, from, (size_t)parts * sizeof *next);
  for (int64_t q = 0; q < g->k; q++)
    order[next[partition_of(u, g->hash[q])]++] = q;
  free(next);
  return 0;
}

static void block_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  numbering *u = ctx;
  for (int64_t b = lo; b < hi; b++) {
    int64_t r0, r1;
    rows_of(u, b, &r0, &r1);
    groups *g = &u->g[b];
    int e = start(g, (int)u->w, 0, u->direct ? u->values : 0);
    if (!e) e = number_rows(u, g, r0, r1);
    if (!e && u->blocks > 1) e = bucketed(u, b);
    if (e) __atomic_store_n(&u->failed, 1, __ATOMIC_RELAXED);
  }
}

/* Spans */

/* Each block's least and greatest word of each column, at span. */
static void span_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  numbering *u = ctx;
  int64_t w = u->w;
  for (int64_t b = lo; b < hi; b++) {
    int64_t r0, r1;
    rows_of(u, b, &r0, &r1);
    uint64_t *s = u->span + 2 * w * b;
    for (int64_t j = 0; j < w; j++) {
      const uint64_t *c = u->x + j * u->s1;
      uint64_t least = ~0ull, most = 0;
      for (int64_t i = r0; i < r1; i++) {
        uint64_t y = c[i * u->s0];
        least = y < least ? y : least;
        most = y > most ? y : most;
      }
      s[2 * j] = least;
      s[2 * j + 1] = most;
    }
  }
}

/* Whether the keys take a direct table: their columns' spans multiply to
   at most [limit] values, written at values with each column's least and
   radix. */
static int directly(numbering *u, int64_t limit) {
  int64_t w = u->w;
  for (int64_t j = 0; j < w; j++) {
    uint64_t least = ~0ull, most = 0;
    for (int b = 0; b < u->blocks; b++) {
      const uint64_t *s = u->span + 2 * w * b + 2 * j;
      least = s[0] < least ? s[0] : least;
      most = s[1] > most ? s[1] : most;
    }
    u->lo[j] = least;
    u->radix[j] = most - least; /* the span, until the radices below */
  }
  uint64_t values = 1;
  for (int64_t j = w - 1; j >= 0; j--) {
    uint64_t span = u->radix[j];
    if (span >= (uint64_t)limit) return 0;
    u->radix[j] = values;
    values *= span + 1;
    if (values > (uint64_t)limit) return 0;
  }
  u->values = (int64_t)values;
  return 1;
}

/* Merging */

/* Partition [p]'s groups, every block's in block order, into u->g's
   partition table, each block's numbers there at prov; and a bit at each
   first row. */
static void merge_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  numbering *u = ctx;
  int w = (int)u->w;
  for (int64_t p = lo; p < hi; p++) {
    groups *t = &u->g[u->blocks + p];
    int e = start(t, w, u->pb, 0);
    for (int b = 0; !e && b < u->blocks; b++) {
      const groups *g = &u->g[b];
      const int64_t *order = u->order[b], *from = u->bucket[b];
      for (int64_t j = from[p]; j < from[p + 1]; j++) {
        int64_t l = order[j];
        int64_t q = find(t, g->hash[l], g->key + l * w, w, g->first[l]);
        if (q < 0) {
          e = -1;
          break;
        }
        t->count[q] += g->count[l];
        u->prov[b][l] = q;
      }
    }
    if (e) {
      __atomic_store_n(&u->failed, 1, __ATOMIC_RELAXED);
      continue;
    }
    for (int64_t q = 0; q < t->k; q++) {
      int64_t f = t->first[q];
      __atomic_fetch_or(&u->bits[f >> 6], 1ull << (f & 63), __ATOMIC_RELAXED);
    }
  }
}

/* The number of first rows before row [f]. */
static inline int64_t rank_of(const numbering *u, int64_t f) {
  uint64_t below = u->bits[f >> 6] & ((1ull << (f & 63)) - 1);
  return u->below[f >> 6] + __builtin_popcountll(below);
}

/* Partition [p]'s groups' final numbers, in their hash slots, and their
   first rows and counts out. */
static void rank_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  numbering *u = ctx;
  for (int64_t p = lo; p < hi; p++) {
    groups *t = &u->g[u->blocks + p];
    for (int64_t q = 0; q < t->k; q++) {
      int64_t r = rank_of(u, t->first[q]);
      t->hash[q] = (uint64_t)r;
      if (r < u->m) {
        u->first_out[r] = t->first[q];
        u->counts_out[r] = t->count[q];
      }
    }
  }
}

/* Block [b]'s ids, provisional, made final. */
static void final_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  numbering *u = ctx;
  for (int64_t b = lo; b < hi; b++) {
    const groups *g = &u->g[b];
    int64_t *prov = u->prov[b];
    for (int64_t l = 0; l < g->k; l++) {
      const groups *t = &u->g[u->blocks + partition_of(u, g->hash[l])];
      prov[l] = (int64_t)t->hash[prov[l]];
    }
    int64_t r0, r1;
    rows_of(u, b, &r0, &r1);
    int64_t *ids = u->ids;
    for (int64_t i = r0; i < r1; i++) ids[i] = prov[ids[i]];
  }
}

/* The number of [u]'s keys after its blocks merge: -1 where memory runs
   out. */
static int64_t merged(numbering *u, int64_t bytes, double cost) {
  int parts = 1 << u->pb;
  int64_t words = (u->n + 63) / 64;
  u->bits = calloc((size_t)words, sizeof *u->bits);
  u->below = malloc((size_t)words * sizeof *u->below);
  if (u->bits == NULL || u->below == NULL) return -1;
  nx_cpu_job(parts, bytes, cost / 4, merge_units, u);
  if (u->failed) return -1;
  int64_t set = 0;
  for (int64_t j = 0; j < words; j++) {
    u->below[j] = set;
    set += __builtin_popcountll(u->bits[j]);
  }
  nx_cpu_job(parts, bytes, cost / 8, rank_units, u);
  nx_cpu_job(u->blocks, bytes, cost / 4, final_units, u);
  return set;
}

/* The external */

/* Frees what [u] holds. */
static void freed(numbering *u, int tables) {
  for (int i = 0; i < tables; i++) drop(&u->g[i]);
  for (int b = 0; u->order != NULL && b < u->blocks; b++) {
    free(u->order[b]);
    free(u->bucket[b]);
    free(u->prov[b]);
  }
  free(u->g);
  free(u->order);
  free(u->bucket);
  free(u->prov);
  free(u->bits);
  free(u->below);
  free(u->span);
}

/* Numbers [u]'s keys into its outputs: the number of keys, or -1 where
   memory runs out. */
static int64_t numbered(numbering *u, int64_t bytes, double cost) {
  int parts = u->blocks > 1 ? 1 << u->pb : 0;
  int tables = u->blocks + parts;
  int64_t k = -1;
  u->g = calloc((size_t)tables, sizeof *u->g);
  if (u->blocks > 1) {
    u->order = calloc((size_t)u->blocks, sizeof *u->order);
    u->bucket = calloc((size_t)u->blocks, sizeof *u->bucket);
    u->prov = calloc((size_t)u->blocks, sizeof *u->prov);
  }
  int ok = u->g != NULL &&
           (u->blocks == 1 ||
            (u->order != NULL && u->bucket != NULL && u->prov != NULL));
  if (ok && u->w <= DIRECT_WORDS) {
    u->span = malloc((size_t)(2 * u->w * u->blocks) * sizeof *u->span);
    ok = u->span != NULL;
    if (ok) {
      nx_cpu_job(u->blocks, bytes, cost / 16, span_units, u);
      int64_t rows = (u->n + u->blocks - 1) / u->blocks;
      int64_t limit = 2 * rows + DIRECT_FEW;
      u->direct = directly(u, limit < DIRECT_SLOTS ? limit : DIRECT_SLOTS);
    }
  }
  if (ok) {
    nx_cpu_job(u->blocks, bytes, cost, block_units, u);
    ok = !u->failed;
  }
  if (ok && u->blocks == 1) {
    const groups *g = &u->g[0];
    k = g->k;
    int64_t kept = k < u->m ? k : u->m;
    memcpy(u->first_out, g->first, (size_t)kept * sizeof *g->first);
    memcpy(u->counts_out, g->count, (size_t)kept * sizeof *g->count);
  } else if (ok)
    k = merged(u, bytes, cost);
  freed(u, u->g == NULL ? 0 : tables);
  return k;
}

value nx_cpu_unique(value vids, value vfirst, value vcounts, value vcount,
                    value vkeys) {
  CAMLparam5(vids, vfirst, vcounts, vcount, vkeys);
  int64_t kd[2 * NX_MAX_RANK], id[2 * NX_MAX_RANK], fd[2 * NX_MAX_RANK],
      cd[2 * NX_MAX_RANK], nd[2 * NX_MAX_RANK], off;
  int rk = nx_array_layout(vkeys, kd, &off);
  int ri = nx_array_layout(vids, id, &off);
  int rf = nx_array_layout(vfirst, fd, &off);
  int rc = nx_array_layout(vcounts, cd, &off);
  int rn = nx_array_layout(vcount, nd, &off);
  if (rk != 2 || ri != 1 || rf != 1 || rc != 1 || rn != 0 || id[0] != kd[0] ||
      fd[0] != cd[0])
    CAMLreturn(Val_int(NX_SHAPE));
  if (kd[0] >= (int64_t)ID_MASK) CAMLreturn(Val_int(NX_DECLINED));
  nx_operand in[5] = {{vids, NX_INT64, 1},
                      {vfirst, NX_INT64, 1},
                      {vcounts, NX_INT64, 1},
                      {vcount, NX_INT64, 1},
                      {vkeys, NX_INT64, 0}};
  nx_array a[5];
  int e = nx_read(5, in, a);
  if (e) CAMLreturn(Val_int(e));
  int64_t n = a[4].dim[0], w = a[4].dim[1], m = a[1].dim[0];
  numbering u = {.x = (const uint64_t *)a[4].base + a[4].offset,
                 .n = n,
                 .w = w,
                 .s0 = a[4].dim[2],
                 .s1 = a[4].dim[3],
                 .ids = (int64_t *)a[0].base + a[0].offset,
                 .blocks = 1,
                 .m = m,
                 .first_out = (int64_t *)a[1].base + a[1].offset,
                 .counts_out = (int64_t *)a[2].base + a[2].offset};
  keys_now(u.key);
  int64_t *count = (int64_t *)a[3].base + a[3].offset;
  int64_t bytes = n * (8 * w + 8) + 16 * m;
  double past = n > CACHED_ROWS ? (double)(n - CACHED_ROWS) : 0;
  double cost = (double)n * (ROW_COST + WORD_COST * (double)w) +
                past * NX_CPU_LINE + 16.0 * m;
  int released = nx_cpu_release(cost);
  int64_t k;
  if (n == 0 || w == 0) {
    /* Every row one key. */
    for (int64_t i = 0; i < n; i++) u.ids[i] = 0;
    k = n > 0;
    if (k && m > 0) {
      u.first_out[0] = 0;
      u.counts_out[0] = n;
    }
  } else {
    int64_t threads = nx_cpu_threads(bytes, cost);
    int64_t most = n / BLOCK_ROWS;
    u.blocks = (int)(threads < most ? threads : most < 1 ? 1 : most);
    /* Four partitions a block, at most 256. */
    while (u.blocks > 1 && (1 << u.pb) < 4 * u.blocks && u.pb < 8) u.pb++;
    k = numbered(&u, bytes, cost);
  }
  if (k < 0) {
    nx_cpu_reacquire(released);
    nx_done(5, a);
    caml_raise_out_of_memory();
  }
  for (int64_t g = k; g < m; g++) {
    u.first_out[g] = n;
    u.counts_out[g] = 0;
  }
  *count = k;
  nx_cpu_reacquire(released);
  nx_done(5, a);
  CAMLreturn(Val_int(NX_OK));
}
