/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_group.c — Group: the rows of a uint64 matrix numbered in order of first
   appearance.

   Row i's id is the number of distinct rows whose first occurrence comes before
   that of row i's words. The ids depend on neither the hash, its seed nor the
   thread count: each is fixed by the order of first rows, whatever tables met
   the rows on the way.

   The rows are cut into blocks of NX_C_GROUP_BLOCK rows, and each block is
   grouped on its own by a table that numbers its groups in order of first
   appearance within the block. With one block, those are the ids. With
   several, the blocks' groups merge:
   1. Grouping, per block in parallel. A block's group records its key and
      first row, and each row its block-local id. A row's key is its word when
      it has one, and its hash otherwise; its partition is its hash's top byte.
   2. Partitioning. The blocks' groups are scattered into NX_C_GROUP_PARTS
      partitions, stably: a partition holds its groups in block order, then in
      block-local order, which is the order of their first rows. The groups of
      block b in partition p are the segment (b, p).
   3. Merging, per partition in parallel. A second table merges the partition's
      groups. The block group that makes a merged group holds its first row,
      so it is flagged.
   4. Numbering. The flagged block groups, in block order, then block-local
      order, are the merged groups in order of first appearance, so counting
      them numbers the merged groups.
   5. Each block group takes its merged group's number, per partition, then
      each row its block group's, per block.
   A block that repeats a few keys has a few groups, so the merge costs in
   proportion to the distinct keys of each block, never to the rows, and a key
   that fills most rows loads no partition more than another. Each phase reads
   and writes within one block or one partition at a time, but for the
   partitioning, which streams to every partition, and the numbering's
   writes.

   A table holds fewer than half as many groups as slots. A block's table has
   room for every row of the block. A partition's starts as large as the last
   table its worker made, since the partitions of one input hold alike numbers
   of groups, and doubles once half full, placing its groups again. A worker
   reuses one region of slots for its blocks or partitions without clearing it:
   a slot holds the generation of the table that wrote it, and a slot of
   another generation is empty.

   Threads are planned as for COMPUTE work, so a short input runs on one.

   The seed is the destination's address, so no run of keys made to collide
   carries over to another call. The hash only places keys: rows are told apart
   by their words.

   Scratch lives across the phases, so it cannot be one region's free_on_exit:
   an asynchronous exception at a region's lock re-acquire leaks it, a leak and
   never a corruption, as in nx_c_fft.c's irfft. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* The probes are inlined into the loops that make them, which take the row
   width as a constant, and the slots a batch will probe are fetched ahead. */
#if defined(__GNUC__) || defined(__clang__)
#define NX_C_GROUP_HOT static inline __attribute__((always_inline))
#define NX_C_GROUP_PREFETCH(p) __builtin_prefetch(p)
#else
#define NX_C_GROUP_HOT static inline
#define NX_C_GROUP_PREFETCH(p) ((void)(p))
#endif

#define NX_C_GROUP_BLOCK ((int64_t)1 << 16)
#define NX_C_GROUP_PARTS 256
#define NX_C_GROUP_BATCH 16
#define NX_C_GROUP_MIN_SLOTS ((int64_t)1024)

/* A slot's tag holds its table's generation above NX_C_GROUP_ID_BITS bits of
   its group's id, which bound the rows. */
#define NX_C_GROUP_ID_BITS 40
#define NX_C_GROUP_ID_MASK (((uint64_t)1 << NX_C_GROUP_ID_BITS) - 1)
#define NX_C_GROUP_GENERATIONS ((uint64_t)1 << (64 - NX_C_GROUP_ID_BITS))
#define NX_C_GROUP_ERR_ROWS "more than 2^40 rows"

/* The rows: [w] words each, at byte strides [rs] between rows and [cs] between
   words. */
typedef struct {
  const char *base;
  int64_t w, rs, cs;
  uint64_t seed;
} nx_c_group_rows;

typedef struct {
  uint64_t key;
  uint64_t tag;
} nx_c_group_slot;

/* A group: its key and its first row. */
typedef struct {
  uint64_t key;
  int64_t row;
} nx_c_group_rep;

/* A block group in its partition: its key, then its merged group, then that
   group's number, and its index among the blocks' groups. */
typedef struct {
  uint64_t key;
  int64_t rep;
} nx_c_group_pair;

/* A worker's slots, the first [cleared] of them zeroed or written, and the
   generation and size of its last table. */
typedef struct {
  nx_c_group_slot *slots;
  int64_t cleared;
  uint64_t generation;
  int64_t size;
} nx_c_group_region;

typedef struct {
  nx_c_group_region *region;
  nx_c_group_slot *slots;
  uint64_t mask;
  uint64_t generation; /* shifted into place */
  int64_t count;
  nx_c_group_rep *groups; /* the table's groups, by id */
} nx_c_group_table;

typedef struct {
  nx_c_group_rows r;
  int64_t n;
  int64_t *out;
  int64_t nblocks;
  /* Block b's groups from b * NX_C_GROUP_BLOCK. Their keys become their merged
     groups, the first of each flagged, then their numbers. */
  nx_c_group_rep *reps;
  int64_t *ngroups;
  /* Per block and partition: the groups, then the end of segment (b, p). */
  int64_t *cursor;
  int64_t *firsts; /* per block: merged groups first met, then their first id */
  int64_t poff[NX_C_GROUP_PARTS + 1];
  nx_c_group_pair *pairs; /* partition p's from poff[p] */
  nx_c_group_rep *merged; /* per merged group: its number, its first row */
  int nregions;
  nx_c_group_region *regions; /* per worker */
} nx_c_group_exec;

/* murmur3's 64-bit finalizer: a bijection that mixes every input bit into every
   output bit. */
static inline uint64_t nx_c_group_mix(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

static inline uint64_t nx_c_group_word(const nx_c_group_rows *r, int64_t row,
                                       int64_t j) {
  uint64_t v;
  memcpy(&v, r->base + row * r->rs + j * r->cs, sizeof v);
  return v;
}

/* [one] is whether rows have one word: the hot loops take it as a constant. */

static inline uint64_t nx_c_group_key(const nx_c_group_rows *r, int64_t row,
                                      const int one) {
  if (one) return nx_c_group_word(r, row, 0);
  uint64_t h = r->seed;
  for (int64_t j = 0; j < r->w; j++)
    h = nx_c_group_mix(h ^ nx_c_group_word(r, row, j));
  return h;
}

static inline uint64_t nx_c_group_hash(const nx_c_group_rows *r, uint64_t key,
                                       const int one) {
  return one ? nx_c_group_mix(key ^ r->seed) : key;
}

static inline int nx_c_group_same(const nx_c_group_rows *r, int64_t a,
                                  int64_t b) {
  for (int64_t j = 0; j < r->w; j++)
    if (nx_c_group_word(r, a, j) != nx_c_group_word(r, b, j)) return 0;
  return 1;
}

/* Tables */

/* Makes [t] a table of [size] slots in its region, of a new generation. */
static void nx_c_group_open(nx_c_group_table *t, int64_t size) {
  nx_c_group_region *g = t->region;
  if (size > g->cleared) {
    memset(g->slots + g->cleared, 0,
           (size_t)(size - g->cleared) * sizeof *g->slots);
    g->cleared = size;
  }
  if (++g->generation == NX_C_GROUP_GENERATIONS) {
    memset(g->slots, 0, (size_t)g->cleared * sizeof *g->slots);
    g->generation = 1;
  }
  g->size = size;
  t->slots = g->slots;
  t->mask = (uint64_t)size - 1;
  t->generation = g->generation << NX_C_GROUP_ID_BITS;
}

/* [t] is the next table of [worker]'s region, of [size] slots, for groups
   listed in [list]. */
static void nx_c_group_table_of(const nx_c_group_exec *e, int worker,
                                int64_t size, nx_c_group_rep *list,
                                nx_c_group_table *t) {
  t->region = &e->regions[worker];
  t->count = 0;
  t->groups = list;
  nx_c_group_open(t, size);
}

/* The slots of a table that never grows for [rows] groups. */
static int64_t nx_c_group_slots(int64_t rows) {
  int64_t slots = NX_C_GROUP_MIN_SLOTS;
  while (slots <= 2 * rows) slots *= 2;
  return slots;
}

/* Doubles [t], placing its groups again, a batch of slots fetched ahead. */
static void nx_c_group_grow(const nx_c_group_rows *r, nx_c_group_table *t) {
  nx_c_group_open(t, 2 * (int64_t)(t->mask + 1));
  int one = r->w == 1;
  for (int64_t g = 0; g < t->count; g += NX_C_GROUP_BATCH) {
    int64_t m = t->count - g < NX_C_GROUP_BATCH ? t->count - g
                                                : NX_C_GROUP_BATCH;
    uint64_t hs[NX_C_GROUP_BATCH];
    for (int64_t j = 0; j < m; j++) {
      hs[j] = nx_c_group_hash(r, t->groups[g + j].key, one);
      NX_C_GROUP_PREFETCH(&t->slots[hs[j] & t->mask]);
    }
    for (int64_t j = 0; j < m; j++) {
      uint64_t s = hs[j] & t->mask;
      while ((t->slots[s].tag & ~NX_C_GROUP_ID_MASK) == t->generation)
        s = (s + 1) & t->mask;
      t->slots[s].key = t->groups[g + j].key;
      t->slots[s].tag = t->generation | (uint64_t)(g + j);
    }
  }
}

/* The id of the group of [row], whose key is [key] and hash [h], or, if [t]
   has none, the next group's id complemented, made that group's. [row] is
   read only to tell rows of several words apart. */
NX_C_GROUP_HOT int64_t nx_c_group_find(const nx_c_group_rows *r,
                                      nx_c_group_table *t, uint64_t key,
                                      uint64_t h, int64_t row, const int one) {
  nx_c_group_slot *slots = t->slots;
  uint64_t mask = t->mask, generation = t->generation, s = h & mask;
  for (;;) {
    uint64_t tag = slots[s].tag;
    if ((tag & ~NX_C_GROUP_ID_MASK) != generation) break;
    int64_t id = (int64_t)(tag & NX_C_GROUP_ID_MASK);
    if (slots[s].key == key &&
        (one || nx_c_group_same(r, t->groups[id].row, row)))
      return id;
    s = (s + 1) & mask;
  }
  int64_t id = t->count++;
  slots[s].key = key;
  slots[s].tag = generation | (uint64_t)id;
  t->groups[id].key = key;
  t->groups[id].row = row;
  if (2 * t->count > (int64_t)mask) nx_c_group_grow(r, t);
  return ~id;
}

static inline int64_t nx_c_group_block_end(const nx_c_group_exec *e,
                                           int64_t b) {
  int64_t end = (b + 1) * NX_C_GROUP_BLOCK;
  return end < e->n ? end : e->n;
}

/* The first pair of segment (b, p). */
static inline int64_t nx_c_group_segment(const nx_c_group_exec *e, int64_t b,
                                         int p) {
  return b == 0 ? e->poff[p] : e->cursor[(b - 1) * NX_C_GROUP_PARTS + p];
}

/* Sets the keys of block [b]'s groups to their pairs'. */
static void nx_c_group_gather(const nx_c_group_exec *e, int64_t b) {
  const int64_t *end = e->cursor + b * NX_C_GROUP_PARTS;
  for (int p = 0; p < NX_C_GROUP_PARTS; p++)
    for (int64_t k = nx_c_group_segment(e, b, p); k < end[p]; k++)
      e->reps[e->pairs[k].rep].key = e->pairs[k].key;
}

/* 1. Grouping */

NX_C_GROUP_HOT void nx_c_group_block(const nx_c_group_exec *e, int worker,
                                    int64_t b, const int one) {
  const nx_c_group_rows r = e->r;
  int64_t first = b * NX_C_GROUP_BLOCK, end = nx_c_group_block_end(e, b);
  int64_t *out = e->out;
  /* A lone block counts its groups' partitions for no one. */
  int64_t lone[NX_C_GROUP_PARTS];
  int64_t *cursor = e->cursor ? e->cursor + b * NX_C_GROUP_PARTS : lone;
  memset(cursor, 0, NX_C_GROUP_PARTS * sizeof *cursor);
  nx_c_group_table t;
  nx_c_group_table_of(e, worker, nx_c_group_slots(end - first), e->reps + first,
                      &t);
  for (int64_t i = first; i < end; i += NX_C_GROUP_BATCH) {
    int64_t m = end - i < NX_C_GROUP_BATCH ? end - i : NX_C_GROUP_BATCH;
    uint64_t keys[NX_C_GROUP_BATCH], hs[NX_C_GROUP_BATCH];
    for (int64_t j = 0; j < m; j++) {
      keys[j] = nx_c_group_key(&r, i + j, one);
      hs[j] = nx_c_group_hash(&r, keys[j], one);
      NX_C_GROUP_PREFETCH(&t.slots[hs[j] & t.mask]);
    }
    for (int64_t j = 0; j < m; j++) {
      int64_t id = nx_c_group_find(&r, &t, keys[j], hs[j], i + j, one);
      if (id < 0) {
        id = ~id;
        cursor[hs[j] >> 56]++;
      }
      out[i + j] = id;
    }
  }
  e->ngroups[b] = t.count;
}

static void nx_c_group_blocks(int64_t lo, int64_t hi, int worker, void *vctx) {
  const nx_c_group_exec *e = vctx;
  for (int64_t b = lo; b < hi; b++)
    if (e->r.w == 1)
      nx_c_group_block(e, worker, b, 1);
    else
      nx_c_group_block(e, worker, b, 0);
}

/* 2. Partitioning */

static void nx_c_group_scatter(int64_t lo, int64_t hi, int worker,
                               void *vctx) {
  (void)worker;
  const nx_c_group_exec *e = vctx;
  int one = e->r.w == 1;
  for (int64_t b = lo; b < hi; b++) {
    int64_t *cursor = e->cursor + b * NX_C_GROUP_PARTS;
    const nx_c_group_rep *reps = e->reps + b * NX_C_GROUP_BLOCK;
    for (int64_t g = 0; g < e->ngroups[b]; g++) {
      uint64_t key = reps[g].key;
      nx_c_group_pair *p =
          &e->pairs[cursor[nx_c_group_hash(&e->r, key, one) >> 56]++];
      p->key = key;
      p->rep = b * NX_C_GROUP_BLOCK + g;
    }
  }
}

/* 3. Merging */

NX_C_GROUP_HOT void nx_c_group_partition(const nx_c_group_exec *e, int worker,
                                        int64_t p, const int one) {
  const nx_c_group_rows r = e->r;
  int64_t first = e->poff[p], end = e->poff[p + 1];
  nx_c_group_table t;
  nx_c_group_table_of(e, worker, e->regions[worker].size, e->merged + first,
                      &t);
  for (int64_t k = first; k < end; k += NX_C_GROUP_BATCH) {
    int64_t m = end - k < NX_C_GROUP_BATCH ? end - k : NX_C_GROUP_BATCH;
    uint64_t hs[NX_C_GROUP_BATCH];
    for (int64_t j = 0; j < m; j++) {
      hs[j] = nx_c_group_hash(&r, e->pairs[k + j].key, one);
      NX_C_GROUP_PREFETCH(&t.slots[hs[j] & t.mask]);
    }
    for (int64_t j = 0; j < m; j++) {
      nx_c_group_pair *pair = &e->pairs[k + j];
      int64_t row = one ? 0 : e->reps[pair->rep].row;
      int64_t id = nx_c_group_find(&r, &t, pair->key, hs[j], row, one);
      /* The flat index, complemented for the block group that made it. */
      pair->key = (uint64_t)(id < 0 ? ~(first + ~id) : first + id);
    }
  }
}

static void nx_c_group_merge(int64_t lo, int64_t hi, int worker, void *vctx) {
  const nx_c_group_exec *e = vctx;
  for (int64_t p = lo; p < hi; p++)
    if (e->r.w == 1)
      nx_c_group_partition(e, worker, p, 1);
    else
      nx_c_group_partition(e, worker, p, 0);
}

/* 4. Numbering */

static void nx_c_group_count(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_group_exec *e = vctx;
  for (int64_t b = lo; b < hi; b++) {
    nx_c_group_gather(e, b);
    const nx_c_group_rep *reps = e->reps + b * NX_C_GROUP_BLOCK;
    int64_t c = 0;
    for (int64_t g = 0; g < e->ngroups[b]; g++) c += (int64_t)reps[g].key < 0;
    e->firsts[b] = c;
  }
}

static void nx_c_group_number(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_group_exec *e = vctx;
  for (int64_t b = lo; b < hi; b++) {
    const nx_c_group_rep *reps = e->reps + b * NX_C_GROUP_BLOCK;
    int64_t id = e->firsts[b];
    for (int64_t g = 0; g < e->ngroups[b]; g++) {
      int64_t m = (int64_t)reps[g].key;
      if (m < 0) e->merged[~m].key = (uint64_t)id++;
    }
  }
}

/* 5. Ids */

static void nx_c_group_numbers(int64_t lo, int64_t hi, int worker,
                               void *vctx) {
  (void)worker;
  const nx_c_group_exec *e = vctx;
  for (int64_t k = e->poff[lo]; k < e->poff[hi]; k++) {
    int64_t m = (int64_t)e->pairs[k].key;
    e->pairs[k].key = e->merged[m < 0 ? ~m : m].key;
  }
}

static void nx_c_group_ids(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_group_exec *e = vctx;
  for (int64_t b = lo; b < hi; b++) {
    nx_c_group_gather(e, b);
    const nx_c_group_rep *reps = e->reps + b * NX_C_GROUP_BLOCK;
    for (int64_t i = b * NX_C_GROUP_BLOCK; i < nx_c_group_block_end(e, b); i++)
      e->out[i] = (int64_t)reps[e->out[i]].key;
  }
}

/* Driver */

/* Gives each worker a region of slots for tables of up to [groups] groups:
   more than twice as many slots. */
static nx_c_status nx_c_group_regions(nx_c_group_exec *e, int64_t groups) {
  int64_t slots = nx_c_group_slots(groups);
  for (int w = 0; w < e->nregions; w++) {
    nx_c_group_region *g = &e->regions[w];
    nx_c_aligned_free(g->slots);
    *g = (nx_c_group_region){.size = NX_C_GROUP_MIN_SLOTS};
    g->slots = nx_c_aligned_alloc((size_t)slots * sizeof *g->slots);
    if (!g->slots) return NX_C_ERR_ALLOC;
  }
  return NX_C_OK;
}

/* Phases 2 to 5, once every block is grouped. */
static nx_c_status nx_c_group_merge_blocks(nx_c_group_exec *e) {
  int nth = e->nregions;
  /* Each partition's groups follow the previous partition's, block by block. */
  int64_t total = 0, widest = 0;
  for (int p = 0; p < NX_C_GROUP_PARTS; p++) {
    e->poff[p] = total;
    for (int64_t b = 0; b < e->nblocks; b++) {
      int64_t *c = &e->cursor[b * NX_C_GROUP_PARTS + p], groups = *c;
      *c = total;
      total += groups;
    }
    if (total - e->poff[p] > widest) widest = total - e->poff[p];
  }
  e->poff[NX_C_GROUP_PARTS] = total;
  e->pairs = malloc((size_t)total * sizeof *e->pairs);
  e->merged = malloc((size_t)total * sizeof *e->merged);
  if (!e->pairs || !e->merged) return NX_C_ERR_ALLOC;
  int64_t traffic = total * (int64_t)(2 * sizeof *e->pairs);
  nx_c_parallel_for(nth, e->nblocks, traffic, nx_c_group_scatter, e, NULL);

  nx_c_status s = nx_c_group_regions(e, widest);
  if (s != NX_C_OK) return s;
  nx_c_parallel_for(nth, NX_C_GROUP_PARTS, traffic, nx_c_group_merge, e, NULL);

  nx_c_parallel_for(nth, e->nblocks, traffic, nx_c_group_count, e, NULL);
  int64_t id = 0;
  for (int64_t b = 0; b < e->nblocks; b++) {
    int64_t c = e->firsts[b];
    e->firsts[b] = id;
    id += c;
  }
  nx_c_parallel_for(nth, e->nblocks, traffic, nx_c_group_number, e, NULL);
  nx_c_parallel_for(nth, NX_C_GROUP_PARTS, traffic, nx_c_group_numbers, e,
                    NULL);
  nx_c_parallel_for(nth, e->nblocks, traffic + e->n * 2 * 8, nx_c_group_ids,
                    e, NULL);
  return NX_C_OK;
}

static nx_c_status nx_c_group_run(nx_c_group_exec *e, int threads) {
  int64_t bytes = e->n * (e->r.w + 1) * 8;
  e->nregions = nx_c_plan_threads(threads, NX_C_COST_COMPUTE, e->nblocks,
                                  NX_C_GROUP_BLOCK, bytes);
  int merging = e->nblocks > 1;
  e->reps = malloc((size_t)e->n * sizeof *e->reps);
  e->ngroups = malloc((size_t)e->nblocks * sizeof *e->ngroups);
  e->regions = calloc((size_t)e->nregions, sizeof *e->regions);
  if (!e->reps || !e->ngroups || !e->regions) return NX_C_ERR_ALLOC;
  if (merging) {
    e->firsts = malloc((size_t)e->nblocks * sizeof *e->firsts);
    e->cursor =
        malloc((size_t)e->nblocks * NX_C_GROUP_PARTS * sizeof *e->cursor);
    if (!e->firsts || !e->cursor) return NX_C_ERR_ALLOC;
  }
  nx_c_status s =
      nx_c_group_regions(e, e->n < NX_C_GROUP_BLOCK ? e->n : NX_C_GROUP_BLOCK);
  if (s != NX_C_OK) return s;
  nx_c_parallel_for(e->nregions, e->nblocks, bytes, nx_c_group_blocks, e,
                    NULL);
  return merging ? nx_c_group_merge_blocks(e) : NX_C_OK;
}

static nx_c_status nx_c_group_drive(const nx_c_ndarray *in,
                                    const nx_c_ndarray *out, int threads) {
  if (in->ndim != 2) return NX_C_ERR_OUT_RANK;
  if (out->ndim != 1 || out->shape[0] != in->shape[0]) return NX_C_ERR_SHAPE;
  if (out->shape[0] > 1 && out->strides[0] != 1) return NX_C_ERR_OUT_ALIASED;
  if ((uint64_t)in->shape[0] > NX_C_GROUP_ID_MASK) return NX_C_GROUP_ERR_ROWS;
  nx_c_group_exec e = {0};
  e.n = in->shape[0];
  e.out = (int64_t *)out->data + out->offset;
  e.r.w = in->shape[1];
  if (e.n == 0) return NX_C_OK;
  if (e.r.w == 0) {
    memset(e.out, 0, (size_t)e.n * sizeof *e.out);
    return NX_C_OK;
  }
  e.r.base = (const char *)in->data + in->offset * 8;
  e.r.rs = in->strides[0] * 8;
  e.r.cs = in->strides[1] * 8;
  e.r.seed = nx_c_group_mix((uint64_t)(uintptr_t)e.out ^ (uint64_t)e.n);
  e.nblocks = (e.n + NX_C_GROUP_BLOCK - 1) / NX_C_GROUP_BLOCK;
  nx_c_status s = nx_c_group_run(&e, threads);
  if (e.regions)
    for (int w = 0; w < e.nregions; w++)
      nx_c_aligned_free(e.regions[w].slots);
  free(e.regions);
  free(e.reps);
  free(e.ngroups);
  free(e.firsts);
  free(e.cursor);
  free(e.pairs);
  free(e.merged);
  return s;
}

/* Stub */

CAMLprim value caml_nx_c_group(value vout, value vin, value vthreads) {
  CAMLparam3(vout, vin, vthreads);
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) nx_c_raise("group", s);
  s = nx_c_group_drive(&in, &out, Int_val(vthreads));
  if (s != NX_C_OK) nx_c_raise_status("group", s);
  CAMLreturn(Val_unit);
}
