/* The Zstandard encoder's match finders, levels 1 to 12. Each turns a
   block into sequences (RFC 8878 section 3.1.1.3.2): literals, then a
   match coded by its Offset_Value.

   - fast (levels 1, 2): one table of the last position per hash of the
     next [mls] bytes. A scan probes positions two at a time, [a] and
     [a + 1], then jumps to the next pair; the jump grows by one every
     128 bytes without a match. Before the pair's own candidates it tries
     the repeat offset at the next pair's start.
   - double fast (3, 4): a table of 8-byte hashes and one of [mls]-byte
     hashes, probed one position at a time with a jump that grows every
     256 bytes. A long candidate is taken at once; a short one only if the
     next position's long candidate is no longer.
   - greedy, lazy and lazy2 (5 to 12): rows of 15 to 63 recent positions
     per hash, a byte of the hash tagging each; the tags of a row are
     compared at once and the matching candidates counted. The rows are
     filled in position order just before a search needs them, with the
     hashes of the next 8 positions kept so that a row is requested 8
     positions before it is written or searched. The lazy strategies look
     one or two positions further for a match whose length, weighed
     against its offset's bits, wins by a margin; a search there skips
     the candidates too short to win.

   The fast finders read the next probe's table entries before testing
   the current probe's candidates, so that the memory reads of
   consecutive probes overlap.

   Every finder tries the repeat offset before its tables and, after a
   match, the second repeat offset at the match's end. A candidate below
   the window, or before the frame, is never used, and no read reaches
   the block's end, so the sequences of a block depend only on its bytes,
   the window before it and the tables. No decision depends on an index's
   value, only on differences between indices, so rebasing them changes
   nothing. */

#include "zstd_enc_int.h"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/* The fast finders' growth of the step (the reference's kSearchStrength),
   and the lazy finders' threshold for skipping insertions. */
#define SEARCH_STRENGTH 8
#define LAZY_SKIP_STEP 8

/* The bytes scanned without a match after which fast's and double fast's
   steps grow by one. */
#define GROW_FAST (1u << (SEARCH_STRENGTH - 1))
#define GROW_DFAST (1u << SEARCH_STRENGTH)

/* The rows' parameters: the tag's bits, the hashes kept ahead, and the
   bounds on the update after a long match. */
#define TAG_BITS 8
#define TAG_MASK ((1u << TAG_BITS) - 1)

/* [tag h] is the hash's tag, never 0: a slot whose tag is 0 has held no
   position, so a row's positions need no clearing, only its tags. */
ZE_INLINE uint8_t tag_of(uint32_t h) {
  uint8_t t = (uint8_t)(h & TAG_MASK);
  return (uint8_t)(t + (t == 0));
}
#define CACHE 8
#define SKIP_THRESHOLD 384
#define SKIP_START 96
#define SKIP_END 32

/* The most positions a block's first search inserts before the block,
   once the gap passes SKIP_THRESHOLD. */
#define CLAMP_KEEP 192

/* The reference's multiplicative hash primes for 4 to 8 bytes. */
static const uint32_t prime4 = 2654435761U;
static const uint64_t prime5 = 889523592379ULL;
static const uint64_t prime6 = 227718039650203ULL;
static const uint64_t prime7 = 58295818150454627ULL;
static const uint64_t prime8 = 0xCF1BBCDCB7A56463ULL;

ZE_INLINE size_t hash(const uint8_t *p, unsigned hlog, unsigned mls) {
  switch (mls) {
  case 5:
    return (size_t)(((ze_rd64(p) << 24) * prime5) >> (64 - hlog));
  case 6:
    return (size_t)(((ze_rd64(p) << 16) * prime6) >> (64 - hlog));
  case 7:
    return (size_t)(((ze_rd64(p) << 8) * prime7) >> (64 - hlog));
  case 8:
    return (size_t)((ze_rd64(p) * prime8) >> (64 - hlog));
  default:
    return (size_t)((ze_rd32(p) * prime4) >> (32 - hlog));
  }
}

/* [count ip m end] is the length of the common prefix of [ip] and [m],
   up to [end]. */
ZE_INLINE size_t count(const uint8_t *ip, const uint8_t *m,
                       const uint8_t *end) {
  const uint8_t *start = ip;
  while (ip + 8 <= end) {
    uint64_t d = ze_rd64(ip) ^ ze_rd64(m);
    if (d) return (size_t)(ip - start) + ((unsigned)__builtin_ctzll(d) >> 3);
    ip += 8;
    m += 8;
  }
  while (ip < end && *ip == *m) {
    ip++;
    m++;
  }
  return (size_t)(ip - start);
}

/* Where a finder appends: the next sequence and the next literal, kept
   in locals while it runs. */
struct out {
  struct ze_seq *q;
  uint8_t *lit;
};

#define OPEN(ss) {(ss)->seq + (ss)->n, (ss)->lit + (ss)->nlit}
#define CLOSE(ss, o)                                                           \
  do {                                                                         \
    (ss)->n = (size_t)((o).q - (ss)->seq);                                     \
    (ss)->nlit = (size_t)((o).lit - (ss)->lit);                                \
  } while (0)

/* [store o ll lit off ml end] appends a sequence and copies its literals,
   with wide copies while [end] is far. */
ZE_INLINE void store(struct out *o, size_t ll, const uint8_t *lit,
                     uint32_t off, size_t ml, const uint8_t *end) {
  struct ze_seq *q = o->q++;
  q->ll = (uint32_t)ll;
  q->ml = (uint32_t)ml;
  q->off = off;
  q->code = ze_code((uint32_t)ll, (uint32_t)ml, off);

  /* 16-byte copies overrun the literals by at most 15 bytes, inside
     ZE_SLACK, and read at most 15 past them, which must stay before
     [end]. */
  uint8_t *d = o->lit;
  o->lit = d + ll;
  if ((size_t)(end - lit) < ll + 16) {
    memcpy(d, lit, ll);
    return;
  }
  size_t i = 0;
  do {
    memcpy(d + i, lit + i, 16);
    i += 16;
  } while (i < ll);
}

#define REP1 1u
#define OFF(o) ((uint32_t)(o) + 3)

/* The index of a position, in a finder whose [base] is in scope. */
#define IDX(p) ((uint32_t)((p) - base))

/* [lowest ms idx] is the lowest index a match may reach from [idx]. */
ZE_INLINE uint32_t lowest(const struct ze_match *ms, uint32_t idx) {
  uint32_t w = 1u << ms->p.wlog;
  return idx - ms->low > w ? idx - w : ms->low;
}

/* Repeat offsets

   [r1], [r2] follow the decoder's first two repeat offsets. [u1], [u2]
   are the same offsets where the block may use them, 0 where it may not:
   an offset reaching below the window at the block's first searchable
   position stays unused for the whole block, however it moves. An offset
   found by a match stays usable, since the window's floor rises by at
   most one per position. */
struct reps {
  uint32_t r1, r2, u1, u2;
};

/* [first ms src] is the first position of the block at [src] a match
   may start at: the frame's first byte is always a literal. */
ZE_INLINE const uint8_t *first(const struct ze_match *ms,
                               const uint8_t *src) {
  return src + ((uint32_t)(src - ms->base) == ms->low);
}

ZE_INLINE struct reps reps_in(const struct ze_match *ms, const uint32_t rep[2],
                              const uint8_t *f) {
  uint32_t i = (uint32_t)(f - ms->base);
  uint32_t reach = i - lowest(ms, i);
  struct reps r = {rep[0], rep[1], rep[0] <= reach ? rep[0] : 0,
                   rep[1] <= reach ? rep[1] : 0};
  return r;
}

ZE_INLINE void reps_out(const struct reps *r, uint32_t rep[2]) {
  rep[0] = r->r1;
  rep[1] = r->r2;
}

/* A new offset goes in front. */
ZE_INLINE void reps_push(struct reps *r, uint32_t off) {
  r->r2 = r->r1;
  r->u2 = r->u1;
  r->r1 = off;
  r->u1 = off;
}

/* Offset_Value 1 without literals names the second offset, which the
   decoder moves in front. */
ZE_INLINE void reps_swap(struct reps *r) {
  uint32_t t = r->r1;
  r->r1 = r->r2;
  r->r2 = t;
  t = r->u1;
  r->u1 = r->u2;
  r->u2 = t;
}

/* [rep_at p u] holds when the offset [u] is usable and the 4 bytes at [p]
   recur [u] back. */
ZE_INLINE bool rep_at(const uint8_t *p, uint32_t u) {
  return u != 0 && ze_rd32(p - u) == ze_rd32(p);
}

/* Fast */

/* [cand4 base j floor p cmov] holds when the entry [j] is in the window
   and its 4 bytes equal those at [p]. With [cmov] the window test is a
   select, not a branch: below 512 KiB windows stale entries are common
   and their test unpredictable. The bytes read are then [floor]'s, inside
   the window, and the result the same. */
ZE_INLINE bool cand4(const uint8_t *base, uint32_t j, uint32_t floor,
                     const uint8_t *p, bool cmov) {
  if (!cmov) return j >= floor && ze_rd32(base + j) == ze_rd32(p);
  bool in = j >= floor;
  uint32_t k = in ? j : floor;
  return in & (ze_rd32(base + k) == ze_rd32(p));
}

ZE_INLINE size_t fast(struct ze_match *ms, struct ze_seqs *ss,
                      uint32_t rep[2], const uint8_t *src, size_t n,
                      unsigned mls, bool cmov) {
  const uint8_t *const base = ms->base;
  uint32_t *const t = ms->t1;
  const unsigned hlog = ms->p.hlog;
  const uint8_t *const end = src + n;
  /* A pair (a, a + 1) whose next pair starts at c is probed while
     c + 1 < lim: c's repeat test and the hashes stay inside the block. */
  const uint8_t *const lim = end - 8;
  const uint32_t floor = lowest(ms, IDX(end));
  const uint8_t *const floor_p = base + floor;
  const size_t step0 = (ms->p.tlen ? ms->p.tlen : 1u) + 1;

  const uint8_t *anchor = src;
  const uint8_t *s = first(ms, src);
  struct reps r = reps_in(ms, rep, s);
  struct out o = OPEN(ss);

  for (;;) {
    /* A scan from [s]. Each pair reads its two entries, then inserts its
       two positions. The next pair's first entry is read as soon as
       this pair is inserted, as the next pair will find it, and its
       candidate's bytes requested before this pair's are tested. */
    const uint8_t *a = s, *x, *m, *from;
    size_t step = step0;
    if (lim - a <= (ptrdiff_t)step + 1) goto done;
    size_t ha = hash(a, hlog, mls);
    uint32_t ja = t[ha];
    uint32_t ov;
    size_t len;
    for (;;) {
      const uint8_t *b = a + 1, *c = a + step;
      size_t hb = hash(b, hlog, mls), hc = hash(c, hlog, mls);
      t[ha] = IDX(a);
      uint32_t jb = t[hb];
      t[hb] = IDX(b);
      uint32_t jc = t[hc];
      __builtin_prefetch(base + jc);

      if (rep_at(c, r.u1)) {
        x = c - (c[-1] == (c - r.u1)[-1]);
        len = 4 + count(c + 4, c + 4 - r.u1, end) + (size_t)(c - x);
        ov = REP1;
        from = a;
        goto found;
      }
      if (cand4(base, ja, floor, a, cmov)) {
        x = a;
        m = base + ja;
        from = a;
        goto hashed;
      }
      if (cand4(base, jb, floor, b, cmov)) {
        x = b;
        m = base + jb;
        from = b;
        goto hashed;
      }

      a = c;
      ha = hc;
      ja = jc;
      step = step0 + (size_t)(a - s) / GROW_FAST;
      if (lim - a <= (ptrdiff_t)step + 1) goto done;
    }

  hashed:
    len = 4 + count(x + 4, m + 4, end);
    while (x > anchor && m > floor_p && x[-1] == m[-1]) {
      x--;
      m--;
      len++;
    }
    ov = OFF(x - m);
    reps_push(&r, (uint32_t)(x - m));

  found:
    store(&o, (size_t)(x - anchor), anchor, ov, len, end);
    anchor = x + len;
    if (anchor <= lim) {
      /* The pair's next pair start, inside the match, would never be
         inserted otherwise. */
      if (from != a && step <= 4) t[hash(a + step, hlog, mls)] = IDX(a + step);
      t[hash(from + 2, hlog, mls)] = IDX(from + 2);
      t[hash(anchor - 2, hlog, mls)] = IDX(anchor - 2);
      while (anchor <= lim && rep_at(anchor, r.u2)) {
        size_t l = 4 + count(anchor + 4, anchor + 4 - r.u2, end);
        store(&o, 0, anchor, REP1, l, end);
        reps_swap(&r);
        t[hash(anchor, hlog, mls)] = IDX(anchor);
        anchor += l;
      }
    }
    s = anchor;
  }

done:
  CLOSE(ss, o);
  reps_out(&r, rep);
  return (size_t)(end - anchor);
}

#define FAST(m, c)                                                             \
  static size_t fast_##m##_##c(struct ze_match *ms, struct ze_seqs *ss,        \
                               uint32_t rep[2], const uint8_t *src,      \
                               size_t n) {                                     \
    return fast(ms, ss, rep, src, n, m, c);                                    \
  }
FAST(4, 0)
FAST(5, 0)
FAST(6, 0)
FAST(7, 0)
FAST(4, 1)
FAST(5, 1)
FAST(6, 1)
FAST(7, 1)

/* Double fast */

ZE_INLINE size_t dfast(struct ze_match *ms, struct ze_seqs *ss,
                       uint32_t rep[2], const uint8_t *src, size_t n,
                       unsigned mls) {
  const uint8_t *const base = ms->base;
  uint32_t *const tl = ms->t1;
  uint32_t *const ts = ms->t2;
  const unsigned hlog = ms->p.hlog, clog = ms->p.clog;
  const uint8_t *const end = src + n;
  /* A probe at p whose next probe is q runs while q <= lim: the 8-byte
     reads at q stay inside the block. */
  const uint8_t *const lim = end - 8;
  const uint32_t floor = lowest(ms, IDX(end));
  const uint8_t *const floor_p = base + floor;

  const uint8_t *anchor = src;
  const uint8_t *s = first(ms, src);
  struct reps r = reps_in(ms, rep, s);
  struct out o = OPEN(ss);

  for (;;) {
    /* A scan from [s]. A probe inserts itself, then reads the next
       probe's entries, as that probe will find them, and requests their
       candidates' bytes before testing its own. */
    const uint8_t *p = s, *q, *x, *m;
    size_t step = 1, len;
    uint32_t ov;
    bool hashed;
    if (p + 1 > lim) goto done;
    size_t hl = hash(p, hlog, 8), hs = hash(p, clog, mls);
    uint32_t jl = tl[hl], js = ts[hs];
    for (;;) {
      q = p + step;
      if (q > lim) goto done;
      size_t hlq = hash(q, hlog, 8), hsq = hash(q, clog, mls);
      tl[hl] = ts[hs] = IDX(p);
      uint32_t jlq = tl[hlq], jsq = ts[hsq];
      __builtin_prefetch(base + jlq);
      __builtin_prefetch(base + jsq);

      if (rep_at(p + 1, r.u1)) {
        x = p + 1;
        len = 4 + count(p + 5, p + 5 - r.u1, end);
        ov = REP1;
        hashed = false;
        break;
      }
      if (jl >= floor && ze_rd64(base + jl) == ze_rd64(p)) {
        x = p;
        m = base + jl;
        len = 8 + count(p + 8, m + 8, end);
        hashed = true;
        break;
      }
      if (js >= floor && ze_rd32(base + js) == ze_rd32(p)) {
        /* The next position's long candidate, as the table holds it now,
           wins when it is longer. */
        x = p;
        m = base + js;
        len = 4 + count(p + 4, m + 4, end);
        uint32_t jq = jlq;
        if (jq > floor && ze_rd64(base + jq) == ze_rd64(q)) {
          size_t lq = 8 + count(q + 8, base + jq + 8, end);
          if (lq > len) {
            x = q;
            m = base + jq;
            len = lq;
          }
        }
        hashed = true;
        break;
      }

      p = q;
      hl = hlq;
      hs = hsq;
      jl = jlq;
      js = jsq;
      step = 1 + (size_t)(p - s) / GROW_DFAST;
    }

    if (hashed) {
      while (x > anchor && m > floor_p && x[-1] == m[-1]) {
        x--;
        m--;
        len++;
      }
      ov = OFF(x - m);
      reps_push(&r, (uint32_t)(x - m));
      if (step < 4) tl[hash(q, hlog, 8)] = IDX(q);
    }

    store(&o, (size_t)(x - anchor), anchor, ov, len, end);
    anchor = x + len;
    if (anchor <= lim) {
      tl[hash(p + 2, hlog, 8)] = ts[hash(p + 2, clog, mls)] = IDX(p + 2);
      tl[hash(anchor - 2, hlog, 8)] = IDX(anchor - 2);
      ts[hash(anchor - 1, clog, mls)] = IDX(anchor - 1);
      while (anchor <= lim && rep_at(anchor, r.u2)) {
        size_t l = 4 + count(anchor + 4, anchor + 4 - r.u2, end);
        store(&o, 0, anchor, REP1, l, end);
        reps_swap(&r);
        tl[hash(anchor, hlog, 8)] = ts[hash(anchor, clog, mls)] = IDX(anchor);
        anchor += l;
      }
    }
    s = anchor;
  }

done:
  CLOSE(ss, o);
  reps_out(&r, rep);
  return (size_t)(end - anchor);
}

#define DFAST(m)                                                               \
  static size_t dfast_##m(struct ze_match *ms, struct ze_seqs *ss,             \
                          uint32_t rep[2], const uint8_t *src,           \
                          size_t n) {                                          \
    return dfast(ms, ss, rep, src, n, m);                                      \
  }
DFAST(4)
DFAST(5)
DFAST(6)
DFAST(7)

/* Rows */

/* [row_mask tags tag head entries] has bit i set when the entry [i]
   places after the row's head carries [tag]: newest first. The SSE2 and
   the generic paths give the same bits. */
ZE_INLINE uint64_t row_mask(const uint8_t *tags, uint8_t tag, unsigned head,
                            unsigned entries) {
  uint64_t m = 0;
#if defined(__SSE2__)
  /* Broadcast from 32 bits: a byte spilled and reloaded wider would
     stall the load on its store. */
  const __m128i t = _mm_set1_epi32((int)(tag * 0x01010101u));
  for (unsigned i = 0; i < entries; i += 16) {
    __m128i c = _mm_loadu_si128((const __m128i *)(const void *)(tags + i));
    m |= (uint64_t)(uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(c, t)) << i;
  }
#else
  const uint64_t ones = 0x0101010101010101ULL, high = 0x8080808080808080ULL;
  for (unsigned i = 0; i < entries; i += 8) {
    uint64_t x = ze_rd64(tags + i) ^ (ones * tag);
    uint64_t z = ~(((x & ~high) + ~high) | x) & high;
    m |= ((z >> 7) * 0x0102040810204080ULL >> 56) << i;
  }
#endif
  if (entries == 64) return head ? (m >> head) | (m << (64 - head)) : m;
  uint64_t full = (1ull << entries) - 1;
  return ((m >> head) | (m << (entries - head))) & full;
}

/* The next slot of a row: a ring over slots 1 to [mask], walking down;
   slot 0 of the tags holds the head. */
ZE_INLINE unsigned row_next(uint8_t *tags, unsigned mask) {
  unsigned head = tags[0] & mask;
  unsigned next = head > 1 ? head - 1 : mask;
  tags[0] = (uint8_t)next;
  return next;
}

ZE_INLINE uint32_t row_hash(const uint8_t *p, unsigned bits, unsigned mls) {
  return (uint32_t)hash(p, bits, mls);
}

/* What a row operation needs, read once per search. A hash [h] names the
   row [h >> TAG_BITS] and the tag [tag_of h]. */
struct rows {
  const uint8_t *base;
  uint32_t *restrict t;
  uint8_t *restrict tags;
  uint32_t *restrict cache;
  unsigned bits, mls, rlog;
};

ZE_INLINE size_t row_of(const struct rows *rw, uint32_t h) {
  return (size_t)(h >> TAG_BITS) << rw->rlog;
}

ZE_INLINE void row_fetch(const struct rows *rw, uint32_t h) {
  size_t row = row_of(rw, h);
  __builtin_prefetch(rw->tags + row);
  __builtin_prefetch(rw->t + row);
}

/* [row_put rw h i] writes position [i], of hash [h], over its row's
   oldest entry. */
ZE_INLINE void row_put(const struct rows *rw, uint32_t h, uint32_t i) {
  size_t row = row_of(rw, h);
  unsigned slot = row_next(rw->tags + row, (1u << rw->rlog) - 1);
  rw->tags[row + slot] = tag_of(h);
  rw->t[row + slot] = i;
}

/* The hash cache. While [ms->cached], [ms->cache] holds the hashes of
   the CACHE positions from [ms->next], each at its index modulo CACHE,
   and their rows have been requested. Taking a position's hash replaces
   it with the hash CACHE positions further, so a row is requested at
   least CACHE positions before it is written or searched. Hashing
   position [k + CACHE] reads up to [k + 2 * CACHE - 1]: the lazy parser
   stops 2 * CACHE before the block's end. Any jump of [ms->next] clears
   [ms->cached]. */

ZE_INLINE void cache_fill(const struct rows *rw, uint32_t k) {
  for (uint32_t i = k; i < k + CACHE; i++) {
    uint32_t h = row_hash(rw->base + i, rw->bits, rw->mls);
    rw->cache[i & (CACHE - 1)] = h;
    row_fetch(rw, h);
  }
}

ZE_INLINE uint32_t cache_take(const struct rows *rw, uint32_t k) {
  uint32_t h = rw->cache[k & (CACHE - 1)];
  uint32_t a = row_hash(rw->base + k + CACHE, rw->bits, rw->mls);
  rw->cache[k & (CACHE - 1)] = a;
  row_fetch(rw, a);
  return h;
}

/* [row_insert rw k stop] inserts positions [k] to [stop - 1] from a
   valid cache at [k], leaving it valid at [stop]. Each insertion
   requests the row CACHE positions ahead as it takes its own hash.
   Requesting the rows of a run of positions at once instead puts more
   misses in flight than the core tracks, and the requests stall. */
ZE_INLINE void row_insert(const struct rows *rw, uint32_t k, uint32_t stop) {
  for (; k < stop; k++) row_put(rw, cache_take(rw, k), k);
}

/* [catch_up ms rw t] brings the rows up to position [t], exclusive, and
   leaves the cache valid at [t]. Of a gap longer than SKIP_THRESHOLD only
   the first SKIP_START and the last SKIP_END positions go in. */
ZE_INLINE void catch_up(struct ze_match *ms, const struct rows *rw,
                        uint32_t t) {
  uint32_t k = ms->next;
  if (!ms->cached) cache_fill(rw, k);
  if (t - k > SKIP_THRESHOLD) {
    row_insert(rw, k, k + SKIP_START);
    k = t - SKIP_END;
    cache_fill(rw, k);
  }
  row_insert(rw, k, t);
  ms->cached = true;
}

/* [row_search ms ip iend off least rlog mls] is the longest match at
   [ip] among the row's candidates longer than [least], at least 3, or 3
   for none, with [*off] its Offset_Value. It inserts [ip]. */
ZE_INLINE size_t row_search(struct ze_match *ms, const uint8_t *ip,
                            const uint8_t *iend, uint32_t *off, size_t least,
                            unsigned rlog, unsigned mls) {
  const uint8_t *const base = ms->base;
  const struct rows rw = {base,     ms->t1, ms->tags,
                          ms->cache, ms->p.hlog - rlog + TAG_BITS,
                          mls,      rlog};
  const uint32_t t = IDX(ip);

  /* In skip mode the positions since the last search are dropped. */
  uint32_t h;
  if (!ms->skipping) {
    catch_up(ms, &rw, t);
    h = cache_take(&rw, t);
  } else if (ms->cached && ms->next == t) {
    h = cache_take(&rw, t);
  } else {
    ms->cached = false;
    h = row_hash(ip, rw.bits, mls);
  }
  ms->next = t + 1;

  const unsigned mask = (1u << rlog) - 1;
  const size_t row = row_of(&rw, h);
  uint8_t *const tags = rw.tags + row;
  uint32_t *const slots = rw.t + row;
  const uint8_t tag = tag_of(h);
  const unsigned head = tags[0] & mask;

  /* Bit i is the slot i places after the head; slot 0 holds the head. */
  uint64_t hits = row_mask(tags, tag, head, mask + 1);
  hits &= ~(1ull << ((mask + 1 - head) & mask));

  /* The candidates in the window, newest first, their bytes requested
     before any is compared. */
  const uint32_t lo = lowest(ms, t);
  const unsigned slog = ms->p.slog;
  const unsigned tries = 1u << (slog < rlog ? slog : rlog);
  uint32_t cand[64];
  unsigned nc = 0;
  for (; hits && nc < tries; hits &= hits - 1) {
    uint32_t j = slots[(head + (unsigned)__builtin_ctzll(hits)) & mask];
    /* The ring runs newest to oldest: past the window, so are the rest. */
    if (j < lo) break;
    __builtin_prefetch(base + j);
    cand[nc++] = j;
  }

  unsigned slot = row_next(tags, mask);
  tags[slot] = tag;
  slots[slot] = t;

  size_t best = least;
  uint32_t bo = 0;
  /* No candidate is longer than the bytes left. */
  if (best >= (size_t)(iend - ip)) nc = 0;
  for (unsigned k = 0; k < nc; k++) {
    const uint8_t *m = base + cand[k];
    /* A candidate is longer only if it matches up to byte [best]: the 4
       bytes ending there reject most in one load, the 4 at its start the
       rest. */
    if (ze_rd32(m + best - 3) != ze_rd32(ip + best - 3) ||
        ze_rd32(m) != ze_rd32(ip))
      continue;
    size_t len = 4 + count(ip + 4, m + 4, iend);
    if (len > best) {
      best = len;
      bo = t - cand[k];
      if (ip + len == iend) break;
    }
  }

  if (!bo) return 3;
  *off = OFF(bo);
  return best;
}

#define ROW(m, r)                                                              \
  static ZE_NOINLINE size_t row_##m##_##r(struct ze_match *ms,                 \
                                          const uint8_t *ip,                   \
                                          const uint8_t *iend,                 \
                                          uint32_t *off, size_t least) {       \
    return row_search(ms, ip, iend, off, least, r, m);                         \
  }
ROW(4, 4)
ROW(4, 5)
ROW(4, 6)
ROW(5, 4)
ROW(5, 5)
ROW(5, 6)
ROW(6, 4)
ROW(6, 5)
ROW(6, 6)

typedef size_t (*row_fn)(struct ze_match *, const uint8_t *, const uint8_t *,
                         uint32_t *, size_t);

static row_fn row_select(unsigned mls, unsigned rlog) {
  static const row_fn fns[3][3] = {{row_4_4, row_4_5, row_4_6},
                                   {row_5_4, row_5_5, row_5_6},
                                   {row_6_4, row_6_5, row_6_6}};
  return fns[mls - 4][rlog - 4];
}

ZE_INLINE unsigned bounded(unsigned lo, unsigned v, unsigned hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/* The lazy parser: at each position the repeat offset, then the row's
   best match; with [depth] 1 or 2 the next positions are searched too, and
   a later match replaces the current one when its length, weighed against
   the offset's bits, wins by the reference's margins. */
ZE_INLINE size_t lazy(struct ze_match *ms, struct ze_seqs *ss,
                      uint32_t rep[2], const uint8_t *src, size_t n,
                      unsigned depth) {
  const uint8_t *const base = ms->base;
  const uint8_t *const end = src + n;
  /* Searches hash CACHE positions ahead of themselves. */
  const uint8_t *const lim = end - 2 * CACHE;
  const uint8_t *const floor_p = base + lowest(ms, IDX(src));
  const row_fn search = row_select(bounded(4, ms->p.mls, 6),
                                   bounded(4, ms->p.slog, 6));

  const uint8_t *anchor = src;
  const uint8_t *ip = first(ms, src);
  struct reps r = reps_in(ms, rep, ip);
  struct out o = OPEN(ss);
  ms->skipping = false;
  ms->cached = false;

  while (ip < lim) {
    const uint8_t *start = ip;
    size_t len = 0;
    uint32_t ov = 0, mo = 0;

    if (rep_at(ip + 1, r.u1)) {
      start = ip + 1;
      len = 4 + count(ip + 5, ip + 5 - r.u1, end);
      ov = REP1;
    }
    if (depth > 0 || len == 0) {
      /* Only a longer match replaces the repeat match. */
      size_t ml = search(ms, ip, end, &mo, len < 4 ? 3 : len);
      if (ml > len) {
        start = ip;
        len = ml;
        ov = mo;
      }
    }
    if (len < 4) {
      size_t step = 1 + ((size_t)(ip - anchor) >> SEARCH_STRENGTH);
      if (step > LAZY_SKIP_STEP) ms->skipping = true;
      ip += step;
      continue;
    }

    /* Look-ahead: [d] positions past the last restart. A match at the
       next position replaces the best when it gains by the margins, in
       quarter (or third) bytes against the bits of the offsets. */
    unsigned d = 1;
    for (const uint8_t *i = ip; depth > 0 && i < lim;) {
      i++;
      int lb = (int)len, c = (int)ze_highbit(ov);
      if (rep_at(i, r.u1)) {
        int lr = 4 + (int)count(i + 4, i + 4 - r.u1, end);
        if (d == 1 ? 3 * lr > 3 * lb - c + 1 : 4 * lr > 4 * lb - c + 1) {
          start = i;
          len = (size_t)lr;
          ov = REP1;
        }
      }
      /* A match of length [ml] wins when 4 * ml - hb(mo) > bar; hb(mo) is
         at least 2, so one no longer than (bar + 2) / 4 never does. */
      lb = (int)len;
      c = (int)ze_highbit(ov);
      int bar = 4 * lb - c + (d == 1 ? 4 : 7);
      size_t least = bar + 2 < 12 ? 3 : (size_t)(bar + 2) / 4;
      size_t ml = search(ms, i, end, &mo, least);
      if (ml >= 4) {
        int gain = 4 * (int)ml - (int)ze_highbit(mo);
        if (gain > bar) {
          start = i;
          len = ml;
          ov = mo;
          d = 1;
          continue;
        }
      }
      if (d >= depth) break;
      d++;
    }

    if (ov > 3) {
      const uint8_t *m = start - (ov - 3);
      while (start > anchor && m > floor_p && start[-1] == m[-1]) {
        start--;
        m--;
        len++;
      }
      reps_push(&r, ov - 3);
    }
    store(&o, (size_t)(start - anchor), anchor, ov, len, end);
    ip = anchor = start + len;
    ms->skipping = false;

    while (ip <= lim && rep_at(ip, r.u2)) {
      size_t l = 4 + count(ip + 4, ip + 4 - r.u2, end);
      store(&o, 0, anchor, REP1, l, end);
      reps_swap(&r);
      ip = anchor = ip + l;
    }
  }

  CLOSE(ss, o);
  reps_out(&r, rep);
  return (size_t)(end - anchor);
}

static size_t greedy_b(struct ze_match *ms, struct ze_seqs *ss,
                       uint32_t rep[2], const uint8_t *src, size_t n) {
  return lazy(ms, ss, rep, src, n, 0);
}

static size_t lazy_b(struct ze_match *ms, struct ze_seqs *ss,
                     uint32_t rep[2], const uint8_t *src, size_t n) {
  return lazy(ms, ss, rep, src, n, 1);
}

static size_t lazy2_b(struct ze_match *ms, struct ze_seqs *ss,
                      uint32_t rep[2], const uint8_t *src, size_t n) {
  return lazy(ms, ss, rep, src, n, 2);
}

/* Entry points */

size_t ze_match_bytes(const struct ze_params *p) {
  size_t h = (size_t)4 << p->hlog;
  switch (p->strat) {
  case ZE_FAST:
    return h;
  case ZE_DFAST:
    return h + ((size_t)4 << p->clog);
  default:
    return h + ((size_t)1 << p->hlog);
  }
}

void ze_match_reset(struct ze_match *ms) {
  const struct ze_params *p = &ms->p;
  if (p->strat >= ZE_GREEDY)
    memset(ms->tags, 0, (size_t)1 << p->hlog);
  else
    memset(ms->t1, 0, (size_t)4 << p->hlog);
  if (p->strat == ZE_DFAST) memset(ms->t2, 0, (size_t)4 << p->clog);
  ms->skipping = false;
  ms->cached = false;
  memset(ms->cache, 0, sizeof ms->cache);
}

/* [clamp_next ms cur] bounds the positions the rows insert before the
   block starting at index [cur]. */
ZE_INLINE void clamp_next(struct ze_match *ms, uint32_t cur) {
  if (ms->next < ms->low) ms->next = ms->low;
  if (cur - ms->next > SKIP_THRESHOLD) {
    uint32_t over = cur - ms->next - SKIP_THRESHOLD;
    ms->next = cur - (over < CLAMP_KEEP ? over : CLAMP_KEEP);
  }
  ms->cached = false;
}

size_t ze_match_block(struct ze_match *ms, struct ze_seqs *ss,
                      uint32_t rep[2], const uint8_t *src, size_t n) {
  /* Too short for a search: every finder needs 16 bytes of lookahead. */
  if (n <= 16) return n;

  clamp_next(ms, (uint32_t)(src - ms->base));

  const struct ze_params *p = &ms->p;
  unsigned mls = p->mls < 4 ? 4 : p->mls > 7 ? 7 : p->mls;
  switch (p->strat) {
  case ZE_FAST: {
    static size_t (*const fns[2][4])(struct ze_match *, struct ze_seqs *,
                                           uint32_t *, const uint8_t *,
                                           size_t) = {
        {fast_4_0, fast_5_0, fast_6_0, fast_7_0},
        {fast_4_1, fast_5_1, fast_6_1, fast_7_1}};
    return fns[p->wlog < 19][mls - 4](ms, ss, rep, src, n);
  }
  case ZE_DFAST: {
    static size_t (*const fns[4])(struct ze_match *, struct ze_seqs *,
                                  uint32_t *, const uint8_t *, size_t) = {
        dfast_4, dfast_5, dfast_6, dfast_7};
    return fns[mls - 4](ms, ss, rep, src, n);
  }
  case ZE_GREEDY:
    return greedy_b(ms, ss, rep, src, n);
  case ZE_LAZY:
    return lazy_b(ms, ss, rep, src, n);
  default:
    return lazy2_b(ms, ss, rep, src, n);
  }
}
