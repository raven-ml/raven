/* The fast loop of deflate_decode.c, compiled once per instruction set
   it is built for. The includer defines:

     FAST          the function's name
     FAST_TARGET   its target attribute, or nothing
     EXTRACT(x,n)  the low (n & 0xff) bits of x, for that count below 64

   The bit buffer lives in [bb] and [bl]: [bl] bits of the stream, the
   next lowest. FILL tops it up to 56-63 bits: it loads the next 8 input
   bytes above the held bits and steps [in] past the whole bytes that fit
   (Giesen's "variant 4"). A fill leaves all 64 bits of [bb] the input's:
   those above [bl] are the next ones, which the following fill loads
   again. So after a fill, a lookup may read past [bl] as long as the
   symbols taken since have left it the bits it reads.

   An iteration starts after a fill, with its first symbol's entry looked
   up. It takes at most three literals from the root, of at most 11 bits
   each, then a literal of at most 15 bits or a length of at most 20:
   53 of the fill's 56 bits, which leaves 11 of the 64 for a lookup. A
   distance's root lookup reads 8 of them, and the iteration fills again
   unless the held bits cover the distance and the next lookup. */

#define FILL()                                                               \
  do {                                                                       \
    bb |= load64(in) << (bl & 63);                                           \
    in += (56 & ~bl) >> 3;                                                   \
    bl |= 56;                                                                \
  } while (0)

/* [bl] counts modulo 64, the held bits being [bl & 63]: a symbol takes
   its whole entry from it, of which only the low 6 bits count. */
#define TAKE(e)                                                              \
  do {                                                                       \
    bb >>= TAKES(e);                                                         \
    bl -= (e);                                                               \
  } while (0)

/* A length's or distance's value: its base and the extra bits above its
   code, which end where the entry's low byte says. Bits 12-13 of such an
   entry are zero, so its second byte is a shift count as it stands. */
#define VAL(e) (VALUE(e) + (uint32_t)(EXTRACT(bb, e) >> (((e) >> 8) & 63)))

/* The entry a subtable pointer leads to. */
#define SUB(t, e) ((t)[VALUE(e) + EXTRACT(bb >> TAKES(e), CODE_LEN(e))])

/* The bits a distance and the next lookup need: a distance resolved in
   the root takes at most its 8 bits and 13 extra, one in a subtable 15
   and 13. */
#define ROOT_DIST_AND_NEXT (DIST_BITS + 13 + LITLEN_BITS)
#define DIST_AND_NEXT (MAX_CODE + 13 + LITLEN_BITS)

FAST_TARGET static enum fast_exit
FAST(const struct lz_deflate_dec *st, struct cursor *c, uint8_t *out0,
     const uint8_t *hist, const uint8_t *in_end, uint8_t *out_end) {
  const uint32_t *lt = st->litlen, *dt = st->dist;
  const uint64_t lmask = (UINT64_C(1) << st->litlen_bits) - 1;
  const uint64_t dmask = (UINT64_C(1) << DIST_BITS) - 1;
  const uint8_t *in = c->in, *in_last = in_end - FAST_IN;
  uint8_t *out = c->out, *out_last = out_end - FAST_OUT;
  uint64_t bb = c->bitbuf;
  uint32_t bl = c->bitsleft, e, d, len, dist;
  enum fast_exit why = FAST_SLOW;

  FILL();
  e = lt[bb & lmask];
  do {
    /* [next] is the buffer once the symbol of [e] is taken, shifted
       before its kind is known, so that the shift does not wait for the
       branch. */
    uint64_t next = bb >> TAKES(e);

    /* Literals, at most three a fill. The branch is laid out for the
       matches, whose path the literals' rejoins. */
    if (unlikely(e & E_LIT)) {
      unsigned k = 0;
      do {
        *out++ = (uint8_t)VALUE(e);
        bb = next;
        bl -= e;
        e = lt[bb & lmask];
        next = bb >> TAKES(e);
      } while (++k < 3 && (e & E_LIT));
      if (e & E_LIT) {
        FILL();
        continue;
      }
    }

    /* A code longer than the root, an end of block, or a symbol no
       stream sends, which the slow loop reports from its first bit. */
    if (unlikely(e & E_X)) {
      if (KIND(e) == K_SUB) {
        e = SUB(lt, e);
        if (e & E_LIT) {
          *out++ = (uint8_t)VALUE(e);
          TAKE(e);
          e = lt[bb & lmask];
          FILL();
          continue;
        }
      }
      if (e & E_X) {
        why = FAST_BAD_LITLEN;
        if (KIND(e) == K_END) {
          TAKE(e);
          why = FAST_END;
        }
        break;
      }
      next = bb >> TAKES(e);
    }
    len = VAL(e);
    bb = next;
    bl -= e;

    d = dt[bb & dmask];
    if ((bl & 63) < ROOT_DIST_AND_NEXT) FILL();
    if (unlikely(d & E_X)) {
      if ((bl & 63) < DIST_AND_NEXT) FILL();
      if (KIND(d) == K_SUB) d = SUB(dt, d);
      if (d & E_X) {
        why = FAST_BAD_DIST;
        break;
      }
    }
    dist = VAL(d);

    /* A match reaching before [out0] reads the window's earlier segment,
       or before the stream's start; the latter is left at its distance
       code for the slow loop to report. */
    size_t have = (size_t)(out - out0);
    if (unlikely(dist > have)) {
      if (dist > have + st->ext) {
        why = FAST_TOO_FAR;
        break;
      }
      TAKE(d);
      e = lt[bb & lmask];
      FILL();
      size_t back = dist - have;
      if (back < len)
        out = far(out, out0, hist, dist, len);
      else
        out = wide(out, hist - back, len, 0);
      continue;
    }

    /* The next symbol's lookup goes before the copy, so that its latency
       overlaps the stores. */
    TAKE(d);
    e = lt[bb & lmask];
    FILL();
    out = wide(out, out - dist, len, dist);
  } while (in < in_last && out < out_last);

  c->in = in;
  c->out = out;
  bl &= 63;
  c->bitbuf = low(bb, bl);
  c->bitsleft = bl;
  return why;
}

#undef FILL
#undef TAKE
#undef VAL
#undef SUB
#undef ROOT_DIST_AND_NEXT
#undef DIST_AND_NEXT
#undef FAST
#undef FAST_TARGET
#undef EXTRACT
