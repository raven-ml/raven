/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The LZ4 block decoder, after the block format description:
  https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
  and XXH32, which frames use for their checksums:
  https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
  Every sequence is checked against both ends before it is copied.
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdint.h>
#include <string.h>

enum {
  OK,
  TRUNCATED,
  DATA_TOO_LONG,
  OFFSET_ZERO,
  OFFSET_BEYOND
};

static inline void copy8(uint8_t *d, const uint8_t *s) {
  uint64_t v;
  memcpy(&v, s, 8);
  memcpy(d, &v, 8);
}

/* Copies [n] bytes from [offset] back. Wide copies may write up to 15 bytes
   past the match, which [room] holds and are not yet produced. */
static inline void copy_match(uint8_t *d, size_t offset, size_t n,
                              size_t room) {
  const uint8_t *s = d - offset;
  if (n + 16 > room) {
    for (size_t k = 0; k < n; k++)
      d[k] = s[k];
    return;
  }
  if (offset < 8) {
    /* The data repeats every [offset] bytes, hence every [step >= 8]. */
    size_t step = offset;
    while (step < 8)
      step += offset;
    for (size_t k = 0; k < step; k++)
      d[k] = s[k];
    if (n <= step)
      return;
    s = d;
    d += step;
    n -= step;
  }
  for (size_t k = 0; k < n; k += 8)
    copy8(d + k, s + k);
}

/* Decodes the block [src[*ip, n)] into [dst[*op, m)]; matches reach back to
   [hist]. A block ends with a sequence of literals alone. On return, [*op] is
   the end of the data written and, on an error, [*ip] the offset where
   decoding failed. */
static int decode(const uint8_t *src, size_t *ip_, size_t n, uint8_t *dst,
                  size_t hist, size_t *op_, size_t m) {
  size_t ip = *ip_, op = *op_;
  int status = TRUNCATED;
  *ip_ = n;
  while (ip < n) {
    size_t sequence = ip;
    unsigned token = src[ip++];
    size_t len = token >> 4;
    if (len == 15) {
      uint8_t b;
      do {
        if (ip == n)
          goto done;
        b = src[ip++];
        len += b;
      } while (b == 255);
    }
    if (len > n - ip)
      goto done;
    if (len > m - op) {
      *ip_ = sequence;
      status = DATA_TOO_LONG;
      goto done;
    }
    if (len <= 16 && n - ip >= 16 && m - op >= 16) {
      copy8(dst + op, src + ip);
      copy8(dst + op + 8, src + ip + 8);
    } else {
      memcpy(dst + op, src + ip, len);
    }
    ip += len;
    op += len;
    if (ip == n) {
      status = OK;
      goto done;
    }
    if (n - ip < 2)
      goto done;
    size_t offset = src[ip] | ((size_t)src[ip + 1] << 8);
    ip += 2;
    if (offset == 0 || offset > op - hist) {
      *ip_ = ip - 2;
      status = offset == 0 ? OFFSET_ZERO : OFFSET_BEYOND;
      goto done;
    }
    len = (token & 15) + 4;
    if ((token & 15) == 15) {
      uint8_t b;
      do {
        if (ip == n)
          goto done;
        b = src[ip++];
        len += b;
      } while (b == 255);
    }
    if (len > m - op) {
      *ip_ = sequence;
      status = DATA_TOO_LONG;
      goto done;
    }
    copy_match(dst + op, offset, len, m - op);
    op += len;
  }
done:
  *op_ = op;
  return status;
}

/* XXH32 */

#define P1 2654435761u
#define P2 2246822519u
#define P3 3266489917u
#define P4 668265263u
#define P5 374761393u

static inline uint32_t rotl(uint32_t x, unsigned r) {
  return (x << r) | (x >> (32 - r));
}

static inline uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline uint32_t round32(uint32_t acc, uint32_t lane) {
  return rotl(acc + lane * P2, 13) * P1;
}

static uint32_t xxh32(const uint8_t *p, size_t len, uint32_t seed) {
  const uint8_t *end = p + len;
  uint32_t h;
  if (len >= 16) {
    uint32_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
    for (; end - p >= 16; p += 16) {
      v1 = round32(v1, le32(p));
      v2 = round32(v2, le32(p + 4));
      v3 = round32(v3, le32(p + 8));
      v4 = round32(v4, le32(p + 12));
    }
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
  } else {
    h = seed + P5;
  }
  h += (uint32_t)len;
  for (; end - p >= 4; p += 4)
    h = rotl(h + le32(p) * P3, 17) * P4;
  for (; p < end; p++)
    h = rotl(h + *p * P5, 11) * P1;
  h ^= h >> 15;
  h *= P2;
  h ^= h >> 13;
  h *= P3;
  h ^= h >> 16;
  return h;
}

/* OCaml entry points. Positions travel in an int array,
   [| src_pos; src_end; dst_hist; dst_pos; dst_end |]; [Compress_lz4] checks
   them. */

#define RELEASE_THRESHOLD 65536

CAMLprim value caml_compress_lz4_overlap(value a, value b) {
  uintptr_t pa = (uintptr_t)Caml_ba_data_val(a);
  uintptr_t pb = (uintptr_t)Caml_ba_data_val(b);
  size_t la = Caml_ba_array_val(a)->dim[0], lb = Caml_ba_array_val(b)->dim[0];
  return Val_bool(la != 0 && lb != 0 && pa < pb + lb && pb < pa + la);
}

CAMLprim value caml_compress_lz4_block(value vsrc, value vdst, value vio) {
  CAMLparam3(vsrc, vdst, vio);
  const uint8_t *src = Caml_ba_data_val(vsrc);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t ip = Long_val(Field(vio, 0)), n = Long_val(Field(vio, 1));
  size_t hist = Long_val(Field(vio, 2));
  size_t op = Long_val(Field(vio, 3)), m = Long_val(Field(vio, 4));
  int status;
  if ((n - ip) + (m - op) > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    status = decode(src, &ip, n, dst, hist, &op, m);
    caml_acquire_runtime_system();
  } else {
    status = decode(src, &ip, n, dst, hist, &op, m);
  }
  Field(vio, 0) = Val_long(ip);
  Field(vio, 3) = Val_long(op);
  CAMLreturn(Val_int(status));
}

CAMLprim value caml_compress_lz4_xxh32(value vb, value voff, value vlen) {
  CAMLparam1(vb);
  const uint8_t *p = (const uint8_t *)Caml_ba_data_val(vb) + Long_val(voff);
  size_t len = Long_val(vlen);
  uint32_t h;
  if (len > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    h = xxh32(p, len, 0);
    caml_acquire_runtime_system();
  } else {
    h = xxh32(p, len, 0);
  }
  CAMLreturn(Val_long(h));
}
