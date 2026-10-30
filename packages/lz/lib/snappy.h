/* Snappy blocks, whole-buffer only, with lz4.h's conventions. The encoder
   takes at most 2^32 - 1 bytes, the largest length a preamble states (the
   caller checks it), allocates its tables with [lz_alloc], and returns
   LZ_ERROR when that refuses. */

#ifndef LZ_SNAPPY_H
#define LZ_SNAPPY_H

#include "lz.h"

enum lz_status lz_snappy_decode(struct lz_io *io, struct lz_error *err);
enum lz_status lz_snappy_encode(struct lz_io *io);
size_t lz_snappy_bound(size_t n);

#endif
