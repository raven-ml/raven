/* LZ4 blocks and frames, whole-buffer only: each entry reads all of [src].
   A decoder keeps its state on its own stack and returns LZ_DONE or
   LZ_ERROR, with LZ_TOO_LONG when the output exceeds [dst_cap]. An encoder
   allocates its hash table with lz_alloc and frees it before returning
   LZ_DONE, LZ_NEED_OUTPUT when the output does not fit, or LZ_ERROR when
   the allocation is refused. */

#ifndef LZ_LZ4_H
#define LZ_LZ4_H

#include "lz.h"

enum lz_status lz_lz4_block_decode(struct lz_io *io, struct lz_error *err);
enum lz_status lz_lz4_block_encode(struct lz_io *io);
size_t lz_lz4_block_bound(size_t n);

/* Reads the frames of [src], skippable frames included, to its end: a
   byte that begins no frame is LZ_MALFORMED at its offset, a cut magic
   number LZ_TRUNCATED. [consumed] is [src_len]. */
enum lz_status lz_lz4_frame_decode(struct lz_io *io, struct lz_error *err);
enum lz_status lz_lz4_frame_encode(struct lz_io *io);
size_t lz_lz4_frame_bound(size_t n);

#endif
