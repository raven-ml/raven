/* LZF blocks, whole-buffer decode only, with lz4.h's conventions: the
   decoder reads all of [src], keeps no state beyond its locals, and
   returns LZ_DONE or LZ_ERROR, with LZ_TOO_LONG when the output exceeds
   [dst_cap]. [consumed] is [src_len]. */

#ifndef LZ_LZF_H
#define LZ_LZF_H

#include "lz.h"

enum lz_status lz_lzf_decode(struct lz_io *io, struct lz_error *err);

#endif
