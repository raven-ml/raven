/* TIFF's LZW, whole-buffer decode only, with lz4.h's conventions: the
   decoder keeps its table on its own stack (40 KiB), reads [src] up to
   the byte that holds EndOfInformation's last bit, sets [consumed] there,
   and returns LZ_DONE or LZ_ERROR, with LZ_TOO_LONG when the output
   exceeds [dst_cap]. */

#ifndef LZ_LZW_H
#define LZ_LZW_H

#include "lz.h"

enum lz_status lz_lzw_decode(struct lz_io *io, struct lz_error *err);

#endif
