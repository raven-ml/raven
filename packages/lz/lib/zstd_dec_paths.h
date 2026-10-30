/* The builds of the Zstandard decoder's hot loops (the Huffman lanes and
   the sequences) that this host can run, for the tests that hold them
   equal: the generic build first, then the BMI2 build where the CPU has
   BMI1, BMI2 and LZCNT. A decoder takes the last one at its init; both
   decode the same bytes to the same output and the same errors. */

#ifndef LZ_ZSTD_DEC_PATHS_H
#define LZ_ZSTD_DEC_PATHS_H

#include "zstd.h"

/* [lz_zstd_dec_paths names] sets [*names] to the paths' names and is
   their number, at least 1. */
size_t lz_zstd_dec_paths(const char *const **names);

/* [lz_zstd_dec_path st i] makes [st] decode with path [i], which is below
   their number. */
void lz_zstd_dec_path(struct lz_zstd_dec *st, size_t i);

#endif
