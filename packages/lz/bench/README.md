# lz benchmarks

`bench_lz.exe` times lz's codecs on three 8 MiB corpora that it builds
when it starts, as the tests build them (`Lz_support.corpus`): `text`,
words of made-up syllables, which compresses two- to threefold with no
long-range repeats; `columns`, int64 keys with small gaps and float64
prices of two decimals, as a Parquet page holds them; and `random`,
incompressible bytes. Its decode rows read lz's own encodings of each
corpus, encoded before the row runs; its encode and checksum rows read the
raw bytes. Two last decode rows read lz's LZ4 block of 8 MiB of zeros,
one long match, into a destination of exactly its size and into one with
room past its output. Run it from the tree's root on a quiet machine:

    dune build @packages/lz/bench/bench

or run `bench_lz.exe` from this directory (`list` prints the rows,
`bless` records them in `lz.thumper`).

## lz against the vendors

Whole-buffer encode and decode of Silesia (its twelve files concatenated,
211,938,580 bytes) by lz and by the vendors, one thread each. MB/s counts
uncompressed bytes. Each row is the faster of two runs of
`compare/vendor_table.sh`, each the fastest of 10 calls for a decoder, 3
for an encoder and 50 for CRC-32, after an untimed call that faults the
destination in and checks the output. Both sides write into one
destination allocated for the run, sized to the format's bound. Both
sides' decoders read the same files, the vendors' encodings of each
corpus: libdeflate level 6 for deflate, zlib levels 6 and 9 for zlib and
gzip, zstd levels 1, 3, 12 and 19, lz4's block and default frame, and
snappy.

- lz: `Lz.<Format>.encode` and `decode` called from OCaml
  (`compare/vendor_table.ml`).
- Vendors, through their C APIs (`compare/vendor_table.c`): libdeflate
  1.26 (CMake Release) for deflate, zlib and gzip; zstd 1.5.7 (its `lib/`
  at the Makefile's `-O3`, single-threaded); lz4 1.10.0 (its Makefile,
  `-O3`); Google's snappy 1.2.2 (CMake Release, `-DSNAPPY_REQUIRE_AVX2=ON`);
  zlib 1.3.1 (Debian's) for reference. Zstandard frames carry the XXH64
  content checksum and content size, as lz's do. LZ4 frames have
  independent blocks of up to 4 MiB, the content size and checksum, as
  lz's do.
- fudge: Intel Core Ultra 5 235, Debian 13, gcc 14.2, OCaml 5.5.2, cores
  0-5, 2026-10-30. Load 0.78 before and 1.06 after the first run, 1.16
  and 1.07 the second.

`text`, `columns` and `random` are lz / vendor on the bench's corpora,
from the same runs.

Re-run, from the tree's root, with the vendors built under the directory
`VENDORS` names (default `~/wt/lz-vendors`). Fetch Silesia once into the
output directory; the script writes the other corpora and every encoding
there:

    mkdir -p OUT
    curl -fLo OUT/silesia.zip https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip
    dune build packages/lz/bench/compare/vendor_table.exe
    packages/lz/bench/compare/vendor_table.sh OUT silesia text columns random

| row | lz MB/s | vendor MB/s | lz / vendor | lz bytes | vendor bytes | zlib MB/s | text | columns | random |
|---|---|---|---|---|---|---|---|---|---|
| decode deflate | 1383 | 1468 | 0.94 |  |  | 519 | 0.98 | 0.98 | 1.00 |
| decode zlib | 1406 | 1422 | 0.99 |  |  | 463 | 0.98 | 0.99 | 1.26 |
| decode gzip | 1416 | 1442 | 0.98 |  |  | 488 | 0.98 | 0.99 | 1.31 |
| decode zstd 1 | 1951 | 1976 | 0.99 |  |  |  | 1.05 | 0.96 | 1.01 |
| decode zstd 3 | 1826 | 1813 | 1.01 |  |  |  | 1.05 | 1.05 | 1.02 |
| decode zstd 12 | 2007 | 2015 | 1.00 |  |  |  | 1.04 | 1.09 | 1.01 |
| decode zstd 19 | 1771 | 1791 | 0.99 |  |  |  | 1.01 | 1.09 | 1.01 |
| decode lz4 block | 4696 | 5393 | 0.87 |  |  |  | 0.97 | 1.00 | 0.98 |
| decode lz4 frame | 4698 | 5377 | 0.87 |  |  |  | 0.97 | 1.00 | 1.00 |
| decode snappy | 2234 | 2507 | 0.89 |  |  |  | 0.73 | 0.91 | 1.00 |
| encode deflate 1 | 331 | 343 | 0.97 | 73504840 | 73504303 | 160 | 1.01 | 0.98 | 0.96 |
| encode deflate 2 | 261 | 258 | 1.01 | 71069707 | 71069287 | 146 | 1.05 | 0.99 | 0.98 |
| encode deflate 3 | 241 | 236 | 1.02 | 70169052 | 70168643 | 115 | 1.09 | 1.05 | 0.97 |
| encode deflate 4 | 229 | 220 | 1.04 | 69472212 | 69471858 | 108 | 1.16 | 1.08 | 0.97 |
| encode deflate 5 | 193 | 184 | 1.05 | 68169690 | 68169406 | 79 | 1.20 | 1.05 | 1.01 |
| encode deflate 6 | 164 | 151 | 1.09 | 67511960 | 67511681 | 53 | 1.28 | 1.12 | 0.99 |
| encode deflate 7 | 125 | 112 | 1.12 | 67141525 | 67141220 | 44 | 1.25 | 1.17 | 1.04 |
| encode deflate 8 | 78 | 70 | 1.11 | 66766136 | 66765833 | 28 | 1.19 | 1.18 | 0.99 |
| encode deflate 9 | 63 | 56 | 1.12 | 66714790 | 66714506 | 21 | 1.16 | 1.14 | 0.99 |
| encode zlib 1 | 327 | 339 | 0.96 | 73504846 | 73504309 |  | 1.01 | 0.98 | 0.96 |
| encode zlib 2 | 257 | 256 | 1.00 | 71069713 | 71069293 |  | 1.04 | 1.00 | 0.97 |
| encode zlib 3 | 239 | 234 | 1.02 | 70169058 | 70168649 |  | 1.08 | 1.04 | 0.97 |
| encode zlib 4 | 226 | 218 | 1.04 | 69472218 | 69471864 |  | 1.10 | 1.08 | 0.97 |
| encode zlib 5 | 193 | 183 | 1.05 | 68169696 | 68169412 |  | 1.21 | 1.05 | 1.00 |
| encode zlib 6 | 163 | 150 | 1.09 | 67511966 | 67511687 |  | 1.13 | 1.13 | 1.01 |
| encode zlib 7 | 125 | 111 | 1.13 | 67141531 | 67141226 |  | 1.23 | 1.16 | 1.04 |
| encode zlib 8 | 78 | 69 | 1.13 | 66766142 | 66765839 |  | 1.16 | 1.18 | 1.00 |
| encode zlib 9 | 63 | 56 | 1.12 | 66714796 | 66714512 |  | 1.16 | 1.14 | 1.00 |
| encode gzip 1 | 327 | 339 | 0.96 | 73504858 | 73504321 |  | 1.00 | 0.98 | 0.95 |
| encode gzip 2 | 258 | 256 | 1.01 | 71069725 | 71069305 |  | 1.05 | 1.00 | 0.97 |
| encode gzip 3 | 239 | 234 | 1.02 | 70169070 | 70168661 |  | 1.12 | 1.04 | 0.97 |
| encode gzip 4 | 227 | 218 | 1.04 | 69472230 | 69471876 |  | 1.16 | 1.08 | 0.97 |
| encode gzip 5 | 192 | 183 | 1.05 | 68169708 | 68169424 |  | 1.21 | 1.06 | 1.00 |
| encode gzip 6 | 164 | 150 | 1.09 | 67511978 | 67511699 |  | 1.16 | 1.12 | 1.01 |
| encode gzip 7 | 125 | 111 | 1.13 | 67141543 | 67141238 |  | 1.23 | 1.16 | 1.03 |
| encode gzip 8 | 78 | 69 | 1.13 | 66766154 | 66765851 |  | 1.16 | 1.18 | 0.99 |
| encode gzip 9 | 63 | 56 | 1.12 | 66714808 | 66714524 |  | 1.16 | 1.14 | 0.99 |
| encode zstd 1 | 686 | 671 | 1.02 | 73180821 | 73213483 |  | 1.05 | 0.99 | 1.12 |
| encode zstd 2 | 583 | 565 | 1.03 | 69310028 | 69316336 |  | 1.04 | 1.02 | 1.09 |
| encode zstd 3 | 451 | 421 | 1.07 | 66127023 | 66121945 |  | 1.04 | 0.97 | 0.91 |
| encode zstd 4 | 380 | 381 | 1.00 | 64948038 | 64943432 |  | 0.97 | 0.90 | 1.05 |
| encode zstd 5 | 208 | 216 | 0.96 | 62722603 | 62742339 |  | 0.95 | 0.92 | 0.87 |
| encode zstd 6 | 154 | 158 | 0.97 | 61229609 | 61230928 |  | 0.96 | 0.91 | 0.89 |
| encode zstd 7 | 137 | 137 | 1.00 | 60247036 | 60250556 |  | 1.03 | 0.96 | 0.83 |
| encode zstd 8 | 109 | 107 | 1.02 | 59628409 | 59699954 |  | 1.07 | 0.99 | 0.89 |
| encode zstd 9 | 105 | 102 | 1.03 | 59006081 | 59073807 |  | 1.09 | 0.99 | 0.93 |
| encode zstd 10 | 78 | 74 | 1.05 | 58312305 | 58381179 |  | 1.07 | 1.10 | 0.88 |
| encode zstd 11 | 60 | 58 | 1.03 | 57901361 | 57967002 |  | 1.07 | 1.12 | 1.02 |
| encode zstd 12 | 53 | 52 | 1.02 | 57845797 | 57911235 |  | 1.04 | 1.03 | 0.30 |
| encode lz4 block | 753 | 878 | 0.86 | 96796920 | 100880026 |  | 0.89 | 0.73 | 0.95 |
| encode lz4 frame | 694 | 726 | 0.96 | 96854228 | 100924488 |  | 0.97 | 0.85 | 1.90 |
| encode snappy | 808 | 872 | 0.93 | 101413154 | 101503311 |  | 0.88 | 0.97 | 0.99 |
| crc32 | 30756 | 30077 | 1.02 |  |  | 6141 | 0.98 | 1.00 | 1.00 |

Google's snappy built without ISA flags decodes Silesia at 2115 MB/s and
encodes at 875 MB/s, writing lz's bytes; the AVX2 build hashes with
CRC32C, so its bytes differ.

Where lz is under its vendor on Silesia:

- decode deflate, 5.8 %; zlib, 1.1 %; gzip, 1.8 %: about half of the
  raw stream's gap is building the decode tables, the rest the fast
  loop, which follows libdeflate's structure; no change tried to its
  copies or refills closed it.
- decode zstd 1, 1.3 %; 12, 0.4 %; 19, 1.1 %: cause not known.
- decode lz4 block, 12.9 %; frame, 12.6 %: Silesia's short matches.
  lz4's 8+8+2-byte copy for matches under 19 bytes gained 3 % here and
  lost 22 % on `columns`, where lz is at parity; 32-byte copies for long
  matches and literals lost on every corpus.
- decode snappy, 10.9 % under the AVX2 build, 5.6 % over the portable
  one: the AVX2 build's 32-byte copies, pattern shuffles and deferred
  copies. Compiling lz with `-mavx2 -mbmi2` changes nothing, so the lead
  is in that code, not the instruction set.
- encode deflate, zlib and gzip 1, 3.5 %: cause not known.
- encode zstd 4, 0.3 %: double fast keeps its hash and chain logs
  spilled, and shifts by a variable without BMI2.
- encode zstd 5, 3.7 %: the row search's candidate loop.
- encode zstd 6, 2.5 %: cause not known; it searches rows as level 5
  does.
- encode lz4 block, 14.2 %; frame, 4.4 %: code generation. Both encoders
  find the same matches, and clang 19 compiles the same kernel 12 %
  faster than gcc 14; writing the literals before counting the match,
  branch hints and gcc's loop options changed nothing. lz writes 4.0 %
  fewer bytes.
- encode snappy, 7.3 % (7.7 % under the portable build, same bytes):
  cause not known.

Off Silesia, Zstandard level 12 encodes `random` at 0.30 of zstd's speed
(440 against 1488 MB/s; level 11 at 1.03, Silesia's level 12 at 1.02;
re-timed on 2026-10-29, load 1.23 before and 1.18 after). Level 12's
match tables take 40 MiB, past the 32 MiB above which glibc maps every
allocation afresh: each `encode` faults in 10,240 pages (11.5 ms) and
unmaps them (1.9 ms) around 5.9 ms of compression. Level 11's 20 MiB
come back from the heap. The vendor keeps one context across calls;
given a fresh one per call, zstd encodes `random` at 388 MB/s. Mapping
the tables populated takes the encode to 608 MB/s; parity needs tables
kept across calls, where `encode` allocates them for the call.
