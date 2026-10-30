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
  0-5, 2026-10-29. Load 1.33 before and 1.28 after the first run, 1.32
  and 1.15 the second. The decoders then read files written through
  the vendors' Python bindings (cramjam's Rust codec for snappy), from
  the same raw corpora.

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
| decode deflate | 1387 | 1469 | 0.94 |  |  | 518 | 0.98 | 0.98 | 1.00 |
| decode zlib | 1409 | 1428 | 0.99 |  |  | 464 | 0.98 | 1.00 | 1.27 |
| decode gzip | 1414 | 1442 | 0.98 |  |  | 485 | 0.98 | 0.98 | 1.31 |
| decode zstd 1 | 1951 | 1975 | 0.99 |  |  |  | 1.05 | 0.96 | 1.01 |
| decode zstd 3 | 1820 | 1817 | 1.00 |  |  |  | 1.05 | 1.05 | 1.01 |
| decode zstd 12 | 2001 | 2009 | 1.00 |  |  |  | 1.05 | 1.09 | 1.01 |
| decode zstd 19 | 1779 | 1806 | 0.99 |  |  |  | 1.02 | 1.09 | 1.01 |
| decode lz4 block | 4725 | 5397 | 0.88 |  |  |  | 0.93 | 1.01 | 0.98 |
| decode lz4 frame | 4691 | 5350 | 0.88 |  |  |  | 0.93 | 1.00 | 1.00 |
| decode snappy | 2225 | 2442 | 0.91 |  |  |  | 0.75 | 0.91 | 1.00 |
| encode deflate 1 | 325 | 343 | 0.95 | 73504303 | 73504303 | 160 | 1.00 | 0.93 | 0.92 |
| encode deflate 2 | 254 | 258 | 0.98 | 71069286 | 71069287 | 146 | 1.01 | 0.96 | 0.97 |
| encode deflate 3 | 232 | 237 | 0.98 | 70168643 | 70168643 | 115 | 1.03 | 0.98 | 0.97 |
| encode deflate 4 | 217 | 220 | 0.99 | 69471858 | 69471858 | 107 | 1.06 | 0.99 | 0.97 |
| encode deflate 5 | 188 | 184 | 1.02 | 68169406 | 68169406 | 79 | 1.01 | 0.97 | 0.98 |
| encode deflate 6 | 156 | 151 | 1.03 | 67511681 | 67511681 | 53 | 1.05 | 0.99 | 0.98 |
| encode deflate 7 | 117 | 111 | 1.05 | 67141220 | 67141220 | 44 | 1.00 | 1.08 | 0.99 |
| encode deflate 8 | 73 | 70 | 1.04 | 66765833 | 66765833 | 28 | 1.06 | 1.10 | 0.99 |
| encode deflate 9 | 60 | 56 | 1.07 | 66714506 | 66714506 | 21 | 1.03 | 1.10 | 0.99 |
| encode zlib 1 | 322 | 339 | 0.95 | 73504309 | 73504309 |  | 1.00 | 0.93 | 0.92 |
| encode zlib 2 | 252 | 256 | 0.98 | 71069292 | 71069293 |  | 1.01 | 0.97 | 0.97 |
| encode zlib 3 | 231 | 234 | 0.99 | 70168649 | 70168649 |  | 1.05 | 0.98 | 0.97 |
| encode zlib 4 | 215 | 218 | 0.99 | 69471864 | 69471864 |  | 1.01 | 0.98 | 0.97 |
| encode zlib 5 | 187 | 183 | 1.02 | 68169412 | 68169412 |  | 1.08 | 0.97 | 0.98 |
| encode zlib 6 | 155 | 150 | 1.03 | 67511687 | 67511687 |  | 1.09 | 0.99 | 0.98 |
| encode zlib 7 | 117 | 111 | 1.05 | 67141226 | 67141226 |  | 1.08 | 1.08 | 0.99 |
| encode zlib 8 | 73 | 69 | 1.06 | 66765839 | 66765839 |  | 1.03 | 1.13 | 0.99 |
| encode zlib 9 | 60 | 56 | 1.07 | 66714512 | 66714512 |  | 1.03 | 1.10 | 0.99 |
| encode gzip 1 | 322 | 339 | 0.95 | 73504321 | 73504321 |  | 1.00 | 0.93 | 0.91 |
| encode gzip 2 | 252 | 256 | 0.98 | 71069304 | 71069305 |  | 1.00 | 0.96 | 0.98 |
| encode gzip 3 | 231 | 234 | 0.99 | 70168661 | 70168661 |  | 1.03 | 0.98 | 0.98 |
| encode gzip 4 | 215 | 218 | 0.99 | 69471876 | 69471876 |  | 1.06 | 0.98 | 0.97 |
| encode gzip 5 | 187 | 183 | 1.02 | 68169424 | 68169424 |  | 1.07 | 0.97 | 0.97 |
| encode gzip 6 | 155 | 150 | 1.03 | 67511699 | 67511699 |  | 1.11 | 0.99 | 0.97 |
| encode gzip 7 | 117 | 111 | 1.05 | 67141238 | 67141238 |  | 1.08 | 1.08 | 0.99 |
| encode gzip 8 | 73 | 69 | 1.06 | 66765851 | 66765851 |  | 1.06 | 1.13 | 0.99 |
| encode gzip 9 | 60 | 56 | 1.07 | 66714524 | 66714524 |  | 0.97 | 1.10 | 0.99 |
| encode zstd 1 | 691 | 670 | 1.03 | 73219178 | 73213483 |  | 1.02 | 1.08 | 1.03 |
| encode zstd 2 | 580 | 565 | 1.03 | 69316795 | 69316336 |  | 1.07 | 1.07 | 1.11 |
| encode zstd 3 | 417 | 422 | 0.99 | 66121367 | 66121945 |  | 0.99 | 0.97 | 1.07 |
| encode zstd 4 | 374 | 386 | 0.97 | 64942289 | 64943432 |  | 0.97 | 0.91 | 1.03 |
| encode zstd 5 | 202 | 217 | 0.93 | 62741383 | 62742339 |  | 0.93 | 0.95 | 0.94 |
| encode zstd 6 | 151 | 158 | 0.96 | 61230200 | 61230928 |  | 0.96 | 0.95 | 0.99 |
| encode zstd 7 | 133 | 137 | 0.97 | 60249571 | 60250556 |  | 1.00 | 0.98 | 0.94 |
| encode zstd 8 | 106 | 108 | 0.98 | 59698482 | 59699954 |  | 1.01 | 0.98 | 0.92 |
| encode zstd 9 | 102 | 103 | 0.99 | 59072133 | 59073807 |  | 1.02 | 0.99 | 0.96 |
| encode zstd 10 | 75 | 74 | 1.01 | 58379525 | 58381179 |  | 1.00 | 1.03 | 0.99 |
| encode zstd 11 | 59 | 58 | 1.02 | 57965595 | 57967002 |  | 1.03 | 1.05 | 1.01 |
| encode zstd 12 | 52 | 52 | 1.00 | 57909916 | 57911235 |  | 1.00 | 0.97 | 0.29 |
| encode lz4 block | 759 | 877 | 0.87 | 96796920 | 100880026 |  | 0.88 | 0.73 | 0.91 |
| encode lz4 frame | 698 | 725 | 0.96 | 96854228 | 100924488 |  | 0.97 | 0.85 | 1.93 |
| encode snappy | 808 | 873 | 0.93 | 101413154 | 101503311 |  | 0.88 | 0.97 | 0.99 |
| crc32 | 30627 | 30151 | 1.02 |  |  | 6133 | 1.00 | 1.00 | 0.99 |

Google's snappy built without ISA flags decodes Silesia at 2081 MB/s and
encodes at 876 MB/s, writing lz's bytes; the AVX2 build hashes with
CRC32C, so its bytes differ.

Where lz is under its vendor on Silesia:

- decode deflate, 5.6 %; zlib, 1.3 %; gzip, 1.9 %: about half of the
  raw stream's gap is building the decode tables, the rest the fast
  loop, which follows libdeflate's structure; no change tried to its
  copies or refills closed it.
- decode zstd 1, 1.2 %; 12, 0.4 %; 19, 1.5 %: cause not known.
- decode lz4 block, 12.5 %; frame, 12.3 %: Silesia's short matches.
  lz4's 8+8+2-byte copy for matches under 19 bytes gained 3 % here and
  lost 22 % on `columns`, where lz is at parity; 32-byte copies for long
  matches and literals lost on every corpus.
- decode snappy, 8.9 % under the AVX2 build, 7 % over the portable one:
  the AVX2 build's 32-byte copies, pattern shuffles and deferred copies.
  Compiling lz with `-mavx2 -mbmi2` changes nothing, so the lead is in
  that code, not the instruction set.
- encode deflate, zlib and gzip 1, 5.0-5.2 %: cause not known.
- encode deflate, zlib and gzip 2-4, 1.3-2.1 %: cause not known.
- encode zstd 3, 1.2 %; 4, 3.1 %: double fast keeps its hash and chain
  logs spilled, and shifts by a variable without BMI2.
- encode zstd 5, 6.9 %: the row search's candidate loop.
- encode zstd 6, 4.4 %; 7, 2.9 %; 8, 1.9 %; 9, 1.0 %: cause not known;
  they search rows as level 5 does.
- encode lz4 block, 13.5 %; frame, 3.7 %: code generation. Both encoders
  find the same matches, and clang 19 compiles the same kernel 12 %
  faster than gcc 14; writing the literals before counting the match,
  branch hints and gcc's loop options changed nothing. lz writes 4.0 % fewer bytes.
- encode snappy, 7.4 % (7.8 % under the portable build, same bytes):
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
