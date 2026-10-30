## Snappy

Fudge (Intel Core Ultra 5 235), single thread, P-cores 0-5, under
`flock ~/wt/TIMING.lock`. Google's snappy 1.2.2 is built from source under
`~/wt/lz-vendors/snappy-1.2.2/` twice: `build-default` (CMake Release,
no ISA flags: no SSSE3, multiplicative hash) and `build-avx2`
(`-DSNAPPY_REQUIRE_AVX2=ON`: SSSE3 pattern shuffles, 32-byte AVX copies,
BMI2, and the CRC32C hash, so its bytes differ). It is timed by
`packages/lz/test/gen/snappy_ref.cc` (`snappy_ref bench RAW COMPRESSED RUNS`,
best of RUNS, MB/s of the raw size), never linked into lz; lz's kernels by
`bench/measure/snappy/whole.c`, the same command line over the kernels
dune compiles. Every decoder decodes the same file,
`data/<corpus>.snappy` (written by python-snappy). One hold runs them all,
from the tree's root:

    dune build packages/lz/lib/liblz_stubs.a
    packages/lz/bench/measure/snappy/compare.sh OUT

lz's rows, `bench_lz.exe check -f snappy` from `packages/lz/bench`, load 8.5
before the run (other sessions' builds on the E-cores), at the first
commit of the kernels; the encoder was rewritten since, at the speed the
table below shows:

| row | time | MB/s | words |
|---|---|---|---|
| text/decode-snappy | 4.38 ms | 1915 | 5 |
| text/encode-snappy | 12.58 ms | 667 | 2 |
| columns/decode-snappy | 3.86 ms | 2173 | 5 |
| columns/encode-snappy | 8.01 ms | 1047 | 2 |
| random/decode-snappy | 367 us | 22800 | 5 |
| random/encode-snappy | 452 us | 18600 | 2 |

Against Google, `compare.sh`, load 5.9 before and 5.8 after, the host 76 %
idle during the run (another session's bench on other cores); MB/s, 10
runs (Silesia 3):

| corpus | encode lz | Google | Google AVX2 | decode lz | Google | Google AVX2 |
|---|---|---|---|---|---|---|
| text | 669 | 758 | 762 | 1904 | 2001 | 2579 |
| columns | 1109 | 1174 | 1145 | 2305 | 1701 | 2469 |
| random | 19729 | 19666 | 19746 | 22973 | 22877 | 22906 |
| silesia | 803 | 866 | 854 | 2182 | 2054 | 2437 |

Ratios: text 2.014, columns 1.776, Silesia 2.090, the same as Google's
portable build (byte-identical output); the AVX2 build's differ slightly
(2.009, 1.775, 2.088). The encoder runs at 0.88-0.94 of Google's; the
decoder at 0.95-1.36 of the portable build and 0.74-0.93 of the AVX2
build. Compiling lz's kernel with `-mavx2 -mbmi2` changed neither number
(within 2 %), so the AVX2 build's lead is its code (32-byte copies,
PSHUFB pattern expansion, the deferred-copy loop), not the instruction set
alone.

### 11. Decode with and without the 16-byte pattern expansion

`expand` in `lib/snappy.c` writes a copy whose offset is under 16 as one
16-byte pattern and then 16-byte loads from a whole number of patterns
back. Without it, such a copy is written byte by byte (`compare.sh`
builds that decoder from `lib/snappy.c` with one line replaced). Same
hold as above, MB/s:

| corpus | with | without | gain |
|---|---|---|---|
| text | 1904 | 1915 | -0.6 % |
| columns | 2305 | 2307 | -0.1 % |
| silesia | 2182 | 2088 | 4.5 % |

Two earlier holds, at loads 7.7 and 3.8, gave text 1890 against 1891 and
1822 against 1816, columns 2293 against 2175 and 2236 against 2163,
Silesia 2091 against 2002 and 2089 against 2002. Text, words of
syllables, has almost no copy under 16 bytes back, and the expansion is
noise there; on Silesia it is worth 4 %, on columns 0 to 5 % by hold.

## Checksums

Fudge (Intel Core Ultra 5 235: PCLMULQDQ, VPCLMULQDQ, AVX2, AVX-VNNI, no
AVX-512), single thread, P-cores 0-5, under `flock ~/wt/TIMING.lock`. lz
takes VPCLMULQDQ for CRC-32 and AVX-VNNI for Adler-32 on this host; XXH32
and XXH64 have one scalar path. The vendors are built from source under
`~/wt/lz-vendors/`: libdeflate 1.26 (CMake Release, `-O2`, its own
dispatch picks VPCLMULQDQ/AVX2 and AVX-VNNI here) and xxHash 0.8.3
(`xxhash.c` at `-O3`). They are timed by
`~/wt/lz-vendors/checksum-bench/bench.c` (`build.sh`, then
`OFFSET=o bench data/random.raw SIZE`), which links lz's checksum units
compiled with lib/dune's flags beside the vendors, checks that each pair
agrees, and times each on the same bytes: the first SIZE bytes of
`data/random.raw` at OFFSET bytes past a 4 KiB boundary, called repeatedly,
best of 9 batches of about 0.2 s. The script of the hold is
`~/wt/lz-logs/checksums/time.sh`; its log `time3.log`.

lz's rows, `bench_lz.exe check checksums` from `packages/lz/bench`, in the same
hold, load 5.7 before and 5.2 after (`top`: other sessions' tests on the
E-cores). The rows' buffers are views of a bigarray, 16-byte aligned:

| row | time | GB/s | words |
|---|---|---|---|
| crc32 of 4 KiB | 34.2 ns | 120 | 3 |
| adler32 of 4 KiB | 38.9 ns | 105 | 3 |
| xxh64 of 4 KiB | 185 ns | 22.2 | 3 |
| crc32 of 64 KiB | 519 ns | 126 | 3 |
| adler32 of 64 KiB | 566 ns | 116 | 3 |
| xxh64 of 64 KiB | 2.88 us | 22.8 | 3 |
| crc32 of 8 MiB | 170 us | 49.4 | 3 |
| adler32 of 8 MiB | 183 us | 46.0 | 3 |
| xxh64 of 8 MiB | 379 us | 22.1 | 3 |

The kernels against the vendors, same hold, GB/s, lz / vendor:

| checksum | size | at a line | 16 bytes past |
|---|---|---|---|
| CRC-32 / libdeflate | 4 KiB | 128.7 / 132.6 | 129.4 / 134.0 |
| | 64 KiB | 133.8 / 134.1 | 131.9 / 121.8 |
| | 8 MiB | 49.0 / 47.1 | 49.2 / 49.3 |
| Adler-32 / libdeflate | 4 KiB | 127.6 / 122.8 | 111.4 / 107.0 |
| | 64 KiB | 121.0 / 112.3 | 119.0 / 84.6 |
| | 8 MiB | 47.4 / 48.3 | 48.1 / 48.4 |
| XXH64 / xxHash | 4 KiB | 23.1 / 23.6 | 23.0 / 24.5 |
| | 64 KiB | 23.7 / 24.9 | 23.7 / 24.9 |
| | 8 MiB | 23.7 / 24.8 | 23.7 / 23.7 |
| XXH32 / xxHash | 4 KiB | 11.7 / 11.3 | 11.6 / 11.6 |
| | 64 KiB | 11.8 / 11.7 | 11.8 / 11.7 |
| | 8 MiB | 11.8 / 11.6 | 11.8 / 11.7 |

At 8 MiB both libraries run at the speed of memory. CRC-32 at 4 KiB is
3 % behind libdeflate, for a cause not measured: its last reduction is a
Barrett step where lz reads the last vector through the table path. XXH64 is 2-5 % behind
xxHash with the same instructions in its loop (eight `imul` per 32-byte
stripe); reordering the loads, unrolling two stripes and a do-while loop
did not close it (a scratch harness at 64 KiB: 23.3 GB/s for each
against xxHash's 24.0).
From 16 KiB lz aligns the vectors of CRC-32 and Adler-32 to 32 bytes;
libdeflate aligns from 64 KiB, so at 64 KiB 16 bytes past a line it
splits half its loads.

The generic paths, 64 KiB, E-cores, load about
10: CRC-32 3.2 GB/s, Adler-32 3.9 GB/s.

## LZ4

lz's LZ4 kernels against lz4 1.10, built from source under
`~/wt/lz-vendors/lz4-1.10.0` (`make lz4`, its default `-O3`), on fudge, one
P-core (`taskset -c 2`), each set under `flock ~/wt/TIMING.lock`. lz4's
numbers are its own benchmark, `lz4 -b1 -i3 FILE.raw`, which cuts the file
into independent blocks of 4 MiB. lz's are a C harness linked with the
kernels dune builds (`_build/default/packages/lz/lib/liblz_stubs.a`), best of
2 s per row, on the same files: the decoders read the Python lz4
package's files (`FILE.lz4b`, one block; `FILE.lz4f`, linked 64 KiB
blocks, no checksum), the encoders write one block of the whole file.

Load average before: 4.1 (lz4), 4.9 (lz; `top`: 89 % idle). MB/s of
uncompressed bytes.

| Corpus | Decode lz | Decode lz4 | Encode lz | Encode lz4 | Ratio lz | Ratio lz4 |
|---|---|---|---|---|---|---|
| text | 3745 | 3998 | 622 | 670 | 1.892 | 1.881 |
| columns | 2969 | 2969 | 875 | 1120 | 1.612 | 1.607 |
| random | 22303 | 22362 | 18183 | 19288 | 0.996 | 0.996 |
| Silesia | 4857 | 5446 | 784 | 873 | 2.190 | 2.101 |

Random's row is from an earlier hold at load 11.8. The bench rows
(`bench_lz.exe check -f lz4`, load 6.9) agree: text decode lz4b 2.37 ms
(3540 MB/s), encode 14.1 ms (594 MB/s); columns decode 3.38 ms ±6.6 %,
encode 10.1 ms (829 MB/s).

lz's own frames carry the content's XXH32, which the decoder verifies in
a second pass over each block's output: decoding them runs at 2785
(text), 2397 (columns) and 3099 (Silesia) MB/s, encoding at 591, 815 and
720.

The encoder trails lz4 by 7-22 %, most on columns, whose short matches
make the per-sequence path the cost. The two encoders find the same
matches (with lz4's 2^12 entries of 32 bits, lz's ratios equal lz4's on
text and columns, 2.105 against 2.101 on Silesia), so the gap is code
generation: clang 19 compiles the same
source 12 % faster than GCC 14 (on an E-core: text 560 against 498 MB/s,
columns 873 against 776), past lz4's own GCC build on text. Writing the
literals before counting the match, branch hints, a checked and an
unchecked copy of the loop, and GCC's loop options each changed nothing.
The decoder trails on Silesia by 11 %; lz4's copy of 8+8+2 bytes for
matches under 19 bytes gained 3 % there and lost 22 % on columns, and
32-byte copies for long matches and literals lost on every corpus.

## Deflate decode

Setup. Every number is from fudge's P-cores (`taskset -c 0-5`), one row
set per hold of `flock ~/wt/TIMING.lock`, single thread, MB/s of
uncompressed bytes, best of runs filling 1 s per row. libdeflate is
1.26 built from source under `~/wt/lz-vendors/libdeflate-1.26` (CMake
Release, `-O2`, its BMI2 path on this CPU), never linked into lz. Both
decoders run through the C programs of `bench/measure/deflate/`, whose
headers say how to build and run them, on the same files: `speed.c`
links lz's kernels as dune builds them (`liblz_stubs.a`, `-O3`) and runs
the stubs' whole-buffer loop or a stepped one; `ldspeed.c` times
libdeflate's `libdeflate_{deflate,zlib,gzip}_decompress` on a buffer of
the known size. The files are `bench/data/*.{deflate6,zlib6,gz9}`:
libdeflate level 6 raw, zlib level 6, zlib level 9 in gzip.

### Where the decoder stands

The committed kernel, both programs built as their headers say, one
hold at load average 7.6 before and 6.8 after (other sessions' builds
on the E-cores), best of two rounds:

| File | lz | libdeflate | lz / libdeflate |
|---|---|---|---|
| silesia.deflate6 | 1408 | 1464 | 0.96 |
| silesia.zlib6 | 1386 | 1417 | 0.98 |
| silesia.gz9 | 1447 | 1434 | 1.01 |
| text.deflate6 | 2253 | 2281 | 0.99 |
| text.zlib6 | 1698 | 1721 | 0.99 |
| text.gz9 | 1992 | 1988 | 1.00 |
| columns.deflate6 | 1340 | 1355 | 0.99 |
| columns.zlib6 | 1361 | 1392 | 0.98 |
| columns.gz9 | 1413 | 1431 | 0.99 |

The remaining gap is 4 % on Silesia's raw stream and at most 2 %
elsewhere. Building the decode tables is about half of Silesia's:
sizing the literal/length root to the longest code took it from 28 to
16 million TSC ticks per pass, about 2 % of the decode, for 2972 blocks.
The rest is in the fast loop, whose root mask is now a register (the
change cost text 1.5 %); the loop is libdeflate's structure, and no
change tried to its copies or refills closed it.

### Earlier holds, before the last three changes

The table below and the steps' tables after it come from holds at load
average 10-14.5 (other sessions' builds on the E-cores; `top` showed
the P-cores held by this run), before the checksum moved to each
block's end and the root to the longest code.

| File | lz | libdeflate | lz / libdeflate |
|---|---|---|---|
| silesia.deflate6 | 1317-1393 | 1401-1462 | 0.94-0.95 |
| silesia.zlib6 | 1393 | 1398 | 1.00 |
| silesia.gz9 | 1376 | 1428 | 0.96 |
| text.deflate6 | 2072-2130 | 2249-2292 | 0.92-0.94 |
| text.zlib6 | 1614 | 1708 | 0.94 |
| text.gz9 | 1756 | 1969 | 0.89 |
| columns.deflate6 | 1240-1311 | 1345-1353 | 0.92-0.97 |
| columns.zlib6 | 1314 | 1381 | 0.95 |
| columns.gz9 | 1315 | 1426 | 0.92 |

Ranges span the holds. libdeflate's own driver,
`programs/benchmark -6 -s 211938580 silesia.raw` (and `-z -C libz`,
`-9 -g -C libz` for zlib's encodings), reported 1047, 1045 and 1074 MB/s
in the same loaded hold, from a single timed pass; the harness's best of
runs is the comparison above.

The bench rows (`bench_lz.exe check zlib6 gz9 deflate6`, load 4.1
before, 5.1 after) read: text zlib6 5.50 ms (1525 MB/s), gz9 4.76 ms,
deflate6 4.23 ms; columns zlib6 6.74 ms, gz9 6.24 ms, deflate6 6.79 ms;
random zlib6 448 us, gz9 431 us. Each whole-buffer row allocates 5 words,
its result.

Compiling the loop a second time for BMI2 gained 15 % on text and 4 %
on Silesia, and `bzhi` for the extra bits another 2-4 %.

Conformance: `zlib_fuzz.py` runs mutated goldens through lz and
Python's zlib. A stream that breaks two rules gets the message of the
rule lz proves first, where zlib, checking code sets only after every
code length, may name the other: the rules and strings are zlib's, the
choice between two faults is not a promise (37 of 3000 mutants, each
with two faults).

### 2. The window copy per step

A streaming decoder copies each step's output, up to 32 KiB, into its
ring. Timed against the same kernel with that copy compiled out (its
output is wrong where a match reaches the ring, so it times the copy
alone), on the harness's stepped loop (each step a destination of the
step's size and that many input bytes):

| File | Step | With copy | Without | Copy's share |
|---|---|---|---|---|
| silesia.deflate6 | 1 MiB | 1376 | 1376 | 0 % |
| silesia.deflate6 | 64 KiB | 1250 | 1234 | under the noise |
| text.deflate6 | 1 MiB | 2026 | 2017 | under the noise |
| text.deflate6 | 64 KiB | 1587 | 1495 | under the noise |

The copy (32 KiB per 64 KiB of output, about 1 us from L2) is below the
run-to-run spread. What 64 KiB steps cost is matches that reach before
the step's destination: on text, 15 % of the output (155 thousand
matches per pass) comes from the ring. Copying those with the fast
loop's wide stores, from a ring whose first 298 bytes are mirrored past
its end, took text's 64 KiB steps from 1360 to 1650 MB/s.

### 3. Whole against 1 MiB steps on Silesia

| File | Whole | 1 MiB steps | 64 KiB steps |
|---|---|---|---|
| silesia.deflate6 | 1372 | 1376 | 1250 |
| silesia.zlib6 | 1392 | 1389 | 1214 |
| text.deflate6 | 2109 | 2026 | 1587 |

1 MiB steps cost nothing measurable on Silesia (0 %, within the model's
3 %) and 4 % on text; 64 KiB steps cost 9-13 % on Silesia and 25 % on
text, from the matches into the ring above and from the byte-at-a-time
loop that ends each step.

## Zstd decode

Setup. Every number is from fudge's P-cores (`taskset -c 0-5`), one row
set per hold of `flock ~/wt/TIMING.lock`, single thread, MB/s of
uncompressed bytes. zstd is 1.5.7 built from source under
`~/wt/lz-vendors/zstd-1.5.7` (`make -C programs zstd-release`, with
its x86-64 assembly Huffman loops) and `~/wt/lz-vendors/zstd-1.5.7-noasm`
(the same with `MOREFLAGS=-DZSTD_DISABLE_ASM`: its C path, which still
dispatches to BMI2 at run time), never linked into lz. The vendor is
timed with its own driver, `zstd -b -d -i2 FILE` (decode only, the
whole frame into one buffer, checksum verified), on the same files.
lz's kernel is timed through the bench rows where they exist and
otherwise through a C harness (sources in `_plans/lz/zstd-dec/dbg/`,
not committed) that runs the kernel exactly as stubs.c's drivers do:
`bench.c` the whole-buffer loop over every frame, best of runs filling
1.5-2 s; `step.c` the stream with input slices and a reused destination
of one step's size; `lit.c` the literals section of a frame's first
block, as zstd's `tests/fullbench -b21` times it. The `.zst3` corpora
are zstd level 3 with a checksum; `silesia.w19/w23/w27.zst` are
`zstd -3 --zstd=wlog=N silesia.raw`. Load is the 1-minute average
before and after each hold, from other sessions' builds on the E-cores.

### Bench rows against zstd 1.5.7

`bench_lz.exe check -f decode-zst` (load 4.9 before, 6.1 after), then
`zstd -b -d -i2` in the same hold, MB/s, C path / with assembly:

| Row | lz | zstd | lz / best zstd |
|---|---|---|---|
| text zst1 | 1836 | 1778 / 1783 | 1.03 |
| text zst3 | 1824 | 1724 / 1761 | 1.04 |
| text zst12 | 2056 | 1958 / 2025 | 1.02 |
| text zst19 | 1965 | 2094 / 2080 | 0.94 |
| columns zst1 | 1474 | 1370 / 1488 | 0.99 |
| columns zst3 | 1351 | 1332 / 1305 | 1.01 |
| columns zst12 | 1812 | 1771 / 1712 | 1.02 |
| columns zst19 | 1924 | 1928 / 1837 | 1.00 |
| random zst3 | 9930 | 10995 / 11482 | 0.86 |
| text zst3, 1 MiB steps | 1674 | | |
| text zst3, 64 KiB steps | 1671 | | |

Silesia (211 MB), C harness against `zstd -b -d -i2`, load 5.7 before
and 6.9 after: silesia.zst1 1950 against 1929 / 1865, zst3 1753 against
1739 / 1713, zst12 1942 against 1865 / 1838, zst19 1721 against 1564 /
1573. Without the run-time BMI2 dispatch the same kernel ran text.zst3
at 1719 and silesia.zst3 at 1625 (load 10-13): BMI2 is worth 10-15 %.

Short of the bar: text zst19 (0.94) and the incompressible corpus
(0.86-0.88), whose frames are raw blocks: the cost there is the copy
and the checksum, and lz's row includes the OCaml call.

### 2. The window copy per step

A streaming decoder decodes each block into a ring of window + 128 KiB
+ 64 bytes and copies it out to the step's destination: one copy of
every output byte, whatever the window, since a block is decoded whole
and a destination may be 1 byte. Copying back `min(window, produced)`
after decoding into the destination would copy as much at these step
sizes (each step produces less than the window) and cannot serve a
destination smaller than a block. Whole (no copy) against steps, C
harness, load 9.5 before and 7.4 after:

| File (window) | Whole | 1 MiB steps | 64 KiB steps | 1 MiB cost |
|---|---|---|---|---|
| silesia.w19 (512 KiB) | 1848 | 1707 | 1697 | 7.6 % |
| silesia.w23 (8 MiB) | 1830 | 1816 | 1808 | 0.8 % |
| silesia.w27 (128 MiB) | 1811 | 1803 | 1779 | 0.4 % |
| silesia.zst3 (2 MiB) | 1818 | 1725 | 1738 | 5.1 % |
| text.zst3 (8 MiB, never wraps) | 1986 | 1762 | 1787 | 11 % |

`zstd -d -c FILE > /dev/null` (the CLI's streaming decoder, one pass,
process start included) ran the three Silesia windows at 1534, 1647 and
1504 (C path), 1558, 1596 and 1357 (assembly) in the same hold.

The cost has two parts. The copy itself, 128 KiB per block from L2,
is what text.zst3 shows: it decodes at 2 GB/s, so a copy at about
20 GB/s is a tenth of the time. The ring's wraps add the rest at small
windows: after a wrap, matches into the earlier segment leave the fast
loop. They took the exact path, byte for byte, at first: 512 KiB cost
13 % then; wide copies from the earlier segment, safe because it lies
at least the slack ahead of the output, brought it to 7.6 %.

### 3. The step entry's cost on the whole loop

Silesia at level 3 (window 2 MiB), whole against steps, load 3.5
before and 4.0 after, after the literals moved into the destination:
1833 whole, 1747 in 1 MiB steps (4.7 %). The model's 3 % holds from an
8 MiB window up (0.4-0.8 % above); below it, the copy and the wraps of
section 2 cost 5-8 %.

### 8. Huffman's four-stream loops

The literals section of Silesia's first block at level 3 (51556 bytes
compressed, 19633 literals, four streams, double-symbol table), MB/s of
the block's 128 KiB, `lit.c` against `fullbench -b21 -l3 -i4` on
Silesia's first 10 MB (the same first block), load 6.7 before and 7.4
after:

| Decoder | MB/s |
|---|---|
| lz, C, five symbols per stream per reload, BMI2 dispatched | 15962 |
| zstd 1.5.7, C path (`ZSTD_DISABLE_ASM`) | 13990 |
| zstd 1.5.7, x86-64 assembly | 16224 |

lz's C loop is 14 % faster than zstd's C loop and 2 % behind its
assembly. Building the double-symbol table per code length (one row
shared by every symbol of that length) took it from about 5900 to 1250
cycles, a fifth of the section's cost.

### 10. XXH64's share of a decode at level 3

The kernel with the checksum against the same kernel with the XXH64
update and check removed, C harness, load 6.7 before and 7.4 after:

| File | With | Without | Share |
|---|---|---|---|
| text.zst3 | 1980 | 2170 | 8.8 % |
| columns.zst3 | 1451 | 1562 | 7.1 % |
| silesia.zst3 | 1823 | 1978 | 7.8 % |

The 2 % claim does not hold on this machine: XXH64 runs at about
20 GB/s (generic path, computed per block as it is written), and a
decode at 2 GB/s spends a tenth as long hashing.

## Measurements for the design (§7 of the brief)

The sections below keep the brief's §7 numbering. Every number is from
fudge: Intel Core Ultra 5 235 (P-cores 0-5, E-cores 6-13; 48 KiB L1d and
3 MB L2 per P-core, 24 MB L3), Debian 13, Linux 6.12, gcc 14.2.0, OCaml
5.5.2, governor `powersave`. Each ran single-threaded under `flock -w 1800
~/wt/TIMING.lock taskset -c 0-5`, one row set per hold, no build inside
it. `~/wt/lz-logs/measure/hold.sh` recorded `uptime` before and after and
one `top` sample during each run, taken from the E-cores. Each measurement
ran twice, on 2026-10-28. The first runs, 18:55-19:43 UTC, were under load
(1-minute load 7-16, other sessions' builds and tests, mostly on the
E-cores). The second runs, 21:04-21:37 UTC, were at load 2.3-10. Tables
give the second run, and the first where it differs. The programs are in
`bench/measure/` and the hold logs in `~/wt/lz-logs/measure/` (`h*` first,
`q*` second).

The vendors run in Python (`uv run --offline --no-project --with deflate
--with zstandard --with lz4 --with python-snappy --with xxhash python -I
vendors.py DATA decode|encode|checksum CORPUS...`, from `bench/measure`):
zlib 1.3.1 (Debian's), libdeflate as the `deflate` 0.9.0 package bundles
it (version not exposed), zstd 1.5.7 (`zstandard` 0.25.0), lz4 1.9.4,
Snappy through `cramjam` (python-snappy 0.7.3, not Google's code), xxHash
0.8.3. The Python bindings allocate their output per call. Above glibc's
32 MiB mmap ceiling that is a fresh, page-faulting mapping, which cut
Silesia's decode rates by a third (zstd level 1: 1221 MB/s, against 1907
for `zstd -b`). Every vendor row here therefore runs with
`GLIBC_TUNABLES=glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=17179869184`,
so freed output is reused as it is at 8 MiB. Rows that use the vendors' own
drivers say so: zstd 1.5.7's `zstd -b`, lz4 1.10.0's `lz4 -b`, and
libdeflate 1.26's `programs/benchmark`, the last two built by other agents
under `~/wt/lz-vendors/`.

## 1. The runtime lock's release

The cost of `caml_enter_blocking_section` and `caml_leave_blocking_section`
around a stub that does nothing, which is what `stubs.c` adds to a call
above its kernel's threshold. `caml_release_runtime_system` is a macro for
`caml_enter_blocking_section` in OCaml 5.5's `caml/threads.h`, so one
variant covers both. `bench/measure/lock_cost.ml` times 1,000,000 calls of
a stub that roots its bigarray argument and returns (`plain`), then
1,000,000 of the same stub with the release (`blocking`). It alternates
the two for 15 rounds; the cost is the median of the per-round
differences. Each mode runs in a fresh process: `alone` (one domain, none
ever spawned), `joined` (a second domain spawned and joined before
timing), `spin` (a second domain running a loop that does not allocate)
and `alloc` (a second domain allocating, so minor collections stop both).
`threads/lock_cost.exe` is the same program with `threads.posix` linked
(`-linkall`); its hooks also release the master lock.

```sh
dune build packages/lz/bench/measure/lock_cost.exe packages/lz/bench/measure/threads/lock_cost.exe
cd _build/default/packages/lz/bench/measure
flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 sh -c \
  'for m in alone joined spin alloc; do ./lock_cost.exe $m; done;
   for m in alone joined spin alloc; do ./threads/lock_cost.exe $m; done'
```

Second run: load 5.82 before, 5.67 after (21:04). First run: load 10.47
before, 10.45 after (18:55), under load. Median of 15 rounds, best round in
parentheses. The no-op call costs 1.6-1.8 ns in every row.

| process | release, second run | release, first run (load 10.5) |
|---|---|---|
| one domain | 9.7 ns (9.3) | 9.8 ns (9.7) |
| a second domain joined | 16.8 ns (16.5) | 14.3 ns (13.9) |
| a second domain spinning | 17.1 ns (17.1) | 14.4 ns (14.4) |
| a second domain allocating | 14.3 ns (14.3) | 17.1 ns (17.1) |
| systhreads, one domain | 40.8 ns (38.8) | 41.7 ns (40.1) |
| systhreads, a second domain joined | 62.7 ns (61.7) | 62.7 ns (61.9) |
| systhreads, a second domain spinning | 60.9 ns (59.9) | 63.5 ns (62.4) |
| systhreads, a second domain allocating | 61.9 ns (60.2) | 62.1 ns (61.7) |

Without systhreads the release costs 10 ns: an unlock and lock of the
domain lock and the pending-signal checks. Once a second domain has
existed, domain 0 keeps a backup thread for the rest of the process. Each
release then also publishes its state to that thread and signals its
condition variable: 14 to 17 ns. With `threads.posix` linked the hooks
release and take the master lock as well: 41 ns, and 61 to 64 ns beside a
second domain.

### Thresholds

Rule: a kernel releases the lock from the smallest power of two of input
bytes whose call lasts at least 100 times the release, so that the release
costs under 1 % of the call. The release is the dearest measured, 64 ns
(systhreads linked, a second domain), so the call must last 6.4 us. The
call is timed at the speed of the fastest vendor build measured on fudge,
on compressible input: the highest input rate over `text`, `columns` and
Silesia. A decoder's input is compressed bytes, so its rate is its decoded
MB/s divided by the ratio, as `vendors.py` and the drivers print it. The
lz kernels are held to match or beat these vendors, so their own speed
gives the same threshold or a larger one. The lead re-measures at
integration.

| kernel | fastest vendor, input MB/s | bytes at 10 ns | at 17 ns | at 64 ns | set |
|---|---|---|---|---|---|
| deflate decode | libdeflate (Python), text level 6 raw: 2138 / 3.321 = 644 | 644 | 1,095 | 4,122 | 8 KiB |
| zstd decode | `zstd -b`, columns level 1: 1554 / 1.958 = 793 | 793 | 1,349 | 5,078 | 8 KiB |
| LZ4 decode | `lz4 -b` 1.10, Silesia: 5341 / 2.101 = 2542 | 2,542 | 4,321 | 16,269 | 16 KiB |
| Snappy decode | snappy 1.2.2 AVX2 build, columns: 2478 / 1.775 = 1396 (the Snappy section) | 1,396 | 2,373 | 8,934 | 16 KiB |
| deflate encode | libdeflate level 1 (Python), columns: 415 | 415 | 706 | 2,656 | 4 KiB |
| zstd encode | zstd level 1 (Python), Silesia: 669 | 669 | 1,137 | 4,282 | 8 KiB |
| LZ4 encode | `lz4 -b` 1.10, columns: 1087 | 1,087 | 1,848 | 6,956 | 8 KiB |
| Snappy encode | snappy 1.2.2, columns: 1120 (the Snappy section) | 1,120 | 1,904 | 7,168 | 8 KiB |
| checksums (`release_bytes`) | CRC-32 of libdeflate 1.26 at 64 KiB: 134,100 (the Checksums section) | 134,100 | 227,970 | 858,240 | 1 MiB |

At the speeds §7.1 names: inflate at libdeflate's 644 MB/s gives 4,122
bytes (8 KiB); LZ4 at 4 GB/s decoded, on columns' ratio 1.607 (the lowest
of the compressible corpora), 2489 MB/s of input and 15,930 bytes (16
KiB); CRC-32 at 30 GB/s, 192,000 bytes (256 KiB). The scaffold held
deflate and zstd decoding to 64 KiB, which at libdeflate's rate keeps the
lock for up to 100 us and makes another domain's minor collection wait
that long.

`stubs.c` and `lz.ml` now hold these values, with the rule as a comment.
`release_bytes` is one constant for all four checksums, so it follows the
fastest. On incompressible input a call runs faster than the rule assumes.
At the threshold, a stored deflate block decodes at 22 GB/s and an
incompressible LZ4, Snappy or zstd input at 11 to 23 GB/s, so the release
costs 9 to 18 % of such a call at 64 ns, 1.4 to 2.8 % at 10 ns.
Without vector instructions the checksums run at 2.5 to 3.9 GB/s, and a
call just under 1 MiB keeps the lock for up to 0.4 ms.

## 4. Staging a bigarray slice in bytes

What a bytesrw filter over lz pays to copy a slice between a bigarray and
`bytes`, by C `memcpy` in a `[@@noalloc]` stub, in ns per byte.

```sh
dune build packages/lz/bench/measure/staging.exe
cd _build/default/packages/lz/bench/measure
flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 ./staging.exe
```

Second run: load 10.23 before, 9.49 after (21:21), other sessions on the
E-cores. First run: load 16.39 before, 16.04 after (19:22), under load.
Each number is the median of 11 rounds of about 20 ms of calls on the
same buffers; the best round is within 4 % of it. Second run, with the
first in parentheses where it differs by more than 5 %:

| bytes | bigarray to bytes | bytes to bigarray | there and back |
|---|---|---|---|
| 4 KiB | 0.0067 ns/B (150 GB/s) | 0.0064 ns/B (156 GB/s; first 0.0093) | 0.0129 ns/B (77 GB/s) |
| 64 KiB | 0.0154 ns/B (65 GB/s) | 0.0143 ns/B (70 GB/s) | 0.0284 ns/B (35 GB/s) |
| 1 MiB | 0.0141 ns/B (71 GB/s; first 0.0151) | 0.0138 ns/B (72 GB/s) | 0.0339 ns/B (30 GB/s) |

A 64 KiB slice there and back costs 1.9 us. The vendors decode at 0.46 to
0.76 ns per decoded byte (libdeflate, zstd) and 0.19 to 0.34 (LZ4), §9.
Staging the output alone (0.015 ns/B) adds 2 to 3 % to a deflate or zstd
decode and 4 to 8 % to an LZ4 decode.

## 5. Bigarray to bytes in OCaml today

The same job without a C blit: as raven's readers do it, and as
`Bytesrw.Bytes.Slice.of_bigbytes` does it, against the C blit. The
program, command and holds are §4's. `staging.exe` depends on `bytesrw`,
which is in the workspace's lock; lz does not depend on it.

`staging.ml` reproduces both loops with their types. Both read an `(int,
int8_unsigned_elt, c_layout) Array1.t`.

`packages/nx/lib/io/zip_archive.ml:596-599`, into a 64 KiB `chunk` made once
per entry:

```ocaml
for i = 0 to n - 1 do
  Bytes.unsafe_set chunk i
    (Char.unsafe_chr (Array1.unsafe_get encoded.data (first + i)))
done;
```

`contrib/ymir/lib/fits/tiles.ml:255-258`, into a `bytes` made per tile:

```ocaml
let input = Bytes.create (A.dim src) in
for i = 0 to A.dim src - 1 do
  Bytes.unsafe_set input i (Char.unsafe_chr (A.unsafe_get src i))
done;
```

Second run, ns per byte; the first run is within 2 % except where shown:

| bytes | zip_archive loop | C blit into reused bytes | tiles loop | C blit into new bytes | `of_bigbytes` |
|---|---|---|---|---|---|
| 4 KiB | 0.365 (first 0.373) | 0.0067 | 0.313 (first 0.325) | 0.0273 | 2.93 |
| 64 KiB | 0.373 | 0.0154 | 0.313 | 0.0204 | 2.91 |
| 1 MiB | 0.373 | 0.0141 | 0.312 | 0.0215 | 2.91 |

The per-byte loops run at 2.7 to 3.2 GB/s, 15 to 24 times slower than the
blit at 64 KiB. Each loop costs about half what libdeflate takes to decode
the same number of bytes. `of_bigbytes` runs at 0.34 GB/s, 107 to 143
times slower than a blit into new bytes, the same job: `bytesrw.ml`'s `of_bigbytes'` fills a
`Bytes.init` through a closure, and reads with `Bigarray.Array1.get` in a
function whose bigarray kind is not fixed there, so each byte goes through
the runtime's generic accessor. A bytesrw filter over lz should blit with
a stub, as §4 does.

## 7. CRC-32 kernels

`bench/measure/crc32_bench.c` is standalone C, built by dune for amd64
with gcc 14.2.0 `-std=c11 -O3 -mpclmul -msse4.2 -mavx2 -mvpclmulqdq`.
Before timing, each CRC-32 variant is checked against the bytewise table
on every length 0 to 1100, at every alignment 0 to 15, with a random
initial value, and on 8 MiB. The check values of `"123456789"`
(`0xCBF43926`, CRC-32C `0xE3069283`) are asserted. The folding constants
(`x^(128n+31) mod G` and `x^(128n-33) mod G`) and Barrett's
`floor(x^96 / G)` are computed at start-up from the polynomial. The fold
variants reduce their last 128 bits by two folds by `x^63 mod G` and a
Barrett step, then finish a tail under 16 bytes by slice-by-8.

```sh
dune build packages/lz/bench/measure/crc32_bench.exe
cd _build/default/packages/lz/bench/measure
flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 ./crc32_bench.exe
```

Second run: load 2.95 before, 4.08 after (21:23). First run: load 9.03
before, 9.19 after (19:34), under load. Each number is the median of 11
rounds of about 20 ms of calls on one buffer of random bytes, each call
continuing the previous call's CRC. GB/s are 1e9 bytes per second.

| kernel | 4 KiB | 64 KiB | 8 MiB | first run, 4 KiB / 64 KiB / 8 MiB |
|---|---|---|---|---|
| bytewise, one table | 0.7 | 0.7 | 0.7 | same |
| slice-by-8 | 2.6 | 2.5 | 2.5 | 2.5 / 2.5 / 2.5 |
| `crc32` instruction, one chain (CRC-32C) | 7.2 | 7.3 | 7.5 | 7.3 / 7.3 / 7.5 |
| `crc32` instruction, three chains, not combined (CRC-32C) | 22.8 | 22.3 | 22.2 | 23.6 / 23.1 / 23.1 |
| PCLMULQDQ, 4 xmm accumulators | 46.9 | 48.1 | 44.2 | 47.0 / 48.1 / 41.7 |
| PCLMULQDQ, 8 xmm | 62.8 | 66.4 | 47.6 | 62.8 / 66.3 / 48.4 |
| PCLMULQDQ, 12 xmm | 59.8 | 69.1 | 46.5 | 59.8 / 69.0 / 46.6 |
| VPCLMULQDQ on ymm, 4 accumulators (8 blocks) | 78.7 | 96.1 | 46.7 | 78.8 / 96.0 / 46.4 |
| VPCLMULQDQ on ymm, 8 accumulators (16 blocks) | 105.5 | 128.4 | 47.5 | 101.5 / 128.6 / 47.1 |

For comparison, the Checksums section above times lz's kernel and
libdeflate 1.26 in C (CRC-32 at 4 KiB 128.7 and 132.6 GB/s). Through
Python, where a call costs about 0.1 us and caps the 4 KiB column: the
`deflate` package's `crc32`, 36.9, 100.5 and 47.5 GB/s; zlib 1.3.1's, 6.8,
8.1 and 8.2.

- The `crc32` instruction computes CRC-32C (polynomial `0x1EDC6F41`), not
  the CRC-32 of gzip, zlib and ZIP (`0x04C11DB7`). Its polynomial is fixed
  in hardware, so gzip's CRC-32 needs a table or a carry-less-multiply
  kernel on x86; the instruction serves only formats that specify CRC-32C.
  Its single chain, one dependent instruction per 8 bytes, is also slower
  than every fold here, and on the P-cores slower than on the E-cores
  (11.7 GB/s in a functional check on cores 6-13). Three chains reach 23
  GB/s before the cost of combining them.
- 8 xmm accumulators run 1.3 to 1.4 times as fast as 4; 12 gain at most
  4 % over 8. On ymm (this CPU has VPCLMULQDQ with AVX2, no AVX-512), 8
  accumulators reach 128 GB/s at 64 KiB, 1.9 times the best xmm variant.
- At 8 MiB every fold variant gives 44 to 48 GB/s. The buffer is past the
  3 MB L2 and is read from the 24 MB L3, so the rate is the cache's, not
  the kernel's.

## 9. otherlibs/compress against the vendors on fudge

The fudge counterpart of the Mac table in
`otherlibs/compress/bench/README.md`, with Silesia added. Every decoder
reads the files `bench/reference.py` wrote to `bench/data/`. Two caveats:
`deflate6` there is libdeflate's level 6, where otherlibs' corpora used
zlib's; and the Snappy input is cramjam's, which decodes the same as
Google's. Nothing in `otherlibs/` was edited.

Two runs of today's library:

- Its own thumper suite, from the build tree, on lz's 8 MiB corpora
  linked under the names it reads (`gz` is lz's `gz9`, both gzip level 9):

  ```sh
  dune build otherlibs/compress/bench/bench_compress.exe
  cd ~/wt/lz-logs/measure/compress-bench   # data/ -> lz's files
  flock -w 1800 ~/wt/TIMING.lock taskset -c 0-5 \
    .../_build/default/otherlibs/compress/bench/bench_compress.exe check \
      --baseline=$PWD/none.thumper
  ```

  Second run at load 5.64 before, 4.99 after (21:27); first at 9.87 and
  9.49 (19:40).
- `bench/measure/compress_corpora.exe DATA decode|encode CORPUS...`, which
  times the same entries on all four corpora with `vendors.py`'s method:
  the fastest of at least 5 calls (3 for an encoder), and as many as fit a
  second, after checking the output. Second run at load 4.26 before, 2.30
  after (21:34); first at 9.49 and 8.95 (19:41), under load.

The vendors: `vendors.py decode`, second run at load 5.74 before, 4.76
after (21:09); first at 14.24 and 16.49 (19:16), under load. `zstd -b1
-i2 --single-thread` and the like on each raw file, at load 8.34 and 9.72
(21:16). `lz4 -b1 -i2 -T1` at load 4.20 and 4.32 (21:36). Google's snappy
1.2.2 AVX2 build at load 3.8, from the Snappy section above.

Decoding, MB/s of decoded bytes, `compress_corpora` (second run, first in
the next column). "Best vendor" is the fastest of the Python row and the
vendor's own driver.

| corpus | codec | compress | first run | zlib | libdeflate | zstd/lz4/snappy driver | Python vendor | compress / best vendor |
|---|---|---|---|---|---|---|---|---|
| text | zlib (level 6) | 780 | 769 | 488 | 1601 | | | 0.49 |
| text | deflate (level 6) | 1146 | 1099 | 627 | 2138 | | | 0.54 |
| text | gzip (level 9) | 799 | 764 | 553 | 1862 | | | 0.43 |
| text | zstd level 1 | 1005 | 971 | | | 1768 | 1890 | 0.53 |
| text | zstd level 3 | 902 | 849 | | | 1748 | 1842 | 0.49 |
| text | zstd level 12 | 1130 | 1088 | | | 2097 | 2171 | 0.52 |
| text | zstd level 19 | 1342 | 1321 | | | 1860 | 2150 | 0.62 |
| text | LZ4 block | 3037 | 3028 | | | 3890 | 3201 | 0.78 |
| text | LZ4 frame | 3057 | 3006 | | | 3890 (block) | 3183 | 0.79 |
| text | Snappy | 1034 | 990 | | | 2444 | 1833 | 0.42 |
| columns | zlib (level 6) | 829 | 818 | 501 | 1369 | | | 0.61 |
| columns | deflate (level 6) | 936 | 921 | 553 | 1324 | | | 0.71 |
| columns | gzip (level 9) | 786 | 783 | 528 | 1408 | | | 0.56 |
| columns | zstd level 1 | 1011 | 959 | | | 1554 | 1411 | 0.65 |
| columns | zstd level 3 | 947 | 908 | | | 1270 | 1393 | 0.68 |
| columns | zstd level 12 | 1239 | 1169 | | | 1772 | 1858 | 0.67 |
| columns | zstd level 19 | 1394 | 1327 | | | 1883 | 1976 | 0.71 |
| columns | LZ4 block | 3353 | 3311 | | | 2933 | 2524 | 1.14 |
| columns | LZ4 frame | 3356 | 3271 | | | 2933 (block) | 2613 | 1.14 |
| columns | Snappy | 3282 | 3240 | | | 2478 | 2149 | 1.32 |
| silesia | zlib (level 6) | 747 | 715 | 449 | 1377 | | | 0.54 |
| silesia | deflate (level 6) | 877 | 845 | 503 | 1432 | | | 0.61 |
| silesia | gzip (level 9) | 710 | 679 | 467 | 1399 | | | 0.51 |
| silesia | zstd level 1 | 1195 | 1155 | | | 1950 | 1888 | 0.61 |
| silesia | zstd level 3 | 1105 | 1074 | | | 1809 | 1782 | 0.61 |
| silesia | zstd level 12 | 1235 | 1204 | | | 1893 | 1977 | 0.62 |
| silesia | zstd level 19 | 1125 | 1059 | | | | 1570 | 0.72 |
| silesia | LZ4 block | 3333 | 3219 | | | 5341 | 4232 | 0.62 |
| silesia | LZ4 frame | 3327 | 3192 | | | 5341 (block) | 4134 | 0.62 |
| silesia | Snappy | 1427 | 1366 | | | 2343 | 1998 | 0.61 |
| random | zlib (level 6) | 4022 | 3995 | 3309 | 15803 | | | 0.25 |
| random | deflate (level 6) | 22921 | 22073 | 11447 | 22366 | | | 1.02 |
| random | gzip (level 9) | 3026 | 3011 | 4744 | 15781 | | | 0.19 |
| random | zstd level 3 | 11847 | 11681 | | | 22528 | 10824 | 0.53 |
| random | LZ4 block | 22670 | 23042 | | | 22288 | 11377 | 1.02 |
| random | Snappy | 22921 | 22198 | | | 22988 | 7299 | 1.00 |

libdeflate 1.26's own `benchmark -6 -s 268435456 silesia.raw` decodes
Silesia at 1080 MB/s, raw (1018 MB/s zlib-wrapped; its zlib engine 400).
That is below the Python binding's 1432 because the driver times one pass
in whole milliseconds, faults included. The Python row is the bar.

The thumper suite gives the same picture to within 9 % (it reports
medians, `compress_corpora` the fastest call), except LZ4 on columns
(second run 2542 MB/s against 3107 in the first, ±1.2 % within each). Second run, MB/s from its medians: text zlib6 741, deflate6 1077,
gz 748, snappy 988, lz4b 2853, lz4f 2873, zst3 842, zst19 1258. Columns:
817, 919, 763, 2996, 2542, 2527, 855, 1252.

Encoding, MB/s of raw bytes and ratio, `compress_corpora` second run (first
run in parentheses) against `vendors.py encode` (second run at load 4.76
before, 5.65 after):

| corpus | compress zlib 1 | zlib 1 | libdeflate 1 | compress zlib 6 | zlib 6 | libdeflate 6 | compress Snappy | cramjam | snappy 1.2.2 |
|---|---|---|---|---|---|---|---|---|---|
| text | 51 (49), 2.628 | 166, 2.755 | 407, 3.030 | 53 (51), 2.884 | 24, 3.195 | 75, 3.321 | 511 (488), 2.001 | 682 | 727 |
| columns | 84 (80), 2.384 | 136, 2.558 | 415, 2.596 | 86 (83), 2.541 | 47, 2.741 | 135, 2.687 | 1798 (1719), 1.773 | 1060 | 1120 |
| silesia | 65 (63), 2.657 | 148, 2.743 | 336, 2.883 | 66 (64), 2.878 | 51, 3.106 | 149, 3.139 | 626 (602), 2.091 | 831 | 840 |

Against the Mac table: on the M1, compress decoded deflate at 0.55 to
0.69 of zlib 1.2.12. On fudge it decodes at 1.6 times Debian's zlib 1.3.1,
which runs here at half the M1's zlib, and at 0.43 to 0.71 of libdeflate,
the bar. zstd is at 0.49 to 0.72 of zstd 1.5.7, as on the Mac (0.59 to
0.66). LZ4 is at 0.62 to 0.79 of lz4 1.10 on text and Silesia and 1.14 on
columns; on the Mac it was at parity with lz4 1.9.4, as it is here with
1.9.4 on text (0.95). Snappy is at 0.42 to 0.61 of Google's AVX2 build,
and 1.32 on columns (on the Mac 0.69 to 1.87 of python-snappy). The
encoders are where they were: zlib level 6 runs at 1.3 to 2.2 times zlib's
speed at 0.90 to 0.93 of its ratio, and at 0.4 to 0.7 of libdeflate's
speed.

## 12. Static size of a program linking lz

`size_lz.exe` links `lz` with `-linkall` and calls one entry.
`size_base.exe` is the same program without `lz`, also with `-linkall`.
Both are OCaml native executables, static in OCaml and lz, dynamic in libc.

```sh
dune build packages/lz/bench/measure/size_base.exe packages/lz/bench/measure/size_lz.exe
size _build/default/packages/lz/bench/measure/size_{base,lz}.exe
```

The tree was at `177719d15` (21:32 UTC), with other agents' uncommitted
changes in `lib/`. The deflate, zstd, LZ4 and Snappy decoders were in,
and the Snappy and LZ4 encoders. The deflate and zstd encoders were
placeholders, 196 bytes of text each.

| | text | data | bss | stripped file |
|---|---|---|---|---|
| size_base.exe | 1,734,693 | 484,484 | 3,464 | 2,230,712 |
| size_lz.exe | 1,857,660 | 492,556 | 11,816 | 2,361,368 |
| lz's share | 122,967 | 8,072 | 8,352 | 130,656 |

The C kernels and stubs are 93 KB of that text (`liblz_stubs.a`: the
deflate decoder 24 KB; the zstd decoder 32 KB in four units; stubs 12 KB;
LZ4 9 KB; Snappy 5 KB; checksums 10 KB in five units), and the OCaml
module is 8 KB (`lz.a`). At 19:33, before the LZ4 and Snappy commits,
lz's share was 113,376 bytes of text. The lead re-measures at
integration, when both encoders are in.

## 13. The lent buffer: steps against the whole loop, and the run

The step entry's cost of §7.3, measured again once the streaming
decoders write their runs into the decoder's own buffer (RFC 0039,
Memory): the RFC's open question of whether one loop serves both
drivers, and the run per format.

Setup. fudge's P-cores (`taskset -c 0-5`), single thread, MB/s of
decoded bytes, best of runs filling 1 s, the bench corpora and Silesia
(`bench/data/`). A C harness, `stepbench.c` (sources and scripts in
`_plans/lz/lent-buffer/`, not committed), links the deflate and
Zstandard kernels and drives them as stubs.c does: `whole()` into a
destination of the decoded size, or `lz_dec_step` and `dec_run` step
after step over the whole input, each run handed to a function that
reads its first and last bytes. Every mode's output is checked against
the raw corpus first. The kernels are compiled with dune's flags
(`-O2 -fno-strict-aliasing -fwrapv -fPIC -pthread -std=c11`) from a copy
of the tree at each run size, and from the commit before, `c1d7364f5`
(HEAD in the tables), whose steps copy each run out of the kernel's
window into a run buffer (deflate: 64 KiB; Zstandard: one block). A
step's decoder is new each time, its buffer allocated as
`Lz.Decoder.t`'s is. Load is the 1-minute average
before and after each hold.

### Steps against the whole loop

The committed runs, deflate 256 KiB and Zstandard 128 KiB, one hold,
load 1.9 before and 1.2 after (`top` 89 % idle):

| File | Whole, HEAD | Whole | Steps, HEAD | Steps | Steps against whole | Steps against HEAD's |
|---|---|---|---|---|---|---|
| text.deflate6 | 2321 | 2327 | 1872 | 2207 | -5.2 % | +17.9 % |
| text.zlib6 | 1752 | 1745 | 1485 | 1681 | -3.7 % | +13.2 % |
| text.gz9 | 2044 | 2048 | 1662 | 1937 | -5.4 % | +16.6 % |
| text.zst1 | 1959 | 1957 | 1820 | 1859 | -5.0 % | +2.1 % |
| text.zst3 | 1962 | 1964 | 1790 | 1828 | -6.9 % | +2.1 % |
| text.zst12 | 2238 | 2220 | 2111 | 2165 | -2.5 % | +2.6 % |
| text.zst19 | 2214 | 2228 | 2139 | 2231 | +0.1 % | +4.3 % |
| columns.deflate6 | 1348 | 1344 | 1308 | 1322 | -1.7 % | +1.0 % |
| columns.zlib6 | 1381 | 1372 | 1342 | 1368 | -0.3 % | +1.9 % |
| columns.gz9 | 1419 | 1414 | 1345 | 1394 | -1.4 % | +3.6 % |
| columns.zst1 | 1424 | 1424 | 1385 | 1407 | -1.2 % | +1.6 % |
| columns.zst3 | 1444 | 1439 | 1404 | 1431 | -0.6 % | +1.9 % |
| columns.zst12 | 1925 | 1924 | 1887 | 1921 | -0.2 % | +1.8 % |
| columns.zst19 | 2036 | 2048 | 1980 | 2055 | +0.4 % | +3.8 % |
| random.deflate6 | 23000 | 22928 | 35064 | 39530 | +72.4 % | +12.7 % |
| random.zlib6 | 20044 | 19947 | 27554 | 29724 | +49.0 % | +7.9 % |
| random.gz9 | 21306 | 21450 | 29613 | 31935 | +48.9 % | +7.8 % |
| random.zst1 | 11770 | 11779 | 12641 | 14686 | +24.7 % | +16.2 % |
| random.zst3 | 11943 | 11785 | 10738 | 12095 | +2.6 % | +12.6 % |
| random.zst12 | 11781 | 11775 | 10303 | 11758 | -0.1 % | +14.1 % |
| random.zst19 | 11773 | 11776 | 10288 | 11711 | -0.6 % | +13.8 % |
| silesia.deflate6 | 1378 | 1380 | 1279 | 1370 | -0.8 % | +7.1 % |
| silesia.zlib6 | 1406 | 1399 | 1298 | 1387 | -0.9 % | +6.9 % |
| silesia.gz9 | 1390 | 1410 | 1300 | 1391 | -1.3 % | +7.0 % |
| silesia.zst1 | 1882 | 1893 | 1840 | 1873 | -1.0 % | +1.8 % |
| silesia.zst3 | 1814 | 1805 | 1715 | 1748 | -3.1 % | +2.0 % |
| silesia.zst12 | 2017 | 2008 | 1818 | 1867 | -7.0 % | +2.7 % |
| silesia.zst19 | 1779 | 1785 | 1688 | 1718 | -3.8 % | +1.8 % |

The whole-buffer decode is HEAD's within 1.5 %, the spread of the
rows. Steps gain 13-18 % on text's deflate family and 7 % on Silesia's,
where HEAD copied 32 KiB of window per 64 KiB run and cut runs four
times as often, and 1.6-4.3 % on Zstandard's compressed corpora, where
it copied every byte out (12-16 % on random's raw blocks).
On random bytes, stored and raw blocks, steps beat the whole decode:
their buffer stays in L2 while the whole decode writes 8 MiB of
destination.

The answer to the open question: steps cost 0-1.7 % on columns and on
Silesia's deflate family, within the model's 3 %, but 5.2-5.4 % on
text's deflate and gzip and up to 7 % on Zstandard frames whose window
is shorter than their content. The whole-buffer path keeps its own loop:
the same kernels with the destination as the window, no run and no
wrap, as before this change.

Where the cost is. A step wraps to its buffer's start once a run no
longer fits, and for the next window of output a match may reach back
past the wrap into the earlier segment. Deflate's fast loop copies such
a match from there with the same wide stores, but which side a match
reads is a branch the predictor misses: the cost scales with the window
over the run (below). A Zstandard buffer holds the window and a run, so
a frame whose window is shorter than its content wraps once per window
of output, and nearly every block lies within a window after a wrap;
matches into the earlier segment take the sequence loop's exact path.
Text at level 3 (2 MiB window) costs 6.9 %, at level 19 (an 8 MiB
window, its whole content) nothing.

Two other causes were ruled out in a hold at load 9.1 before and 4.7
after: keeping the stepped decoder's buffer across runs, so that its
pages are not faulted in again, changed text.zst3 from 1822 to 1827 and
Silesia's from 1731 to 1739; decoding the whole-buffer path's literals
into a separate buffer, as a stream's are, left it at 1984 against
1980. Selecting the deflate source with a conditional move, not a
branch, slowed the whole decode of text by 7 % (2326 to 2171) and the
steps too (2096 to 1972 at 128 KiB); it is not kept.

### The run

One hold per corpus, the four holds at loads 5.4 to 4.0 (text), 3.8 to
2.8 (columns), 8.6 to 5.6 (random) and 3.1 to 2.6 (Silesia). Steps at
each run against the whole decode of the same hold; a Zstandard block
decodes whole and holds up to 128 KiB, so its run is at least 128 KiB:

| File | Whole | 64 KiB | 128 KiB | 256 KiB | 512 KiB |
|---|---|---|---|---|---|
| text.deflate6 | 2321 | 1903 (-18.0 %) | 2094 (-9.8 %) | 2202 (-5.1 %) | 2265 (-2.4 %) |
| text.zlib6 | 1736 | 1487 (-14.3 %) | 1600 (-7.8 %) | 1666 (-4.0 %) | 1703 (-1.9 %) |
| text.gz9 | 2031 | 1672 (-17.7 %) | 1834 (-9.7 %) | 1933 (-4.8 %) | 1982 (-2.4 %) |
| text.zst1 | 1948 | | 1856 (-4.7 %) | 1879 (-3.6 %) | 1898 (-2.5 %) |
| text.zst3 | 1952 | | 1820 (-6.8 %) | 1823 (-6.6 %) | 1825 (-6.5 %) |
| text.zst12 | 2211 | | 2139 (-3.3 %) | 2146 (-2.9 %) | 2130 (-3.7 %) |
| text.zst19 | 2131 | | 2097 (-1.6 %) | 2106 (-1.2 %) | 2101 (-1.4 %) |
| columns.deflate6 | 1339 | 1311 (-2.1 %) | 1322 (-1.3 %) | 1324 (-1.1 %) | 1338 (-0.1 %) |
| columns.zlib6 | 1375 | 1360 (-1.1 %) | 1367 (-0.6 %) | 1370 (-0.4 %) | 1374 (-0.1 %) |
| columns.gz9 | 1414 | 1358 (-3.9 %) | 1389 (-1.7 %) | 1400 (-1.0 %) | 1406 (-0.5 %) |
| columns.zst1 | 1421 | | 1416 (-0.4 %) | 1407 (-1.0 %) | 1410 (-0.8 %) |
| columns.zst3 | 1429 | | 1438 (+0.6 %) | 1434 (+0.3 %) | 1421 (-0.6 %) |
| columns.zst12 | 1923 | | 1923 (-0.0 %) | 1890 (-1.7 %) | 1930 (+0.4 %) |
| columns.zst19 | 2042 | | 2018 (-1.2 %) | 2042 (-0.0 %) | 2030 (-0.6 %) |
| random.deflate6 | 22886 | 52226 (+128.2 %) | 47828 (+109.0 %) | 39510 (+72.6 %) | 38707 (+69.1 %) |
| random.zlib6 | 19844 | 33982 (+71.2 %) | 32853 (+65.6 %) | 29796 (+50.2 %) | 29296 (+47.6 %) |
| random.gz9 | 21093 | 36840 (+74.7 %) | 34504 (+63.6 %) | 31904 (+51.3 %) | 31358 (+48.7 %) |
| random.zst1 | 11769 | | 14689 (+24.8 %) | 14516 (+23.3 %) | 14299 (+21.5 %) |
| random.zst3 | 11761 | | 12113 (+3.0 %) | 12010 (+2.1 %) | 11872 (+0.9 %) |
| random.zst12 | 11758 | | 11768 (+0.1 %) | 11772 (+0.1 %) | 11744 (-0.1 %) |
| random.zst19 | 11778 | | 11699 (-0.7 %) | 11840 (+0.5 %) | 11685 (-0.8 %) |
| silesia.deflate6 | 1383 | 1297 (-6.2 %) | 1344 (-2.8 %) | 1370 (-0.9 %) | 1384 (+0.1 %) |
| silesia.zlib6 | 1406 | 1301 (-7.4 %) | 1356 (-3.6 %) | 1387 (-1.3 %) | 1403 (-0.2 %) |
| silesia.gz9 | 1411 | 1301 (-7.7 %) | 1361 (-3.5 %) | 1394 (-1.2 %) | 1404 (-0.4 %) |
| silesia.zst1 | 1886 | | 1872 (-0.7 %) | 1882 (-0.2 %) | 1888 (+0.1 %) |
| silesia.zst3 | 1785 | | 1745 (-2.2 %) | 1743 (-2.4 %) | 1763 (-1.2 %) |
| silesia.zst12 | 1969 | | 1829 (-7.1 %) | 1852 (-5.9 %) | 1856 (-5.7 %) |
| silesia.zst19 | 1759 | | 1598 (-9.2 %) | 1596 (-9.3 %) | 1666 (-5.3 %) |

Silesia's zst19 row is the noisiest: the final hold above, at load 1.9,
read 3.8 % at 128 KiB.

Deflate's cost halves as the run doubles (text: 18, 10, 5, 2.4 %): it
is the matches into the earlier segment over the first 32 KiB after
each wrap. The run is 256 KiB: a quarter of 128 KiB's cost on Silesia
and half on text, and its buffer, 288 KiB with the window, stays in a
512 KiB L2 with room for what `f` does with the run; 512 KiB halves
text's cost again for twice the memory, past such an L2. Zstandard's
cost does not move with the run, which only sets how many blocks a step
writes. Its run was 128 KiB, one block, at first; it is now 256 KiB:
a compressed block's output is known only once decoded, so a step
places one only while the run has room for 128 KiB, and a run of one
block ended every step after one block, however small (83,887 steps on
8 MiB of 100-byte blocks). With room for two, a step ends only once its
run holds a block's maximum: 64 steps on that stream, 7.84 to 1.47 ms
through `Lz.Decoder.step` in 1 MiB slices, 1000-byte blocks 1.02 to
0.35 ms, the CLI's 128 KiB blocks 4.58 and 4.61 ms (load 2.2 before,
2.1 after). The run-size table above shows no cost to 256 KiB on the
corpora's frames.

### Through the bench rows

`bench_lz.exe check -f decode-zlib6 -f decode-zst3`, one hold, load 2.1
before and 1.9 after: text zlib6 5.04 ms whole, 5.24 ms in 1 MiB steps
(+4.0 %), 5.30 ms in 64 KiB steps; text zst3 4.27 ms whole, 4.69 ms
(+9.8 %) and 4.79 ms; columns zlib6 6.08 and 6.12 ms; columns zst3 5.87
and 5.94 ms; random zlib6 420 us whole and 290 us in 1 MiB steps. Each
stepped row makes a new decoder per iteration, whose buffer, for text
zst3 the 2 MiB window and a run, is a new bigarray; the whole rows reuse
their destination.

## 6. Deflate encoder constants: libdeflate's levels against zlib's

The RFC left open whether the encoder takes libdeflate's level
parameters or zlib's at levels 1, 6 and 9. Both ran in our encoder,
raw deflate, whole buffer, fudge cores 0-5, best of 2 (Silesia once),
load 1.2 before and 1.4 after. zlib's set is its chain length as the
search depth and its nice length: level 1 greedy, depth 4, nice 8;
level 6 lazy, 128, 128; level 9 lazy, 4096, 258. "zlib, lazy2" keeps
libdeflate's parser at 9 with zlib's depth, and nice 32 at 1, 65 at 6.
MB/s and ratio; zlib 1.3.1's own ratio last.

| corpus, level | libdeflate's | zlib's | zlib's, lazy2 | zlib 1.3.1 |
|---|---|---|---|---|
| text 1 | 410, 3.030 | 303, 3.132 | 307, 3.132 | 2.755 |
| text 6 | 76, 3.321 | 44, 3.343 | 44, 3.343 | 3.195 |
| text 9 | 29, 3.310 | 41, 3.342 | 29, 3.310 | 3.313 |
| columns 1 | 383, 2.596 | 299, 2.636 | 301, 2.636 | 2.558 |
| columns 6 | 127, 2.687 | 71, 2.703 | 72, 2.703 | 2.741 |
| columns 9 | 18.2, 2.748 | 4.1, 2.775 | 3.1, 2.776 | 2.781 |
| silesia 1 | 312, 2.883 | 255, 2.953 | 252, 2.987 | 2.743 |
| silesia 6 | 145, 3.139 | 98, 3.159 | 96, 3.158 | 3.106 |
| silesia 9 | 51, 3.177 | 39, 3.169 | 27, 3.180 | 3.133 |

libdeflate's parameters stay. They pass zlib's ratio everywhere but
columns at 6 to 9 (-2.0 % at 6, -1.2 % at 9) and text at 9 (-0.1 %).
zlib's depths buy 0.5 to 3.4 % of ratio for 1.2 to 5.9 times the time
(text at 9, where zlib's lazy parser is faster than lazy2, aside) and
still miss zlib on columns at 6 and 9, so the gap there is the
parser's, not the depth's. libdeflate
1.26 itself writes the same sizes on these corpora. Two of its
heuristics were tried off on columns at 6, the 8 KiB limit on 3-byte
matches (2.646) and the literal-count minimum length (2.687, text
3.319), and blocks capped at 50,000 and 20,000 bytes (2.681, 2.667):
none closes it.

A stepped encode costs 1 to 2 % against the whole one (text 6: 75.2
whole, 73.6 in 64 KiB steps, 73.4 in 4 KiB steps; Silesia 1: 312.2,
305.6, 305.6), the copy into the window and its slides; the quota stays
at 64 KiB.
