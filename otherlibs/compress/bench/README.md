# compress benchmarks

The suite times each decoder on three 8 MiB corpora compressed by the
reference encoders, and the encoders on the raw corpora. `reference.py` writes
the corpora to `data/`, which git ignores, and prints the references' own
throughput:

```sh
cd otherlibs/compress/bench
uv run --with python-snappy --with lz4 --with zstandard python reference.py
dune build @otherlibs/compress/bench/bench
```

The corpora: `text` is words drawn by a seeded xorshift from 512 made-up
words; `columns` is int64 keys with small gaps and float64 prices of two
decimals, as a Parquet page holds them; `random` is incompressible bytes.

## Results

Apple M1 Max, macOS 26.3.1, OCaml 5.5.1 release build; Python 3.13.1 with zlib
1.2.12, python-snappy (snappy 1.x), lz4 1.9.4 and zstandard 0.25.0. Measured on
2026-10-01 with a 1-minute load average between 9 and 33 on 10 cores: the
ratios hold better than the absolute numbers, which a quiet machine raises.

Decoding, MB/s of decompressed data:

| Corpus | Codec | compress | Reference | Ratio |
|---|---|---|---|---|
| text | zlib (level 6) | 550 | 994 | 0.55 |
| text | deflate (level 6) | 658 | 1058 | 0.62 |
| text | gzip (level 9) | 674 | 1127 | 0.60 |
| text | Snappy | 867 to 978 | 1248 | 0.69 to 0.78 |
| text | LZ4 block | 1985 | 1963 | 1.01 |
| text | LZ4 frame | 1961 | 2122 | 0.92 |
| text | Zstandard level 3 | 602 | 1024 | 0.59 |
| text | Zstandard level 19 | 647 | 1032 | 0.63 |
| columns | zlib (level 6) | 428 | 799 | 0.54 |
| columns | deflate (level 6) | 600 | 881 | 0.68 |
| columns | gzip (level 9) | 605 | 879 | 0.69 |
| columns | Snappy | 2766 | 1479 | 1.87 |
| columns | LZ4 block | 2640 | 2533 | 1.04 |
| columns | LZ4 frame | 2633 | 2687 | 0.98 |
| columns | Zstandard level 3 | 558 | 844 | 0.66 |
| columns | Zstandard level 19 | 765 | 1246 | 0.61 |

Encoding, MB/s of raw data, and compression ratio:

| Corpus | Codec | compress | Ratio | Reference | Ratio |
|---|---|---|---|---|---|
| text | zlib level 6 | 39 | 2.88 | 15 | 3.19 |
| text | Snappy | 401 | 2.00 | 494 | 2.00 |
| columns | zlib level 6 | 65 | 2.42 | 20 | 2.74 |
| columns | Snappy | 978 | 1.77 | 370 | 1.77 |

Snappy blocks are byte for byte those of the reference. The zlib encoder at
level 6 follows a match chain of 4 links where zlib's follows 128, trading
ratio for speed.
