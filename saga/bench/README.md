# Saga Benchmarks

This directory contains micro-benchmarks for the `saga.tokenizers` library.
The suite mirrors HuggingFace's `tokenizers` so we can compare wall-clock
throughput for realistic workloads and catch regressions.

## Fixtures

Benchmark inputs live in `./data/`:

- `news_1k.txt`, `wiki_64k.txt`, `code_excerpt.txt` — sample corpora used for
  encoding workloads.
- `byte_bpe.json`, `wordpiece.json`, `wordlevel.json` — tokenizers trained on
  the fixture texts via `scripts/generate_fixtures.py`.

Regenerate fixtures after editing the generator or the base texts:

```bash
uv run python saga/bench/scripts/generate_fixtures.py
```

## Running the Benchmarks

### Saga (OCaml)

```bash
dune exec saga/bench/bench_tokenizers.exe
```

### tokenizers (Python/Rust FFI)

```bash
uv run saga/bench/bench_tokenizers.py
```

## Results Saga

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ WordPiece/Decode/long         │  31.06ns │  41.48ns │   22.00w │   1.00x │       100% │
│ WordLevel/Decode/long         │  33.35ns │  34.81ns │   22.00w │   0.93x │       107% │
│ WordLevel/Encode/single_short │ 109.30μs │ 109.05μs │  16.70kw │   0.00x │    351950% │
│ WordPiece/Encode/single_short │ 115.92μs │ 115.78μs │  20.81kw │   0.00x │    373261% │
│ BPE/Encode/single_short       │ 707.51μs │ 707.09μs │  55.81kw │   0.00x │   2278162% │
│ BPE/Decode/long               │   2.08ms │   2.08ms │ 326.58kw │   0.00x │   6712213% │
│ WordLevel/Encode/batch_32     │   3.29ms │   3.29ms │ 534.18kw │   0.00x │  10598314% │
│ WordPiece/Encode/batch_32     │   4.07ms │   3.98ms │ 665.58kw │   0.00x │  13119989% │
│ WordLevel/Encode/single_long  │   5.99ms │   5.99ms │ 787.81kw │   0.00x │  19293154% │
│ WordPiece/Encode/single_long  │   7.01ms │   7.00ms │ 984.48kw │   0.00x │  22585690% │
│ BPE/Encode/batch_32           │  22.27ms │  22.25ms │   1.79Mw │   0.00x │  71701143% │
│ BPE/Encode/single_long        │  49.91ms │  49.83ms │   2.66Mw │   0.00x │ 160719732% │
└───────────────────────────────┴──────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results HF tokenizers

```
┌───────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ WordLevel/Encode/single_short │ 124.90µs │ 124.91µs │ 103.05w │   1.00x │       100% │
│ WordPiece/Encode/single_short │ 200.99µs │ 200.98µs │ 128.79w │   0.62x │       161% │
│ BPE/Encode/single_short       │ 251.01µs │ 251.02µs │ 115.08w │   0.50x │       201% │
│ WordLevel/Decode/long         │ 651.97µs │ 651.26µs │ 560.72w │   0.19x │       522% │
│ WordPiece/Decode/long         │ 689.25µs │ 689.23µs │ 599.26w │   0.18x │       552% │
│ BPE/Decode/long               │   1.32ms │   1.32ms │ 702.18w │   0.09x │      1059% │
│ WordLevel/Encode/batch_32     │   2.16ms │  18.35ms │  2.90kw │   0.06x │      1732% │
│ WordPiece/Encode/batch_32     │   3.23ms │  27.55ms │  2.88kw │   0.04x │      2589% │
│ BPE/Encode/batch_32           │   5.12ms │  45.75ms │  2.22kw │   0.02x │      4097% │
│ WordLevel/Encode/single_long  │   7.30ms │   7.30ms │  2.55kw │   0.02x │      5843% │
│ WordPiece/Encode/single_long  │   8.07ms │   8.07ms │  2.33kw │   0.02x │      6465% │
│ BPE/Encode/single_long        │  12.16ms │  12.16ms │  2.08kw │   0.01x │      9735% │
└───────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```
