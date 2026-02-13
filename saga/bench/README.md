# Saga Benchmarks

This directory contains micro-benchmarks for the `saga` library.
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
│ WordLevel/Encode/single_short │ 152.58μs │ 152.28μs │  42.07kw │   1.00x │       100% │
│ BPE/Encode/single_short       │ 170.22μs │ 143.07μs │  40.96kw │   0.90x │       112% │
│ WordPiece/Encode/single_short │ 280.83μs │ 279.61μs │  70.86kw │   0.54x │       184% │
│ WordPiece/Decode/long         │ 488.30μs │ 485.52μs │ 109.39kw │   0.31x │       320% │
│ WordLevel/Decode/long         │ 542.80μs │ 499.65μs │ 109.39kw │   0.28x │       356% │
│ BPE/Decode/long               │   1.38ms │   1.37ms │ 446.95kw │   0.11x │       906% │
│ BPE/Encode/batch_32           │   3.79ms │   3.79ms │   1.31Mw │   0.04x │      2487% │
│ BPE/Encode/single_long        │   5.97ms │   5.90ms │   1.54Mw │   0.03x │      3912% │
│ WordPiece/Encode/batch_32     │   8.93ms │   8.91ms │   2.27Mw │   0.02x │      5852% │
│ WordLevel/Encode/batch_32     │   9.97ms │   5.71ms │   1.35Mw │   0.02x │      6531% │
│ WordLevel/Encode/single_long  │  10.03ms │  10.02ms │   1.93Mw │   0.02x │      6571% │
│ WordPiece/Encode/single_long  │  11.18ms │  11.14ms │   2.33Mw │   0.01x │      7326% │
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
