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
│ BPE/Encode/single_short       │ 102.22μs │ 101.45μs │  23.38kw │   1.00x │       100% │
│ WordLevel/Encode/single_short │ 124.40μs │ 124.22μs │  25.34kw │   0.82x │       122% │
│ WordPiece/Encode/single_short │ 252.80μs │ 252.37μs │  54.14kw │   0.40x │       247% │
│ WordPiece/Decode/long         │ 472.96μs │ 471.40μs │ 109.39kw │   0.22x │       463% │
│ WordLevel/Decode/long         │ 492.19μs │ 489.37μs │ 109.39kw │   0.21x │       481% │
│ BPE/Decode/long               │   1.44ms │   1.43ms │ 446.95kw │   0.07x │      1405% │
│ BPE/Encode/batch_32           │   3.08ms │   3.06ms │ 748.01kw │   0.03x │      3010% │
│ BPE/Encode/single_long        │   4.49ms │   4.47ms │ 686.74kw │   0.02x │      4392% │
│ WordLevel/Encode/batch_32     │   4.56ms │   4.54ms │ 810.82kw │   0.02x │      4457% │
│ WordPiece/Encode/batch_32     │   8.58ms │   8.52ms │   1.73Mw │   0.01x │      8390% │
│ WordPiece/Encode/single_long  │   8.66ms │   8.65ms │   1.50Mw │   0.01x │      8473% │
│ WordLevel/Encode/single_long  │   8.89ms │   8.83ms │   1.11Mw │   0.01x │      8697% │
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
