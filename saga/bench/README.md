# Saga Benchmarks

This directory contains micro-benchmarks for the `saga.tokenizers` library.
The suite mirrors HuggingFace's `tokenizers` so we can compare wall-clock
throughput for realistic workloads and catch regressions.

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
│ WordPiece/Decode/long         │  31.10ns │  67.99ns │   22.00w │   1.00x │       100% │
│ WordLevel/Decode/long         │  39.11ns │  64.80ns │   22.00w │   0.80x │       126% │
│ WordLevel/Encode/single_short │ 109.36μs │ 108.71μs │  16.70kw │   0.00x │    351653% │
│ WordPiece/Encode/single_short │ 136.05μs │ 125.82μs │  20.81kw │   0.00x │    437479% │
│ BPE/Encode/single_short       │ 698.17μs │ 697.98μs │  57.90kw │   0.00x │   2245047% │
│ BPE/Decode/long               │   1.96ms │   1.96ms │ 326.58kw │   0.00x │   6289266% │
│ WordLevel/Encode/batch_32     │   3.33ms │   3.33ms │ 534.18kw │   0.00x │  10709213% │
│ WordPiece/Encode/batch_32     │   3.84ms │   3.83ms │ 665.58kw │   0.00x │  12353067% │
│ WordLevel/Encode/single_long  │   5.97ms │   5.97ms │ 787.81kw │   0.00x │  19193110% │
│ WordPiece/Encode/single_long  │   6.67ms │   6.67ms │ 984.48kw │   0.00x │  21452022% │
│ BPE/Encode/batch_32           │  24.52ms │  23.12ms │   1.85Mw │   0.00x │  78842659% │
│ BPE/Encode/single_long        │  52.12ms │  51.93ms │   2.79Mw │   0.00x │ 167604161% │
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
