# Saga Tokenizers Benchmarks

```bash
# Build and run
dune exec saga/bench/bench_tokenizers.exe

# With time quota
dune exec saga/bench/bench_tokenizers.exe -- -q 5s
```

## Results

```
┌───────────────────────────────────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                                  │ Time/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────────────┼──────────┼──────────┼─────────┼────────────┤
│ Encoding/BPE encode[100 chars]        │ 173.73ns │   76.00w │   1.00x │       100% │
│ Serialization/to_json                 │ 431.63ns │  652.00w │   0.40x │       248% │
│ Encoding/WordLevel encode[100 chars]  │   1.14μs │  484.00w │   0.15x │       656% │
│ Vocab/add_tokens (100)                │   3.16μs │    9.00w │   0.05x │      1818% │
│ Decoding/BPE decode[100 tokens]       │   3.24μs │   1.02kw │   0.05x │      1868% │
│ Decoding/WordPiece decode[100 tokens] │   3.53μs │   1.07kw │   0.05x │      2030% │
│ Encoding/WordPiece encode[100 chars]  │   6.03μs │   1.76kw │   0.03x │      3473% │
│ Encoding/WordLevel encode[1K chars]   │   8.91μs │   3.54kw │   0.02x │      5126% │
│ Encoding/WordPiece encode[1K chars]   │  11.70μs │   3.16kw │   0.01x │      6735% │
│ Batch/BPE batch[10 items]             │  17.00μs │  14.62kw │   0.01x │      9787% │
│ Decoding/BPE decode[1K tokens]        │  32.16μs │  10.14kw │   0.01x │     18514% │
│ Batch/WordPiece batch[10 items]       │  32.55μs │   9.04kw │   0.01x │     18733% │
│ Decoding/WordPiece decode[1K tokens]  │  38.67μs │  10.01kw │   0.00x │     22261% │
│ Encoding/BPE encode[1K chars]         │  53.05μs │   5.17kw │   0.00x │     30538% │
│ Encoding/WordLevel encode[10K chars]  │  92.08μs │  30.42kw │   0.00x │     53004% │
│ Encoding/WordPiece encode[10K chars]  │ 114.79μs │  30.16kw │   0.00x │     66073% │
│ Serialization/from_file               │ 225.55μs │   2.52kw │   0.00x │    129827% │
│ Batch/BPE batch[100 items]            │ 242.70μs │ 146.15kw │   0.00x │    139699% │
│ Batch/WordPiece batch[100 items]      │ 328.46μs │  90.39kw │   0.00x │    189065% │
│ Encoding/BPE encode[10K chars]        │ 518.13μs │  50.17kw │   0.00x │    298242% │
└───────────────────────────────────────┴──────────┴──────────┴─────────┴────────────┘
```
