# Saga Tokenizers Benchmarks

```bash
# Build and run
dune exec saga/bench/bench_tokenizers.exe

# With time quota
dune exec saga/bench/bench_tokenizers.exe -- -q 5s
```

## Results

┌───────────────────────────────────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                                  │ Time/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────────────┼──────────┼──────────┼─────────┼────────────┤
│ Encoding/BPE encode[100 chars]        │ 299.58ns │   76.00w │   1.00x │       100% │
│ Serialization/to_json                 │ 914.42ns │  652.00w │   0.33x │       305% │
│ Encoding/WordLevel encode[100 chars]  │   1.97μs │  684.00w │   0.15x │       658% │
│ Vocab/add_tokens (100)                │   3.08μs │    9.00w │   0.10x │      1027% │
│ Decoding/BPE decode[100 tokens]       │   3.22μs │   1.02kw │   0.09x │      1076% │
│ Decoding/WordPiece decode[100 tokens] │   3.64μs │   1.07kw │   0.08x │      1215% │
│ Encoding/WordPiece encode[100 chars]  │   6.28μs │   1.76kw │   0.05x │      2096% │
│ Encoding/WordPiece encode[1K chars]   │  11.54μs │   3.16kw │   0.03x │      3852% │
│ Encoding/WordLevel encode[1K chars]   │  17.23μs │   5.54kw │   0.02x │      5751% │
│ Batch/BPE batch[10 items]             │  17.83μs │  14.62kw │   0.02x │      5950% │
│ Batch/WordPiece batch[10 items]       │  32.44μs │   9.04kw │   0.01x │     10829% │
│ Decoding/BPE decode[1K tokens]        │  32.49μs │  10.14kw │   0.01x │     10845% │
│ Decoding/WordPiece decode[1K tokens]  │  38.18μs │  10.01kw │   0.01x │     12745% │
│ Encoding/BPE encode[1K chars]         │  54.07μs │   5.17kw │   0.01x │     18049% │
│ Encoding/WordPiece encode[10K chars]  │ 108.28μs │  30.16kw │   0.00x │     36143% │
│ Encoding/WordLevel encode[10K chars]  │ 177.29μs │  50.42kw │   0.00x │     59178% │
│ Serialization/from_file               │ 214.57μs │   2.52kw │   0.00x │     71623% │
│ Batch/BPE batch[100 items]            │ 257.77μs │ 146.15kw │   0.00x │     86043% │
│ Batch/WordPiece batch[100 items]      │ 332.73μs │  90.39kw │   0.00x │    111063% │
│ Encoding/BPE encode[10K chars]        │ 536.73μs │  50.17kw │   0.00x │    179159% │
└───────────────────────────────────────┴──────────┴──────────┴─────────┴────────────┘