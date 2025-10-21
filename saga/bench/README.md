# Saga Benchmarks

This directory contains benchmarks for the `saga` library. We provide comparative benchmarks against HuggingFace's `tokenizers`.

## Results Saga Tokenizers

```
┌────────────────────────────────────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                                   │ Time/Run │  mWd/Run │ Speedup │ vs Fastest │
├────────────────────────────────────────┼──────────┼──────────┼─────────┼────────────┤
│ Serialization/to_json                  │ 477.54ns │  652.00w │   1.00x │       100% │
│ Vocab/add_tokens (100)                 │   3.11μs │    9.00w │   0.15x │       651% │
│ Encoding/WordLevel encode[10K chars]   │  92.75μs │  30.42kw │   0.01x │     19422% │
│ Encoding/WordPiece encode[10K chars]   │ 109.03μs │  30.16kw │   0.00x │     22833% │
│ Serialization/from_file                │ 216.81μs │   2.52kw │   0.00x │     45402% │
│ Batch/BPE batch[100 items]             │ 240.22μs │ 146.15kw │   0.00x │     50305% │
│ Batch/WordPiece batch[100 items]       │ 323.71μs │  90.39kw │   0.00x │     67788% │
│ Decoding/BPE decode[10K tokens]        │ 383.90μs │ 100.01kw │   0.00x │     80391% │
│ Decoding/WordPiece decode[10K tokens]  │ 489.96μs │ 100.01kw │   0.00x │    102601% │
│ Encoding/BPE encode[10K chars]         │ 597.47μs │  50.17kw │   0.00x │    125114% │
│ Encoding/WordLevel encode[100K chars]  │ 915.16μs │ 300.43kw │   0.00x │    191641% │
│ Encoding/WordPiece encode[100K chars]  │   1.13ms │ 300.16kw │   0.00x │    237521% │
│ Batch/BPE batch[1K items]              │   3.02ms │   1.46Mw │   0.00x │    633409% │
│ Batch/WordPiece batch[1K items]        │   3.37ms │ 903.81kw │   0.00x │    706025% │
│ Encoding/BPE encode[100K chars]        │   5.08ms │ 500.17kw │   0.00x │   1063609% │
│ Decoding/BPE decode[100K tokens]       │   6.02ms │   1.00Mw │   0.00x │   1260889% │
│ Decoding/WordPiece decode[100K tokens] │   6.59ms │   1.00Mw │   0.00x │   1380669% │
│ Encoding/WordLevel encode[1M chars]    │   8.70ms │   3.00Mw │   0.00x │   1821629% │
│ Encoding/WordPiece encode[1M chars]    │  11.71ms │   3.00Mw │   0.00x │   2451589% │
│ Encoding/BPE encode[1M chars]          │  52.45ms │   5.00Mw │   0.00x │  10984025% │
└────────────────────────────────────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results HF Tokenizers

```
┌────────────────────────────────────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                   │ Time/Run │ mWd/Run │ Speedup │ vs Fastest │
├────────────────────────────────────────┼──────────┼─────────┼─────────┼────────────┤
│ Serialization/to_json                  │ 824.16ns │   2.60w │   1.00x │       100% │
│ Vocab/add_tokens (100)                 │ 223.06µs │ 750.48w │   0.00x │     27065% │
│ Serialization/from_file                │ 225.30µs │ 750.48w │   0.00x │     27337% │
│ Encoding/WordPiece encode[10K chars]   │ 484.56µs │ 647.16w │   0.00x │     58794% │
│ Encoding/BPE encode[10K chars]         │ 500.77µs │  29.50w │   0.00x │     60762% │
│ Encoding/WordLevel encode[10K chars]   │ 531.24µs │  1.18kw │   0.00x │     64458% │
│ Decoding/BPE decode[10K tokens]        │ 676.79µs │  2.07kw │   0.00x │     82119% │
│ Decoding/WordPiece decode[10K tokens]  │ 750.57µs │  2.47kw │   0.00x │     91071% │
│ Batch/WordPiece batch[100 items]       │   2.03ms │  6.94kw │   0.00x │    246514% │
│ Batch/BPE batch[100 items]             │   2.91ms │  8.28kw │   0.00x │    353612% │
│ Encoding/WordPiece encode[100K chars]  │   4.92ms │  5.48kw │   0.00x │    597163% │
│ Encoding/WordLevel encode[100K chars]  │   5.59ms │ 12.10kw │   0.00x │    677667% │
│ Encoding/BPE encode[100K chars]        │   5.66ms │ 752.90w │   0.00x │    686922% │
│ Decoding/BPE decode[100K tokens]       │   7.25ms │ 18.59kw │   0.00x │    879523% │
│ Decoding/WordPiece decode[100K tokens] │   8.02ms │ 20.33kw │   0.00x │    973542% │
│ Batch/WordPiece batch[1K items]        │  11.21ms │ 32.47kw │   0.00x │   1360464% │
│ Batch/BPE batch[1K items]              │  17.45ms │ 39.86kw │   0.00x │   2116785% │
│ Encoding/WordPiece encode[1M chars]    │  62.38ms │ 26.46kw │   0.00x │   7568359% │
│ Encoding/WordLevel encode[1M chars]    │  67.64ms │ 41.21kw │   0.00x │   8207479% │
│ Encoding/BPE encode[1M chars]          │  68.86ms │ 11.58kw │   0.00x │   8355246% │
└────────────────────────────────────────┴──────────┴─────────┴─────────┴────────────┘
```
