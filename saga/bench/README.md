# Saga Benchmarks

This directory contains micro-benchmarks for the `saga` library.
The suite mirrors HuggingFace's `tokenizers` so we can compare wall-clock
throughput for realistic workloads and catch regressions.

## Fixtures

Benchmark inputs live in `./data/`:

- `news_1k.txt`, `wiki_64k.txt`, `code_excerpt.txt` — sample corpora used for
  encoding workloads.
- `gpt2.json` — OpenAI GPT-2 (BPE, 50K vocab, 50K merges)
- `bert_base.json` — Google BERT-base-uncased (WordPiece, 30K vocab)
- `llama.json` — Meta LLaMA (BPE, 32K vocab, 61K merges, no pre-tokenizer)

Download the tokenizer model files:

```bash
saga/bench/download_data.sh
```

## Running the Benchmarks

### Saga (OCaml)

```bash
dune exec saga/bench/bench_saga.exe
```

### tokenizers (Python/Rust FFI)

```bash
uv run --with tokenizers saga/bench/bench_tokenizers.py
```

## Results Saga

```
┌───────────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ GPT-2/Encode/single_short     │  68.98μs │  69.19μs │  15.54kw │   1.00x │       100% │
│ BERT-base/Encode/single_short │ 132.04μs │ 132.03μs │  33.40kw │   0.52x │       191% │
│ LLaMA/Encode/single_short     │ 354.22μs │ 354.08μs │  25.04kw │   0.19x │       514% │
│ GPT-2/Encode/batch_32         │   1.52ms │   5.24ms │  63.10kw │   0.05x │      2206% │
│ GPT-2/Decode/long             │   1.63ms │   1.62ms │ 497.96kw │   0.04x │      2359% │
│ LLaMA/Decode/long             │   1.70ms │   1.70ms │ 310.77kw │   0.04x │      2465% │
│ BERT-base/Decode/long         │   2.20ms │   2.20ms │ 458.64kw │   0.03x │      3195% │
│ BERT-base/Encode/batch_32     │   2.73ms │  13.47ms │ 134.56kw │   0.03x │      3955% │
│ LLaMA/Encode/batch_32         │   4.07ms │  23.08ms │ 101.10kw │   0.02x │      5898% │
│ GPT-2/Encode/single_long      │   5.20ms │   5.19ms │ 760.38kw │   0.01x │      7531% │
│ BERT-base/Encode/single_long  │   9.82ms │   9.80ms │   1.59Mw │   0.01x │     14232% │
│ LLaMA/Encode/single_long      │  20.77ms │  20.74ms │   1.18Mw │   0.00x │     30113% │
└───────────────────────────────┴──────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results HF tokenizers

```
┌───────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                          │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ LLaMA/Encode/single_short     │ 246.99µs │ 246.89µs │ 223.81w │   1.00x │       100% │
│ GPT-2/Encode/single_short     │ 250.09µs │ 249.96µs │ 187.33w │   0.99x │       101% │
│ BERT-base/Encode/single_short │ 325.43µs │ 325.12µs │ 264.58w │   0.76x │       132% │
│ LLaMA/Encode/batch_32         │   1.51ms │  11.80ms │  3.23kw │   0.16x │       611% │
│ GPT-2/Decode/long             │   1.58ms │   1.58ms │  1.02kw │   0.16x │       639% │
│ BERT-base/Encode/batch_32     │   2.66ms │  19.17ms │  3.14kw │   0.09x │      1079% │
│ GPT-2/Encode/batch_32         │   3.91ms │  31.18ms │  2.99kw │   0.06x │      1584% │
│ LLaMA/Decode/long             │   5.03ms │   5.02ms │  2.28kw │   0.05x │      2036% │
│ BERT-base/Decode/long         │   7.76ms │   7.76ms │  2.79kw │   0.03x │      3144% │
│ GPT-2/Encode/single_long      │  13.27ms │  13.26ms │  2.70kw │   0.02x │      5371% │
│ LLaMA/Encode/single_long      │  16.23ms │  16.22ms │  3.18kw │   0.02x │      6572% │
│ BERT-base/Encode/single_long  │  16.64ms │  16.63ms │  3.04kw │   0.01x │      6739% │
└───────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```

## Comparison

| Benchmark                            | Saga      | HF tokenizers | Ratio           |
| ------------------------------------ | --------- | ------------- | --------------- |
| **GPT-2** (BPE, 50K vocab)           |           |               |                 |
| Encode/short (1KB)                   | 68.98μs   | 250.09μs      | **3.6x faster** |
| Encode/long (64KB)                   | 5.20ms    | 13.27ms       | **2.6x faster** |
| Encode/batch_32                      | 1.52ms    | 3.91ms        | **2.6x faster** |
| Decode/long                          | 1.63ms    | 1.58ms        | ~par            |
| **BERT-base** (WordPiece, 30K vocab) |           |               |                 |
| Encode/short (1KB)                   | 132.04μs  | 325.43μs      | **2.5x faster** |
| Encode/long (64KB)                   | 9.82ms    | 16.64ms       | **1.7x faster** |
| Encode/batch_32                      | 2.73ms    | 2.66ms        | ~par            |
| Decode/long                          | 2.20ms    | 7.76ms        | **3.5x faster** |
| **LLaMA** (BPE, 32K vocab)           |           |               |                 |
| Encode/short (1KB)                   | 354.22μs  | 246.99μs      | 1.4x slower     |
| Encode/long (64KB)                   | 20.77ms   | 16.23ms       | 1.3x slower     |
| Encode/batch_32                      | 4.07ms    | 1.51ms        | 2.7x slower     |
| Decode/long                          | 1.70ms    | 5.03ms        | **3.0x faster** |

Notes:
- Both saga and HF tokenizers use multi-threading for batch encoding (CPU/Run >>
  Wall/Run in both cases).
- LLaMA has no pre-tokenizer, so the entire text goes through BPE as a single
  sequence.
