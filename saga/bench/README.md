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
│ GPT-2/Encode/single_short     │  72.41μs │  72.31μs │  15.54kw │   1.00x │       100% │
│ BERT-base/Encode/single_short │ 132.50μs │ 132.50μs │  33.40kw │   0.55x │       183% │
│ LLaMA/Encode/single_short     │ 315.95μs │ 315.55μs │  25.04kw │   0.23x │       436% │
│ GPT-2/Decode/long             │   1.70ms │   1.70ms │ 497.96kw │   0.04x │      2342% │
│ LLaMA/Decode/long             │   1.73ms │   1.72ms │ 310.77kw │   0.04x │      2383% │
│ GPT-2/Encode/batch_32         │   2.27ms │   2.27ms │ 497.06kw │   0.03x │      3133% │
│ BERT-base/Decode/long         │   2.28ms │   2.28ms │ 458.64kw │   0.03x │      3152% │
│ BERT-base/Encode/batch_32     │   4.39ms │   4.38ms │   1.07Mw │   0.02x │      6058% │
│ GPT-2/Encode/single_long      │   5.21ms │   5.20ms │ 760.38kw │   0.01x │      7193% │
│ LLaMA/Encode/batch_32         │   9.17ms │   9.16ms │ 801.06kw │   0.01x │     12663% │
│ BERT-base/Encode/single_long  │   9.76ms │   9.76ms │   1.59Mw │   0.01x │     13484% │
│ LLaMA/Encode/single_long      │  22.01ms │  21.99ms │   1.18Mw │   0.00x │     30394% │
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
| Encode/short (1KB)                   | 72.41μs   | 250.09μs      | **3.5x faster** |
| Encode/long (64KB)                   | 5.21ms    | 13.27ms       | **2.5x faster** |
| Decode/long                          | 1.70ms    | 1.58ms        | ~par            |
| **BERT-base** (WordPiece, 30K vocab) |           |               |                 |
| Encode/short (1KB)                   | 132.50μs  | 325.43μs      | **2.5x faster** |
| Encode/long (64KB)                   | 9.76ms    | 16.64ms       | **1.7x faster** |
| Decode/long                          | 2.28ms    | 7.76ms        | **3.4x faster** |
| **LLaMA** (BPE, 32K vocab)           |           |               |                 |
| Encode/short (1KB)                   | 315.95μs  | 246.99μs      | 1.3x slower     |
| Encode/long (64KB)                   | 22.01ms   | 16.23ms       | 1.4x slower     |
| Decode/long                          | 1.73ms    | 5.03ms        | **2.9x faster** |

Notes:
- HF tokenizers batch benchmarks use multi-threading (CPU/Run >> Wall/Run), so
  batch wall-clock times are not directly comparable.
- LLaMA has no pre-tokenizer, so the entire text goes through BPE as a single
  sequence.
