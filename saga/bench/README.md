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
│ GPT-2/Encode/single_short     │  90.60μs │  90.53μs │  15.54kw │   1.00x │       100% │
│ BERT-base/Encode/single_short │ 648.37μs │ 648.35μs │ 119.16kw │   0.14x │       716% │
│ GPT-2/Decode/long             │   1.66ms │   1.66ms │ 497.96kw │   0.05x │      1829% │
│ LLaMA/Decode/long             │   1.82ms │   1.82ms │ 310.77kw │   0.05x │      2005% │
│ LLaMA/Encode/single_short     │   1.95ms │   1.95ms │  88.37kw │   0.05x │      2152% │
│ BERT-base/Decode/long         │   2.25ms │   2.24ms │ 458.64kw │   0.04x │      2479% │
│ GPT-2/Encode/batch_32         │   2.37ms │   2.36ms │ 497.06kw │   0.04x │      2611% │
│ GPT-2/Encode/single_long      │   6.17ms │   6.17ms │ 760.38kw │   0.01x │      6811% │
│ BERT-base/Encode/batch_32     │  19.01ms │  18.99ms │   3.81Mw │   0.00x │     20981% │
│ BERT-base/Encode/single_long  │  40.35ms │  40.30ms │   5.79Mw │   0.00x │     44536% │
│ LLaMA/Encode/batch_32         │  64.22ms │  64.19ms │   2.83Mw │   0.00x │     70883% │
│ LLaMA/Encode/single_long      │ 111.44ms │ 111.35ms │   4.16Mw │   0.00x │    122994% │
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

| Benchmark                            | Saga     | HF tokenizers | Ratio           |
| ------------------------------------ | -------- | ------------- | --------------- |
| **GPT-2** (BPE, 50K vocab)           |          |               |                 |
| Encode/short (1KB)                   | 90.60μs  | 250.09μs      | **2.8x faster** |
| Encode/long (64KB)                   | 6.17ms   | 13.27ms       | **2.1x faster** |
| Decode/long                          | 1.66ms   | 1.58ms        | ~par            |
| **BERT-base** (WordPiece, 30K vocab) |          |               |                 |
| Encode/short (1KB)                   | 648.37μs | 325.43μs      | 2.0x slower     |
| Encode/long (64KB)                   | 40.35ms  | 16.64ms       | 2.4x slower     |
| Decode/long                          | 2.25ms   | 7.76ms        | **3.4x faster** |
| **LLaMA** (BPE, 32K vocab)           |          |               |                 |
| Encode/short (1KB)                   | 1.95ms   | 246.99μs      | 7.9x slower     |
| Encode/long (64KB)                   | 111.44ms | 16.23ms       | 6.9x slower     |
| Decode/long                          | 1.82ms   | 5.03ms        | **2.8x faster** |

Notes:
- HF tokenizers batch benchmarks use multi-threading (CPU/Run >> Wall/Run), so
  batch wall-clock times are not directly comparable.
- LLaMA has no pre-tokenizer, so the entire text goes through BPE as a single
  sequence. This is a known weak spot in our BPE implementation for long inputs.
