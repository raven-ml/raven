# Brot Benchmarks

This directory contains micro-benchmarks for the `brot` library.
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
brot/bench/download_data.sh
```

## Running the Benchmarks

### Brot (OCaml)

```bash
dune exec brot/bench/bench_brot.exe -- --gc
```

### tokenizers — Rust native

```bash
cd brot/bench/bench_rust && cargo run --release
```

### tokenizers — Python (Rust FFI)

```bash
uv run --with tokenizers brot/bench/bench_tokenizers.py
```

## Comparison

Wall-clock time per run. Lower is better. Apple M3 Pro, macOS.

| Benchmark                            | Brot (OCaml) | Rust native | Python (Rust FFI) | Brot vs Rust |
| ------------------------------------ | ------------ | ----------- | ----------------- | ------------ |
| **GPT-2** (BPE, 50K vocab)           |              |             |                   |              |
| Encode/short (1KB)                   | 46μs         | 209μs       | 250μs             | **4.5x**     |
| Encode/long (64KB)                   | 5.26ms       | 10.25ms     | 13.27ms           | **1.9x**     |
| Encode/batch_32                      | 1.38ms       | 3.05ms      | 3.91ms            | **2.2x**     |
| Decode/long                          | 1.19ms       | 1.50ms      | 1.58ms            | **1.3x**     |
| **BERT-base** (WordPiece, 30K vocab) |              |             |                   |              |
| Encode/short (1KB)                   | 137μs        | 278μs       | 325μs             | **2.0x**     |
| Encode/long (64KB)                   | 10.87ms      | 13.95ms     | 16.64ms           | **1.3x**     |
| Encode/batch_32                      | 2.06ms       | 2.31ms      | 2.66ms            | **1.1x**     |
| Decode/long                          | 1.25ms       | 7.63ms      | 7.76ms            | **6.1x**     |
| **LLaMA** (BPE, 32K vocab)           |              |             |                   |              |
| Encode/short (1KB)                   | 51μs         | 207μs       | 247μs             | **4.1x**     |
| Encode/long (64KB)                   | 20.15ms      | 13.41ms     | 16.23ms           | 1.5x slower  |
| Encode/batch_32                      | 1.43ms       | 1.56ms      | 1.51ms            | ~par         |
| Decode/long                          | 1.12ms       | 5.02ms      | 5.03ms            | **4.5x**     |

Notes:
- The "Rust native" column calls the `tokenizers` crate directly, no Python FFI.
  Source: `bench_rust/main.rs`.
- Both brot and HF tokenizers use multi-threading for batch encoding (wall < CPU).
- LLaMA has no pre-tokenizer, so the entire text goes through BPE as a single
  sequence — this is where brot's BPE is slower on long inputs.
