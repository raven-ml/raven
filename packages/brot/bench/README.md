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
- `roberta_base.json` — FacebookAI RoBERTa-base (BPE, 50K vocab,
  RobertaProcessing)
- `t5_base.json` — Google T5-base (Unigram, 32K vocab, SentencePiece
  `Precompiled` normalizer, Metaspace)
- `mistral.json` — Mistral-7B-v0.1 (BPE with byte fallback, 32K vocab,
  non-splitting Metaspace pre-tokenizer)
- `llama3.json` — Llama 3.2 (byte-level BPE, 128K vocab, `ignore_merges`,
  regular-expression `Split` on the cl100k pattern)
- `qwen2_5.json`, `deepseek_v3.json`, `gpt_oss.json`, `phi4.json` — the other
  regular-expression `Split` tokenizers the parity suite reads: Qwen2.5,
  DeepSeek-V3, gpt-oss (the o200k pattern) and Phi-4

The benchmarks read `gpt2.json`, `bert_base.json`, `llama.json`,
`llama3.json`, `t5_base.json` and `mistral.json`; the others are read by the
HuggingFace parity test, and `test/gen/gen_parity_expected.py` documents the
whole set.

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
uv run packages/brot/bench/bench_tokenizers.py
```

### Throughput over a corpus

`bench_owt.exe` encodes the documents of a corpus split on `<|endoftext|>`
with `encode_batch_ids`; `bench_owt_ref.py` does the same with gigatoken or HF
tokenizers. Both print the id count and sum of each run, which agree when the
outputs do.

```bash
_build/default/packages/brot/bench/bench_owt.exe TOKENIZER.json CORPUS.txt -domains 1
uv run packages/brot/bench/bench_owt_ref.py gt TOKENIZER.json CORPUS.txt st 3
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

## Against gigatoken

`bench_owt.exe` against gigatoken's `Tokenizer.encode_batch` on OpenWebText,
ids equal on every pipeline. gigatoken refuses WordPiece and Unigram models,
so BERT-base runs against HF tokenizers' `encode_batch_fast`, and T5 has no
reference.

Measured 2026-11-03 on fudge: Intel Core Ultra 5 235 (P-cores 0-5, E-cores
6-13), Debian 13, OCaml 5.5.2. brot at 64e0c2a22. gigatoken at cd5ee648c6
(0.11.0), built from source with its release profile by rustc 1.101.0-nightly
(2026-11-02); tokenizers 0.23.3.

- Counts: callgrind with its cache and branch models, one thread, over the
  first 4,983,259 bytes (1,003 documents), without `bench_owt`'s summing of
  the ids. brot: the mean of four warm passes (a run of six passes minus a run
  of two), since the collector's cycles are whole and fall in some passes
  only. The reference: one warm pass (three minus two).
- Wall: the best of three passes over 99,729,892 bytes (20,177 documents).
  The two engines alternate, one process at a time; one-thread runs are
  pinned to the P-cores. The machine was shared: the 1-minute load during
  the pair is shown.

Ratios are brot's cost over the reference's: above 1, brot is slower.

| Pipeline                           | brot instr/B | ref instr/B | ratio | brot mispredicts/KB | ref mispredicts/KB | ratio | brot LL misses/KB | ref LL misses/KB | ratio |
| ---------------------------------- | ------------ | ----------- | ----- | ------------------- | ------------------ | ----- | ----------------- | ---------------- | ----- |
| GPT-2, gigatoken                   | 31.4         | 28.8        | 1.09  | 37.5                | 20.2               | 1.86  | 43.1              | 16.3             | 2.64  |
| LLaMA (Prepend/Replace normalizer) | 61.6         | 109.2       | 0.56  | 238.7               | 1516.5             | 0.16  | 133.6             | 156.0            | 0.86  |
| Mistral (Metaspace)                | 61.2         | 62.4        | 0.98  | 234.7               | 574.2              | 0.41  | 137.8             | 165.2            | 0.83  |
| BERT-base, HF tokenizers           | 140.6        | 3221.7      | 0.04  | 894.9               |                    |       | 71.1              |                  |       |

| Pipeline                           | Threads | brot MB/s | ref MB/s | ratio | load  |
| ---------------------------------- | ------- | --------- | -------- | ----- | ----- |
| GPT-2, gigatoken                   | 1       | 229.4     | 804.9    | 3.51  | 23-31 |
| GPT-2, gigatoken                   | 8       | 502.0     | 975.1    | 1.94  | 56-63 |
| LLaMA (Prepend/Replace normalizer) | 1       | 107.5     | 66.5     | 0.62  | 33-35 |
| Mistral (Metaspace)                | 1       | 65.0      | 62.8     | 0.97  | 49-51 |
| BERT-base, HF tokenizers           | 1       | 62.7      | 2.7      | 0.04  | 28-29 |

- Before 64e0c2a22, at c51b4b858, GPT-2 counted 41.6 instructions per byte,
  228.4 mispredictions and 108.7 last-level misses per KB (four-pass mean): the
  walker popped a span at a time and left every byte beyond ASCII to the
  scalar walker, the cache probe branched on a front table and on each way,
  and each chunk of a batch allocated an ids buffer that kept the collector
  cycling. LLaMA and Mistral counted 64 and 62 instructions and 359 and 351
  mispredictions per KB.
- Wall ratios moved with the load: the second GPT-2 pair, under load 41-43,
  gave 2.27. Over the 4.98 MB prefix, pinned to one P-core under load 31,
  three rounds alternating the two engines gave 0.88-1.42 (589 against 774
  MB/s at best), where c51b4b858 gave 1.70-3.58. The 100 MB corpus holds
  more distinct pretokens than a 262,144-entry cache: brot evicts and merges
  again where gigatoken's table grows, and writes 90 MB of fresh result per
  pass.
- GPT-2, four warm passes: the kernel (`brot_byte_level_encode_ids32`, the
  batch walk `harvest`, the merge of misses) is 87% of brot's instructions,
  major GC marking and sweeping 5%, the added-token scan (`skip_one`) 6%. Of
  its mispredictions the batch walk's per-batch tests (bytes beyond ASCII, an
  apostrophe, the loops' exits) are 27%, the merges of the cache's misses 16%,
  the probe loop 13%, the collector 20% and OCaml's calls per document
  (indirect branches through `caml_c_call` and `caml_apply`) 8%.
- LLaMA and Mistral, one warm pass: the SentencePiece merge (`Bpe.go` with
  `merge_run`, `sp_load`, `sp_merge_short`) is about half of brot's
  instructions and the `Replace` normalizer (`replace_byte`, `count_byte`) a
  quarter.
