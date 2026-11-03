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
6-13), Debian 13, OCaml 5.5.2. brot at c51b4b858. gigatoken at cd5ee648c6
(0.11.0), built from source with its release profile by rustc 1.101.0-nightly
(2026-11-02); tokenizers 0.23.3.

- Instructions: callgrind, one warm one-thread pass (a run of three passes
  minus a run of two) over the first 4,983,259 bytes (1,003 documents),
  without `bench_owt`'s summing of the ids. Over the first 19,941,865 bytes
  GPT-2 counts 40.9 and 28.6 per byte: the ratio holds as the corpus grows.
- Wall: the best of three passes over 99,729,892 bytes (20,177 documents).
  The two engines alternate, one process at a time; one-thread runs are
  pinned to the P-cores. The machine was shared: the 1-minute load during
  the pair is shown.

Ratios are brot's cost over the reference's: above 1, brot is slower.

| Pipeline                           | Threads | brot instr/B | ref instr/B | ratio | brot MB/s | ref MB/s | ratio | load  |
| ---------------------------------- | ------- | ------------ | ----------- | ----- | --------- | -------- | ----- | ----- |
| GPT-2, gigatoken                   | 1       | 40.5         | 28.9        | 1.40  | 262.5     | 768.2    | 2.93  | 18-22 |
| GPT-2, gigatoken                   | 8       |              |             |       | 1112      | 3088     | 2.78  | 10-12 |
| LLaMA (Prepend/Replace normalizer) | 1       | 64.9         | 108.7       | 0.60  | 109.6     | 66.2     | 0.60  | 13-16 |
| Mistral (Metaspace)                | 1       | 64.0         | 62.8        | 1.02  | 116.9     | 158.4    | 1.36  | 9-10  |
| BERT-base, HF tokenizers           | 1       | 142.3        | 3221.7      | 0.04  | 115.1     | 4.9      | 0.04  | 6-8   |

- Wall ratios moved with the load: GPT-2 one-thread pairs gave 2.66-2.93,
  eight-thread pairs 2.26-2.78, pinned Mistral pairs 1.14-1.50. One
  eight-domain brot run under load 19 fell to 220 MB/s. Unpinned one-thread
  runs of the SentencePiece pipelines, which can land on an E-core, ran up to
  4x slower and are left out.
- GPT-2, one warm pass: brot's kernel (`brot_byte_level_encode_ids32`) is
  52% of its instructions, the batch classifier (`mask_refill`) 17%, major
  GC marking and sweeping 14%, the added-token scan
  (`Added_tokens.candidate`) 8%. Under callgrind's cache and branch model
  brot takes 11x gigatoken's branch mispredicts (1.13M against 0.10M) and
  6x its last-level misses (0.52M against 0.08M), so its wall gap is twice
  its instruction gap.
- LLaMA and Mistral, one warm pass: the SentencePiece merge (`Bpe.go` with
  `merge_run`, `sp_load`, `sp_merge_short`) is 49% of brot's instructions,
  the `Replace` normalizer (`replace_byte`, `count_byte`) 25%, major GC 10%,
  the added-token scan 5-6%. On Mistral brot matches gigatoken's
  instructions and simulated last-level misses with fewer mispredicts, so
  callgrind does not show where its wall gap comes from.
