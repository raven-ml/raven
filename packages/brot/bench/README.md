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
6-13), Debian 13, OCaml 5.5.2. brot at f988690df. gigatoken at cd5ee648c6
(0.11.0), built from source with its release profile by rustc 1.101.0-nightly
(2026-11-02); tokenizers 0.23.3.

- Counts: callgrind with its cache and branch models, one thread, over the
  first 4,983,259 bytes (1,003 documents), without `bench_owt`'s summing of
  the ids. brot: the mean of eight warm passes (a run of ten passes minus a
  run of two), since the collector's cycles are whole and fall in some passes
  only; four passes left GPT-2's count 1.4 instructions per byte apart between
  two builds whose kernels counted the same. The reference: one warm pass
  (three minus two). BERT-base's row is from 64e0c2a22.
- Wall: the best of three passes over 99,729,892 bytes (20,177 documents).
  The two engines alternate, one process at a time; one-thread runs are
  pinned to the P-cores. The machine was shared: the 1-minute load during
  the pair is shown, and a pair under one load can disagree with the next by
  a factor of two. Two eight-thread GPT-2 pairs under load 16-24 disagreed by
  a factor of seven and are left out.

Ratios are brot's cost over the reference's: above 1, brot is slower.

| Pipeline                           | brot instr/B | ref instr/B | ratio | brot mispredicts/KB | ref mispredicts/KB | ratio | brot LL misses/KB | ref LL misses/KB | ratio |
| ---------------------------------- | ------------ | ----------- | ----- | ------------------- | ------------------ | ----- | ----------------- | ---------------- | ----- |
| GPT-2, gigatoken                   | 32.1         | 28.8        | 1.11  | 41.8                | 20.2               | 2.07  | 48.9              | 16.3             | 3.00  |
| LLaMA (Prepend/Replace normalizer) | 43.9         | 109.2       | 0.40  | 200.1               | 1516.5             | 0.13  | 94.9              | 156.0            | 0.61  |
| Mistral (Metaspace)                | 43.2         | 62.4        | 0.69  | 191.1               | 574.2              | 0.33  | 89.2              | 165.2            | 0.54  |
| Llama 3 (cl100k), brot alone       | 35.7         |             |       | 63.6                |                    |       | 60.5              |                  |       |
| BERT-base, HF tokenizers           | 140.6        | 3221.7      | 0.04  | 894.9               |                    |       | 71.1              |                  |       |

| Pipeline                           | Threads | brot MB/s | ref MB/s | ratio | load  |
| ---------------------------------- | ------- | --------- | -------- | ----- | ----- |
| GPT-2, gigatoken                   | 1       | 474.4     | 816.6    | 1.72  | 24-27 |
| LLaMA (Prepend/Replace normalizer) | 1       | 190.1     | 71.4     | 0.38  | 18-20 |
| Mistral (Metaspace)                | 1       | 203.5     | 145.4    | 0.71  | 13    |
| BERT-base, HF tokenizers           | 1       | 62.7      | 2.7      | 0.04  | 28-29 |

- At 47055829c, before brot read SentencePiece spaces in place of ▁ and
  before its BPE cache grew, LLaMA and Mistral counted 61.1 and 61.2
  instructions per byte, 236.4 and 234.1 mispredictions and 129.7 and 138.9
  last-level misses per KB (eight-pass mean), a quarter of the instructions
  spent marking the spaces with ▁ (`replace_byte`, `count_byte`) and most of
  the misses on the two copies of each document that marking made. GPT-2
  counted 32.1 instructions per byte, as now: its path did not change. Its
  one-thread wall pair then gave 229.4 against 804.9 MB/s, LLaMA's 107.5
  against 66.5 and Mistral's 65.0 against 62.8.
- GPT-2 over the 100 MB corpus, which holds 320k distinct pretokens of up to
  15 bytes: with the 8 MB cache, brot evicted and merged again what
  gigatoken's table kept. Grown, the cache peaks at 32 MB, and a pass after
  the first takes 0.195-0.20 s of user time where the 8 MB cache took
  0.305-0.31 (gigatoken 0.115-0.14), under load 13-37. The first pass, whose
  misses are mostly a pretoken's first occurrence, moves within the noise.
- Pages: gigatoken asks for 2 MB pages for its pretoken table
  (`MADV_HUGEPAGE`), and with fudge's transparent huge pages set to
  `defrag=madvise` gets them. brot takes its large blocks (the pretoken table,
  each batch's sinks and result) from `malloc`, so the allocator's settings and
  the kernel's transparent huge pages decide their pages and no brot code does:
  under glibc's defaults on fudge under load they land on 4 KB pages
  (`AnonHugePages` 0), so a probe can miss the TLB, which callgrind does not
  model. Over the 100 MB corpus under load 32-39, gigatoken with huge pages
  turned off spent 7-17% more user time per Mistral pass, and brot with its
  blocks of 4 MB or more on huge pages (an `LD_PRELOAD` experiment) 1-7% less.
  With that, brot's Mistral passes took 0.76-0.79 s of user time against
  gigatoken's 0.76-0.87.
- Allocator: the tables above ran under glibc's defaults, with no
  `GLIBC_TUNABLES`. A comparison states the settings it ran under, since they
  decide the share of page faults: `encode_batch_ids` over the 100 MB corpus
  on one thread, GPT-2 (1235428d3), the faults of passes 2, 3 and 4, with the
  same ids in each row. The corpus's 90 MB result alone is 22,120 4 KB pages.

  | `GLIBC_TUNABLES`                                                 | Faults per pass          |
  | ---------------------------------------------------------------- | ------------------------ |
  | none                                                             | 25,573 / 29,354 / 11,506 |
  | `glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=4294967296` | 2,036 / 97 / 0           |
  | `glibc.malloc.hugetlb=1`                                         | 19,104 / 2,998 / 775     |

- GPT-2, eight warm passes: the kernel (`brot_byte_level_encode_ids32`, the
  batch walk `harvest`, the merge of misses) is 84% of brot's instructions,
  major GC marking and sweeping 7%, the added-token scan (`skip_one`) 6%.
- Mistral, eight warm passes: the SentencePiece kernel (`Bpe.go` with
  `merge_run`, `sp_load`, `sp_merge_short`) is 76% of brot's instructions, the
  search for a ▁ to read as a space (`index_byte`) 6%, the collector 6% and
  the added-token scan 4%. Of its mispredictions the merge of the cache's
  misses (`merge_run`) is 44% and the unit walk 27%; of its last-level read
  misses the collector's marking is 35% and the added-token scan, which reads
  each document first, 30%.

- Stages, 2026-11-09, GPT-2, one thread, 99,729,892 bytes, a warm pass in
  10^9 TSC cycles (best-of-pass, load 3-8; both engines with their tables on
  huge pages; gigatoken's stages by `rdtsc` in a patched build, its other
  costs the rest of its 0.40):

  | Stage                                | brot  | gigatoken       |
  | ------------------------------------ | ----- | --------------- |
  | Walk, then key and prefetch          | 0.221 | 0.188           |
  | Probe and emit, misses included      | 0.146 | 0.172           |
  | Added-token scan                     | 0.020 | in the rest     |
  | Copy into the result                 | 0.045 | in the rest     |
  | Driver and collector                 | 0.048 | in the rest     |
  | Pass                                 | 0.48  | 0.40            |

  The scan was 0.040 before it took `memchr`; the 0.020 left is reading the
  text from memory. Warm, the 20 MB prefix counts 31.9 instructions per byte
  before and 30.2 after. The copy is 0.019 of `memcpy` and 0.03 to 0.11 of
  first touching the result, by whether the kernel gives it huge pages, which
  moves it from 0.045 to 0.135 between runs. Tried and left out: prefetching
  sets into L2 (`T1`), as gigatoken does, 4-7% slower here; `-march=x86-64-v3`,
  no gain; handing the sink to the caller to skip the copy, at most the 0.019;
  sizing the cache for the batch up front, 0.04 of a first pass's 0.87, too
  small for wall time to show, and it would take the table to its limit for
  text whose pretokens would not have filled it. A first pass takes 0.27-0.32 s
  against gigatoken's 0.19-0.20, where merging the new pretokens is 0.25 of 0.87.
