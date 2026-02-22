# Brot

Brot tokenizes text into token IDs for language models and reverses the
process. It supports BPE, WordPiece, Unigram, word-level, and
character-level algorithms, loads and saves HuggingFace `tokenizer.json`
files, and is 1.3-6x faster than HuggingFace tokenizers on most
benchmarks.

## Features

- **Tokenization algorithms**: BPE, WordPiece, Unigram, word-level, character-level
- **HuggingFace compatible**: load and save `tokenizer.json`, load vocab/merges model files
- **Composable pipeline**: normalizer, pre-tokenizer, post-processor, decoder — each stage independently configurable
- **Rich encoding output**: token IDs, string tokens, byte offsets, attention masks, type IDs, word IDs, special token masks
- **Training**: train BPE, WordPiece, Unigram, and word-level tokenizers from scratch
- **Performance**: 1.3-6x faster than HuggingFace tokenizers (Rust native)

## Quick Start

Build a BPE tokenizer from a vocabulary and merge rules, encode text,
and decode it back:

```ocaml
open Brot

let tokenizer =
  bpe
    ~vocab:
      [ ("h", 0); ("e", 1); ("l", 2); ("o", 3); (" ", 4); ("w", 5);
        ("r", 6); ("d", 7); ("he", 8); ("ll", 9); ("llo", 10);
        ("hello", 11); ("wo", 12); ("rl", 13); ("rld", 14); ("world", 15) ]
    ~merges:
      [ ("h", "e"); ("l", "l"); ("ll", "o"); ("he", "llo");
        ("w", "o"); ("r", "l"); ("rl", "d"); ("wo", "rld") ]
    ()

let encoding = encode tokenizer "hello world"
let ids = Encoding.ids encoding         (* [| 11; 4; 15 |] *)
let tokens = Encoding.tokens encoding   (* [| "hello"; " "; "world" |] *)
let decoded = decode tokenizer ids      (* "hello world" *)
```

Load a pretrained tokenizer from a HuggingFace `tokenizer.json` file:

<!-- $MDX skip -->
```ocaml
open Brot

let tokenizer = from_file "tokenizer.json" |> Result.get_ok
let encoding = encode tokenizer "Hello world!"
let ids = Encoding.ids encoding
```

Train a tokenizer from a text corpus:

```ocaml
open Brot

let tokenizer =
  train_bpe ~vocab_size:100 ~show_progress:false
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked at the fox";
         "Quick brown foxes are rare" ]))

let size = vocab_size tokenizer
let ids = encode_ids tokenizer "The quick fox"
```

## Next Steps

- [Getting Started](01-getting-started/) — encode, decode, pipeline basics, training
- [The Tokenization Pipeline](02-pipeline/) — how the 5 pipeline stages work
- [Pretrained Tokenizers](03-pretrained/) — loading, saving, and building known model pipelines
- [Batch Processing](04-batch-processing/) — padding, truncation, encoding metadata
- [Choosing an Algorithm](05-algorithms/) — BPE vs WordPiece vs Unigram and when to use each
