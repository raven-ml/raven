# Saga

Modern text tokenization and processing library for NLP in OCaml.

## Overview

Saga provides state-of-the-art text processing capabilities for machine learning applications, with support for modern tokenization algorithms like BPE and WordPiece, comprehensive Unicode handling, and seamless integration with the Nx tensor library.

## Key Features

- **Modern Tokenizers**: BPE, WordPiece, Unigram, word-level, and character-level tokenization
- **HuggingFace Compatible**: Load and save tokenizers in HuggingFace JSON format
- **Efficient Vocabulary Management**: Fast token-to-index mapping with frequency-based filtering
- **Configurable Pipeline**: Normalizers, pre-tokenizers, post-processors, and decoders
- **Batch Processing**: Efficient batch encoding with padding and truncation

## Installation

Saga is part of the Raven ecosystem. Install it with:

<!-- $MDX skip -->
```bash
opam install saga
```

Or build from source:

<!-- $MDX skip -->
```bash
dune build saga/
```

## Quick Example

```ocaml
open Saga

let () =
  (* Train a word-level tokenizer from a corpus *)
  let corpus = ["Hello world"; "How are you"; "Hello again"; "world peace"] in
  let tokenizer =
    Tokenizer.train_wordlevel ~vocab_size:20 ~show_progress:false
      (`Seq (List.to_seq corpus))
  in
  Printf.printf "Vocabulary size: %d\n" (Tokenizer.vocab_size tokenizer);

  (* Encode text to token IDs *)
  let ids = Tokenizer.encode_ids tokenizer "Hello world" in
  Printf.printf "Encoded: ";
  Array.iter (fun id -> Printf.printf "%d " id) ids;
  print_newline ();

  (* Decode back to text *)
  let text = Tokenizer.decode tokenizer ids in
  Printf.printf "Decoded: %s\n" text
```

Load a pretrained tokenizer from files:

<!-- $MDX skip -->
```ocaml
(* Load pretrained BPE tokenizer (e.g. GPT-2 format) *)
let tokenizer =
  Tokenizer.from_model_file ~vocab:"vocab.json" ~merges:"merges.txt" ()

let encoding = Tokenizer.encode tokenizer "Hello world"
let ids = Encoding.get_ids encoding
```

## Architecture

Saga follows the HuggingFace Tokenizers design with composable pipeline stages:

- **Tokenizer Module**: Main API for encoding, decoding, and training
- **Normalizers**: Text normalization (lowercase, NFD/NFC, accent stripping)
- **Pre_tokenizers**: Text splitting before vocabulary-based encoding
- **Processors**: Post-processing (adding special tokens like [CLS]/[SEP])
- **Decoders**: Converting token IDs back to text
- **Encoding**: Rich output with token IDs, offsets, attention masks

## When to Use Saga

Use Saga when you need:

- Text preprocessing for NLP models
- Tokenization compatible with pretrained models (BPE, WordPiece)
- Building custom vocabularies from corpora
- Unicode-aware text processing
- Batch encoding text for neural networks

## Next Steps

- [Getting Started](/docs/saga/getting-started/) - Learn the basics
