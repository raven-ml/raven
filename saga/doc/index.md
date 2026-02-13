# Saga

Modern text tokenization and processing library for NLP in OCaml.

## Overview

Saga provides state-of-the-art text processing capabilities for machine learning applications, with support for modern tokenization algorithms like BPE and WordPiece, comprehensive Unicode handling, and seamless integration with the Nx tensor library.

## Key Features

- **Modern Tokenizers**: BPE, WordPiece, word-level, and character-level tokenization
- **Unicode-aware**: Proper handling of multilingual text, emoji, and special characters
- **Efficient Vocabulary Management**: Fast token-to-index mapping with frequency-based filtering
- **Tensor Integration**: Direct encoding to Nx tensors for ML pipelines
- **Batch Processing**: Efficient batch encoding with padding and truncation

## Installation

Saga is part of the Raven ecosystem. Install it with:

```bash
opam install saga
```

Or build from source:

```bash
dune build saga/
```

## Quick Example

```ocaml
open Saga

(* Simple tokenization *)
let tokens = tokenize "Hello world! 你好世界"
(* ["Hello"; "world"; "!"; "你好世界"] *)

(* Build vocabulary and encode *)
let vocab = vocab ~max_size:10000 tokens in
let encoded = encode ~vocab "Hello world"
(* [4; 5] *)

(* Batch encode for neural networks *)
let texts = ["Hello world"; "How are you?"] in
let tensor = encode_batch ~vocab ~max_len:128 texts
(* Returns Nx tensor shape [2; 128] *)

(* Use advanced tokenizers *)
let bpe = Bpe.from_files ~vocab:"vocab.json" ~merges:"merges.txt" in
let tokens = Bpe.tokenize bpe "Hello world"
```

## Architecture

Saga is designed with modularity in mind:

- **Core API**: Simple functions for common use cases
- **Tokenizer Module**: Composable tokenizer pipelines
- **Unicode Module**: Low-level Unicode processing utilities
- **BPE Module**: Byte Pair Encoding implementation
- **WordPiece Module**: WordPiece tokenization (BERT-style)
- **Vocab Module**: Vocabulary management and persistence

## When to Use Saga

Use Saga when you need:

- Text preprocessing for NLP models
- Tokenization compatible with pretrained models (BPE, WordPiece)
- Building custom vocabularies from corpora
- Unicode-aware text processing
- Batch encoding text for neural networks

## Next Steps

- [Getting Started](/docs/saga/getting-started/) - Learn the basics
