# saga

Fast tokenization library for ML in OCaml with comprehensive Unicode support.

## Features

- **Simple API**: Direct functions for common use cases
- **Multiple tokenizers**: Whitespace, character-level, regex-based
- **Unicode support**: Proper text normalization, CJK handling, emoji removal
- **Efficient vocabulary**: O(1) lookups with frequency-based building
- **Nx integration**: Direct tensor encoding/decoding for ML pipelines
- **Error handling**: Consistent error messages with helpful hints

## Installation

```bash
dune build saga/
```

## Quick Start

```ocaml
open Saga

(* Simple tokenization *)
let tokens = tokenize "Hello world!"
(* ["Hello"; "world"; "!"] *)

(* Direct encoding - builds vocab automatically *)
let encoded = encode "hello world hello"
(* [4; 5; 4] *)

(* Batch encoding to tensor *)
let texts = ["Hello world"; "How are you?"]
let tensor = encode_batch texts
(* Returns Nx tensor of shape [2; 512] with padding *)

(* With Unicode normalization *)
let text = "Café RÉSUMÉ"
let normalized = normalize ~lowercase:true ~strip_accents:true text
(* "cafe resume" *)
let tokens = tokenize normalized
```

## Advanced Usage

### Custom Tokenization

```ocaml
(* Regex tokenizer *)
let tokens = tokenize ~method_:(`Regex "\\w+|[^\\w\\s]+") "don't stop!"
(* ["don"; "'"; "t"; "stop"; "!"] *)

(* Character-level with Unicode *)
let tokens = tokenize ~method_:`Chars "Hello 世界"
(* ["H"; "e"; "l"; "l"; "o"; " "; "世"; "界"] *)
```

### Vocabulary Management

```ocaml
(* Build vocabulary with constraints *)
let vocab = vocab ~max_size:10000 ~min_freq:2 all_tokens

(* Save/load vocabulary *)
vocab_save my_vocab "vocab.txt"
let loaded_vocab = vocab_load "vocab.txt"

(* Encode with specific vocabulary *)
let encoded = encode ~vocab:my_vocab "hello unknown world"
(* Unknown words become <unk> token *)
```

### Unicode Text Processing

```ocaml
(* The Unicode module provides advanced text processing *)
open Saga.Unicode

(* Split words with Unicode awareness *)
let words = split_words "Hello世界"
(* ["Hello"; "世"; "界"] - CJK chars split individually *)

(* Check character properties *)
let is_chinese = is_cjk (Uchar.of_int 0x4E00)
(* true *)

(* Clean text *)
let cleaned = clean_text ~remove_control:true "Hello\x00World"
(* "HelloWorld" *)
```

## API Overview

### Core Functions

- `tokenize ?method_ text` - Split text into tokens
- `encode ?vocab text` - Encode text to indices
- `encode_batch ?vocab ?max_len ?pad texts` - Batch encode to tensor
- `decode vocab indices` - Decode indices to text
- `normalize ?lowercase ?strip_accents ?collapse_whitespace text` - Normalize text

### Vocabulary

- `vocab ?max_size ?min_freq tokens` - Build vocabulary
- `vocab_size vocab` - Get vocabulary size
- `vocab_save vocab path` - Save to file
- `vocab_load path` - Load from file

### Advanced Modules

- `Tokenizer` - Custom tokenizers with normalizers
- `Vocab` - Direct vocabulary manipulation
- `Unicode` - Unicode text processing utilities

## Performance

- Fast tokenization using OCaml's native string operations
- Efficient vocabulary with hash tables
- Zero-copy tensor integration
- Parallel-ready batch processing