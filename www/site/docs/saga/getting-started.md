# Saga

Fast tokenization and text processing for machine learning in OCaml.

## Overview

Saga provides:
- Simple tokenization (word, character, regex-based)
- Vocabulary management with special tokens
- Batch encoding to tensors for ML models
- Unicode-aware text processing

## Quick Start

### Basic Tokenization

Tokenize text into words (default):

```ocaml
let tokens = Saga.tokenize "Hello world!" in
(* ["Hello"; "world!"] *)
```

Character-level tokenization:

```ocaml
let chars = Saga.tokenize ~method_:`Chars "Hi!" in
(* ["H"; "i"; "!"] *)
```

### Encoding Text for ML

Encode text to indices automatically:

```ocaml
let indices = Saga.encode "hello world hello" in
(* [0; 1; 0] *)
```

Batch encoding for multiple texts:

```ocaml
let texts = ["hi there"; "hello world"; "good morning"] in
let tensor = Saga.encode_batch ~max_len:5 ~pad:true texts in
(* Returns int32 tensor of shape [3; 5] with padding *)
```

## Vocabulary Management

### Building Vocabularies

Create vocabulary from texts:

```ocaml
let texts = ["hello world"; "hello there"; "world peace"] in
let vocab = Saga.vocab texts in
Printf.printf "Vocab size: %d\n" (Saga.vocab_size vocab)
```

Control vocabulary size and frequency:

```ocaml
let vocab = Saga.vocab 
  ~max_size:1000      (* Keep top 1000 tokens *)
  ~min_freq:2         (* Tokens must appear at least twice *)
  texts
```

### Special Tokens

Every vocabulary includes special tokens:
- `<pad>`: Padding token (index 0)
- `<unk>`: Unknown token (index 1)
- `<bos>`: Beginning of sequence (index 2)
- `<eos>`: End of sequence (index 3)

### Saving and Loading

```ocaml
(* Save vocabulary *)
Saga.vocab_save vocab "vocab.txt"

(* Load vocabulary *)
let vocab = Saga.vocab_load "vocab.txt"
```

## Text Preprocessing

### Normalization

Clean and normalize text:

```ocaml
let text = "  Hello   WORLD!  " in
let normalized = Saga.normalize 
  ~lowercase:true 
  ~collapse_whitespace:true 
  text in
(* "hello world!" *)
```

Remove accents:

```ocaml
let normalized = Saga.normalize 
  ~strip_accents:true 
  "café naïve" in
(* "cafe naive" *)
```

## Advanced Tokenization

### Custom Tokenizers

Use regex-based tokenization:

```ocaml
open Saga.Tokenizer

let tokenizer = regex "\\w+|[^\\w\\s]+" in
let tokens = run tokenizer "Hello, world!" in
(* ["Hello"; ","; "world"; "!"] *)
```

Get token offsets for alignment:

```ocaml
let tokens_with_offsets = run_with_offsets tokenizer "Hello world" in
(* [("Hello", 0, 5); ("world", 6, 11)] *)
```

### Tokenizer Pipeline

Add normalization to tokenizer:

```ocaml
let tokenizer = 
  words
  |> with_normalizer (Saga.normalize ~lowercase:true) in
let tokens = run tokenizer "Hello WORLD!" in
(* ["hello"; "world!"] *)
```

## Unicode Processing

### Character Classification

```ocaml
open Saga.Unicode

let is_emoji = not (is_word_char (Uchar.of_char '😀'))
let is_chinese = is_cjk (Uchar.of_int 0x4E00)  (* 一 *)
```

### Text Cleaning

Remove control characters and normalize whitespace:

```ocaml
let cleaned = clean_text 
  ~remove_control:true 
  ~normalize_whitespace:true 
  "Hello\x00\tworld" in
(* "Hello world" *)
```

### Unicode Normalization

Apply Unicode normalization forms:

```ocaml
(* Canonical composition (NFC) *)
let normalized = normalize NFC "é" in  (* é as single character *)

(* Remove emoji *)
let text_no_emoji = remove_emoji "Hello 😀 world 🌍!" in
(* "Hello  world !" *)
```

### Word Splitting

Unicode-aware word boundary detection:

```ocaml
let words = split_words "Hello-world 你好世界" in
(* ["Hello"; "-"; "world"; "你"; "好"; "世"; "界"] *)
```

## Working with Vocabularies

### Vocabulary Module

For fine-grained control:

```ocaml
open Saga.Vocab

(* Create empty vocabulary *)
let vocab = create () in

(* Add tokens manually *)
add vocab "hello";
add_batch vocab ["world"; "foo"; "bar"];

(* Query vocabulary *)
let idx = get_index vocab "hello" in  (* Some 4 *)
let token = get_token vocab 4 in      (* Some "hello" *)

(* Access special tokens *)
let pad_token_idx = pad_idx vocab in  (* 0 *)
let unk_token_idx = unk_idx vocab in  (* 1 *)
```
