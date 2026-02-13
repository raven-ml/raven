# Getting Started with Saga

This guide shows you how to use Saga for text processing and tokenization in OCaml.

## Installation

Saga isn't released yet. When it is, you'll install it with:

```bash
opam install saga
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build saga
```

## First Steps

Here's a simple example that actually works with Saga:

```ocaml
open Saga

(* Tokenize text *)
let tokens = tokenize "Hello world! How are you?"
(* ["Hello"; "world"; "!"; "How"; "are"; "you"; "?"] *)

(* Build vocabulary from corpus *)
let corpus = ["Hello world"; "Hello there"; "world peace"] in
let vocab = vocab (List.concat_map tokenize corpus)

(* Encode text to indices *)
let encoded = encode ~vocab "Hello world"
(* [4; 5] *)

(* Batch encode for neural networks *)
let texts = ["Hello world"; "How are you?"] in
let tensor = encode_batch ~vocab ~max_len:10 texts
(* Returns Nx int32 tensor of shape [2; 10] *)
```

## Key Concepts

**Tokenization methods.** Choose between word-level (default), character-level, or regex-based tokenization:
```ocaml
let words = tokenize "Hello world"                    (* Word-level *)
let chars = tokenize ~method_:`Chars "Hi!"           (* Character-level *)
let custom = tokenize ~method_:(`Regex "\\w+") text  (* Regex-based *)
```

**Vocabularies are essential.** Build them from your corpus or load pretrained ones:
```ocaml
let vocab = vocab ~max_size:10000 ~min_freq:2 tokens
```

**Special tokens are automatic.** Every vocabulary includes `<pad>`, `<unk>`, `<bos>`, and `<eos>` at indices 0-3.

**Batch processing returns tensors.** Use `encode_batch` to get Nx tensors ready for ML models:
```ocaml
let tensor = encode_batch ~vocab ~pad:true ~max_len:128 texts
```

## Common Operations

```ocaml
(* Tokenization *)
let tokens = tokenize "Hello world! 你好世界"
let tokens = tokenize ~method_:`Chars "Hello"
let tokens = tokenize ~method_:(`Regex "\\w+|[^\\w\\s]+") text

(* Normalization *)
let clean = normalize ~lowercase:true ~strip_accents:true text
let clean = normalize ~collapse_whitespace:true "  hello   world  "

(* Vocabulary *)
let vocab = vocab tokens
let vocab = vocab ~max_size:30000 ~min_freq:2 tokens
vocab_save vocab "vocab.txt"
let vocab = vocab_load "vocab.txt"

(* Encoding *)
let indices = encode ~vocab text
let tensor = encode_batch ~vocab texts
let tensor = encode_batch ~vocab ~pad:true ~max_len:512 texts

(* Decoding *)
let text = decode vocab indices
let texts = decode_batch vocab tensor

(* Advanced tokenizers *)
let bpe = Bpe.from_files ~vocab:"vocab.json" ~merges:"merges.txt" in
let tokens = Bpe.tokenize bpe text

let wp = Wordpiece.from_files ~vocab:"vocab.txt" in
let tokens = Wordpiece.tokenize wp text
```

## Subword Tokenization

Saga supports modern subword tokenizers used by transformer models:

```ocaml
(* BPE (used by GPT models) *)
let bpe = Bpe.from_files ~vocab:"vocab.json" ~merges:"merges.txt" in
let tokens = Bpe.tokenize bpe "unrecognizable"
(* ["un", "##rec", "##ogn", "##iz", "##able"] *)

(* WordPiece (used by BERT) *)
let wp = Wordpiece.from_files ~vocab:"vocab.txt" in
let tokens = Wordpiece.tokenize wp "unrecognizable"
(* ["un", "##recogniz", "##able"] *)

(* Configure subword tokenizers *)
let config = {
  Bpe.default_config with
  cache_capacity = 50000;
  continuing_subword_prefix = Some "##";
  dropout = Some 0.1;  (* For training *)
} in
let bpe = Bpe.create config
```

## Unicode Support

Handle multilingual text properly:

```ocaml
open Saga.Unicode

(* Clean text *)
let clean = clean_text ~remove_control:true text
let no_emoji = remove_emoji "Hello 👋 World 🌍!"

(* Check character types *)
let is_chinese = is_cjk (Uchar.of_int 0x4E00)
let is_letter = categorize_char (Uchar.of_char 'A') = Letter

(* Split words with Unicode awareness *)
let words = split_words "Hello世界"
(* ["Hello"; "世"; "界"] - CJK chars split individually *)
```

## Next Steps

Check out the [Tokenizers Guide](/docs/saga/tokenizers/) for detailed information about BPE and WordPiece.

For Unicode processing, see the [Unicode Guide](/docs/saga/unicode/).

When Saga is released, full API documentation will be available. For now, the source code in `saga/src/saga.mli` is your best reference.