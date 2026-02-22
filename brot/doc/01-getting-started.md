# Getting Started

This guide covers the basics: encoding text to token IDs, decoding back
to text, configuring the pipeline, and training tokenizers from scratch.

## Installation

<!-- $MDX skip -->
```bash
opam install brot
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build brot
```

## Encoding and Decoding

A tokenizer converts text to token IDs and back. Build one from a
vocabulary and merge rules, then encode and decode:

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

(* Encode text to an Encoding *)
let encoding = encode tokenizer "hello world"
let ids = Encoding.ids encoding       (* [| 11; 4; 15 |] *)
let tokens = Encoding.tokens encoding (* [| "hello"; " "; "world" |] *)

(* Decode back to text *)
let text = decode tokenizer ids (* "hello world" *)
```

`encode` returns an `Encoding.t`. For just the IDs, use `encode_ids`:

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

let ids = encode_ids tokenizer "hello world" (* [| 11; 4; 15 |] *)
```

## Encoding Output

An `Encoding.t` carries more than just token IDs. Every field is a
parallel array of the same length:

- `ids` — integer token IDs for model input
- `tokens` — string representation of each token
- `offsets` — `(start, end)` byte positions in the original text
- `type_ids` — segment IDs (0 for first sentence, 1 for second in pair tasks)
- `attention_mask` — 1 for real tokens, 0 for padding
- `special_tokens_mask` — 1 for special tokens (`[CLS]`, `[SEP]`, padding), 0 for content
- `word_ids` — maps each token to its source word index, or `None` for special tokens

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2);
        ("the", 3); ("cat", 4); ("play", 5); ("##ing", 6) ]
    ~specials:(List.map special [ "[UNK]"; "[CLS]"; "[SEP]" ])
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~pre:(Pre_tokenizer.whitespace ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer "the cat playing"
(* tokens: [| "[CLS]"; "the"; "cat"; "play"; "##ing"; "[SEP]" |] *)
let ids = Encoding.ids enc
let type_ids = Encoding.type_ids enc
let attention_mask = Encoding.attention_mask enc
let special_tokens_mask = Encoding.special_tokens_mask enc
let offsets = Encoding.offsets enc
let word_ids = Encoding.word_ids enc
```

See [Batch Processing](04-batch-processing/) for a deeper look at encoding
metadata, sentence pairs, padding, and truncation.

## The Pipeline

Tokenization proceeds through up to 5 configurable stages:

1. **Normalizer** — text cleanup (lowercase, accent removal, Unicode normalization)
2. **Pre-tokenizer** — split text into pieces with byte offsets
3. **Algorithm** — apply vocabulary-based encoding (BPE, WordPiece, Unigram, etc.)
4. **Post-processor** — add special tokens and set type IDs
5. **Decoder** — reverse the encoding back to text

Each stage is optional. Here is a complete BERT-style pipeline:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~normalizer:(Normalizer.bert ~lowercase:true ())
    ~pre:(Pre_tokenizer.bert ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2); ("[PAD]", 3);
        ("the", 4); ("cat", 5); ("sat", 6); ("on", 7);
        ("play", 8); ("##ing", 9); ("##ed", 10) ]
    ~specials:(List.map special [ "[UNK]"; "[CLS]"; "[SEP]"; "[PAD]" ])
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

(* The normalizer lowercases "The Cat" before tokenization *)
let enc = encode tokenizer "The Cat Sat"
let tokens = Encoding.tokens enc
(* [| "[CLS]"; "the"; "cat"; "sat"; "[SEP]" |] *)

(* Decode, skipping special tokens *)
let text = decode tokenizer ~skip_special_tokens:true (Encoding.ids enc)
(* "the cat sat" *)
```

See [The Tokenization Pipeline](02-pipeline/) for a detailed guide to each
stage.

## Training

Train a tokenizer from a text corpus. Brot supports training BPE,
WordPiece, Unigram, and word-level tokenizers:

```ocaml
open Brot

let tokenizer =
  train_bpe ~vocab_size:80 ~show_progress:false
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked loudly at the brown fox";
         "Quick brown foxes are jumping over lazy dogs";
         "The lazy dog slept while the fox jumped" ]))

let size = vocab_size tokenizer
let enc = encode tokenizer "The quick fox"
```

See [Choosing an Algorithm](05-algorithms/) for guidance on which algorithm
to use and how to configure training.

## Loading Pretrained Tokenizers

Load a HuggingFace `tokenizer.json` file:

<!-- $MDX skip -->
```ocaml
open Brot

let tokenizer = from_file "tokenizer.json" |> Result.get_ok
let encoding = encode tokenizer "Hello world!"
```

Load from separate vocabulary and merges files:

<!-- $MDX skip -->
```ocaml
open Brot

let tokenizer =
  from_model_file ~vocab:"vocab.json" ~merges:"merges.txt"
    ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
    ~decoder:(Decoder.byte_level ())
    ()
```

See [Pretrained Tokenizers](03-pretrained/) for complete pipeline
configurations for BERT, GPT-2, and SentencePiece-style models.

## Batch Processing

Encode multiple texts at once with padding to uniform length:

```ocaml
open Brot

let tokenizer =
  train_bpe ~vocab_size:80 ~show_progress:false
    ~specials:(List.map special [ "[PAD]" ])
    ~pad_token:"[PAD]"
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked loudly at the brown fox";
         "Quick brown foxes are jumping over lazy dogs" ]))

let encodings =
  encode_batch tokenizer
    ~padding:(padding `Batch_longest)
    [ "The quick fox"; "The lazy dog barked" ]

(* All encodings now have the same length *)
let lengths = List.map Encoding.length encodings
```

See [Batch Processing](04-batch-processing/) for padding strategies,
truncation, sentence pairs, and offset alignment.

## Next Steps

- [The Tokenization Pipeline](02-pipeline/) — how the 5 pipeline stages work
- [Pretrained Tokenizers](03-pretrained/) — loading, saving, and building known model pipelines
- [Batch Processing](04-batch-processing/) — padding, truncation, encoding metadata
- [Choosing an Algorithm](05-algorithms/) — BPE vs WordPiece vs Unigram and when to use each
