# Batch Processing

Real-world usage requires encoding multiple texts into uniform-length
sequences for model input. This guide covers encoding metadata, sentence
pairs, batch encoding, padding, truncation, and offset alignment.

## Encoding Metadata

`Encoding.t` carries parallel arrays that all share the same length. Each
field serves a specific purpose in model input preparation:

| Field                 | Type                | Description                                     |
| --------------------- | ------------------- | ----------------------------------------------- |
| `ids`                 | `int array`         | Token IDs for model input                       |
| `tokens`              | `string array`      | String representation of each token             |
| `offsets`             | `(int * int) array` | `(start, end)` byte positions in source text    |
| `type_ids`            | `int array`         | Segment IDs: 0 for sentence A, 1 for sentence B |
| `attention_mask`      | `int array`         | 1 for real tokens, 0 for padding                |
| `special_tokens_mask` | `int array`         | 1 for special tokens, 0 for content             |
| `word_ids`            | `int option array`  | Source word index, or `None` for special tokens |

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2);
        ("the", 3); ("cat", 4); ("play", 5); ("##ing", 6) ]
    ~specials:(List.map special [ "[UNK]"; "[CLS]"; "[SEP]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer "the cat playing"
(* tokens: [| "[CLS]"; "the"; "cat"; "play"; "##ing"; "[SEP]" |] *)
let ids = Encoding.ids enc
let type_ids = Encoding.type_ids enc
let attention_mask = Encoding.attention_mask enc
let special_tokens_mask = Encoding.special_tokens_mask enc
let offsets = Encoding.offsets enc
let word_ids = Encoding.word_ids enc
(* word_ids: [| None; Some 0; Some 1; Some 2; Some 2; None |]
   "play" and "##ing" share word index 2 *)
```

## Sentence Pairs

Many NLP tasks (question answering, natural language inference, sentence
similarity) operate on pairs of sentences. Use `encode ~pair` to encode
both sequences together:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2);
        ("the", 3); ("cat", 4); ("sat", 5); ("how", 6);
        ("are", 7); ("you", 8) ]
    ~specials:(List.map special [ "[UNK]"; "[CLS]"; "[SEP]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer ~pair:"how are you" "the cat sat"
(* tokens: [| "[CLS]"; "the"; "cat"; "sat"; "[SEP]"; "how"; "are"; "you"; "[SEP]" |] *)
let type_ids = Encoding.type_ids enc
(* [| 0; 0; 0; 0; 0; 1; 1; 1; 1 |] *)
```

Type IDs distinguish the two sentences: 0 for the first sequence
(including `[CLS]` and first `[SEP]`), 1 for the second (including
final `[SEP]`).

## Batch Encoding

Encode multiple texts at once with `encode_batch`, or multiple sentence
pairs with `encode_pairs_batch`:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2);
        ("the", 3); ("cat", 4); ("sat", 5);
        ("how", 6); ("are", 7); ("you", 8); ("good", 9) ]
    ~specials:(List.map special [ "[UNK]"; "[CLS]"; "[SEP]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

(* Batch of single sentences *)
let encodings =
  encode_batch tokenizer [ "the cat"; "the cat sat"; "good" ]
let lengths = List.map Encoding.length encodings
(* [4; 5; 3] — each includes [CLS] and [SEP] *)

(* Batch of sentence pairs *)
let pairs =
  encode_pairs_batch tokenizer
    [ ("the cat sat", "how are you"); ("good", "the cat") ]
```

## Padding

Models require uniform sequence lengths within a batch. Padding extends
shorter sequences with padding tokens. Three strategies are available:

- **`Batch_longest`** — pad to the longest sequence in the batch
- **`Fixed n`** — pad every sequence to exactly `n` tokens
- **`To_multiple n`** — pad to the smallest multiple of `n` that fits

Padding tokens have `attention_mask = 0` and `special_tokens_mask = 1`.

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[PAD]", 0); ("[UNK]", 1); ("the", 2); ("cat", 3);
        ("sat", 4); ("on", 5); ("a", 6); ("mat", 7) ]
    ~specials:(List.map special [ "[PAD]"; "[UNK]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

let texts = [ "the cat"; "the cat sat on a mat"; "cat" ]

(* Pad to longest in batch — all encodings have length 6 *)
let batch1 =
  encode_batch tokenizer ~padding:(padding `Batch_longest) texts

(* Pad to fixed length — all encodings have length 8 *)
let batch2 =
  encode_batch tokenizer ~padding:(padding (`Fixed 8)) texts

(* Pad to multiple of 4 — lengths rounded up to nearest multiple *)
let batch3 =
  encode_batch tokenizer ~padding:(padding (`To_multiple 4)) texts
```

By default, padding is applied to the right. Use `` ~direction:`Left ``
for left-padding, which is common for autoregressive generation:

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[PAD]", 0); ("[UNK]", 1); ("the", 2); ("cat", 3); ("sat", 4) ]
    ~specials:(List.map special [ "[PAD]"; "[UNK]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

let encodings =
  encode_batch tokenizer
    ~padding:(padding ~direction:`Left (`Fixed 5))
    [ "the cat"; "the cat sat" ]
(* tokens: [| "[PAD]"; "[PAD]"; "[PAD]"; "the"; "cat" |]
           [| "[PAD]"; "[PAD]"; "the"; "cat"; "sat" |] *)
```

## Truncation

Truncation limits sequences to a maximum length. Excess tokens are
trimmed from the specified direction:

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[UNK]", 0); ("the", 1); ("quick", 2); ("brown", 3);
        ("fox", 4); ("jumps", 5); ("over", 6) ]
    ~specials:(List.map special [ "[UNK]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~unk_token:"[UNK]" ()

let text = "the quick brown fox jumps over"

(* Truncate from the right (default) *)
let enc_right = encode tokenizer ~truncation:(truncation 4) text
let tokens_right = Encoding.tokens enc_right
(* [| "the"; "quick"; "brown"; "fox" |] *)

(* Truncate from the left *)
let enc_left =
  encode tokenizer ~truncation:(truncation ~direction:`Left 4) text
let tokens_left = Encoding.tokens enc_left
(* [| "brown"; "fox"; "jumps"; "over" |] *)
```

When using a post-processor that adds special tokens, account for the
tokens it adds. Use `Post_processor.added_tokens` to calculate the
budget:

```ocaml
open Brot

let post = Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ()
let added_single = Post_processor.added_tokens post ~is_pair:false (* 2 *)
let added_pair = Post_processor.added_tokens post ~is_pair:true    (* 3 *)
```

## Padding and Truncation Together

The common pattern for model input: truncate long sequences and pad short
ones to a uniform length:

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[PAD]", 0); ("[UNK]", 1); ("the", 2); ("cat", 3);
        ("sat", 4); ("on", 5); ("a", 6); ("mat", 7);
        ("dog", 8); ("ran", 9); ("fast", 10) ]
    ~specials:(List.map special [ "[PAD]"; "[UNK]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

let encodings =
  encode_batch tokenizer
    ~truncation:(truncation 4)
    ~padding:(padding (`Fixed 4))
    [ "the cat sat on a mat"; "the dog ran"; "cat" ]
(* All encodings have exactly 4 tokens.
   Long sequences are truncated, short ones are padded.
   attention_mask distinguishes real tokens (1) from padding (0). *)
let masks = List.map Encoding.attention_mask encodings
```

## Offsets and Alignment

`Encoding.offsets` maps each token back to its `(start, end)` byte span
in the original text. This is useful for tasks like named entity
recognition where you need to extract the source text for each token:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("hello", 1); ("world", 2);
        ("play", 3); ("##ing", 4) ]
    ~pre:(Pre_tokenizer.whitespace ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

let text = "hello playing world"
let enc = encode tokenizer text
let offsets = Encoding.offsets enc
(* offsets.(0) = (0, 5)   -> "hello"
   offsets.(1) = (6, 13)  -> "playing" (start of "play")
   offsets.(2) = (6, 13)  -> "playing" (extent of "##ing")
   offsets.(3) = (14, 19) -> "world" *)

(* Extract source span for a token *)
let start, end_ = offsets.(0)
let source = String.sub text start (end_ - start) (* "hello" *)
```

`Encoding.word_ids` groups subword tokens back to their source word.
Tokens that belong to the same word share the same word index:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("the", 1); ("cat", 2);
        ("play", 3); ("##ing", 4); ("##s", 5) ]
    ~pre:(Pre_tokenizer.whitespace ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer "the cat playing"
let word_ids = Encoding.word_ids enc
(* [| Some 0; Some 1; Some 2; Some 2 |]
   "play" and "##ing" share word index 2,
   indicating they come from the same source word *)
```
