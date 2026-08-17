# The Tokenization Pipeline

Brot processes text through up to 6 stages, each optional and independently
configurable:

```
text
 │
 ├─ 0. Added tokens     — split the special tokens out of the input
 ├─ 1. Normalizer       — clean and transform text
 ├─ 2. Pre-tokenizer    — split into pieces with byte offsets
 ├─ 3. Algorithm        — map pieces to token IDs (BPE, WordPiece, …)
 ├─ 4. Post-processor   — add special tokens, set type IDs
 └─ 5. Decoder          — reverse the encoding back to text
 │
 ▼
Encoding.t (ids, tokens, offsets, masks, …)
```

Each stage is set when constructing the tokenizer. Omit any stage and it
is skipped.

## Added tokens

The tokens passed as `~added_tokens` are matched in the input ahead of the
pre-tokenizer and the model, and emitted with their own ID. Only the text
around them reaches the rest of the pipeline:

```ocaml
open Brot

let tokenizer =
  bpe
    ~vocab:[ ("a", 0); ("b", 1) ]
    ~added_tokens:[ added_token "<|endoftext|>" ]
    ()

let ids = encode_ids tokenizer "a<|endoftext|>b" (* [| 0; 2; 1 |] *)
```

A token that the model does not already hold is numbered from the end of the
model vocabulary, as above. At a given position the longest token wins, and
earlier positions win over later ones.

`added_token` takes optional parameters that refine matching: `~single_word`
restricts a token to whole words, `~lstrip` and `~rstrip` extend the match
over the neighbouring whitespace, and `~normalized` matches against the
normalized text rather than against the raw input, which is matched first.
`~special:false` marks a token that is matched atomically but never dropped by
`decode ~skip_special_tokens:true`.

A `~bos_token`, `~eos_token` or `~pad_token` the model holds is a special
token in its own right, matched the same way. `~unk_token` is not: it
configures the model's unknown handling and is never matched in the input.

## Normalization

Normalizers transform text before tokenization. They handle lowercasing,
accent removal, Unicode normalization, whitespace cleanup, and
model-specific preprocessing.

Available normalizers:

- **Unicode**: `nfc`, `nfd`, `nfkc`, `nfkd`
- **Text transforms**: `lowercase`, `strip_accents`, `strip`, `replace`, `prepend`
- **Byte-level**: `byte_level` (GPT-2 style byte-to-Unicode mapping)
- **Model-specific**: `bert` (clean text, CJK padding, optional lowercasing and accent stripping)

Compose normalizers with `sequence`:

```ocaml
open Brot

let n =
  Normalizer.sequence
    [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]

let r1 = Normalizer.apply n "Café Résumé" (* "cafe resume" *)
let r2 = Normalizer.apply n "HELLO"        (* "hello" *)
```

The BERT normalizer combines several transforms:

```ocaml
open Brot

let n = Normalizer.bert ~lowercase:true ()
(* Lowercases, cleans control characters, pads CJK *)
let r1 = Normalizer.apply n "Hello World" (* "hello world" *)
let r2 = Normalizer.apply n "Café"        (* "cafe" *)
```

## Pre-tokenization

Pre-tokenizers split text into pieces before the algorithm runs. Each
piece carries byte offsets into the original text. The algorithm then
tokenizes each piece independently.

Available pre-tokenizers:

| Pre-tokenizer         | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `whitespace ()`       | Split on `\w+\|[^\w\s]+` (word chars grouped, non-word grouped) |
| `whitespace_split ()` | Split on whitespace (simplest)                                  |
| `bert ()`             | BERT-style: whitespace + punctuation isolation + CJK separation |
| `byte_level ()`       | GPT-2 style byte-level encoding with regex splitting            |
| `punctuation ()`      | Separate punctuation from alphanumeric content                  |
| `split ~pattern ()`   | Split on a literal string pattern                               |
| `char_delimiter c`    | Split on a single character                                     |
| `digits ()`           | Split on digit boundaries                                       |
| `metaspace ()`        | Replace whitespace with a visible marker (SentencePiece)        |
| `unicode_scripts ()`  | Split on Unicode script boundaries                              |
| `fixed_length n`      | Fixed-size character chunks                                     |

Use `pre_tokenize` to inspect how a pre-tokenizer splits text. It returns
a list of `(piece, (start_offset, end_offset))` pairs:

```ocaml
open Brot

let text = "Hello, world! How's it going?"

let whitespace_pieces =
  Pre_tokenizer.pre_tokenize (Pre_tokenizer.whitespace ()) text
(* [("Hello", (0,5)); (",", (5,6)); ("world", (7,12)); ("!", (12,13)); ...] *)

let bert_pieces =
  Pre_tokenizer.pre_tokenize (Pre_tokenizer.bert ()) text

let punct_pieces =
  Pre_tokenizer.pre_tokenize (Pre_tokenizer.punctuation ()) text
```

Compose pre-tokenizers with `sequence`. Each pre-tokenizer in the chain
processes the pieces from the previous one:

```ocaml
open Brot

let pre =
  Pre_tokenizer.sequence
    [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.digits () ]

let pieces = Pre_tokenizer.pre_tokenize pre "order 42 shipped"
(* [("order", _); ("4", _); ("2", _); ("shipped", _)] *)
```

## Tokenization Algorithms

The algorithm maps pre-tokenized pieces to token IDs using the vocabulary.
Brot supports 5 algorithms:

| Algorithm       | How it splits                               | Notable models                 |
| --------------- | ------------------------------------------- | ------------------------------ |
| BPE             | Iterative merge of most frequent pairs      | GPT-2, GPT-3/4, RoBERTa, LLaMA |
| WordPiece       | Greedy longest-match with `##` prefix       | BERT, DistilBERT, Electra      |
| Unigram         | Probabilistic segmentation (max likelihood) | T5, ALBERT, mBART, XLNet       |
| Word-level      | Whole words, no subword splitting           | Simple models, prototyping     |
| Character-level | Each byte is a token                        | Byte-level fallback            |

See [Choosing an Algorithm](05-algorithms/) for details on each algorithm,
when to use it, and how to configure training.

## Post-processing

Post-processors add special tokens and set type IDs after tokenization.
They handle model-specific requirements like `[CLS]`/`[SEP]` for BERT or
`<s>`/`</s>` for RoBERTa.

Available post-processors:

- `bert ~sep ~cls ()` — `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]`, type IDs 0/1
- `roberta ~sep ~cls ()` — `<s> A </s>` or `<s> A </s> </s> B </s>`, all type IDs 0
- `byte_level ()` — adjust offsets for byte-level encoding
- `template ~single ()` — custom template with `$A`, `$B`, and literal token placeholders
- `sequence processors` — chain multiple post-processors

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("[CLS]", 1); ("[SEP]", 2);
        ("the", 3); ("cat", 4); ("sat", 5); ("how", 6); ("are", 7); ("you", 8) ]
    ~added_tokens:(List.map added_token [ "[UNK]"; "[CLS]"; "[SEP]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

(* Single sentence: [CLS] the cat sat [SEP] *)
let single = encode tokenizer "the cat sat"

(* Sentence pair: [CLS] the cat sat [SEP] how are you [SEP] *)
let pair = encode tokenizer ~pair:"how are you" "the cat sat"
(* type_ids: 0 for first sentence + [CLS]/[SEP], 1 for second + [SEP] *)
let type_ids = Encoding.type_ids pair
```

The `template` post-processor gives full control over the format. Use `$A`
and `$B` as sequence placeholders, and literal token names in brackets.
Append `:N` to set type IDs:

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[BOS]", 0); ("[EOS]", 1); ("hello", 2); ("world", 3) ]
    ~added_tokens:(List.map added_token [ "[BOS]"; "[EOS]" ])
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:
      (Post_processor.template
         ~single:"[BOS]:0 $A:0 [EOS]:0"
         ~pair:"[BOS]:0 $A:0 [EOS]:0 $B:1 [EOS]:1"
         ~special_tokens:[ ("[BOS]", 0); ("[EOS]", 1) ]
         ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer "hello world"
let tokens = Encoding.tokens enc   (* [| "[BOS]"; "hello"; "world"; "[EOS]" |] *)
let type_ids = Encoding.type_ids enc (* [| 0; 0; 0; 0 |] *)
```

## Decoding

Decoders reverse encoding-specific transformations to produce natural text
from token strings. They operate on token *strings* (looked up from the
vocabulary), not IDs.

Decoders fall into two categories:

- **Per-token** — transform each token independently: `bpe`, `byte_fallback`, `metaspace`
- **Collapsing** — process the entire token list as a whole: `byte_level`, `wordpiece`, `replace`, `strip`, `fuse`

This distinction matters when composing with `sequence`: per-token decoders
pass a list of transformed tokens to the next decoder, while collapsing
decoders produce a single result.

Available decoders:

| Decoder                   | Type       | Description                                      |
| ------------------------- | ---------- | ------------------------------------------------ |
| `bpe ()`                  | Per-token  | Strip end-of-word suffix, insert spaces          |
| `byte_fallback ()`        | Per-token  | Convert `<0x41>` hex tokens to bytes             |
| `metaspace ()`            | Per-token  | Convert metaspace markers to spaces              |
| `byte_level ()`           | Collapsing | Reverse GPT-2 byte-to-Unicode encoding           |
| `wordpiece ()`            | Collapsing | Strip `##` prefix, join subwords                 |
| `replace ~pattern ~by ()` | Collapsing | Replace literal pattern in joined text           |
| `strip ()`                | Collapsing | Remove leading/trailing characters               |
| `fuse ()`                 | Collapsing | Concatenate all tokens with no delimiter         |
| `ctc ()`                  | Per-token  | CTC output decoding (deduplication, pad removal) |

```ocaml
open Brot

(* WordPiece decoder: strips ## prefix and joins subwords *)
let wp = Decoder.wordpiece ()
let text = Decoder.decode wp [ "[CLS]"; "play"; "##ing"; "cat"; "##s"; "[SEP]" ]
(* "[CLS] playing cats [SEP]" *)

(* Sequence of decoders *)
let seq = Decoder.sequence [ Decoder.fuse (); Decoder.replace ~pattern:"_" ~by:" " () ]
let text2 = Decoder.decode seq [ "_Hello"; "_world" ]
(* " Hello world" *)
```

When using `Brot.decode`, the tokenizer looks up token strings from the
vocabulary and then applies the configured decoder automatically.

## Complete Example

Here is a complete BERT-style tokenizer using all 5 pipeline stages:

```ocaml
open Brot

let tokenizer =
  wordpiece
    (* 1. Normalizer: lowercase and clean text *)
    ~normalizer:(Normalizer.bert ~lowercase:true ())
    (* 2. Pre-tokenizer: BERT-style splitting *)
    ~pre:(Pre_tokenizer.bert ())
    (* 3. Algorithm: WordPiece with ## prefix *)
    ~vocab:
      [ ("[PAD]", 0); ("[UNK]", 1); ("[CLS]", 2); ("[SEP]", 3);
        ("the", 4); ("cat", 5); ("sat", 6); ("on", 7); ("mat", 8);
        ("play", 9); ("##ing", 10); ("##ed", 11); ("a", 12) ]
    ~added_tokens:(List.map added_token [ "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]" ])
    ~unk_token:"[UNK]" ~pad_token:"[PAD]"
    (* 4. Post-processor: add [CLS] and [SEP] *)
    ~post:(Post_processor.bert ~cls:("[CLS]", 2) ~sep:("[SEP]", 3) ())
    (* 5. Decoder: strip ## and join *)
    ~decoder:(Decoder.wordpiece ())
    ()

(* "The Cat" is normalized to "the cat" before tokenization *)
let enc = encode tokenizer "The Cat Played On A Mat"
let tokens = Encoding.tokens enc
(* [| "[CLS]"; "the"; "cat"; "play"; "##ed"; "on"; "a"; "mat"; "[SEP]" |] *)

(* Decode back, skipping special tokens *)
let text = decode tokenizer ~skip_special_tokens:true (Encoding.ids enc)
(* "the cat played on a mat" *)
```
