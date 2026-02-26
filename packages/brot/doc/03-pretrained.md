# Pretrained Tokenizers

Most users start by loading an existing tokenizer rather than building one
from scratch. Brot reads and writes HuggingFace `tokenizer.json` files and
separate vocabulary/merges model files.

## Loading from tokenizer.json

HuggingFace models ship a `tokenizer.json` that contains the algorithm,
vocabulary, merge rules, and full pipeline configuration. Load it with
`from_file`:

<!-- $MDX skip -->
```ocaml
open Brot

let tokenizer = from_file "path/to/tokenizer.json" |> Result.get_ok
let encoding = encode tokenizer "Hello world!"
let ids = Encoding.ids encoding
```

`from_file` returns `(t, string) result`. Handle errors explicitly when
the file may be missing or malformed:

<!-- $MDX skip -->
```ocaml
let tokenizer =
  match Brot.from_file "tokenizer.json" with
  | Ok t -> t
  | Error msg -> failwith msg
```

## Loading from Model Files

Older models ship separate `vocab.json` and `merges.txt` files instead
of a single `tokenizer.json`. Use `from_model_file`:

<!-- $MDX skip -->
```ocaml
open Brot

(* BPE: provide both vocab and merges *)
let tokenizer =
  from_model_file ~vocab:"vocab.json" ~merges:"merges.txt"
    ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
    ~decoder:(Decoder.byte_level ())
    ()

(* WordPiece: vocab only, no merges *)
let tokenizer =
  from_model_file ~vocab:"vocab.txt"
    ~pre:(Pre_tokenizer.bert ())
    ~decoder:(Decoder.wordpiece ())
    ()
```

When `merges` is provided, a BPE tokenizer is created. Without it,
WordPiece is used. The pipeline stages (normalizer, pre-tokenizer,
post-processor, decoder) must be configured explicitly since model files
do not include them.

## Building Known Pipelines

When you need full control over the pipeline or want to understand what
each stage does, build the tokenizer from scratch with an inline
vocabulary. The following examples show the standard configurations for
well-known models.

### BERT (uncased)

BERT uses WordPiece with `##` continuation prefix, BERT normalization
(lowercase, clean text, CJK padding), BERT pre-tokenization (whitespace +
punctuation), and `[CLS]`/`[SEP]` post-processing:

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[PAD]", 0); ("[UNK]", 1); ("[CLS]", 2); ("[SEP]", 3);
        ("the", 4); ("cat", 5); ("sat", 6); ("on", 7); ("mat", 8);
        ("play", 9); ("##ing", 10); ("##ed", 11); ("a", 12);
        ("is", 13); ("good", 14) ]
    ~normalizer:(Normalizer.bert ~lowercase:true ())
    ~pre:(Pre_tokenizer.bert ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 2) ~sep:("[SEP]", 3) ())
    ~decoder:(Decoder.wordpiece ())
    ~specials:(List.map special [ "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]" ])
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

let enc = encode tokenizer "The Cat Is Playing"
let tokens = Encoding.tokens enc
(* [| "[CLS]"; "the"; "cat"; "is"; "play"; "##ing"; "[SEP]" |] *)
let decoded = decode tokenizer ~skip_special_tokens:true (Encoding.ids enc)
(* "the cat is playing" *)
```

### GPT-2

GPT-2 uses BPE with byte-level pre-tokenization (no information loss,
handles any Unicode input) and byte-level decoding:

```ocaml
open Brot

let tokenizer =
  bpe
    ~vocab:
      [ ("H", 0); ("e", 1); ("l", 2); ("o", 3); ("Ġ", 4); ("w", 5);
        ("r", 6); ("d", 7); ("He", 8); ("ll", 9); ("llo", 10);
        ("Hello", 11); ("Ġw", 12); ("or", 13); ("ld", 14);
        ("orld", 15); ("Ġworld", 16) ]
    ~merges:
      [ ("H", "e"); ("l", "l"); ("ll", "o"); ("He", "llo");
        ("Ġ", "w"); ("o", "r"); ("l", "d"); ("or", "ld");
        ("Ġw", "orld") ]
    ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
    ~decoder:(Decoder.byte_level ())
    ()

let enc = encode tokenizer "Hello world"
let tokens = Encoding.tokens enc (* [| "Hello"; "Ġworld" |] *)
let decoded = decode tokenizer (Encoding.ids enc) (* "Hello world" *)
```

### SentencePiece-style (T5, ALBERT)

SentencePiece models use Unigram with metaspace pre-tokenization (spaces
replaced by a visible marker) and metaspace decoding:

```ocaml
open Brot

let tokenizer =
  unigram
    ~vocab:
      [ ("<unk>", -1.0); ("\xe2\x96\x81", -2.0);
        ("\xe2\x96\x81the", -1.5); ("\xe2\x96\x81cat", -1.8);
        ("\xe2\x96\x81is", -1.6); ("\xe2\x96\x81play", -2.0);
        ("ing", -2.5); ("\xe2\x96\x81a", -1.4); ("\xe2\x96\x81good", -2.1) ]
    ~pre:(Pre_tokenizer.metaspace ~replacement:'\xe2' ())
    ~decoder:(Decoder.metaspace ~replacement:'\xe2' ())
    ~unk_token:"<unk>" ()

let enc = encode tokenizer "the cat is playing"
```

## Saving Tokenizers

Save a tokenizer in HuggingFace format for later use or sharing:

<!-- $MDX skip -->
```ocaml
(* Save as tokenizer.json (full pipeline) *)
Brot.save_pretrained tokenizer ~path:"./my_tokenizer"

(* Save just the vocabulary and merges files *)
let files = Brot.save_model_files tokenizer ~folder:"./model" ()

(* Export BPE merges in tiktoken format *)
Brot.export_tiktoken tokenizer
  ~merges_path:"./tiktoken_merges.txt"
  ~vocab_path:"./tiktoken_vocab.txt"
```

## Training from Scratch

Train a tokenizer from a text corpus. Configure the full pipeline
alongside the training parameters:

```ocaml
open Brot

let tokenizer =
  train_bpe
    ~vocab_size:120
    ~min_frequency:1
    ~show_progress:false
    ~pre:(Pre_tokenizer.whitespace ())
    ~specials:(List.map special [ "[PAD]"; "[UNK]" ])
    ~unk_token:"[UNK]" ~pad_token:"[PAD]"
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog.";
         "Machine learning models need good tokenizers.";
         "Subword tokenization handles unknown words gracefully.";
         "The fox jumped over the lazy dog again.";
         "Tokenizers convert text to numerical representations." ]))

let size = vocab_size tokenizer
let enc = encode tokenizer "The quick fox"
```

See [Choosing an Algorithm](05-algorithms/) for guidance on which algorithm
to train and how to tune parameters like `vocab_size`, `min_frequency`,
and algorithm-specific options.
