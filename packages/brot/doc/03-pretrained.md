# Pretrained tokenizers

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

## Loading from model files

Older models ship separate `vocab.json` and `merges.txt` files instead
of a single `tokenizer.json`. Use `from_model_file`:

<!-- $MDX skip -->
```ocaml
open Brot

(* BPE: provide both vocab and merges *)
let tokenizer =
  from_model_file ~vocab:"vocab.json" ~merges:"merges.txt"
    ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
    ~decoder:(Decoder.byte_level)
    ()

(* WordPiece: vocab only, no merges *)
let tokenizer =
  from_model_file ~vocab:"vocab.txt"
    ~pre:(Pre_tokenizer.bert)
    ~decoder:(Decoder.wordpiece ())
    ()
```

When `merges` is provided, a BPE tokenizer is created. Without it,
WordPiece is used. The pipeline stages (normalizer, pre-tokenizer,
post-processor, decoder) must be configured explicitly since model files
do not include them.

## Loading tiktoken encodings

OpenAI's encodings, and Whisper's, ship as tiktoken rank files: one token
per line, the base64 of its bytes and its rank. `of_tiktoken` reads one
with its pattern and the special tokens it numbers by hand; the tiktoken
encodings section of [`brot.mli`](../lib/brot.mli) says how the rank file,
the pattern and the specials are read.

Whisper numbers its special tokens after its ranks, so they are added with
`add_tokens`. It encodes transcripts with no special token matched, and
drops the timestamp tokens, the ids from `<|0.00|>` on, before it decodes:

<!-- $MDX skip -->
```ocaml
let ranks =
  In_channel.with_open_bin "multilingual.tiktoken" In_channel.input_all

let pattern =
  String.concat "|"
    [ "'s"; "'t"; "'re"; "'ve"; "'m"; "'ll"; "'d"; {| ?\p{L}+|};
      {| ?\p{N}+|}; {| ?[^\s\p{L}\p{N}]+|}; {|\s+(?!\S)|}; {|\s+|} ]

(* Whisper's special tokens in the order it numbers them; [languages] is its
   list of language codes. *)
let specials languages =
  [ "<|endoftext|>"; "<|startoftranscript|>" ]
  @ List.map (Printf.sprintf "<|%s|>") languages
  @ [ "<|translate|>"; "<|transcribe|>"; "<|startoflm|>"; "<|startofprev|>";
      "<|nospeech|>"; "<|notimestamps|>" ]
  @ List.init 1501 (fun i -> Printf.sprintf "<|%.2f|>" (float i *. 0.02))

let tokenizer languages =
  let t = Brot.of_tiktoken ~pattern ranks |> Result.get_ok in
  Brot.add_tokens t (List.map Brot.added_token (specials languages))

let encode t text = Brot.encode_ids t ~match_special_tokens:false text

let decode t ids =
  let timestamp_begin = Option.get (Brot.token_to_id t "<|0.00|>") in
  let ids = List.filter (fun id -> id < timestamp_begin) (Array.to_list ids) in
  Brot.decode t (Array.of_list ids)
```

cl100k_base and o200k_base give their specials' ids, which leave gaps, so
they are passed as `~specials`:

<!-- $MDX skip -->
```ocaml
let o200k ~pattern ranks =
  Brot.of_tiktoken ~pattern ranks
    ~specials:[ ("<|endoftext|>", 199999); ("<|endofprompt|>", 200018) ]
```

## Building known pipelines

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
    ~pre:(Pre_tokenizer.bert)
    ~post:(Post_processor.bert ~cls:("[CLS]", 2) ~sep:("[SEP]", 3))
    ~decoder:(Decoder.wordpiece ())
    ~added_tokens:(List.map added_token [ "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]" ])
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
    ~decoder:(Decoder.byte_level)
    ()

let enc = encode tokenizer "Hello world"
let tokens = Encoding.tokens enc (* [| "Hello"; "Ġworld" |] *)
let decoded = decode tokenizer (Encoding.ids enc) (* "Hello world" *)
```

### SentencePiece-style (LLaMA, T5, ALBERT)

SentencePiece models write spaces as a visible marker, `▁` (U+2581). The
normalizer prepends one to the text and turns every space into one; decoding
turns them back into spaces and drops the one that was prepended. This is the
pipeline LLaMA's `tokenizer.json` carries, minus the byte fallback:

```ocaml
open Brot

let tokenizer =
  unigram
    ~vocab:
      [ ("<unk>", -1.0); ("\xe2\x96\x81", -2.0);
        ("\xe2\x96\x81the", -1.5); ("\xe2\x96\x81cat", -1.8);
        ("\xe2\x96\x81is", -1.6); ("\xe2\x96\x81play", -2.0);
        ("ing", -2.5); ("\xe2\x96\x81a", -1.4); ("\xe2\x96\x81good", -2.1) ]
    ~normalizer:
      (Normalizer.sequence
         [ Normalizer.prepend "\xe2\x96\x81";
           Normalizer.replace ~pattern:" " ~replacement:"\xe2\x96\x81" ])
    ~decoder:
      (Decoder.sequence
         [ Decoder.replace ~pattern:"\xe2\x96\x81" ~by:" " ();
           Decoder.strip ~content:" " ~start:1 () ])
    ~unk_token:"<unk>" ()

let enc = encode tokenizer "the cat is playing"
```

## Saving tokenizers

Save a tokenizer in HuggingFace format for later use or sharing:

<!-- $MDX skip -->
```ocaml
(* Save as tokenizer.json (full pipeline) *)
Brot.save_pretrained tokenizer ~path:"./my_tokenizer"

(* Save just the vocabulary and merges files *)
let files = Brot.save_model_files tokenizer ~folder:"./model"
```

## Training from scratch

Train a tokenizer from a text corpus. Configure the full pipeline
alongside the training parameters:

```ocaml
open Brot

let tokenizer =
  train_bpe
    ~vocab_size:120
    ~min_frequency:1
    ~pre:(Pre_tokenizer.whitespace)
    ~added_tokens:(List.map added_token [ "[PAD]"; "[UNK]" ])
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

See [Choosing an Algorithm](05-algorithms.md) for guidance on which algorithm
to train and how to tune parameters like `vocab_size`, `min_frequency`,
and algorithm-specific options.
