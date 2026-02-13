# Getting Started with Saga

This guide shows you how to use Saga for text processing and tokenization in OCaml.

## Installation

Saga isn't released yet. When it is, you'll install it with:

<!-- $MDX skip -->
```bash
opam install saga
```

For now, build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build saga
```

## First Steps

Train a word-level tokenizer and use it to encode and decode text:

```ocaml
open Saga

let () =
  (* Train a word-level tokenizer from a corpus *)
  let corpus =
    [ "Hello world"; "How are you"; "Hello again"; "The world is big" ]
  in
  let tokenizer =
    Tokenizer.train_wordlevel ~vocab_size:50 ~show_progress:false
      (`Seq (List.to_seq corpus))
  in

  (* Inspect the vocabulary *)
  Printf.printf "Vocabulary size: %d\n" (Tokenizer.vocab_size tokenizer);

  (* Encode text to token IDs *)
  let ids = Tokenizer.encode_ids tokenizer "Hello world" in
  Printf.printf "Token IDs: ";
  Array.iter (fun id -> Printf.printf "%d " id) ids;
  print_newline ();

  (* Get full encoding with metadata *)
  let encoding = Tokenizer.encode tokenizer "Hello world" in
  let tokens = Encoding.get_tokens encoding in
  Printf.printf "Tokens: ";
  Array.iter (fun t -> Printf.printf "%S " t) tokens;
  print_newline ();

  (* Decode back to text *)
  let text = Tokenizer.decode tokenizer ids in
  Printf.printf "Decoded: %s\n" text
```

## Key Concepts

**Tokenizer types.** Saga supports multiple tokenization algorithms:

```ocaml
open Saga

let () =
  (* Word-level: maps whole words to IDs *)
  let wl =
    Tokenizer.word_level
      ~pre:(Pre_tokenizers.whitespace ())
      ~vocab:[ ("hello", 0); ("world", 1) ]
      ~unk_token:"[UNK]" ()
  in
  let ids = Tokenizer.encode_ids wl "hello world" in
  Printf.printf "Word-level IDs: ";
  Array.iter (fun id -> Printf.printf "%d " id) ids;
  print_newline ();

  (* Character-level: each character is a token *)
  let cl = Tokenizer.chars () in
  let encoding = Tokenizer.encode cl "Hi!" in
  Printf.printf "Char tokens: ";
  Array.iter (fun t -> Printf.printf "%S " t) (Encoding.get_tokens encoding);
  print_newline ()
```

**Rich encoding output.** The `Encoding` module provides token IDs, offsets, attention masks, and more:

```ocaml
open Saga

let () =
  let tokenizer =
    Tokenizer.word_level
      ~pre:(Pre_tokenizers.whitespace ())
      ~vocab:[ ("hello", 0); ("world", 1); ("[UNK]", 2) ]
      ~unk_token:"[UNK]" ()
  in
  let encoding = Tokenizer.encode tokenizer "hello world" in
  Printf.printf "IDs: ";
  Array.iter (fun id -> Printf.printf "%d " id) (Encoding.get_ids encoding);
  print_newline ();
  Printf.printf "Tokens: ";
  Array.iter (fun t -> Printf.printf "%S " t) (Encoding.get_tokens encoding);
  print_newline ();
  Printf.printf "Attention mask: ";
  Array.iter (fun m -> Printf.printf "%d " m)
    (Encoding.get_attention_mask encoding);
  print_newline ()
```

**Training tokenizers.** Train BPE or WordPiece tokenizers from your corpus:

```ocaml
open Saga

let () =
  let texts =
    [
      "The quick brown fox jumps over the lazy dog";
      "The dog barked at the fox";
      "Quick brown foxes are rare";
    ]
  in
  (* Train BPE tokenizer *)
  let bpe =
    Tokenizer.train_bpe ~vocab_size:100 ~show_progress:false
      (`Seq (List.to_seq texts))
  in
  let ids = Tokenizer.encode_ids bpe "The quick fox" in
  Printf.printf "BPE vocabulary size: %d\n" (Tokenizer.vocab_size bpe);
  Printf.printf "BPE token count: %d\n" (Array.length ids)
```

**Configurable pipeline.** Customize normalization, pre-tokenization, and post-processing:

```ocaml
open Saga

(* Build a BERT-style tokenizer *)
let tokenizer =
  Tokenizer.wordpiece
    ~normalizer:
      (Normalizers.sequence
         [ Normalizers.nfd (); Normalizers.lowercase (); Normalizers.strip_accents () ])
    ~pre:(Pre_tokenizers.whitespace ())
    ~post:
      (Processors.bert ~sep:("[SEP]", 102) ~cls:("[CLS]", 101) ())
    ~decoder:(Decoders.wordpiece ())
    ~vocab:[ ("[CLS]", 101); ("[SEP]", 102); ("[UNK]", 100) ]
    ~unk_token:"[UNK]" ()
```

## Loading Pretrained Tokenizers

Load HuggingFace-format tokenizers:

<!-- $MDX skip -->
```ocaml
open Saga

(* From tokenizer.json (HuggingFace format) *)
let tokenizer =
  match Tokenizer.from_file "tokenizer.json" with
  | Ok t -> t
  | Error e -> raise e

(* From vocab + merges files *)
let tokenizer =
  Tokenizer.from_model_file ~vocab:"vocab.json" ~merges:"merges.txt" ()

(* Save a trained tokenizer *)
let () = Tokenizer.save_pretrained tokenizer ~path:"./my_tokenizer"
```

## Batch Encoding

Encode multiple texts at once with padding and truncation:

<!-- $MDX skip -->
```ocaml
open Saga

let tokenizer = (* ... trained or loaded tokenizer ... *)

(* Batch encode with padding *)
let texts = [ "Hello world"; "How are you doing today" ] in
let encodings =
  Tokenizer.encode_batch tokenizer texts
    ~padding:
      { length = `Batch_longest;
        direction = `Right;
        pad_id = Some 0;
        pad_type_id = Some 0;
        pad_token = Some "[PAD]" }
    ~truncation:{ max_length = 512; direction = `Right }
```

## Next Steps

Check out the source code in `saga/lib/saga.mli` for the full API reference.
