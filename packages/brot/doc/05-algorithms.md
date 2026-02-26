# Choosing a Tokenization Algorithm

Brot supports 5 tokenization algorithms. The three subword algorithms
(BPE, WordPiece, Unigram) handle open vocabulary by splitting rare words
into smaller pieces. Word-level and character-level are simpler
alternatives.

## BPE (Byte Pair Encoding)

BPE starts with individual characters and iteratively merges the most
frequent adjacent pairs. The merge rules, learned during training, define
how text is split. Used by GPT-2, GPT-3/4, RoBERTa, and LLaMA.

Constructor: `Brot.bpe`. Trainer: `Brot.train_bpe`.

Key parameters:
- `vocab_size` — target vocabulary size (default: 30000)
- `min_frequency` — minimum pair frequency for merging (default: 0)
- `dropout` — probability of skipping merges for data augmentation
- `byte_fallback` — use `<0x00>` byte tokens instead of unknown token
- `continuing_subword_prefix` — prefix for non-initial subwords
- `end_of_word_suffix` — suffix marking word boundaries (e.g., `</w>`)

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

let enc = encode tokenizer "hello world"
let tokens = Encoding.tokens enc (* [| "hello"; " "; "world" |] *)
```

Training BPE:

```ocaml
open Brot

let tokenizer =
  train_bpe ~vocab_size:80 ~min_frequency:1 ~show_progress:false
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked at the brown fox";
         "Quick brown foxes are rare and beautiful" ]))

let size = vocab_size tokenizer
let enc = encode tokenizer "The brown fox"
```

## WordPiece

WordPiece uses a greedy longest-match-first algorithm. For each word, it
finds the longest prefix in the vocabulary, then continues with the
remainder prefixed by a continuation marker (default: `##`). Used by BERT,
DistilBERT, and Electra.

Constructor: `Brot.wordpiece`. Trainer: `Brot.train_wordpiece`.

Key parameters:
- `vocab_size` — target vocabulary size (default: 30000)
- `continuing_subword_prefix` — prefix for non-initial subwords (default: `##`)
- `max_input_chars_per_word` — words longer than this become unknown (default: 100)

```ocaml
open Brot

let tokenizer =
  wordpiece
    ~vocab:
      [ ("[UNK]", 0); ("the", 1); ("cat", 2); ("play", 3);
        ("##ing", 4); ("##ed", 5); ("##s", 6); ("un", 7);
        ("##happy", 8); ("##ly", 9) ]
    ~pre:(Pre_tokenizer.whitespace ())
    ~decoder:(Decoder.wordpiece ())
    ~unk_token:"[UNK]" ()

let enc = encode tokenizer "the cat playing unhappily"
let tokens = Encoding.tokens enc
(* [| "the"; "cat"; "play"; "##ing"; "un"; "##happy"; "##ly" |] *)
let decoded = decode tokenizer (Encoding.ids enc)
(* "the cat playing unhappily" *)
```

Training WordPiece:

```ocaml
open Brot

let tokenizer =
  train_wordpiece ~vocab_size:80 ~show_progress:false
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked at the brown fox";
         "Quick brown foxes are rare and beautiful" ]))

let size = vocab_size tokenizer
let enc = encode tokenizer "The brown fox"
```

## Unigram

Unigram uses probabilistic segmentation: given a vocabulary of subwords
with log-probabilities, it finds the segmentation that maximizes the
total likelihood. Training uses the EM algorithm to iteratively prune the
vocabulary. Used by T5, ALBERT, mBART, and XLNet.

Constructor: `Brot.unigram`. Trainer: `Brot.train_unigram`.

Key parameters:
- `vocab_size` — target vocabulary size (default: 8000)
- `shrinking_factor` — fraction of vocabulary to retain per pruning round (default: 0.75)
- `max_piece_length` — maximum subword length (default: 16)
- `n_sub_iterations` — EM sub-iterations per pruning round (default: 2)

Vocabulary entries are `(token, score)` pairs where scores are negative
log probabilities:

```ocaml
open Brot

let tokenizer =
  unigram
    ~vocab:
      [ ("<unk>", 0.0); ("the", -1.0); ("cat", -1.5);
        ("th", -2.0); ("e", -2.5); ("c", -3.0); ("a", -3.0);
        ("t", -3.0); ("at", -2.0); ("he", -2.0);
        ("sat", -1.8); ("on", -1.5) ]
    ~unk_token:"<unk>" ()

let enc = encode tokenizer "the cat sat on"
```

Training Unigram:

```ocaml
open Brot

let tokenizer =
  train_unigram ~vocab_size:60 ~show_progress:false
    (`Seq (List.to_seq
       [ "The quick brown fox jumps over the lazy dog";
         "The dog barked at the brown fox";
         "Quick brown foxes are rare and beautiful" ]))

let size = vocab_size tokenizer
let enc = encode tokenizer "The brown fox"
```

## Word-level

Word-level tokenization maps each word directly to a token ID. No
subword splitting is performed — words not in the vocabulary are replaced
by the unknown token.

Constructor: `Brot.word_level`. Trainer: `Brot.train_wordlevel`.

Best suited for small controlled vocabularies and prototyping. For
production use with open vocabulary, prefer a subword algorithm.

When no pre-tokenizer is specified, `word_level` defaults to
`Pre_tokenizer.whitespace`.

```ocaml
open Brot

let tokenizer =
  word_level
    ~vocab:
      [ ("[UNK]", 0); ("the", 1); ("cat", 2); ("sat", 3);
        ("on", 4); ("a", 5); ("mat", 6) ]
    ~unk_token:"[UNK]" ()

(* Known words get their IDs, unknown words become [UNK] *)
let enc = encode tokenizer "the cat sat on a rug"
let tokens = Encoding.tokens enc
(* [| "the"; "cat"; "sat"; "on"; "a"; "[UNK]" |] *)
let ids = Encoding.ids enc
(* [| 1; 2; 3; 4; 5; 0 |] *)
```

## Character-level

Character-level tokenization maps each byte to a token with ID equal to
its ordinal value. No vocabulary or training is needed.

Constructor: `Brot.chars`.

Useful as a byte-level fallback or for models that operate directly on
characters:

```ocaml
open Brot

let tokenizer = chars ()

let enc = encode tokenizer "Hi!"
let tokens = Encoding.tokens enc (* [| "H"; "i"; "!" |] *)
let ids = Encoding.ids enc       (* [| 72; 105; 33 |] *)
```

## Quick Reference

| Algorithm       | Splitting strategy                        | Typical vocab | Notable models            | Constructor  | Trainer           |
| --------------- | ----------------------------------------- | ------------- | ------------------------- | ------------ | ----------------- |
| BPE             | Iterative merge of frequent pairs         | 30K-50K       | GPT-2, RoBERTa, LLaMA     | `bpe`        | `train_bpe`       |
| WordPiece       | Greedy longest-match with `##` prefix     | 30K           | BERT, DistilBERT, Electra | `wordpiece`  | `train_wordpiece` |
| Unigram         | Probabilistic max-likelihood segmentation | 8K-32K        | T5, ALBERT, mBART, XLNet  | `unigram`    | `train_unigram`   |
| Word-level      | Whole words, no splitting                 | Varies        | Simple models             | `word_level` | `train_wordlevel` |
| Character-level | Each byte is a token                      | 256           | Byte-level models         | `chars`      | —                 |
