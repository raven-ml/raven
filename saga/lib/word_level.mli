(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Word-level tokenization algorithm.

    This module implements simple word-level tokenization where each unique word
    in the vocabulary maps to a single token. No subword splitting occurs;
    out-of-vocabulary words are replaced with an unknown token.

    {1 Algorithm Overview}

    Word-level tokenization performs direct vocabulary lookup: 1. Text is
    assumed pre-tokenized into words (by whitespace or pre-tokenizer) 2. Each
    word is looked up in vocabulary 3. If found: emit token ID 4. If not found:
    emit unknown token ID

    This is the simplest tokenization approach but requires large vocabularies
    for good coverage and cannot handle unseen words well.

    {1 Use Cases}

    - Simple baselines and prototypes
    - Languages with small vocabularies
    - Cases where all inputs are known in advance
    - Legacy systems requiring word-level tokens

    {1 Performance Characteristics}

    - Tokenization: O(n) where n = number of words
    - Vocabulary lookup: O(1) average case (hash table)
    - Memory: O(vocab_size)

    {1 Limitations}

    - Cannot handle OOV words except via unknown token
    - Vocabulary size grows with training corpus size
    - No morphological awareness (treats "run" and "running" as unrelated) *)

type t
(** Word-level tokenization model.

    Contains vocabulary mapping words to IDs and optional unknown token. *)

val create : ?vocab:(string * int) list -> ?unk_token:string -> unit -> t
(** [create ?vocab ?unk_token ()] constructs word-level model.

    @param vocab Initial vocabulary as (word, id) pairs. Default: empty.
    @param unk_token Unknown token string. Default: "[UNK]". *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize model text] encodes text by direct vocabulary lookup.

    Returns list of (id, token_string, (start_offset, end_offset)) tuples.
    Assumes text is already split into words (no automatic splitting). *)

val tokenize_ids : t -> string -> int array
(** [tokenize_ids model text] returns only token IDs, skipping string and offset
    metadata. Faster than [tokenize] when only IDs are needed. *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] looks up token ID. *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] retrieves token string by ID. *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns vocabulary as (token, id) pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns vocabulary size. *)

val add_tokens : t -> string list -> int
(** [add_tokens model tokens] adds new tokens to vocabulary.

    Returns number of tokens added (duplicates skipped). Mutates model. *)

val save : t -> folder:string -> unit -> string list
(** [save model ~folder ()] writes vocabulary to JSON file.

    Returns list of created filenames. *)

val train :
  vocab_size:int ->
  min_frequency:int ->
  show_progress:bool ->
  special_tokens:string list ->
  string list ->
  t option ->
  t * string list
(** [train ~vocab_size ~min_frequency ~show_progress ~special_tokens texts
     existing] learns vocabulary from corpus by frequency counting.

    Selects most frequent words up to vocab_size limit. Words below
    min_frequency are excluded.

    @return (trained model, special tokens list). *)
