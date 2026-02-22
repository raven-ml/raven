(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** BPE (Byte Pair Encoding) tokenization model.

    {b Internal module.} Iteratively merges the most frequent adjacent character
    pairs to build a subword vocabulary. Used by GPT-2, GPT-3, and RoBERTa.

    A word is first split into characters, then merge rules are applied in
    priority order (earlier rules have higher priority). Merging continues until
    no more rules apply.

    Tokenized words are cached in a direct-mapped bounded cache for amortized
    performance. *)

type t
(** The type for BPE models. Internally mutable due to the merge cache. *)

type vocab = (string, int) Hashtbl.t
(** The type for vocabularies mapping token strings to IDs. *)

type merges = (string * string) list
(** The type for merge rules in priority order (earlier rules have higher
    priority). *)

(** {1:creation Creation} *)

val create :
  vocab:vocab ->
  merges:merges ->
  ?cache_capacity:int ->
  ?dropout:float ->
  ?unk_token:string ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?fuse_unk:bool ->
  ?byte_fallback:bool ->
  ?ignore_merges:bool ->
  unit ->
  t
(** [create ~vocab ~merges ()] is a BPE model.

    - [cache_capacity] is the number of slots in the direct-mapped word cache.
      Defaults to [10000]. Set to [0] to disable caching. Words longer than 4096
      bytes bypass the cache.
    - [dropout] is the probability of randomly skipping a merge during
      tokenization (BPE-dropout regularization). Defaults to [0.] (no dropout).
    - [unk_token] is emitted for characters not in [vocab] (when
      {!byte_fallback} is off). No default.
    - [continuing_subword_prefix] is prepended to non-initial subwords. No
      default.
    - [end_of_word_suffix] is appended to the final subword of each word. No
      default.
    - [fuse_unk], when [true], merges consecutive unknown bytes into a single
      [unk_token] instead of emitting one per byte. Defaults to [false].
    - [byte_fallback], when [true], falls back to byte-level tokens (e.g.
      ["<0xFF>"]) for characters not in [vocab] instead of emitting [unk_token].
      Defaults to [false].
    - [ignore_merges], when [true], skips the merge step entirely and returns
      raw character-level tokens. Defaults to [false]. *)

val from_files : vocab_file:string -> merges_file:string -> t
(** [from_files ~vocab_file ~merges_file] loads a BPE model from
    HuggingFace-format files.

    - [vocab_file] is a JSON object mapping token strings to integer IDs.
    - [merges_file] is a text file with one space-separated merge pair per line.
      An optional [#version:] header line is skipped. *)

(** {1:tokenization Tokenization} *)

type token = { id : int; value : string; offsets : int * int }
(** The type for tokens. [id] is the vocabulary index, [value] the string
    content, and [offsets] the [(start, stop)] byte span in the source text. *)

val tokenize : t -> string -> token list
(** [tokenize t s] is the BPE tokenization of [s]. *)

val tokenize_ids : t -> string -> int array
(** [tokenize_ids t s] is like {!tokenize} but returns only token IDs. *)

val tokenize_encoding : t -> string -> type_id:int -> Encoding.t
(** [tokenize_encoding t s ~type_id] tokenizes [s] and builds an {!Encoding.t}
    directly, avoiding intermediate list allocation. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * int) list
(** [get_vocab t] is the vocabulary as [(token, id)] pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

val get_unk_token : t -> string option
(** [get_unk_token t] is the unknown token, if configured. *)

val get_continuing_subword_prefix : t -> string option
(** [get_continuing_subword_prefix t] is the subword prefix, if configured (e.g.
    ["##"]). *)

val get_end_of_word_suffix : t -> string option
(** [get_end_of_word_suffix t] is the word-end suffix, if configured (e.g.
    ["</w>"]). *)

val get_merges : t -> (string * string) list
(** [get_merges t] is the merge rules in priority order. *)

(** {1:serialization Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> unit
(** [save t ~path ()] writes the model to [path] as two files:

    - [vocab.json]: a JSON object mapping token strings to IDs.
    - [merges.txt]: merge pairs, one per line, with a [#version: 0.2] header. *)

(** {1:training Training} *)

val train :
  min_frequency:int ->
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  limit_alphabet:int option ->
  initial_alphabet:char list ->
  continuing_subword_prefix:string option ->
  end_of_word_suffix:string option ->
  max_token_length:int option ->
  string list ->
  t option ->
  t * string list
(** [train ~min_frequency ~vocab_size ~show_progress ~special_tokens
     ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
     ~end_of_word_suffix ~max_token_length texts init] learns BPE merges from
    [texts].

    The algorithm counts word frequencies, builds an initial character alphabet,
    then iteratively finds and merges the highest-frequency adjacent pair until
    [vocab_size] is reached or pair frequency drops below [min_frequency].

    - [min_frequency] is the minimum pair frequency to merge.
    - [vocab_size] is the target vocabulary size.
    - [show_progress] enables progress output on [stderr].
    - [special_tokens] are added to the vocabulary first.
    - [limit_alphabet] caps the number of distinct initial characters kept.
    - [initial_alphabet] seeds the character set.
    - [continuing_subword_prefix] is set on the resulting model.
    - [end_of_word_suffix] is set on the resulting model.
    - [max_token_length] limits the byte length of merged tokens.
    - [init], when provided, seeds the vocabulary from an existing model.

    Returns [(model, special_tokens)]. *)
