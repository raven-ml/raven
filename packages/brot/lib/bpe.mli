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
(** The type for BPE models. Internally mutable due to the merge cache.

    Several domains may tokenize with the same model: cache entries are
    immutable and published by a single store, and the merge scratch buffers are
    held per domain and claimed by one thread at a time. *)

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
      A non-zero value disables the cache and [ignore_merges], both of which
      would reuse a choice that is drawn per occurrence.
    - [unk_token] is emitted for characters whose affixed form is not in [vocab]
      (when {!byte_fallback} is off). No default.
    - [continuing_subword_prefix] is prepended to every character of a word but
      the first before merging, so continuation subwords carry it. No default;
      [""] is the same as none.
    - [end_of_word_suffix] is appended to the last character of a word before
      merging, so the final subword carries it. A one-character word takes the
      suffix but not the prefix. No default; [""] is the same as none.
    - [fuse_unk], when [true], merges consecutive unknown bytes into a single
      [unk_token] instead of emitting one per byte. Defaults to [false].
    - [byte_fallback], when [true], falls back to byte-level tokens (e.g.
      ["<0xFF>"]) for the bytes of the affixed character, prefix and suffix
      bytes included, instead of emitting [unk_token]. Defaults to [false].
    - [ignore_merges], when [true], emits a word that is itself in [vocab] as
      that single token, without applying any merge; other words are merged as
      usual. Defaults to [false]. *)

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

val tokenize_encoding : t -> string -> type_id:int -> base:int -> Encoding.t
(** [tokenize_encoding t s ~type_id ~base] tokenizes [s] and builds an
    {!Encoding.t} directly, avoiding intermediate list allocation. Offsets count
    from [base], which is where [s] starts in the text being encoded. *)

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

    Words are the space-separated runs of [texts]. Each word starts out as its
    characters, in the form the affixes give them, and the most frequent
    adjacent pair is merged over and over until [vocab_size] is reached, no pair
    is left, or the best pair falls below [min_frequency]. Pairs of equal
    frequency are merged lowest vocabulary id first.

    - [min_frequency] is the number of occurrences a pair needs to be merged.
    - [vocab_size] is the target size, special tokens and alphabet included.
    - [show_progress] is ignored.
    - [special_tokens] take the first ids, in order.
    - [limit_alphabet] caps how many distinct characters are kept; the rarest go
      first, and words drop the characters that did not make the cut.
    - [initial_alphabet] are characters kept whatever their frequency.
    - [continuing_subword_prefix] goes on every character of a word but the
      first and [end_of_word_suffix] on the last, before any pair is counted, so
      merges are learned — and written — over the affixed forms: a model trained
      with [end_of_word_suffix:"</w>"] holds ["w</w>"] and merges ["lo w</w>"].
    - [max_token_length] holds a merge back once the run of characters it would
      join reaches that many. The merges of single characters that open the
      training are not held back.
    - [init] is ignored.

    Returns [(model, special_tokens)]. *)
