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

    Tokenizations are cached per pretoken in a direct-mapped table seeded with
    the whole vocabulary, so a word the vocabulary holds is answered without
    merging. *)

type t
(** The type for BPE models. Immutable after creation.

    Several domains may tokenize with the same model: the merge buffers and the
    pretoken cache are held per (model, domain) and claimed by one thread at a
    time. *)

type vocab = (string, int) Hashtbl.t
(** The type for vocabularies mapping token strings to IDs. *)

type merges = (string * string) list
(** The type for merge rules in priority order (earlier rules have higher
    priority). *)

(** {1:creation Creation} *)

val create :
  vocab:vocab ->
  merges:merges ->
  ?byte_level:bool ->
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

    - [byte_level], when [true], matches pretokens against the {e bytes} each
      vocabulary entry stands for rather than against the entry itself, so a
      byte-level pipeline can hand the model raw text instead of encoding every
      pretoken first. Every entry is decoded through the byte-to-unicode table
      once, at creation; the vocabulary, {!token_to_id}, {!id_to_token},
      {!get_vocab} and {!save} keep speaking the encoded form. Raises
      [Invalid_argument] together with [continuing_subword_prefix] or
      [end_of_word_suffix]. Defaults to [false].
    - [cache_capacity] is the number of slots in the direct-mapped pretoken
      cache, rounded up to a power of two. Each slot costs 32 bytes and each
      domain encoding with the model keeps a table of its own. Defaults to
      [262144] (8 MB). Set to [0] to disable caching. Pretokens over 4096 bytes
      are never cached.
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

val from_files :
  ?byte_level:bool -> vocab_file:string -> merges_file:string -> unit -> t
(** [from_files ~vocab_file ~merges_file ()] loads a BPE model from
    HuggingFace-format files. [byte_level] is as in {!create}.

    - [vocab_file] is a JSON object mapping token strings to integer IDs.
    - [merges_file] is a text file with one space-separated merge pair per line.
      An optional [#version:] header line is skipped. *)

(** {1:tokenization Tokenization} *)

type state
(** The type for a model's encoding scratch: merge buffers and the pretoken
    cache. Every one of them has a single writer, so a state is never shared. *)

val with_state : t -> (state -> 'a) -> 'a
(** [with_state t f] is [f st] for a state [st] held for the duration of the
    call, and released even if [f] raises.

    The calling domain has one state per model, built and seeded when it first
    asks. A thread that finds it already held — another thread of the same
    domain is encoding — gets unshared buffers and no cache instead, so [f] is
    always correct and only sometimes fast. [st] must not escape [f]. *)

val encode_into :
  t -> state -> Ints.t -> opaque:Ints.t -> string -> pos:int -> len:int -> unit
(** [encode_into t st ids ~opaque text ~pos ~len] appends the ids of
    [text.\[pos..pos+len)] to [ids]. The range must lie within [text]; it is not
    checked.

    An id that stands for bytes other than the ones {!len_table} gives it — the
    unknown token for the run it covers, a fallback token merged from the text
    spelling its name — is recorded in [opaque] as three integers: its index in
    [ids], [1], and the number of bytes it stands for.

    This is the throughput path: nothing is allocated per pretoken that hits the
    cache. *)

val token_table : t -> string array
(** [token_table t] maps an id to its token string. Owned by [t]; do not mutate.
*)

val len_table : t -> int array
(** [len_table t] maps an id to the number of source bytes an occurrence of it
    accounts for unless {!encode_into} recorded otherwise: the bytes an entry
    stands for in a byte-level model, and the entry stripped of its affixes
    otherwise. A byte fallback token reads [1]. Owned by [t]; do not mutate. *)

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

val get_dropout : t -> float option
(** [get_dropout t] is the merge dropout probability, if configured. *)

val get_byte_level : t -> bool
(** [get_byte_level t] is [true] iff [t] matches pretokens against the bytes its
    entries stand for rather than against the entries. *)

val get_cache_capacity : t -> int
(** [get_cache_capacity t] is the number of pretoken cache slots asked for at
    creation, [0] when caching is off. *)

val get_fuse_unk : t -> bool
(** [get_fuse_unk t] is [true] iff consecutive unknown bytes are fused into a
    single unknown token. *)

val get_byte_fallback : t -> bool
(** [get_byte_fallback t] is [true] iff a character absent from the vocabulary
    falls back to byte tokens rather than the unknown token. *)

val get_ignore_merges : t -> bool
(** [get_ignore_merges t] is [true] iff a word that is itself in the vocabulary
    is emitted as one token without merging. *)

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
  initial_alphabet:string list ->
  continuing_subword_prefix:string option ->
  end_of_word_suffix:string option ->
  max_token_length:int option ->
  (string * int) list ->
  t * string list
(** [train ~min_frequency ~vocab_size ~show_progress ~special_tokens
     ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
     ~end_of_word_suffix ~max_token_length word_counts] learns BPE merges from
    [word_counts], each word paired with the number of times it occurs.

    Each word starts out as its characters, in the form the affixes give them,
    and the most frequent adjacent pair is merged over and over until
    [vocab_size] is reached, no pair is left, or the best pair falls below
    [min_frequency]. Pairs of equal frequency are merged lowest vocabulary id
    first.

    - [min_frequency] is the number of occurrences a pair needs to be merged.
    - [vocab_size] is the target size, special tokens and alphabet included.
    - [show_progress] is ignored.
    - [special_tokens] take the first ids, in order.
    - [limit_alphabet] caps how many distinct characters are kept; the rarest go
      first, and words drop the characters that did not make the cut.
    - [initial_alphabet] are characters kept whatever their frequency. Each
      string is one code point.
    - [continuing_subword_prefix] goes on every character of a word but the
      first and [end_of_word_suffix] on the last, before any pair is counted, so
      merges are learned — and written — over the affixed forms: a model trained
      with [end_of_word_suffix:"</w>"] holds ["w</w>"] and merges ["lo w</w>"].
    - [max_token_length] holds a merge back once the run of characters it would
      join reaches that many. The merges of single characters that open the
      training are not held back.

    Returns [(model, special_tokens)]. *)
