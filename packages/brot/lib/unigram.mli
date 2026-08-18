(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Unigram language model tokenization.

    {b Internal module.} Probabilistic subword tokenization using token
    log-probabilities. Used by SentencePiece, AlBERT, T5, and mBART.

    A pretoken is cut into the pieces whose scores add up to the most: every
    entry of the vocabulary that starts at a byte is found through a
    double-array trie, one step per byte whatever the fanout, and the best path
    through them is read off by dynamic programming. Where no entry is a whole
    character, the path may spend an unknown token on it, scored ten below the
    rarest entry of the vocabulary. *)

type t
(** The type for unigram models. Immutable after creation, and shared freely
    between domains: the scratch encoding needs is passed in. *)

(** {1:creation Creation} *)

val create : ?unk_id:int -> ?byte_fallback:bool -> (string * float) list -> t
(** [create vocab] is a unigram model from [(token, log_probability)] pairs,
    identified by their position in the list. The trie is built at creation
    time.

    - [unk_id] is the entry standing for a run of characters the vocabulary does
      not hold, one token for the whole run. Without it, encoding raises
      [Failure] as soon as the best path into some character would spend an
      unknown token on it. Raises [Invalid_argument] when it is not the
      identifier of an entry. No default.
    - [byte_fallback], when [true], spells such a run out as byte tokens (e.g.
      ["<0xFF>"]) when the vocabulary holds one for every byte of it, and falls
      back to the unknown token when it does not. Defaults to [false]. *)

(** {1:tokenization Tokenization} *)

type state
(** The type for encoding scratch: the arrays a pretoken's best path is worked
    out in. A state has a single writer, so it is never shared; it grows to the
    longest pretoken it is asked for and keeps the room. *)

val with_state : (state -> 'a) -> 'a
(** [with_state f] is [f st] for a state [st] held for the duration of the call,
    and released even if [f] raises. The calling domain has one, built when it
    first asks; a thread that finds it already held gets a state of its own.
    [st] must not escape [f]. *)

val encode_into : t -> state -> Ints.t -> string -> pos:int -> len:int -> unit
(** [encode_into t st ids text ~pos ~len] appends the ids of
    [text.\[pos..pos+len)] to [ids]. The range must lie within [text]; it is not
    checked. Nothing is allocated per pretoken.

    Raises [Failure] when the best path into some character would spend an
    unknown token and [t] has none. *)

val token_table : t -> string array
(** [token_table t] maps an id to its token string. Owned by [t]; do not mutate.
*)

val len_table : t -> int array
(** [len_table t] maps an id to the number of source bytes an occurrence of it
    accounts for: the entry itself, [1] for a byte fallback token, and [0] for
    the unknown token, whose length is a property of the text rather than of the
    id. Owned by [t]; do not mutate. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * float) list
(** [get_vocab t] is the vocabulary as [(token, score)] pairs, in id order. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

val get_unk_id : t -> int option
(** [get_unk_id t] is the id of the unknown token, if configured. *)

val get_byte_fallback : t -> bool
(** [get_byte_fallback t] is [true] iff a character absent from the vocabulary
    falls back to byte tokens rather than the unknown token. *)

(** {1:serialization Serialization} *)

val save : t -> folder:string -> unit -> string list
(** [save t ~folder ()] writes [unigram.json] to [folder]. The file contains
    each token with its ID and log-probability in JSON format, together with the
    unknown token's id and whether byte fallback is on. Returns the list of
    created filenames. *)

(** {1:training Training} *)

val train :
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  shrinking_factor:float ->
  unk_token:string option ->
  max_piece_length:int ->
  n_sub_iterations:int ->
  (string * int) list ->
  t * string list
(** [train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
     ~unk_token ~max_piece_length ~n_sub_iterations word_counts] learns a
    unigram model from [word_counts], each word paired with the number of times
    it occurs.

    - [vocab_size] is the target vocabulary size.
    - [show_progress] is ignored.
    - [special_tokens] are added to the vocabulary first.
    - [unk_token] names the entry the model spends on a character it does not
      hold; it is not added to the vocabulary, so it only takes effect when
      [special_tokens] already carries it.
    - [shrinking_factor], [max_piece_length] and [n_sub_iterations] are ignored:
      no EM training is run. The vocabulary is the most frequent words, each
      scored by how often it occurs.

    Returns [(model, special_tokens)]. *)
