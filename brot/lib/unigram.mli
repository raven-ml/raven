(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Unigram language model tokenization.

    {b Internal module.} Probabilistic subword tokenization using token
    log-probabilities. Used by SentencePiece, AlBERT, T5, and mBART.

    Tokenization uses greedy longest-prefix matching via a compact trie with
    sorted edges and binary-search dispatch. At each byte position the longest
    vocabulary match is consumed. Unknown single characters default to ID [0].
*)

type t
(** The type for unigram models. *)

(** {1:creation Creation} *)

val create : (string * float) list -> t
(** [create vocab] is a unigram model from [(token, log_probability)] pairs. The
    trie is built at creation time. *)

(** {1:tokenization Tokenization} *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize t s] is the tokenization of [s] as [(id, token, (start, stop))]
    triples. Offsets are byte positions in [s]. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * float) list
(** [get_vocab t] is the vocabulary as [(token, score)] pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

(** {1:serialization Serialization} *)

val save : t -> folder:string -> unit -> string list
(** [save t ~folder ()] writes [unigram.json] to [folder]. The file contains
    each token with its ID and log-probability in JSON format. Returns the list
    of created filenames. *)

(** {1:training Training} *)

val train :
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  shrinking_factor:float ->
  unk_token:string option ->
  max_piece_length:int ->
  n_sub_iterations:int ->
  string list ->
  t option ->
  t * string list
(** [train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
     ~unk_token ~max_piece_length ~n_sub_iterations texts init] learns a unigram
    model from [texts].

    - [vocab_size] is the target vocabulary size.
    - [show_progress] enables progress output on [stderr].
    - [special_tokens] are added to the vocabulary first.
    - [shrinking_factor] controls vocabulary pruning rate.
    - [unk_token] is the unknown token, if any.
    - [max_piece_length] limits the byte length of vocabulary pieces.
    - [n_sub_iterations] is the number of EM sub-iterations.
    - [init], when provided, seeds the vocabulary from an existing model.

    Returns [(model, special_tokens)]. *)
