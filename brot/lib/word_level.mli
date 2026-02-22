(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Word-level tokenization model.

    {b Internal module.} Direct vocabulary lookup with no subword splitting.
    Each input word is mapped to a single token ID via exact string match. Words
    not in the vocabulary are replaced by [unk_token]. *)

type t
(** The type for word-level models. *)

(** {1:creation Creation} *)

val create : ?vocab:(string * int) list -> ?unk_token:string -> unit -> t
(** [create ?vocab ?unk_token ()] is a word-level model.

    {ul
    {- [vocab] is the initial vocabulary as [(token, id)] pairs.
       Defaults to [[]].}
    {- [unk_token] is the token emitted for unknown words.
       Defaults to {["[UNK]"]}.}} *)

(** {1:tokenization Tokenization} *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize t s] is [[(id, token, (start, stop))]] for [s]. If [s] is not in
    the vocabulary, [unk_token] is used. If [unk_token] itself is not in the
    vocabulary, the empty list is returned. *)

val tokenize_ids : t -> string -> int array
(** [tokenize_ids t s] is like {!tokenize} but returns only token IDs. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * int) list
(** [get_vocab t] is the vocabulary as [(token, id)] pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

val add_tokens : t -> string list -> int
(** [add_tokens t toks] adds [toks] to the vocabulary, assigning consecutive IDs
    starting after the current maximum. Returns the number of new tokens
    actually added (duplicates are skipped). Mutates [t]. *)

(** {1:serialization Serialization} *)

val save : t -> folder:string -> unit -> string list
(** [save t ~folder ()] writes [wordlevel.json] to [folder]. The file contains
    the vocabulary and [unk_token] in JSON format. Returns the list of created
    filenames. *)

(** {1:training Training} *)

val train :
  vocab_size:int ->
  min_frequency:int ->
  show_progress:bool ->
  special_tokens:string list ->
  string list ->
  t option ->
  t * string list
(** [train ~vocab_size ~min_frequency ~show_progress ~special_tokens texts init]
    learns a vocabulary from [texts] by counting word frequencies.

    - [vocab_size] is the target vocabulary size.
    - [min_frequency] is the minimum word frequency to include.
    - [show_progress] enables progress output on [stderr].
    - [special_tokens] are added to the vocabulary first.
    - [init], when provided, seeds the vocabulary from an existing model.

    Returns [(model, special_tokens)]. *)
