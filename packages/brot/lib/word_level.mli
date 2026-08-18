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

    - [vocab] is the initial vocabulary as [(token, id)] pairs. Defaults to
      [[]].
    - [unk_token] is the token emitted for unknown words. Defaults to ["[UNK]"].
*)

(** {1:tokenization Tokenization} *)

val encode_into : t -> Ints.t -> string -> pos:int -> len:int -> unit
(** [encode_into t ids text ~pos ~len] appends the id of the word
    [text.\[pos..pos+len)] to [ids]. The range must lie within [text]; it is not
    checked.

    A word absent from the vocabulary is [unk_token], and nothing at all when
    the vocabulary does not hold that either. *)

val token_table : t -> string array
(** [token_table t] maps an id to its token string. Owned by [t]; do not mutate.
*)

val len_table : t -> int array
(** [len_table t] maps an id to the byte length of its token string, and the
    unknown token to [0], the word it stands for being a property of the text
    rather than of the id. Owned by [t]; do not mutate. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * int) list
(** [get_vocab t] is the vocabulary as [(token, id)] pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

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
  (string * int) list ->
  t * string list
(** [train ~vocab_size ~min_frequency ~show_progress ~special_tokens
     word_counts] learns a vocabulary from [word_counts], each word paired with
    the number of times it occurs.

    - [vocab_size] is the target vocabulary size.
    - [min_frequency] is the minimum word frequency to include.
    - [show_progress] is ignored.
    - [special_tokens] take the first ids, in order; the words are numbered
      after them.

    Returns [(model, special_tokens)]. *)
