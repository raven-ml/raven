(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Character-level tokenization model.

    {b Internal module.} Each byte maps to its ordinal value as token ID.
    Stateless: no vocabulary storage, no training. *)

type t
(** The type for character-level models. *)

(** {1:creation Creation} *)

val create : unit -> t
(** [create ()] is a character-level tokenizer. *)

(** {1:tokenization Tokenization} *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize t s] is the tokenization of [s] as
    [(byte_value, char_string, (start, stop))] triples, one per byte. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t s] is the byte value of [s] when [s] is a single byte. *)

val id_to_token : t -> int -> string option
(** [id_to_token t b] is the single-byte string for byte value [b]. *)

val get_vocab : t -> (string * int) list
(** [get_vocab t] is [[]] (no explicit vocabulary). *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is [1114112] (all Unicode code points). *)

(** {1:serialization Serialization} *)

val save : t -> folder:string -> unit -> string list
(** [save t ~folder ()] is [[]] (no files to write). *)
