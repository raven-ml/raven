(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** WordPiece tokenization model.

    {b Internal module.} Greedy longest-match-first subword decomposition
    against a fixed vocabulary. Used by BERT, DistilBERT, and Electra.

    A word is decomposed left-to-right: at each position the longest vocabulary
    match is consumed. Continuation pieces are prefixed with
    {!get_continuing_subword_prefix} (typically ["##"]). If no subword is found
    at any position the {e entire} word falls back to {!get_unk_token}.

    Vocabulary lookup uses a hybrid trie: dense nodes (more than 16 children)
    use a 256-element flat array for O(1) byte dispatch, sparse nodes use binary
    search on sorted edges. *)

type t
(** The type for WordPiece models. *)

(** {1:creation Creation} *)

val create :
  vocab:(string, int) Hashtbl.t ->
  ?unk_token:string ->
  ?continuing_subword_prefix:string ->
  ?max_input_chars_per_word:int ->
  unit ->
  t
(** [create ~vocab ()] is a WordPiece model backed by [vocab].

    - [unk_token] is the token emitted for words that cannot be decomposed.
      Defaults to ["[UNK]"].
    - [continuing_subword_prefix] is prepended to non-initial subwords. Defaults
      to ["##"].
    - [max_input_chars_per_word] is the UTF-8 character count above which a word
      is replaced by [unk_token] without attempting decomposition. Defaults to
      [100].

    Raises [Invalid_argument] if [unk_token] is not in [vocab]. *)

val from_file : vocab_file:string -> t
(** [from_file ~vocab_file] loads a model from a BERT-style [vocab.txt] file
    (one token per line, ID equals line number). Uses BERT defaults:
    [unk_token = "[UNK]"], [continuing_subword_prefix = "##"],
    [max_input_chars_per_word = 100]. *)

(** {1:tokenization Tokenization} *)

val encode_into :
  t -> Ints.t -> opaque:Ints.t -> string -> pos:int -> len:int -> unit
(** [encode_into t ids ~opaque text ~pos ~len] appends the ids of the word
    [text.\[pos..pos+len)] to [ids]. The range must lie within [text]; it is not
    checked.

    A word longer than {!create}'s [max_input_chars_per_word] in UTF-8
    characters, and one that no run of subwords covers, is the unknown token
    alone, recorded in [opaque] as three integers: its index in [ids], [1], and
    [len], the bytes of the word it stands for. *)

val token_table : t -> string array
(** [token_table t] maps an id to its token string, prefix included. Owned by
    [t]; do not mutate. *)

val len_table : t -> int array
(** [len_table t] maps an id to the number of source bytes an occurrence of it
    accounts for when it is a match of its entry: the entry stripped of the
    continuation prefix. The unknown token spent on a word stands for the word,
    which {!encode_into} records. Owned by [t]; do not mutate. *)

(** {1:vocabulary Vocabulary} *)

val token_to_id : t -> string -> int option
(** [token_to_id t tok] is the ID of [tok] in the vocabulary. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id]. *)

val get_vocab : t -> (string * int) list
(** [get_vocab t] is the vocabulary as [(token, id)] pairs. *)

val get_vocab_size : t -> int
(** [get_vocab_size t] is the number of tokens in the vocabulary. *)

val get_unk_token : t -> string
(** [get_unk_token t] is the unknown token string. *)

val get_continuing_subword_prefix : t -> string
(** [get_continuing_subword_prefix t] is the subword continuation prefix (e.g.
    ["##"]). *)

(** {1:serialization Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> string
(** [save t ~path ()] writes the vocabulary as a plain-text [vocab.txt] file
    (one token per line) to [path]. If [name] is given the file is named
    [{name}-vocab.txt]. Returns the filepath written. *)

(** {1:training Training} *)

val train :
  min_frequency:int ->
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  limit_alphabet:int option ->
  initial_alphabet:string list ->
  continuing_subword_prefix:string ->
  end_of_word_suffix:string option ->
  (string * int) list ->
  t * string list
(** [train ~min_frequency ~vocab_size ~show_progress ~special_tokens
     ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
     ~end_of_word_suffix word_counts] learns a WordPiece vocabulary from
    [word_counts], each word paired with the number of times it occurs, using
    BPE merge training internally.

    - [min_frequency] is the minimum pair frequency to merge.
    - [vocab_size] is the target vocabulary size.
    - [show_progress] is ignored.
    - [special_tokens] are added to the vocabulary first.
    - [limit_alphabet] caps the number of distinct initial characters kept.
    - [initial_alphabet] seeds the character set, one code point per string.
    - [continuing_subword_prefix] is set on the resulting model.
    - [end_of_word_suffix] appended to final subwords if given.

    Returns [(model, special_tokens)]. *)
