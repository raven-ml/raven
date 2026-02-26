(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tokenization encodings.

    An encoding bundles token IDs for model input with alignment metadata: byte
    offsets, word indices, segment type IDs, attention masks, and special-token
    flags.

    Encodings are produced by {!Brot.encode} and post-processed with
    {!val-truncate} and {!val-pad}. All parallel arrays ({!val-ids},
    {!val-type_ids}, {!val-tokens}, {!val-word_ids}, {!val-offsets},
    {!val-special_tokens_mask}, {!val-attention_mask}) share the same length,
    equal to {!val-length}. *)

type t
(** The type for tokenization encodings. *)

(** {1:construct Construction} *)

val empty : t
(** [empty] is the encoding with no tokens. *)

val create :
  ids:int array ->
  type_ids:int array ->
  tokens:string array ->
  words:int option array ->
  offsets:(int * int) array ->
  special_tokens_mask:int array ->
  attention_mask:int array ->
  ?overflowing:t list ->
  unit ->
  t
(** [create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
     ~attention_mask ()] is an encoding from the given arrays.

    All arrays must have the same length; no validation is performed.
    [overflowing] defaults to [[]]. *)

val token :
  id:int -> token:string -> offset:int * int -> type_id:int -> special:bool -> t
(** [token ~id ~token ~offset ~type_id ~special] is a single-token encoding.
    When [special] is [true], {!val-special_tokens_mask} is [1] and
    {!val-word_ids} is [None]; otherwise {!val-special_tokens_mask} is [0].
    {!val-attention_mask} is always [1]. *)

val from_tokens : (int * string * (int * int)) list -> type_id:int -> t
(** [from_tokens tokens ~type_id] is an encoding from a list of
    [(id, token_string, (start, end_offset))] triples. Every token gets the
    given [type_id], {!val-attention_mask} [1], {!val-special_tokens_mask} [0]
    and {!val-word_ids} [None]. *)

val concat : t -> t -> t
(** [concat a b] is the encoding with [a]'s tokens followed by [b]'s.
    {!val-overflowing} and sequence ranges are taken from [a]. *)

val concat_list : t list -> t
(** [concat_list encs] is the concatenation of [encs] in order.
    {!val-overflowing} and sequence ranges are taken from the first element.
    Allocates once rather than creating intermediate arrays per pair. *)

(** {1:access Accessors} *)

val ids : t -> int array
(** [ids enc] is the token ID array. *)

val type_ids : t -> int array
(** [type_ids enc] is the segment ID array. Typically [0] for the first sequence
    and [1] for the second in sentence-pair tasks. *)

val tokens : t -> string array
(** [tokens enc] is the string representation of each token. *)

val word_ids : t -> int option array
(** [word_ids enc] maps each token to its source word index, or [None] for
    special tokens. *)

val offsets : t -> (int * int) array
(** [offsets enc] is the [(start, end_)] byte offset spans into the original
    text for each token. *)

val special_tokens_mask : t -> int array
(** [special_tokens_mask enc] is [1] for special tokens ([CLS], [SEP], padding)
    and [0] for content tokens. *)

val attention_mask : t -> int array
(** [attention_mask enc] is [1] for real tokens and [0] for padding tokens. *)

val overflowing : t -> t list
(** [overflowing enc] is the list of overflow encodings produced by
    {!val-truncate} when the input exceeds [max_length]. Each element is a
    sliding window over the excess tokens. *)

val is_empty : t -> bool
(** [is_empty enc] is [true] iff [enc] has no tokens. *)

val length : t -> int
(** [length enc] is the number of tokens in [enc]. *)

(** {1:ops Operations} *)

val truncate :
  t -> max_length:int -> stride:int -> direction:[ `Left | `Right ] -> t
(** [truncate enc ~max_length ~stride ~direction] limits [enc] to at most
    [max_length] tokens.

    Excess tokens are split into sliding windows of size [max_length] with
    overlap [stride] and stored in {!val-overflowing}. If
    [length enc <= max_length], [enc] is returned unchanged.

    [stride] must be strictly less than [max_length]. When [max_length] is [0],
    all tokens move to {!val-overflowing} and {!val-empty} is returned. *)

val pad :
  t ->
  target_length:int ->
  pad_id:int ->
  pad_type_id:int ->
  pad_token:string ->
  direction:[ `Left | `Right ] ->
  t
(** [pad enc ~target_length ~pad_id ~pad_type_id ~pad_token ~direction] extends
    [enc] to exactly [target_length] tokens.

    Padding tokens have {!val-attention_mask} [0] and {!val-special_tokens_mask}
    [1]. If [length enc >= target_length], [enc] is returned unchanged. Padding
    is applied recursively to {!val-overflowing} encodings. When [direction] is
    [`Left], {!val-offsets} and sequence ranges are shifted accordingly. *)
