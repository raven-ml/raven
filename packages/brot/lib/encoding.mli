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
    equal to {!val-length}.

    {!val-tokens}, {!val-word_ids} and {!val-offsets} are derived when they are
    first asked for and kept afterwards: a caller that only reads {!val-ids}
    pays for none of them. Deriving is a mutation: do not race the first read of
    {!val-tokens}, {!val-offsets} or {!val-word_ids} from several domains — read
    them from one domain, or read them before sharing. *)

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
     ~attention_mask ()] is an encoding from the given arrays. [overflowing]
    defaults to [[]].

    Raises [Invalid_argument] if the seven arrays do not all have the same
    length. *)

val concat : t list -> t
(** [concat encs] is the encoding with the tokens of each element of [encs] in
    order. {!val-overflowing} is taken from the first element. Allocates once
    rather than creating intermediate arrays per pair. *)

(** {1:access Accessors} *)

val ids : t -> int array
(** [ids enc] is the token ID array. *)

val type_ids : t -> int array
(** [type_ids enc] is the segment ID array. Typically [0] for the first sequence
    and [1] for the second in sentence-pair tasks. *)

val tokens : t -> string array
(** [tokens enc] is the string representation of each token. *)

val word_ids : t -> int option array
(** [word_ids enc] maps each token to the index of the pretoken it came from,
    counting from [0] in each sequence. The tokens a post-processor inserts, and
    padding, are [None]. *)

val offsets : t -> (int * int) array
(** [offsets enc] is the [(start, end_)] byte span of each token in the text
    that was encoded, before normalization: a token of ["café"] normalized to
    ["cafe"] reports the bytes of the accented text. Spans are ascending but
    need not tile the text, since a pre-tokenizer drops its delimiters and a
    normalizer may remove characters. The tokens a post-processor inserts, and
    padding, report [(0, 0)]. *)

val special_tokens_mask : t -> int array
(** [special_tokens_mask enc] is [1] for special tokens ([CLS], [SEP], padding)
    and [0] for content tokens. An added token found in the input is [0],
    special or not: only what a post-processor inserts is masked. *)

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

val with_type_id : t -> int -> t
(** [with_type_id enc type_id] is [enc] with every {!val-type_ids} entry set to
    [type_id]. Nothing else is read, so an encoding that has not worked out its
    {!val-tokens}, {!val-offsets} or {!val-word_ids} still has not. *)

val with_overflowing : t -> t list -> t
(** [with_overflowing enc windows] is [enc] with [windows] as its
    {!val-overflowing}. Nothing else is read. *)

val truncate :
  ?stride:int -> ?direction:[ `Left | `Right ] -> t -> max_length:int -> t
(** [truncate enc ~max_length] limits [enc] to at most [max_length] tokens.

    The tokens kept are the first [max_length] when [direction] is [`Right] (the
    default) and the last [max_length] when it is [`Left]. The excess is split
    into windows of [max_length] tokens overlapping the previous window by
    [stride] (default [0]) and stored in {!val-overflowing}, walking away from
    the kept tokens; the last window stops at the encoding's edge, so it may be
    shorter. If [length enc <= max_length], [enc] is returned unchanged. When
    [max_length] is [0], all tokens move to {!val-overflowing} and {!val-empty}
    is returned.

    Raises [Invalid_argument] if [enc] is truncated and [stride] is not less
    than [max_length]. *)

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
    [`Left], {!val-offsets} are shifted accordingly. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf enc] formats [enc] as a table for inspection, one row per token:
    index, token, ID, byte offsets, word ID, type ID, attention and
    special-token masks. Reading the table forces the derived {!val-tokens},
    {!val-offsets} and {!val-word_ids}. *)

(**/**)

(* Internals. Building an encoding from the run of spans the encode path
   produced; [Run.t] belongs to a module the library does not export, so
   [of_run] has no caller outside it. [token]'s only caller is
   [post_processor.ml]. *)

val token :
  id:int -> token:string -> offset:int * int -> type_id:int -> special:bool -> t
(** [token ~id ~token ~offset ~type_id ~special] is a single-token encoding.
    When [special] is [true], {!val-special_tokens_mask} is [1] and
    {!val-word_ids} is [None]; otherwise {!val-special_tokens_mask} is [0].
    {!val-attention_mask} is always [1]. *)

val of_run : Run.t -> ids:int array -> t
(** [of_run run ~ids] is the encoding of [ids], whose {!val-tokens},
    {!val-word_ids} and {!val-offsets} are derived from [run] when they are
    asked for. Every token gets type id [0], {!val-attention_mask} [1] and
    {!val-special_tokens_mask} [0]. *)

(**/**)
