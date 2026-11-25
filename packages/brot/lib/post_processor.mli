(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Post-processing tokenization output with special tokens.

    Post-processors add special tokens and type IDs to tokenized sequences after
    core tokenization. They handle model-specific requirements like [[CLS]] and
    [[SEP]] for BERT, sentence pair formatting, and byte-level offset
    adjustments. *)

type t
(** The type for post-processors. *)

type token = string * int
(** A special token as [(text, id)]. *)

(** {1:constructors Constructors} *)

val bert : sep:token -> cls:token -> t
(** [bert ~sep ~cls] is a BERT-style post-processor.

    Single: [[CLS] A [SEP]]. Pair: [[CLS] A [SEP] B [SEP]]. Type IDs: [0] for
    the first sequence, [1] for the second. *)

val roberta :
  sep:token ->
  cls:token ->
  ?trim_offsets:bool ->
  ?add_prefix_space:bool ->
  unit ->
  t
(** [roberta ~sep ~cls ()] is a RoBERTa-style post-processor.

    Single: [<s> A </s>]. Pair: [<s> A </s> </s> B </s>]. All type IDs are [0].

    [trim_offsets] removes leading and trailing whitespace from the offsets of
    byte-level tokens; it defaults to [true]. [add_prefix_space] tells the
    trimming that a single leading space on a token that starts at offset [0]
    was added by the pre-tokenizer and must be kept; it defaults to [true]. *)

val byte_level : ?add_prefix_space:bool -> ?trim_offsets:bool -> unit -> t
(** [byte_level ()] is a byte-level post-processor that adjusts character
    offsets for byte-level encoding.

    [trim_offsets] removes leading and trailing whitespace from offsets.
    Defaults to [true]. [add_prefix_space] tells the trimming that a single
    leading space on a token that starts at offset [0] was added by the
    pre-tokenizer and must be kept; it defaults to [true]. *)

val template :
  single:string -> ?pair:string -> ?special_tokens:token list -> unit -> t
(** [template ~single ()] is a template-based post-processor.

    Templates use [$A] and [$B] as sequence placeholders and literal special
    token names (e.g. [[CLS]]). Type IDs can be specified with a colon suffix:
    [$A:0], [[SEP]:1].

    [pair] must reference both [$A] and [$B]; it defaults to ["$A:0 $B:1"].
    [special_tokens] defaults to [[]]. *)

val sequence : t list -> t
(** [sequence processors] chains [processors] left to right. Each hands the next
    the sequences it gives, and {!process} merges them after the last. *)

(** {1:processing Processing} *)

val process :
  t -> ?pair:Encoding.t -> Encoding.t -> add_special_tokens:bool -> Encoding.t
(** [process t enc ~add_special_tokens] adds special tokens and sets type IDs on
    [enc]. [enc] enters with type ID [0], and [pair] with [1], whatever they
    arrive with; [t] may override either.

    When [~pair] is provided, both sequences are merged into a single encoding.
    Offsets are not shifted: each sequence keeps the offsets into its own text.

    When [~add_special_tokens] is [false], no special token is inserted, but
    everything else still happens: the two sequences are still merged, type IDs
    are still assigned, and byte-level offsets are still trimmed.

    Each window in [enc]'s and [pair]'s {!Encoding.overflowing} is processed the
    same way: each window of [enc] with [pair]'s kept tokens and then with each
    of its windows, then [enc]'s kept tokens with each window of [pair]. *)

val added_tokens : t -> is_pair:bool -> int
(** [added_tokens t ~is_pair] is the number of special tokens {!process} inserts
    into one sequence, or into a pair when [is_pair]. Useful for calculating the
    truncation budget. Raises [Invalid_argument] where {!process} does. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a post-processor for inspection. *)

(** {1:serialization Serialization} *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a post-processor from HuggingFace [tokenizer.json] format.
    Errors if [json] is not an object, has a missing or unknown ["type"] field,
    or has invalid parameters. *)

val to_json : t -> Jsont.json
(** [to_json t] is [t] serialized to HuggingFace [tokenizer.json] format. *)

(* Internals. The ids fast path's view of a processor; the public API is the
   subset written into [brot.mli]. *)

val affixes : t -> add_special_tokens:bool -> (int array * int array) option
(** [affixes t ~add_special_tokens] is the [(prefix, suffix)] of special token
    ids [t] puts around a single sequence, and [None] when [t] does more to that
    sequence's ids than wrap them — a template naming the sequence more than
    once, or not at all. Raises [Invalid_argument] where {!process} does.
    Everything else a processor does — assigning type ids, trimming byte-level
    offsets — leaves the ids untouched.

    Where {!added_tokens} counts what a processor adds, this says what it adds,
    so a caller that wants ids alone can produce them without an {!Encoding.t}.
*)
