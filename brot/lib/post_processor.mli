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

val bert : sep:token -> cls:token -> unit -> t
(** [bert ~sep ~cls ()] is a BERT-style post-processor.

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

    [trim_offsets] defaults to [true]. [add_prefix_space] defaults to [true]. *)

val byte_level : ?trim_offsets:bool -> unit -> t
(** [byte_level ()] is a byte-level post-processor that adjusts character
    offsets for byte-level encoding.

    [trim_offsets] removes leading and trailing whitespace from offsets.
    Defaults to [true]. *)

val template :
  single:string -> ?pair:string -> ?special_tokens:token list -> unit -> t
(** [template ~single ()] is a template-based post-processor.

    Templates use [$A] and [$B] as sequence placeholders and literal special
    token names (e.g. [[CLS]]). Type IDs can be specified with a colon suffix:
    [$A:0], [[SEP]:1].

    [special_tokens] defaults to [[]]. *)

val sequence : t list -> t
(** [sequence processors] chains [processors] left-to-right. *)

(** {1:processing Processing} *)

val process :
  t -> ?pair:Encoding.t -> Encoding.t -> add_special_tokens:bool -> Encoding.t
(** [process t enc ~add_special_tokens] adds special tokens and sets type IDs on
    [enc].

    When [~pair] is provided, both sequences are merged into a single encoding
    with appropriate type IDs. When [~add_special_tokens] is [false], special
    token insertion is skipped but byte-level offset trimming still applies. *)

val added_tokens : t -> is_pair:bool -> int
(** [added_tokens t ~is_pair] is the number of special tokens [t] adds. Useful
    for calculating the truncation budget. *)

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
