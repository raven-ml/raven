(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Post-processing tokenization output with special tokens.

    Post-processors add special tokens and formatting to tokenized sequences
    after the core tokenization step. They handle model-specific requirements
    like [CLS] and [SEP] tokens for BERT, sentence pair formatting, and type
    IDs.

    Post-processing occurs after tokenization but before returning results to
    the user. The typical flow is: 1. Core tokenization produces token IDs and
    strings 2. Post-processor adds special tokens (e.g., [CLS], [SEP]) 3.
    Post-processor sets type IDs and attention masks 4. Result is final encoding
    ready for model input *)

type encoding = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  offsets : (int * int) array;
  special_tokens_mask : int array;
  attention_mask : int array;
  overflowing : encoding list;
  sequence_ranges : (int * int * int) list;
}
(** Encoding representation for post-processing.

    Contains all information needed for model input: token IDs, type IDs
    (segment IDs), token strings, character offsets, special token mask,
    attention mask, overflowing tokens (from truncation), and sequence range
    markers. *)

type t
(** Post-processor that adds special tokens and formatting to encodings. *)

(** {1 Processor Types} *)

val bert : sep:string * int -> cls:string * int -> unit -> t
(** [bert ~sep ~cls ()] creates BERT-style post-processor.

    Formats sequences as: [[CLS] sequence [SEP]] Formats pairs as:
    [[CLS] sequence_a [SEP] sequence_b [SEP]]

    Sets type IDs: 0 for first sequence (including [CLS] and first [SEP]), 1 for
    second sequence.

    @param sep Separator token and ID (typically ("[SEP]", 102)).
    @param cls Classification token and ID (typically ("[CLS]", 101)). *)

val roberta :
  sep:string * int ->
  cls:string * int ->
  ?trim_offsets:bool ->
  ?add_prefix_space:bool ->
  unit ->
  t
(** [roberta ~sep ~cls ?trim_offsets ?add_prefix_space ()] creates RoBERTa-style
    post-processor.

    Similar to BERT but with different special token placement: Formats
    sequences as: [<s> sequence </s>] Formats pairs as:
    [<s> sequence_a </s> </s> sequence_b </s>]

    @param sep Separator/end token and ID (typically ("</s>", 2)).
    @param cls Start token and ID (typically ("<s>", 0)).
    @param trim_offsets
      Adjust offsets for byte-level tokenization (default: true).
    @param add_prefix_space
      Whether prefix space handling is enabled (default: true). *)

val byte_level : ?trim_offsets:bool -> unit -> t
(** [byte_level ?trim_offsets ()] creates byte-level post-processor.

    Adjusts character offsets to account for byte-level encoding
    transformations.

    @param trim_offsets
      Remove leading/trailing spaces from offsets (default: true). *)

val template :
  single:string ->
  ?pair:string ->
  ?special_tokens:(string * int) list ->
  unit ->
  t
(** [template ~single ?pair ?special_tokens ()] creates template-based
    post-processor.

    Flexible processor using templates to define special token placement.
    Templates use placeholders:
    - [$A]: First sequence
    - [$B]: Second sequence (for pairs)
    - Special tokens by name (e.g., "[CLS]")

    @param single Template for single sequences (e.g., "[CLS] $A [SEP]").
    @param pair
      Template for sequence pairs (e.g., "[CLS] $A [SEP] $B [SEP]"). If not
      provided, pairs are rejected.
    @param special_tokens
      List of (token_string, token_id) pairs for special tokens used in
      templates. *)

val sequence : t list -> t
(** [sequence processors] chains multiple post-processors.

    Applies processors left-to-right. Each processor modifies the encoding
    before passing to next. Useful for combining transformations. *)

(** {1 Operations} *)

val process : t -> encoding list -> add_special_tokens:bool -> encoding list
(** [process processor encodings ~add_special_tokens] applies post-processing.

    Adds special tokens, sets type IDs, and updates masks according to processor
    configuration.

    @param processor Post-processor to apply.
    @param encodings
      List of encodings (typically one for single sequence, two for pairs).
    @param add_special_tokens Whether to add special tokens (allows disabling).
    @return Processed encodings with special tokens and updated fields. *)

val added_tokens : t -> is_pair:bool -> int
(** [added_tokens processor ~is_pair] counts special tokens added by processor.

    Useful for calculating maximum sequence length before truncation.

    @param processor Post-processor to query.
    @param is_pair Whether processing a pair (affects token count).
    @return Number of special tokens that will be added. *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
(** [to_json processor] serializes processor to HuggingFace JSON format. *)

val of_json : Yojson.Basic.t -> t
(** [of_json json] deserializes processor from HuggingFace JSON format.

    @raise Yojson.Json_error if JSON is malformed. *)
