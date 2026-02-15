(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tokenization output representation.

    Encodings are the result of tokenization, containing all information needed
    for model input: token IDs, type IDs, token strings, character offsets,
    attention masks, and metadata for alignment and debugging.

    This module provides both construction (for internal use by tokenizers) and
    access methods (for users extracting information from tokenized text). *)

type t
(** Encoding representing tokenized text.

    Contains:
    - Token IDs for model input
    - Type IDs (segment IDs) for distinguishing sequences
    - Token strings for debugging and display
    - Character offsets for alignment with original text
    - Special token mask identifying special tokens
    - Attention mask for padding
    - Overflowing tokens from truncation
    - Sequence ranges for multi-sequence inputs *)

val create :
  ids:int array ->
  type_ids:int array ->
  tokens:string array ->
  words:int option array ->
  offsets:(int * int) array ->
  special_tokens_mask:int array ->
  attention_mask:int array ->
  overflowing:t list ->
  sequence_ranges:(int, int * int) Hashtbl.t ->
  t
(** [create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
     ~attention_mask ~overflowing ~sequence_ranges] constructs encoding.

    For internal use by tokenizers. Most users should obtain encodings via
    {!Saga.Tokenizer.encode}. *)

val with_capacity : int -> t
(** [with_capacity capacity] creates empty encoding with preallocated capacity.

    For internal use during encoding construction. *)

val from_tokens : (int * string * (int * int)) list -> type_id:int -> t
(** [from_tokens tokens ~type_id] creates encoding from token list.

    Useful for testing or simple cases. Sets all tokens to same type_id.

    @param tokens List of (id, token_string, (start, end)) tuples.
    @param type_id Type ID to assign to all tokens. *)

val is_empty : t -> bool
(** [is_empty encoding] checks if encoding contains tokens.

    Returns true if token array is empty. *)

val length : t -> int
(** [length encoding] returns number of tokens.

    Includes special tokens added by post-processing. *)

val n_sequences : t -> int
(** [n_sequences encoding] returns number of input sequences.

    1 for single sequence, 2 for pairs. *)

val set_sequence_id : t -> int -> t
(** [set_sequence_id encoding id] assigns sequence ID to all tokens.

    For internal use when constructing encodings. Returns new encoding. *)

(** {1 Accessors} *)

val get_ids : t -> int array
(** [get_ids encoding] retrieves token IDs.

    These are the primary model inputs. *)

val get_type_ids : t -> int array
(** [get_type_ids encoding] retrieves type IDs (segment IDs).

    Used to distinguish sequences in models like BERT. Typically 0 for first
    sequence, 1 for second sequence. *)

val set_type_ids : t -> int array -> t
(** [set_type_ids encoding type_ids] replaces type IDs.

    Returns new encoding with updated type IDs. Array length must match token
    count. *)

val get_tokens : t -> string array
(** [get_tokens encoding] retrieves token strings.

    Useful for debugging and displaying tokenization results. *)

val get_word_ids : t -> int option array
(** [get_word_ids encoding] retrieves word IDs.

    Maps each token to its source word index in original text. [None] indicates
    special tokens. Useful for word-level alignment. *)

val get_sequence_ids : t -> int option array
(** [get_sequence_ids encoding] retrieves sequence ID for each token.

    0 for first sequence, 1 for second sequence (in pairs). [None] for special
    tokens not belonging to either sequence. *)

val get_offsets : t -> (int * int) array
(** [get_offsets encoding] retrieves character offsets.

    Each tuple (start, end) indicates token's span in original text. Offsets are
    character-based (not byte-based). *)

val get_special_tokens_mask : t -> int array
(** [get_special_tokens_mask encoding] retrieves special token mask.

    1 indicates special token (e.g., [CLS], [SEP]), 0 indicates regular token.
    Useful for filtering special tokens in processing. *)

val get_attention_mask : t -> int array
(** [get_attention_mask encoding] retrieves attention mask.

    1 indicates real token, 0 indicates padding. Used in model attention
    mechanisms to ignore padding. *)

val get_overflowing : t -> t list
(** [get_overflowing encoding] retrieves overflowing tokens.

    When truncation is enabled, tokens exceeding max length are stored here as
    separate encodings. Empty list if no truncation occurred. *)

val set_overflowing : t -> t list -> t
(** [set_overflowing encoding overflowing] replaces overflowing encodings.

    Returns new encoding with updated overflowing list. *)

val take_overflowing : t -> t * t list
(** [take_overflowing encoding] extracts and removes overflowing encodings.

    Returns (encoding without overflowing, overflowing list). Useful for
    processing overflowing tokens separately. *)

(** {1 Alignment and Mapping} *)

val token_to_sequence : t -> int -> int option
(** [token_to_sequence encoding token_index] finds sequence containing token.

    Returns [Some 0] for first sequence, [Some 1] for second sequence, [None]
    for special tokens or out of bounds. *)

val token_to_word : t -> int -> (int * int) option
(** [token_to_word encoding token_index] finds word containing token.

    Returns [Some (sequence_id, word_index)] or [None] for special tokens. *)

val token_to_chars : t -> int -> (int * (int * int)) option
(** [token_to_chars encoding token_index] retrieves token's character span.

    Returns [Some (sequence_id, (start, end))] where offsets are relative to
    that sequence, or [None] for special tokens. *)

val word_to_tokens : t -> word:int -> sequence_id:int -> (int * int) option
(** [word_to_tokens encoding ~word ~sequence_id] finds tokens for word.

    Returns [Some (start_token, end_token)] (exclusive end), or [None] if word
    not found. *)

val word_to_chars : t -> word:int -> sequence_id:int -> (int * int) option
(** [word_to_chars encoding ~word ~sequence_id] finds character span for word.

    Returns [Some (start, end)] or [None] if word not found. *)

val char_to_token : t -> pos:int -> sequence_id:int -> int option
(** [char_to_token encoding ~pos ~sequence_id] finds token at character
    position.

    Returns token index or [None] if position is outside tokens (e.g., in
    whitespace). *)

val char_to_word : t -> pos:int -> sequence_id:int -> int option
(** [char_to_word encoding ~pos ~sequence_id] finds word at character position.

    Returns word index or [None] if position is outside words. *)

(** {1 Operations} *)

type truncation_direction =
  | Left
  | Right
      (** Direction for truncation: remove tokens from left (beginning) or right
          (end). *)

val truncate :
  t -> max_length:int -> stride:int -> direction:truncation_direction -> t
(** [truncate encoding ~max_length ~stride ~direction] limits encoding length.

    Tokens beyond [max_length] are moved to overflowing encodings.

    @param max_length Maximum tokens to keep in main encoding.
    @param stride
      Overlap between main encoding and first overflowing encoding. Allows
      sliding window processing. For example, with [max_length=512] and
      [stride=128], the first overflow starts at token 384 (512-128), creating a
      128-token overlap for context continuity.
    @param direction Which end to truncate from.

    {[
      (* Example: Process long document with overlapping windows *)
      let encoding = Tokenizer.encode tokenizer long_text in
      let truncated = Encoding.truncate encoding
        ~max_length:512 ~stride:128 ~direction:Right in
      let main_tokens = Encoding.get_ids truncated in
      let overflow_encodings = Encoding.get_overflowing truncated in
      (* main_tokens: [0..511], first overflow: [384..895], etc. *)
    ]} *)

val merge : t list -> growing_offsets:bool -> t
(** [merge encodings ~growing_offsets] combines encodings into one.

    Concatenates token arrays and adjusts metadata.

    @param growing_offsets
      If true, adjust character offsets for each encoding (assuming they're from
      different texts). If false, keep original offsets (assuming they're from
      same text). *)

val merge_with : t -> t -> growing_offsets:bool -> t
(** [merge_with encoding other ~growing_offsets] merges two encodings.

    Similar to merge but for exactly two encodings. Returns new encoding. *)

type padding_direction =
  | Left
  | Right
      (** Direction for padding: add padding tokens at left (beginning) or right
          (end). *)

val pad :
  t ->
  target_length:int ->
  pad_id:int ->
  pad_type_id:int ->
  pad_token:string ->
  direction:padding_direction ->
  t
(** [pad encoding ~target_length ~pad_id ~pad_type_id ~pad_token ~direction]
    extends encoding to target length.

    Adds padding tokens until length reaches [target_length]. Pads attention
    mask with zeros for padding positions.

    @param target_length Desired length (must be >= current length).
    @param pad_id Token ID for padding.
    @param pad_type_id Type ID for padding tokens.
    @param pad_token Token string for padding (typically "[PAD]").
    @param direction Which end to add padding to. *)
