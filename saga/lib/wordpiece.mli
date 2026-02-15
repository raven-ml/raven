(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** WordPiece tokenization algorithm.

    This module implements the WordPiece subword tokenization algorithm used by
    BERT and related models. WordPiece employs a greedy longest-match-first
    strategy to decompose words into subword units from a learned vocabulary.

    {1 Algorithm Overview}

    WordPiece tokenization uses a greedy left-to-right scan:

    1. For each word (pre-tokenized by whitespace or punctuation):
    - Start at beginning of word
    - Find longest substring in vocabulary starting from current position
    - If found: emit token, advance position
    - If not found and position > 0: add continuing subword prefix, retry
    - If still not found: emit unknown token for entire word, skip to next word
      2. Repeat until word fully tokenized

    Unlike BPE which learns and applies merge rules, WordPiece directly looks up
    substrings in a fixed vocabulary. This makes tokenization faster but
    vocabulary construction more complex.

    {1 Key Characteristics}

    - Deterministic: Always produces same output for same input (no dropout)
    - Greedy: Takes longest matching substring at each step (not globally
      optimal)
    - Fallback: Words with no vocabulary matches become single unknown token
    - Prefix marking: Non-initial subwords marked with continuing_subword_prefix
      (typically "##")

    {1 Performance Characteristics}

    - Tokenization: O(n * m²) where n = number of words, m = average word length
    - Worst case: O(n * m * vocab_size) for vocabulary lookups
    - Memory: O(vocab_size)

    The quadratic word length complexity comes from trying progressively shorter
    substrings. Vocabulary hash table lookups are O(1) average case.

    {1 Configuration Options}

    - [unk_token]: Token for unknown words (required, typically "[UNK]")
    - [continuing_subword_prefix]: Prefix for non-initial subwords (typically
      "##")
    - [max_input_chars_per_word]: Reject words longer than this limit (prevents
      quadratic blowup on very long words)

    {1 Compatibility}

    Matches HuggingFace Tokenizers WordPiece implementation. Token IDs use OCaml
    integers (int) instead of Rust u32. Vocabularies from BERT models are
    directly compatible.

    {1 Implementation Notes}

    The algorithm maintains no mutable state during tokenization (unlike BPE's
    symbol list). Each word is processed independently. The continuing subword
    prefix is prepended during substring lookup, not stored separately. *)

(** {1 Core Types} *)

type t
(** WordPiece tokenization model.

    Contains vocabulary (string → int mapping), configuration parameters, and
    reverse vocabulary (int → string) for decoding. Model is immutable after
    construction. *)

type vocab = (string, int) Hashtbl.t
(** Vocabulary mapping tokens to integer IDs.

    Includes both initial subwords and continuing subwords (with prefix). For
    example, with prefix "##", vocabulary contains both "ing" and "##ing" as
    separate entries. *)

type config = {
  vocab : vocab;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}
(** WordPiece model configuration.

    @param vocab Token to ID mappings (must be non-empty).
    @param unk_token
      Unknown token string (must exist in vocabulary). Used when word cannot be
      decomposed.
    @param continuing_subword_prefix
      Prefix for non-initial subwords (typically "##"). Added during substring
      lookup.
    @param max_input_chars_per_word
      Maximum characters per word. Words exceeding this are treated as unknown.
      Prevents quadratic complexity on very long inputs. *)

(** {1 Model Creation} *)

val create : config -> t
(** [create config] constructs WordPiece model from configuration.

    Builds reverse vocabulary for decoding. Validates that [unk_token] exists in
    vocabulary.

    @raise Invalid_argument if unk_token not in vocabulary. *)

val from_file : vocab_file:string -> t
(** [from_file ~vocab_file] loads model from BERT-style vocab.txt file.

    File format: one token per line, IDs assigned by line number (0-indexed).
    Uses default settings: unk_token="[UNK]", continuing_subword_prefix="##",
    max_input_chars_per_word=100.

    @raise Sys_error if file cannot be read.
    @raise Invalid_argument if required tokens missing. *)

val from_file_with_config :
  vocab_file:string ->
  unk_token:string ->
  continuing_subword_prefix:string ->
  max_input_chars_per_word:int ->
  t
(** [from_file_with_config ~vocab_file ~unk_token ~continuing_subword_prefix
     ~max_input_chars_per_word] loads model with custom configuration.

    @raise Sys_error if file cannot be read.
    @raise Invalid_argument if unk_token not in loaded vocabulary. *)

val default : unit -> t
(** [default ()] creates model with empty vocabulary and BERT defaults.

    Useful as base for training. Not usable for tokenization until vocabulary
    populated. *)

(** {1 Tokenization} *)

type token = { id : int; value : string; offsets : int * int }
(** Token with ID, string value, and character offsets. *)

val tokenize : t -> string -> token list
(** [tokenize model text] encodes text using greedy longest-match-first.

    For each substring starting at current position, try decreasing lengths
    until match found in vocabulary. Add continuing_subword_prefix for
    non-initial subwords. If no match found, emit unk_token for entire word.

    Words exceeding max_input_chars_per_word are treated as single unk_token. *)

val tokenize_ids : t -> string -> int array
(** [tokenize_ids model text] returns only token IDs, skipping string and offset
    metadata. Faster than [tokenize] when only IDs are needed. *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] looks up token ID in vocabulary.

    Exact string match required including any continuing_subword_prefix. *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] retrieves token string by ID.

    Returns [None] if ID out of bounds. *)

(** {1 Vocabulary Management} *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns vocabulary as (token, id) pairs.

    Order is unspecified. *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns vocabulary size. *)

val get_unk_token : t -> string
(** [get_unk_token model] retrieves configured unknown token. *)

val get_continuing_subword_prefix : t -> string
(** [get_continuing_subword_prefix model] retrieves subword continuation prefix.
*)

val get_max_input_chars_per_word : t -> int
(** [get_max_input_chars_per_word model] retrieves word length limit. *)

(** {1 Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> string
(** [save model ~path ?name ()] writes vocabulary to BERT-format vocab.txt file.

    Returns filepath of saved file. One token per line, order preserves IDs.

    @raise Sys_error if directory not writable. *)

val read_file : vocab_file:string -> vocab
(** [read_file ~vocab_file] reads vocabulary from BERT-format file.

    IDs assigned by line number (0-indexed).

    @raise Sys_error if file cannot be read. *)

val read_bytes : bytes -> vocab
(** [read_bytes bytes] parses vocabulary from in-memory bytes.

    Same format as [read_file]. *)

val to_json : t -> Jsont.json
(** [to_json model] serializes model to HuggingFace JSON format. *)

val of_json : Jsont.json -> t
(** [of_json json] deserializes model from HuggingFace JSON format.

    @raise Failure if JSON malformed. *)

val from_bytes : bytes -> t
(** [from_bytes bytes] deserializes model from bytes.

    Expects vocab.txt format in bytes. *)

(** {1 Training} *)

val train :
  min_frequency:int ->
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  limit_alphabet:int option ->
  initial_alphabet:char list ->
  continuing_subword_prefix:string ->
  end_of_word_suffix:string option ->
  string list ->
  t option ->
  t * string list
(** [train ~min_frequency ~vocab_size ~show_progress ~special_tokens
     ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
     ~end_of_word_suffix texts existing] learns WordPiece vocabulary from
    corpus.

    Training algorithm builds vocabulary by frequency counting and likelihood
    maximization, following original WordPiece paper.

    @param min_frequency Minimum word frequency to include in training.
    @param vocab_size Target vocabulary size.
    @param show_progress Print progress (not yet implemented).
    @param special_tokens Tokens to preserve without splitting.
    @param limit_alphabet
      Limit base alphabet to most frequent characters. None means no limit.
    @param initial_alphabet Starting character set.
    @param continuing_subword_prefix Prefix for non-initial subwords.
    @param end_of_word_suffix
      Optional suffix for word-final tokens (rarely used in WordPiece).
    @param texts Training corpus.
    @param existing Optional existing model to extend (currently unused).
    @return (trained model, special tokens list). *)

(** {1 Conversion} *)

val from_bpe : Bpe.t -> t
(** [from_bpe bpe] converts BPE model to WordPiece format.

    Extracts vocabulary from BPE model and creates WordPiece model with same
    tokens. Useful for format conversion. Loses BPE-specific merge rules. *)
