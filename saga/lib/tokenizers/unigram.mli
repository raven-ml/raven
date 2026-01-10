(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Unigram language model tokenization algorithm.

    This module implements the Unigram tokenization algorithm used by
    SentencePiece. Unlike BPE and WordPiece which use deterministic rules,
    Unigram uses a probabilistic language model to select the most likely
    tokenization.

    {1 Algorithm Overview}

    Unigram tokenization finds the tokenization that maximizes likelihood under
    a unigram language model:

    1. Each token has an associated probability (log probability stored) 2. For
    each input, find all possible tokenizations using vocabulary 3. Select
    tokenization with highest probability (sum of token log probs) 4. Use
    dynamic programming (Viterbi algorithm) for efficient search

    The vocabulary is learned by starting with a large character set and
    iteratively removing tokens that contribute least to the model likelihood.

    {1 Key Characteristics}

    - Probabilistic: Chooses globally optimal tokenization under unigram model
    - Deterministic output: Same input always produces same tokenization (no
      sampling)
    - Subword-aware: Can split unknown words into known subwords
    - Language-agnostic: Works for any language with appropriate training data

    {1 Performance Characteristics}

    - Tokenization: O(n²) where n = input length (Viterbi dynamic programming)
    - Vocabulary lookup: O(1) per candidate substring
    - Memory: O(vocab_size + n) for Viterbi table

    Current implementation uses greedy longest-match as a simplification.

    {1 Vocabulary Format}

    Vocabulary entries are (token, probability) pairs where probability is
    typically negative (log probability). Higher values indicate more common
    tokens.

    {1 Compatibility}

    Matches HuggingFace Tokenizers Unigram implementation. Compatible with
    SentencePiece models. *)

type t
(** Unigram tokenization model.

    Contains vocabulary with token probabilities and reverse mapping for
    decoding. *)

val create : (string * float) list -> t
(** [create vocab] constructs Unigram model from vocabulary.

    @param vocab List of (token, log_probability) pairs. *)

val tokenize : t -> string -> (int * string * (int * int)) list
(** [tokenize model text] encodes text using Unigram algorithm.

    Current implementation uses greedy longest-match. Future versions will
    implement Viterbi algorithm for optimal tokenization.

    Returns list of (id, token_string, (start_offset, end_offset)) tuples. *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] looks up token ID. *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] retrieves token string by ID. *)

val get_vocab : t -> (string * float) list
(** [get_vocab model] returns vocabulary with probabilities. *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns vocabulary size. *)

val save : t -> folder:string -> unit -> string list
(** [save model ~folder ()] writes vocabulary with probabilities to JSON.

    Returns list of created filenames. *)

val train :
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  shrinking_factor:float ->
  unk_token:string option ->
  max_piece_length:int ->
  n_sub_iterations:int ->
  string list ->
  t option ->
  t * string list
(** [train ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
     ~unk_token ~max_piece_length ~n_sub_iterations texts existing] learns
    Unigram model from corpus.

    Training uses EM algorithm to iteratively refine token probabilities and
    prune vocabulary.

    @param vocab_size Target vocabulary size.
    @param show_progress Print progress (not yet implemented).
    @param special_tokens Tokens to preserve without modification.
    @param shrinking_factor
      Fraction of vocabulary to remove per iteration (typically 0.75).
    @param unk_token Optional unknown token.
    @param max_piece_length Maximum characters per token.
    @param n_sub_iterations Number of EM iterations per pruning step.
    @return (trained model, special tokens list). *)
