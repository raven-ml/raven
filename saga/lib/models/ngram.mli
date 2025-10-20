(** Low-level n-gram language models over integer token sequences.

    Implements n-gram models with configurable smoothing for language modeling,
    text generation, and perplexity evaluation. Operates on pre-tokenized
    integer sequences for efficiency.

    {1 Overview}

    An n-gram model predicts the probability of a token given the previous n-1
    tokens. The model maintains counts of all observed n-grams during training
    and uses smoothing to handle unseen contexts.

    Key features:
    - Multiple smoothing strategies (Add-k, Stupid Backoff)
    - Incremental training with {!add_sequence}
    - Efficient probability computation with context normalization
    - Perplexity evaluation for model comparison

    {1 Usage}

    Train a trigram model:
    {[
      (* Pre-tokenized sequences as integer arrays *)
      let sequences = [
        [|0; 1; 2; 3|];  (* "the cat sat down" *)
        [|0; 4; 5; 3|]   (* "the dog ran down" *)
      ] in

      let model = Ngram.of_sequences ~order:3 sequences
    ]}

    Compute next-token probabilities:
    {[
      let context = [|0; 1|] in  (* "the cat" *)
      let logits = Ngram.logits model ~context in
      (* logits.(token) = log P(token | context) *)
    ]}

    Evaluate model quality:
    {[
      let test_seq = [| 0; 1; 2 |] in
      let ppl = Ngram.perplexity model test_seq in
      Printf.printf "Perplexity: %.2f\n" ppl
    ]}

    {1 Smoothing Strategies}

    {b Add-k smoothing} adds a constant k to all counts before normalization.
    This prevents zero probabilities but can over-smooth for large vocabularies.

    {b Stupid backoff} recursively falls back to lower-order n-grams when a
    context is unseen, multiplying by a scaling factor alpha at each backoff
    level. More efficient than interpolation-based smoothing.

    {1 Performance}

    - Training is O(T * N) where T is total tokens and N is the order
    - Probability computation is O(N) per token with backoff, O(1) with Add-k
    - Memory usage is O(unique n-grams), typically linear in training data size
*)

type smoothing = [ `Add_k of float | `Stupid_backoff of float ]
(** Smoothing strategy for handling unseen n-grams.

    - [\`Add_k k]: Add-k (Laplace) smoothing. Adds [k] to all counts. Common
      values are 0.01 to 0.1. Larger values increase smoothing strength.
    - [\`Stupid_backoff alpha]: Backoff to lower-order n-grams with scaling
      factor [alpha]. Typical values are 0.4 to 0.6. Does not produce normalized
      probabilities but works well in practice. *)

type stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }
(** Summary statistics describing the trained model.

    - [vocab_size]: Number of unique token IDs observed during training. This is
      the size of the output space for {!logits}.
    - [total_tokens]: Total count of all tokens across all training sequences,
      including overlaps.
    - [unique_ngrams]: Count of distinct highest-order n-grams. Lower-order
      n-grams are not included in this count. *)

type t
(** N-gram language model.

    Stores n-gram counts and smoothing configuration. Training functions mutate
    internal counts but return the same model for convenience in pipelines. *)

val empty : order:int -> ?smoothing:smoothing -> unit -> t
(** [empty ~order ()] creates an untrained model of the given order.

    @param order
      The n-gram order (1 for unigram, 2 for bigram, etc.). Must be >= 1.
    @param smoothing Smoothing strategy. Default: [\`Add_k 0.01].

    @raise Invalid_argument if [order < 1]. *)

val of_sequences : order:int -> ?smoothing:smoothing -> int array list -> t
(** [of_sequences ~order sequences] builds and trains a model in one step.

    Equivalent to creating an empty model and calling {!add_sequence} on each
    element of [sequences].

    @param order The n-gram order. Must be >= 1.
    @param smoothing Smoothing strategy. Default: [\`Add_k 0.01].

    @raise Invalid_argument if [order < 1]. *)

val add_sequence : t -> int array -> t
(** [add_sequence model tokens] updates the model with a single token sequence.

    Increments counts for all n-grams observed in [tokens]. Token IDs must be
    non-negative integers. The vocabulary size expands automatically to include
    any new token IDs.

    Returns the same model (which has been mutated) for use in pipelines.

    {4 Example}

    {[
      let model = Ngram.empty ~order:2 () in
      let model = add_sequence model [|0; 1; 2|] in
      let model = add_sequence model [|0; 2; 1|]
    ]} *)

val order : t -> int
(** [order model] returns the n-gram order. *)

val smoothing : t -> smoothing
(** [smoothing model] returns the configured smoothing strategy. *)

val is_trained : t -> bool
(** [is_trained model] returns [true] if the model has seen at least one token.

    Untrained models cannot compute probabilities. *)

val stats : t -> stats
(** [stats model] returns summary statistics about the trained model.

    Useful for diagnosing model size and vocabulary coverage. *)

val logits : t -> context:int array -> float array
(** [logits model ~context] computes log probabilities for the next token.

    Returns an array where [logits.(token)] is the natural logarithm of P(token
    | context). The array length equals the vocabulary size.

    The [context] is automatically truncated or padded to match the model order.
    If [context] is shorter than [order - 1], the model uses the available
    context. For unigram models (order=1), [context] is ignored.

    {b Smoothing behavior:}
    - Add-k: All tokens have finite probability, even if unseen.
    - Stupid backoff: Recursively backs off to lower orders for unseen contexts.
      Returns [-infinity] for tokens unseen at all orders.

    @raise Invalid_argument if the model is not trained. *)

val log_prob : t -> int array -> float
(** [log_prob model tokens] computes the log probability of a token sequence.

    Returns the sum of log probabilities for each token given its context. The
    first [order - 1] tokens are skipped since they lack full context.

    For a sequence [t_0, t_1, ..., t_n], computes:
    sum_{i=order-1}^{n} log P(t_i | t_{i-order+1}, ..., t_{i-1})

    Token IDs outside the vocabulary are skipped.

    @raise Invalid_argument if the model is not trained. *)

val perplexity : t -> int array -> float
(** [perplexity model tokens] computes per-token perplexity.

    Returns exp(-log_prob / N) where N is the number of tokens scored (length
    minus [order - 1]). Lower perplexity indicates better model fit.

    Returns [infinity] for empty sequences.

    {4 Example}

    {[
      let ppl = Ngram.perplexity model test_tokens in
      Printf.printf "Perplexity: %.2f\n" ppl
      (* Lower is better; typical values range from 10 to 1000+ *)
    ]}

    @raise Invalid_argument if the model is not trained. *)

val save : t -> string -> unit
(** [save model path] serializes the model to a binary file.

    Uses OCaml's [Marshal] module for serialization. The file includes all
    n-gram counts, vocabulary size, and configuration. *)

val load : string -> t
(** [load path] deserializes a model from a file created by {!save}.

    The loaded model is immediately usable for inference.

    @raise Sys_error if the file does not exist or cannot be read.
    @raise Failure if the file is not a valid marshaled n-gram model. *)
