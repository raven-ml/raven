(** N-gram language models for text generation *)

(** {1 Types} *)

type t
(** An n-gram model *)

type vocab_stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }
(** Statistics about the trained model *)

(** {1 Common Models} *)

module Unigram : sig
  type model
  (** Unigram (1-gram) model - no context *)

  val train : int array -> model
  (** [train tokens] builds a unigram model from token IDs *)

  val train_from_corpus : int array list -> model
  (** [train_from_corpus corpus] builds model from a list of token arrays *)

  val logits : model -> int -> float array
  (** [logits model _prev_token] returns log probabilities for all tokens. The
      previous token is ignored in unigram models. *)

  val log_prob : model -> int -> float
  (** [log_prob model token] returns the log probability of a token *)

  val sample : model -> ?temperature:float -> ?seed:int -> unit -> int
  (** [sample model ?temperature ?seed ()] samples a token from the distribution
  *)

  val stats : model -> vocab_stats
  (** [stats model] returns statistics about the model *)

  val save : model -> string -> unit
  (** [save model path] saves the model to a file *)

  val load : string -> model
  (** [load path] loads a model from a file *)
end

module Bigram : sig
  type model
  (** Bigram (2-gram) model - one token of context *)

  val train : ?smoothing:float -> int array -> model
  (** [train ?smoothing tokens] builds a bigram model from token IDs.
      @param smoothing Add-k smoothing parameter (default: 1.0 for Laplace) *)

  val train_from_corpus : ?smoothing:float -> int array list -> model
  (** [train_from_corpus ?smoothing corpus] builds model from a list of token
      arrays *)

  val logits : model -> int -> float array
  (** [logits model prev_token] returns log probabilities for next token given
      the previous token *)

  val log_prob : model -> prev:int -> next:int -> float
  (** [log_prob model ~prev ~next] returns conditional log probability
      P(next|prev) *)

  val sample :
    model -> prev:int -> ?temperature:float -> ?seed:int -> unit -> int
  (** [sample model ~prev ?temperature ?seed ()] samples next token given
      previous *)

  val stats : model -> vocab_stats
  (** [stats model] returns statistics about the model *)

  val save : model -> string -> unit
  (** [save model path] saves the model to a file *)

  val load : string -> model
  (** [load path] loads a model from a file *)
end

module Trigram : sig
  type model
  (** Trigram (3-gram) model - two tokens of context *)

  val train : ?smoothing:float -> int array -> model
  (** [train ?smoothing tokens] builds a trigram model from token IDs *)

  val train_from_corpus : ?smoothing:float -> int array list -> model
  (** [train_from_corpus ?smoothing corpus] builds model from a list of token
      arrays *)

  val logits : model -> prev1:int -> prev2:int -> float array
  (** [logits model ~prev1 ~prev2] returns log probabilities given two previous
      tokens *)

  val log_prob : model -> prev1:int -> prev2:int -> next:int -> float
  (** [log_prob model ~prev1 ~prev2 ~next] returns P(next|prev1,prev2) *)

  val sample :
    model ->
    prev1:int ->
    prev2:int ->
    ?temperature:float ->
    ?seed:int ->
    unit ->
    int
  (** [sample model ~prev1 ~prev2] samples next token given two previous *)

  val stats : model -> vocab_stats
  (** [stats model] returns statistics about the model *)

  val save : model -> string -> unit
  (** [save model path] saves the model to a file *)

  val load : string -> model
  (** [load path] loads a model from a file *)
end

(** {1 Generic N-gram} *)

val create : n:int -> ?smoothing:float -> int array -> t
(** [create ~n ?smoothing tokens] creates an n-gram model of order n *)

val logits : t -> context:int array -> float array
(** [logits model ~context] returns log probabilities given context. Context
    length should be n-1 for an n-gram model. *)

val perplexity : t -> int array -> float
(** [perplexity model tokens] computes perplexity on test tokens *)

val generate :
  t ->
  ?max_tokens:int ->
  ?temperature:float ->
  ?seed:int ->
  ?start:int array ->
  unit ->
  int array
(** [generate model ?max_tokens ?temperature ?seed ?start ()] generates tokens
*)
