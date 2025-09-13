(** N-gram language models for text generation *)

(** {1 Types} *)

type t
(** An n-gram model *)

type vocab_stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }
(** Statistics about the trained model *)

(** {1 N-gram} *)

(** Smoothing strategies:
    - [Add_k k]: classic add-k (Laplace) smoothing
    - [Stupid_backoff alpha]: back off to lower orders scaled by [alpha] *)
type smoothing = Add_k of float | Stupid_backoff of float

val create :
  n:int -> ?smoothing:smoothing -> ?cache_capacity:int -> int array -> t
(** [create ~n ?smoothing ?cache_capacity tokens] builds a model with
    configurable smoothing and an optional logits cache. *)

val logits : t -> context:int array -> float array
(** [logits model ~context] returns log probabilities given context. Context
    length should be n-1 for an n-gram model. *)

val perplexity : t -> int array -> float
(** [perplexity model tokens] computes perplexity on test tokens *)

val log_prob : t -> int array -> float
(** [log_prob model tokens] returns the sum of log-probabilities of the observed
    tokens under the model. *)

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

val stats : t -> vocab_stats
(** [stats model] returns statistics about the highest-order n-grams. *)

val save : t -> string -> unit
(** [save model filename] serializes the model to a file. *)

val load : string -> t
(** [load filename] deserializes the model from a file. *)

val save_text : t -> string -> unit
(** [save_text model filename] serializes the model to a text file. *)

val load_text : string -> t
(** [load_text filename] deserializes the model from a text file. *)

val n : t -> int
(** [n model] returns the n-gram order of the model. *)
