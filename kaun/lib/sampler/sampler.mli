(** Text generation and sampling utilities for language models *)

(** {1 Types} *)

type logits_fn = int -> float array
(** Function that returns logits (unnormalized log probabilities) for the next token 
    given the previous token ID. For unigram models, the input is ignored. *)

type config = {
  max_tokens : int;
  (** Maximum number of tokens to generate *)
  
  temperature : float;
  (** Temperature for sampling. 1.0 = no change, <1.0 = more deterministic, >1.0 = more random *)
  
  top_k : int option;
  (** If set, only sample from the top-k most likely tokens *)
  
  top_p : float option;
  (** If set, use nucleus sampling - sample from smallest set of tokens whose cumulative probability exceeds p *)
  
  seed : int option;
  (** Random seed for reproducibility *)
}

val default_config : config
(** Default configuration: max_tokens=100, temperature=1.0, top_k=None, top_p=None *)

(** {1 Sampling Functions} *)

val sample_token : 
  ?temperature:float -> 
  ?top_k:int -> 
  ?top_p:float -> 
  ?seed:int ->
  float array -> 
  int
(** [sample_token ?temperature ?top_k ?top_p ?seed logits] samples a token index from logits.
    
    @param temperature Softmax temperature (default: 1.0)
    @param top_k Only consider top k tokens
    @param top_p Use nucleus sampling with this threshold
    @param seed Random seed
    @param logits Unnormalized log probabilities
    @return Sampled token index *)

val greedy : float array -> int
(** [greedy logits] returns the token with highest probability (argmax) *)

(** {1 Generation} *)

val generate : 
  ?max_tokens:int ->
  ?temperature:float ->
  ?top_k:int ->
  ?top_p:float ->
  ?seed:int ->
  ?start:string ->
  logits_fn:logits_fn ->
  tokenizer:(string -> int array) ->
  unit ->
  int array
(** [generate ?max_tokens ?temperature ?top_k ?top_p ?seed ?start ~logits_fn ~tokenizer]
    generates a sequence of token IDs.
    
    @param start Optional starting text to condition on
    @param logits_fn Function that returns logits for next token given previous
    @param tokenizer Function to convert text to token IDs
    @return Array of generated token IDs *)

val generate_text :
  ?max_tokens:int ->
  ?temperature:float ->
  ?top_k:int ->
  ?top_p:float ->
  ?seed:int ->
  ?start:string ->
  logits_fn:logits_fn ->
  tokenizer:(string -> int array) ->
  decoder:(int array -> string) ->
  unit ->
  string
(** [generate_text] is like [generate] but returns decoded text.
    
    @param decoder Function to convert token IDs back to text
    @return Generated text string *)