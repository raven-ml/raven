(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Text generation with composable logits processors.

    Provides the autoregressive decode loop, composable logits processors,
    stopping criteria, and generation configuration for language model
    inference. Operates on nx tensors for logits. *)

(** {1 Core Types} *)

type logits = (float, Bigarray.float32_elt) Nx.t
(** 1D float32 tensor of unnormalized token probabilities. Length equals
    vocabulary size. *)

type token_ids = int array
(** Sequence of token IDs representing encoded text. *)

type logits_processor = {
  name : string;
  process : prompt_length:int -> token_ids -> logits -> logits;
}
(** Transforms logits before sampling. *)

type logits_processor_list = logits_processor list

type stopping_criterion = {
  name : string;
  should_stop : prompt_length:int -> start_time:float -> token_ids -> bool;
}
(** Determines when to end generation. *)

type stopping_criteria_list = stopping_criterion list

(** {1 Generation Configuration} *)

type generation_config = {
  max_length : int;
  max_new_tokens : int option;
  min_length : int;
  min_new_tokens : int;
  do_sample : bool;
  temperature : float;
  top_k : int;
  top_p : float;
  repetition_penalty : float;
  no_repeat_ngram_size : int;
  bad_words_ids : int list list;
  force_words_ids : int list list;
  pad_token_id : int option;
  bos_token_id : int option;
  eos_token_id : int option;
  eos_token_ids : int list;
}

val default : generation_config

(** {2 Builder Pattern} *)

val with_temperature : float -> generation_config -> generation_config
val with_top_k : int -> generation_config -> generation_config
val with_top_p : float -> generation_config -> generation_config
val with_repetition_penalty : float -> generation_config -> generation_config
val with_max_length : int -> generation_config -> generation_config
val with_max_new_tokens : int -> generation_config -> generation_config
val with_min_length : int -> generation_config -> generation_config
val with_min_new_tokens : int -> generation_config -> generation_config
val with_no_repeat_ngram : int -> generation_config -> generation_config
val with_do_sample : bool -> generation_config -> generation_config

(** {2 Presets} *)

val creative_writing : generation_config
val chat : generation_config
val code_generation : generation_config
val factual : generation_config
val from_preset : string -> generation_config

(** {1 Logits Processors} *)

val temperature_warper : temperature:float -> logits_processor
val top_k_warper : k:int -> logits_processor
val top_p_warper : p:float -> logits_processor
val repetition_penalty : penalty:float -> logits_processor
val no_repeat_ngram : ngram_size:int -> logits_processor
val min_length : min_length:int -> eos_token_ids:int list -> logits_processor

val min_new_tokens :
  min_new_tokens:int -> eos_token_ids:int list -> logits_processor

val bad_words : bad_words_ids:int list list -> logits_processor

val force_words :
  force_words_ids:int list list -> iteration:int -> logits_processor

val custom :
  name:string ->
  process:(prompt_length:int -> token_ids -> logits -> logits) ->
  logits_processor

(** {1 Stopping Criteria} *)

val max_length_criteria : max_length:int -> stopping_criterion
val max_new_tokens_criteria : max_new_tokens:int -> stopping_criterion
val eos_token_criteria : eos_token_ids:int list -> stopping_criterion
val max_time_criteria : max_time:float -> stopping_criterion

val stop_strings_criteria :
  stop_strings:string list ->
  decoder:(token_ids -> string) ->
  stopping_criterion

val custom_criteria :
  name:string ->
  should_stop:(prompt_length:int -> start_time:float -> token_ids -> bool) ->
  stopping_criterion

(** {1 Generation} *)

type generation_output = {
  sequences : int array list;
  scores : float list list option;
}

val generate :
  model:(token_ids -> logits) ->
  ?input_ids:token_ids ->
  ?generation_config:generation_config ->
  ?logits_processor:logits_processor_list ->
  ?stopping_criteria:stopping_criteria_list ->
  unit ->
  generation_output

val generate_text :
  model:(token_ids -> logits) ->
  tokenizer:(string -> token_ids) ->
  decoder:(token_ids -> string) ->
  ?prompt:string ->
  ?generation_config:generation_config ->
  ?logits_processor:logits_processor_list ->
  ?stopping_criteria:stopping_criteria_list ->
  unit ->
  string

(** {1 Utilities} *)

val apply_processors :
  processors:logits_processor_list ->
  prompt_length:int ->
  tokens:token_ids ->
  logits:logits ->
  logits

val check_stopping :
  criteria:stopping_criteria_list ->
  prompt_length:int ->
  start_time:float ->
  tokens:token_ids ->
  bool
