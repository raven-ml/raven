(** Modern text generation with composable logits processors.

    This module provides a text generation pipeline. It features composable
    logits processors, stopping criteria, and configuration builders.

    {1 Overview}

    Text generation works by repeatedly applying a language model to generate
    logits, processing those logits through a pipeline of processors
    (temperature scaling, top-k filtering, etc.), and sampling the next token.
    This continues until stopping criteria are met.

    Key concepts:
    - {!logits_processor}: Transform logits before sampling
    - {!stopping_criterion}: Determine when to stop generation
    - {!generation_config}: Bundle common parameters

    {1 Quick Start}

    Basic text generation:
    {[
      let model token_ids = (* your model function *) [||]
      let tokenizer text = (* your tokenizer *) []
      let decoder ids = (* your decoder *) ""

      let result =
        Sampler.generate_text ~model ~tokenizer ~decoder
          ~prompt:"The quick brown fox"
          ~generation_config:(Sampler.default |> Sampler.with_temperature 0.8)
          ()
    ]}

    With custom processors:
    {[
      let processors = [
        Sampler.temperature_warper ~temperature:0.9;
        Sampler.top_k_warper ~k:50;
        Sampler.repetition_penalty ~penalty:1.1;
      ] in
      let result =
        Sampler.generate_text ~model ~tokenizer ~decoder
          ~prompt:"Once upon a time"
          ~logits_processor:processors
          ()
    ]} *)

(** {1 Core Types}

    Basic types used throughout the sampler API. *)

type logits = float array
(** Logits array representing unnormalized token probabilities from a language
    model. Array length equals vocabulary size. Higher values indicate higher
    probability. *)

type token_ids = int list
(** Sequence of token IDs representing encoded text. Each ID corresponds to a
    token in the model's vocabulary. *)

(** {2 Logits Processors}

    Processors that modify logits before sampling. Can implement temperature
    scaling, filtering strategies, penalties, and custom logic. *)

type logits_processor = {
  name : string;  (** Human-readable name for debugging and logging *)
  process : prompt_length:int -> token_ids -> logits -> logits;
      (** Processing function that transforms logits.

          Parameters:
          - [prompt_length]: Length of original prompt (for context)
          - [token_ids]: Current token sequence being generated
          - [logits]: Input logits array to process

          Must return logits array of the same length. *)
}
(** Base logits processor type. Processors transform logits before sampling to
    implement various generation strategies like temperature scaling, top-k
    filtering, repetition penalties, etc. *)

type logits_processor_list = logits_processor list
(** List of logits processors applied in sequence. Each processor transforms the
    logits before the next processor receives them. Empty list means no
    processing. *)

(** {2 Stopping Criteria}

    Criteria that determine when to end generation. Evaluated after each
    generated token. *)

type stopping_criterion = {
  name : string;  (** Human-readable name for debugging and logging *)
  should_stop : prompt_length:int -> start_time:float -> token_ids -> bool;
      (** Function that determines if generation should stop.

          Parameters:
          - [prompt_length]: Length of original prompt
          - [start_time]: Generation start time (Unix timestamp)
          - [token_ids]: Current token sequence

          Return true to stop generation. *)
}
(** Stopping criterion type. Determines when text generation should end based on
    sequence length, time limits, special tokens, or custom logic. *)

type stopping_criteria_list = stopping_criterion list
(** List of stopping criteria. Generation stops when ANY criterion returns true.
    Empty list means no custom stopping criteria (relies on config defaults). *)

(** {2 Generation Configuration}

    Comprehensive configuration record bundling all generation parameters. *)

type generation_config = {
  max_length : int;
      (** Maximum total sequence length (prompt + generated tokens) *)
  max_new_tokens : int option;
      (** Maximum new tokens to generate (overrides max_length when set) *)
  min_length : int;  (** Minimum total sequence length *)
  min_new_tokens : int;  (** Minimum new tokens to generate *)
  do_sample : bool;  (** Enable sampling (true) vs greedy search (false) *)
  early_stopping : bool;  (** Enable early stopping in beam search *)
  num_beams : int;  (** Number of beams for beam search (1 = greedy) *)
  temperature : float;  (** Temperature for sampling (higher = more random) *)
  top_k : int;  (** Top-k filtering (0 = disabled) *)
  top_p : float;  (** Top-p nucleus sampling (1.0 = disabled) *)
  repetition_penalty : float;
      (** Repetition penalty factor (1.0 = no penalty) *)
  length_penalty : float;  (** Length penalty for beam search *)
  no_repeat_ngram_size : int;
      (** Size of n-grams that cannot repeat (0 = disabled) *)
  encoder_repetition_penalty : float;
      (** Repetition penalty for encoder-decoder models *)
  bad_words_ids : int list list;
      (** Token sequences that cannot be generated *)
  force_words_ids : int list list;
      (** Token sequences that must be generated *)
  num_return_sequences : int;  (** Number of sequences to return *)
  output_scores : bool;  (** Include generation scores in output *)
  output_attentions : bool;  (** Include attention weights in output *)
  output_hidden_states : bool;  (** Include hidden states in output *)
  pad_token_id : int option;  (** Padding token ID *)
  bos_token_id : int option;  (** Beginning-of-sequence token ID *)
  eos_token_id : int option;
      (** End-of-sequence token ID (deprecated, use eos_token_ids) *)
  eos_token_ids : int list;  (** End-of-sequence token IDs *)
}
(** Configuration for text generation. Bundles all generation parameters in one
    record.

    Use {!default} as a starting point and customize with builder functions.
    Many parameters correspond directly to Hugging Face Transformers generation
    config for compatibility. *)

val default : generation_config
(** [default] provides sensible default configuration.

    Values:
    - [max_length]: 100
    - [max_new_tokens]: None (uses max_length)
    - [min_length]: 0
    - [min_new_tokens]: 0
    - [do_sample]: false (greedy decoding)
    - [early_stopping]: false
    - [num_beams]: 1 (greedy)
    - [temperature]: 1.0
    - [top_k]: 0 (disabled)
    - [top_p]: 1.0 (disabled)
    - [repetition_penalty]: 1.0 (disabled)
    - [length_penalty]: 1.0
    - [no_repeat_ngram_size]: 0 (disabled)
    - All token IDs: None

    Use builder functions like {!with_temperature} to customize. *)

(** {2 Builder Pattern for Configuration} *)

val with_temperature : float -> generation_config -> generation_config
(** [with_temperature temp config] sets temperature for sampling.

    Higher values (> 1.0) increase randomness, lower values (< 1.0) make output
    more deterministic. Must be positive. Temperature 1.0 uses raw probability
    distribution. *)

val with_top_k : int -> generation_config -> generation_config
(** [with_top_k k config] enables top-k filtering.

    Only considers the top k most likely tokens at each step. Set to 0 to
    disable. Typical values: 40-100. Higher k allows more diversity. *)

val with_top_p : float -> generation_config -> generation_config
(** [with_top_p p config] enables top-p (nucleus) sampling.

    Considers tokens whose cumulative probability mass is p. Value between 0.0
    and 1.0. Set to 1.0 to disable. Typical values: 0.9-0.95. Lower p increases
    focus. *)

val with_repetition_penalty : float -> generation_config -> generation_config
(** [with_repetition_penalty penalty config] sets repetition penalty.

    Penalizes tokens that have already appeared. Values > 1.0 discourage
    repetition, < 1.0 encourage it. 1.0 disables penalty. Typical range:
    1.05-1.2. *)

val with_max_length : int -> generation_config -> generation_config
(** [with_max_length len config] sets maximum total sequence length.

    Includes both prompt and generated tokens. Generation stops when this length
    is reached. Must be positive. *)

val with_max_new_tokens : int -> generation_config -> generation_config
(** [with_max_new_tokens tokens config] sets maximum new tokens to generate.

    Limits only newly generated tokens, not including prompt. When set,
    overrides max_length behavior. Must be positive. *)

val with_min_length : int -> generation_config -> generation_config
(** [with_min_length len config] sets minimum total sequence length.

    Prevents generation from stopping (e.g., on EOS tokens) until this length is
    reached. Includes prompt tokens. Must be non-negative. *)

val with_min_new_tokens : int -> generation_config -> generation_config
(** [with_min_new_tokens tokens config] sets minimum new tokens to generate.

    Similar to min_length but counts only newly generated tokens. Prevents early
    stopping until this many new tokens are produced. *)

val with_no_repeat_ngram : int -> generation_config -> generation_config
(** [with_no_repeat_ngram size config] prevents n-gram repetition.

    Ensures no n-gram of the specified size repeats in the output. Set to 0 to
    disable. Useful for reducing repetitive text. *)

val with_num_beams : int -> generation_config -> generation_config
(** [with_num_beams beams config] enables beam search.

    Number of beams for beam search decoding. 1 means greedy decoding. Higher
    values explore more possibilities but increase computation. *)

val with_do_sample : bool -> generation_config -> generation_config
(** [with_do_sample sample config] toggles sampling vs greedy decoding.

    When true, uses sampling (temperature, top-k, top-p). When false, uses
    greedy decoding (always picks most likely token). *)

(** {2 Preset Configurations} *)

val creative_writing : generation_config
(** [creative_writing] preset for creative text generation.

    Optimized for creative writing, storytelling, and imaginative content. Uses
    higher temperature and moderate filtering for diverse, engaging output.

    Settings:
    - Temperature: 0.8 (creative but coherent)
    - Top-p: 0.9 (good diversity)
    - Repetition penalty: 1.2 (reduce repetition)
    - N-gram blocking: 3
    - Max new tokens: 512
    - Sampling: enabled *)

val chat : generation_config
(** [chat] preset for conversational AI responses.

    Balanced configuration for chat applications. Provides coherent responses
    while maintaining some creativity and avoiding repetitive patterns.

    Settings:
    - Temperature: 0.7 (conversational balance)
    - Top-p: 0.95 (moderate diversity)
    - Repetition penalty: 1.1 (avoid repetitive responses)
    - Max new tokens: 512
    - Sampling: enabled *)

val code_generation : generation_config
(** [code_generation] preset for generating source code.

    Conservative settings optimized for code completion and generation. Lower
    temperature ensures syntactically correct and logical code.

    Settings:
    - Temperature: 0.2 (focused, deterministic)
    - Top-k: 5 (very conservative filtering)
    - Repetition penalty: 1.0 (no penalty)
    - Max new tokens: 1024
    - Sampling: enabled *)

val factual : generation_config
(** [factual] preset for factual, informative text.

    Very conservative settings for factual question answering and informative
    content. Minimizes hallucination and ensures consistent, reliable output.

    Settings:
    - Temperature: 0.3 (relatively deterministic)
    - Top-k: 10 (strict filtering)
    - Repetition penalty: 1.1 (mild penalty)
    - Max new tokens: 256
    - Sampling: enabled *)

val from_preset : string -> generation_config
(** [from_preset name] loads configuration by preset name.

    Convenience function to get preset configurations by string name. Useful for
    configuration files or user input.

    Supported presets:
    - ["creative_writing"]
    - ["chat"]
    - ["code_generation"]
    - ["factual"]

    Returns {!default} for unrecognized preset names. *)

(** {1 Logits Processors} *)

val temperature_warper : temperature:float -> logits_processor
(** [temperature_warper ~temperature] creates a temperature scaling processor.

    Applies temperature scaling to logits: logits[i] = logits[i] / temperature.
    Higher temperature (> 1.0) increases randomness by flattening the
    distribution. Lower temperature (< 1.0) sharpens the distribution, making
    high-probability tokens more likely.

    @param temperature
      Scaling factor. Must be positive. 1.0 leaves distribution unchanged. *)

val top_k_warper : k:int -> logits_processor
(** [top_k_warper ~k] creates a top-k filtering processor.

    Keeps only the k most likely tokens, setting all others to negative
    infinity. This limits the sampling vocabulary to the most promising
    candidates.

    @param k
      Number of top tokens to keep. Must be positive. Use 0 to disable
      filtering. *)

val top_p_warper : p:float -> logits_processor
(** [top_p_warper ~p] creates a top-p (nucleus) sampling processor.

    Keeps tokens whose cumulative probability mass is at most p, setting others
    to negative infinity. Adapts vocabulary size based on probability
    distribution shape.

    @param p Probability threshold between 0.0 and 1.0. 1.0 disables filtering.
*)

val repetition_penalty : penalty:float -> logits_processor
(** [repetition_penalty ~penalty] creates a repetition penalty processor.

    Applies exponential penalty to tokens that have already appeared in the
    sequence. For each token that appeared, its logit is divided by the penalty
    factor.

    @param penalty
      Penalty factor. > 1.0 discourages repetition, < 1.0 encourages it. 1.0
      applies no penalty. Typical values: 1.05-1.2. *)

val no_repeat_ngram : ngram_size:int -> logits_processor
(** [no_repeat_ngram ~ngram_size] prevents n-gram repetition.

    Sets logits to negative infinity for tokens that would create repeating
    n-grams. Prevents exact sequences of the specified length from repeating.

    @param ngram_size Size of n-grams to prevent. Must be positive. 0 disables.
*)

val min_length : min_length:int -> eos_token_ids:int list -> logits_processor
(** [min_length ~min_length ~eos_token_ids] enforces minimum total sequence
    length.

    Sets EOS token logits to negative infinity until the sequence reaches
    min_length. Prevents premature termination of generation.

    @param min_length Minimum total tokens (including prompt).
    @param eos_token_ids List of end-of-sequence token IDs to block. *)

val min_new_tokens :
  min_new_tokens:int -> eos_token_ids:int list -> logits_processor
(** [min_new_tokens ~min_new_tokens ~eos_token_ids] enforces minimum new tokens.

    Similar to min_length but counts only newly generated tokens, not including
    prompt. Blocks EOS until enough new content is generated.

    @param min_new_tokens Minimum new tokens to generate.
    @param eos_token_ids List of end-of-sequence token IDs to block. *)

val bad_words : bad_words_ids:int list list -> logits_processor
(** [bad_words ~bad_words_ids] blocks specific token sequences.

    Sets logits to negative infinity for tokens that would complete any of the
    forbidden sequences. Useful for content filtering and avoiding unwanted
    outputs.

    @param bad_words_ids
      List of forbidden token sequences. Each sequence is a list of token IDs.
*)

val force_words :
  force_words_ids:int list list -> iteration:int -> logits_processor
(** [force_words ~force_words_ids ~iteration] forces specific tokens to appear.

    Boosts probability of tokens that start required sequences. More complex
    than bad_words as it tracks generation progress and required sequence
    completion.

    @param force_words_ids List of required token sequences.
    @param iteration Current generation step for sequence tracking. *)

val custom :
  name:string ->
  process:(prompt_length:int -> token_ids -> logits -> logits) ->
  logits_processor
(** [custom ~name ~process] creates a custom logits processor.

    Allows implementation of arbitrary logits transformations. The process
    function receives context about the generation state and must return
    modified logits.

    @param name Human-readable name for debugging.
    @param process Function that transforms logits given generation context. *)

(** {1 Stopping Criteria} *)

val max_length_criteria : max_length:int -> stopping_criterion
(** [max_length_criteria ~max_length] stops when total sequence length is
    reached.

    Counts both prompt tokens and generated tokens. Generation stops when the
    sequence length equals or exceeds max_length.

    @param max_length Maximum total tokens in the sequence. *)

val max_new_tokens_criteria : max_new_tokens:int -> stopping_criterion
(** [max_new_tokens_criteria ~max_new_tokens] stops after generating enough new
    tokens.

    Counts only newly generated tokens, excluding the original prompt. Useful
    when prompt length varies but you want consistent output length.

    @param max_new_tokens Maximum new tokens to generate. *)

val eos_token_criteria : eos_token_ids:int list -> stopping_criterion
(** [eos_token_criteria ~eos_token_ids] stops when end-of-sequence token is
    generated.

    Monitors the last generated token and stops if it matches any of the
    specified EOS token IDs. This is the natural way for models to signal
    completion.

    @param eos_token_ids List of token IDs that signal end-of-sequence. *)

val max_time_criteria : max_time:float -> stopping_criterion
(** [max_time_criteria ~max_time] stops generation after time limit.

    Compares current time against generation start time. Useful for preventing
    generation from running indefinitely and ensuring responsive applications.

    @param max_time Maximum generation time in seconds. *)

val stop_strings_criteria :
  stop_strings:string list ->
  decoder:(token_ids -> string) ->
  stopping_criterion
(** [stop_strings_criteria ~stop_strings ~decoder] stops when specific strings
    appear.

    Decodes the current token sequence to text and checks if it ends with any of
    the stop strings. Useful for stopping on specific phrases or patterns.

    @param stop_strings List of strings that should stop generation.
    @param decoder Function to convert token IDs back to text. *)

val custom_criteria :
  name:string ->
  should_stop:(prompt_length:int -> start_time:float -> token_ids -> bool) ->
  stopping_criterion
(** [custom_criteria ~name ~should_stop] creates custom stopping logic.

    Allows implementation of arbitrary stopping conditions. The should_stop
    function receives generation context and returns whether to halt generation.

    @param name Human-readable name for debugging.
    @param should_stop
      Function that determines when to stop based on generation state. *)

(** {1 Main Generation Functions} *)

type generation_output = {
  sequences : int list list;
      (** Generated token sequences. For single sequence generation, contains
          one list. For beam search or multiple sequences, contains multiple
          lists. *)
  scores : float list list option;
      (** Generation scores for each step, if output_scores is enabled in
          config. Each inner list corresponds to scores for one sequence. *)
  attentions : float array list option;
      (** Attention weights for each layer, if output_attentions is enabled.
          Format depends on model architecture. *)
  hidden_states : float array list option;
      (** Hidden states for each layer, if output_hidden_states is enabled.
          Format depends on model architecture. *)
}
(** Output structure for the generate function containing all requested
    generation artifacts. *)

val generate :
  model:(token_ids -> logits) ->
  ?input_ids:token_ids ->
  ?generation_config:generation_config ->
  ?logits_processor:logits_processor_list ->
  ?stopping_criteria:stopping_criteria_list ->
  unit ->
  generation_output
(** [generate ~model ?input_ids ?generation_config ?logits_processor
     ?stopping_criteria ()] generates token sequences using the provided model.

    Core generation function that provides full control and detailed output.
    Suitable for advanced use cases requiring access to scores, attention, or
    multiple sequences.

    @param model
      Function that takes token sequence and returns logits for next token.
    @param input_ids Initial token sequence to extend. Defaults to empty list.
    @param generation_config Configuration parameters. Defaults to {!default}.
    @param logits_processor
      List of processors to apply to logits. Defaults to empty.
    @param stopping_criteria
      List of custom stopping conditions. Defaults to empty.

    @return Generation output with sequences and optional artifacts.

    Generation process:
    + Start with input_ids (or empty sequence)
    + Apply model to get logits for next token
    + Process logits through processor pipeline
    + Sample next token based on processed logits
    + Check stopping criteria and config limits
    + Repeat until stopping condition met *)

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
(** [generate_text ~model ~tokenizer ~decoder ?prompt ?generation_config
     ?logits_processor ?stopping_criteria ()] generates text using the provided
    model and tokenizer.

    Simplified text generation interface that handles tokenization
    automatically. Most convenient for typical text generation tasks. Returns
    the generated text as a string.

    @param model
      Function that takes token sequence and returns logits for next token.
    @param tokenizer Function to convert text to token IDs.
    @param decoder Function to convert token IDs back to text.
    @param prompt Initial text prompt to extend. Defaults to empty string.
    @param generation_config Configuration parameters. Defaults to {!default}.
    @param logits_processor
      List of processors to apply to logits. Defaults to empty.
    @param stopping_criteria
      List of custom stopping conditions. Defaults to empty.

    @return Generated text as a string.

    The function:
    + Tokenizes the prompt using the provided tokenizer
    + Calls {!generate} with the tokenized input
    + Decodes the result back to text
    + Returns only the newly generated portion (excluding original prompt) *)

(** {1 Utilities} *)

val apply_processors :
  processors:logits_processor_list ->
  prompt_length:int ->
  tokens:token_ids ->
  logits:logits ->
  logits
(** [apply_processors ~processors ~prompt_length ~tokens ~logits] applies all
    processors sequentially.

    Executes each processor in the list in order, passing the output of each
    processor as input to the next. Returns the final processed logits array.

    @param processors List of processors to apply in sequence.
    @param prompt_length Length of the original prompt (for processor context).
    @param tokens Current token sequence being generated.
    @param logits Input logits array to process.

    @return Processed logits array of the same length as input.

    Processing pipeline:
    + Start with original logits
    + Apply first processor: [logits' = processor_1(logits)]
    + Apply second processor: [logits'' = processor_2(logits')]
    + Continue until all processors applied
    + Return final logits *)

val check_stopping :
  criteria:stopping_criteria_list ->
  prompt_length:int ->
  start_time:float ->
  tokens:token_ids ->
  bool
(** [check_stopping ~criteria ~prompt_length ~start_time ~tokens] evaluates all
    stopping criteria.

    Checks each stopping criterion in the list. Returns true if ANY criterion
    indicates that generation should stop. Returns false only if all criteria
    indicate generation should continue.

    @param criteria List of stopping criteria to evaluate.
    @param prompt_length Length of the original prompt.
    @param start_time Generation start time (Unix timestamp).
    @param tokens Current token sequence.

    @return true if generation should stop, false to continue.

    Evaluation logic:
    - Empty criteria list returns false (continue generation)
    - Criteria are evaluated in order until one returns true
    - Short-circuit evaluation: stops checking after first true result *)
