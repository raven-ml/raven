(** High-level language model API.

    This module provides a simple, batteries-included interface for training and
    using statistical language models. It handles tokenization, vocabulary
    building, model training, and text generation with sensible defaults.

    Designed for ease of use while maintaining flexibility through optional
    parameters. Perfect for prototyping, experimentation, and educational
    purposes.

    {1 Overview}

    The workflow follows three main steps: 1. Create a model with {!ngram} 2.
    Train it on text data with {!train} 3. Generate new text with {!generate} or
    evaluate with {!perplexity}

    All models are immutable - training returns a new model instance.

    {1 Quick Start}

    Train a bigram model on names and generate new ones:
    {[
      let names = [ "alice"; "bob"; "charlie"; "diana"; "eve" ]
      let model = Saga.ngram ~n:2 ~tokenizer:(Saga.tokenizer `Chars) ()
      let trained_model = Saga.train model names
      let new_name = Saga.generate trained_model ~num_tokens:10 ()
      (* Returns: "alicia" or similar character-level generation *)
    ]}

    Word-level model with custom settings:
    {[
      let texts = [ "the cat sat"; "the dog ran"; "the cat ran" ]

      let model =
        Saga.ngram ~n:3 ~smoothing:0.05 ~min_freq:2
          ~tokenizer:(Saga.tokenizer `Words) ()
        |> Saga.train texts

      let story =
        Saga.generate model ~prompt:"the cat" ~num_tokens:20 ~temperature:0.9 ()
    ]}

    {1 Key Concepts}

    {2 N-grams}

    N-gram models predict the next token based on the previous n-1 tokens.
    Higher n captures more context but requires more training data:
    - n=1 (unigram): No context, just token frequencies
    - n=2 (bigram): Depends on 1 previous token
    - n=3 (trigram): Depends on 2 previous tokens
    - n=4,5: Higher context, needs lots of data

    {2 Smoothing}

    Smoothing handles unseen token sequences:
    - Add-k smoothing: Adds small count to all n-grams (default: 0.01)
    - Higher values = more uniform distribution
    - Lower values = sharper predictions

    {2 Tokenization}

    Models work on token sequences. Built-in tokenizers:
    - [`Words]: Split on whitespace and punctuation
    - [`Chars]: Unicode character-level
    - Custom tokenizers for domain-specific needs

    {1 Advanced Usage}

    Save and load trained models:
    {[
      Saga.save trained_model "my_model.bin"

      let loaded_model = Saga.load "my_model.bin"
    ]}

    Evaluate model quality:
    {[
      let test_texts = [ "the quick brown fox" ]
      let perplexity = Saga.perplexity trained_model (List.hd test_texts)
      (* Lower perplexity = better model fit *)
    ]}

    Batch evaluation and generation:
    {[
      let samples =
        Saga.pipeline model training_texts ~num_samples:50 ~temperature:1.2 ()
          List.iter
          (fun (text, perp) ->
            Printf.printf "%s (perplexity: %.2f)\n" text perp)
          samples
    ]} *)

(** {1 Core Types} *)

type model
(** Opaque language model that can be trained and used for generation.

    Models are immutable - training operations return new model instances.
    Supports n-gram models (n=1-5) with potential for extension to other
    statistical models like Markov chains or probabilistic context-free
    grammars. *)

(** {1 Training} *)

val train : model -> string list -> model
(** [train model texts] trains the model on a list of text strings.

    The training process: 1. Tokenizes each text using the model's tokenizer 2.
    Automatically adds BOS/EOS tokens (tokenizer-specific) 3. Builds or updates
    the vocabulary 4. Fits the statistical backend (n-gram counts, etc.)

    BOS/EOS tokens are tokenizer-aware:
    - Character tokenizer: Uses "." as both BOS and EOS
    - Word tokenizer: Uses "<bos>" and "<eos>"
    - Custom tokenizers: Uses configured special tokens

    Returns a new trained model instance (original is unchanged).

    {[
      let untrained = Saga.ngram ~n:2 ~tokenizer:(Saga.tokenizer `Words) ()
      let trained = Saga.train untrained [ "hello world"; "world peace" ]
      (* trained model now knows bigrams: <bos>+hello, hello+world, world+<eos>,
         etc. *)
    ]} *)

(** {1 Text Generation} *)

val generate :
  model ->
  ?num_tokens:int ->
  ?temperature:float ->
  ?top_k:int ->
  ?top_p:float ->
  ?seed:int ->
  ?min_new_tokens:int ->
  ?prompt:string ->
  unit ->
  string
(** [generate model ?num_tokens ?temperature ?top_k ?top_p ?seed ?min_new_tokens
     ?prompt ()] generates text from the trained model.

    Generation continues until:
    - Maximum tokens reached ([num_tokens])
    - EOS token generated (unless blocked by [min_new_tokens])
    - Model produces invalid continuation

    @param num_tokens Maximum tokens to generate (default: 20)
    @param temperature
      Sampling randomness: 0.1 = conservative, 2.0 = very random (default: 1.0)
    @param top_k
      Keep only top-k most likely tokens, 0 = disabled (default: None)
    @param top_p
      Nucleus sampling: keep tokens with cumulative probability ≤ p (default:
      None)
    @param seed Random seed for reproducible generation (default: None = random)
    @param min_new_tokens
      Block EOS tokens until at least this many tokens generated (default: None)
    @param prompt
      Initial text prompt, tokenizer-specific (default: empty = auto BOS)
    @return Generated text as clean, decoded string

    The generated text is automatically cleaned:
    - BOS/EOS tokens removed
    - Tokenizer-specific post-processing applied
    - Invalid unicode sequences handled gracefully

    Examples:
    {[
      (* Conservative, deterministic generation *)
      let result = Saga.generate model ~temperature:0.1 ~num_tokens:10 ()

      (* Creative generation with nucleus sampling *)
      let story =
        Saga.generate model ~temperature:1.2 ~top_p:0.9 ~num_tokens:50 ()

      (* Prompted generation *)
      let completion =
        Saga.generate model ~prompt:"Once upon a time" ~num_tokens:30 ()

      (* Ensure minimum length *)
      let long_text = Saga.generate model ~min_new_tokens:100 ~num_tokens:200 ()
    ]} *)

(** {1 Model Evaluation} *)

val score : model -> string -> float
(** [score model text] computes the log-probability of the text sequence under
    the model.

    Returns the sum of log-probabilities for each token in the sequence. More
    negative values indicate lower probability (worse fit).

    The text is automatically tokenized and BOS/EOS tokens added as needed. For
    unseen n-grams, uses the model's smoothing strategy.

    To convert to perplexity manually: [exp(-score / token_count)]

    {[
      let score = Saga.score model "hello world"
      (* Returns: -5.234 (example negative log-probability) *)

      let manual_perplexity =
        let tokens = (* tokenize "hello world" *) 3 in
        exp (-.score /. float_of_int tokens)
    ]} *)

val perplexity : model -> string -> float
(** [perplexity model text] computes perplexity, a standard metric for language
    model quality.

    Perplexity = exp(-average_log_probability)
    - Lower values indicate better model fit
    - Perplexity ≈ average branching factor at each step
    - Perfect prediction gives perplexity = 1.0
    - Random guessing gives perplexity = vocabulary_size

    Text is automatically wrapped with BOS/EOS tokens for proper evaluation.

    {[
      let perp1 = Saga.perplexity model "the cat sat"  (* Returns: 12.5 *)
      let perp2 = Saga.perplexity model "xyz qwerty"   (* Returns: 45.2 (worse fit) *)

      (* Lower perplexity = better model *)
      assert (perp1 < perp2)
    ]} *)

val perplexities : model -> string list -> float list
(** [perplexities model texts] computes perplexity for multiple texts
    efficiently.

    Equivalent to [List.map (perplexity model) texts] but may be optimized for
    batch processing in the future.

    {[
      let test_set = [ "the cat sat"; "dogs run fast"; "hello world" ]
      let perps = Saga.perplexities model test_set

      let avg_perp =
        List.fold_left ( +. ) 0. perps
        /. float_of_int (List.length perps) Printf.printf
             "Average test perplexity: %.2f\n" avg_perp
    ]} *)

(** {1 Model Creation} *)

val ngram :
  n:int ->
  ?smoothing:float ->
  ?min_freq:int ->
  ?specials:string list ->
  ?tokenizer:'a Saga_tokenizers.Tokenizer.t ->
  unit ->
  model
(** [ngram ~n ?smoothing ?min_freq ?specials ?tokenizer ()] creates an n-gram
    language model.

    N-gram models predict tokens based on the previous n-1 tokens. This
    implementation uses Maximum Likelihood Estimation (MLE) with add-k
    smoothing, similar to NLTK's approach.

    @param n
      Order of the n-gram model (1-5 supported, higher orders possible via
      custom backends)
    @param smoothing
      Add-k smoothing parameter (default: 0.01). Higher values create more
      uniform distributions
    @param min_freq
      Minimum frequency threshold for n-gram inclusion (default: 1)
    @param specials
      Special tokens for BOS/EOS, auto-detected from tokenizer if omitted
    @param tokenizer
      Tokenizer to use, auto-inferred if omitted (characters for short texts)

    Special tokens are tokenizer-aware:
    - Character tokenizer: [".", "."] (period for both BOS and EOS)
    - Word tokenizer: ["<bos>", "<eos>"] (explicit start/end markers)
    - Custom tokenizers: Use provided specials or tokenizer defaults

    The returned model is untrained - use {!train} to fit it on data.

    {4 Usage Examples}

    NLTK-style workflow:
    {[
      let model =
        Saga.ngram ~n:2 ~tokenizer:(Saga.tokenizer `Chars) ~min_freq:2 ()

      let trained_model = Saga.train model names

      let generated =
        Saga.generate trained_model ~num_tokens:15 ~temperature:0.8 ()

      let quality = Saga.perplexity trained_model "emma" (* Lower = better *)
    ]}

    Fluent chaining style:
    {[
      let model = Saga.ngram ~n:2 ~smoothing:0.01 () |> Saga.train names

      let generated_name =
        Saga.generate model ~num_tokens:20 ~temperature:1.0 () Printf.printf
          "Generated: %s\n" generated_name
    ]}

    Custom configuration:
    {[
      let model =
        Saga.ngram ~n:3 ~smoothing:0.05 ~min_freq:3
          ~specials:[ "<start>"; "<end>" ]
          ~tokenizer:(Saga.tokenizer (`Regex {|\\w+|[.,!?]|}))
          ()
    ]} *)

(** {1 Model Persistence} *)

val save : model -> string -> unit
(** [save model filename] saves the trained model to a binary file.

    Serializes the complete model state including:
    - Vocabulary mappings
    - N-gram counts and statistics
    - Tokenizer configuration
    - Model hyperparameters

    Use {!load} to restore the model later.

    {[
      Saga.save trained_model "my_language_model.bin"
    ]} *)

val load : string -> model
(** [load filename] loads a previously saved model from disk.

    The loaded model is immediately ready for generation and evaluation without
    requiring retraining.

    @raise Sys_error if file doesn't exist or is corrupted
    @raise Invalid_argument if file format is incompatible

    {[
      let model = Saga.load "my_language_model.bin"
      let text = Saga.generate model ~num_tokens:50 ()
    ]} *)

(** {1 Convenience Functions} *)

val pipeline :
  model ->
  string list ->
  ?num_samples:int ->
  ?temperature:float ->
  ?top_k:int ->
  ?top_p:float ->
  ?seed:int ->
  unit ->
  (string * float) list
(** [pipeline model texts ?num_samples ?temperature ?top_k ?top_p ?seed ()] is a
    convenience function that trains a model and generates samples with their
    perplexities.

    This function: 1. Trains the model on the provided texts 2. Generates
    [num_samples] text samples using the trained model 3. Computes perplexity
    for each generated sample 4. Returns (generated_text, perplexity) pairs

    Useful for quick experimentation and model evaluation.

    @param model Untrained or partially trained model
    @param texts Training texts
    @param num_samples Number of samples to generate (default: 20)
    @param temperature,top_k,top_p,seed
      Generation parameters (same as {!generate})
    @return
      List of (generated_text, perplexity) pairs, sorted by perplexity (best
      first)

    {4 Usage Examples}

    Quick model evaluation:
    {[
      let names = Saga.IO.read_lines "names.txt"
      let model = Saga.ngram ~n:2 ()

      let samples =
        Saga.pipeline model names ~num_samples:20 ~temperature:1.0 ()
          (* Print best samples (lowest perplexity) *)
          List.iter
          (fun (name, perp) ->
            Printf.printf "%s (perplexity: %.2f)\\n" name perp)
          (List.take 5 samples)
    ]}

    Parameter exploration:
    {[
      let compare_temperatures temps texts =
        List.map (fun temp ->
          let samples = Saga.pipeline model texts ~temperature:temp ~num_samples:10 ()
          let avg_perp =
            List.fold_left (fun acc (_, p) -> acc +. p) 0. samples /. 10.
          (temp, avg_perp)
        ) temps
    ]} *)
