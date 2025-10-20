(** Saga - Fast tokenization and text processing for ML in OCaml.

    Saga is a comprehensive text processing library for machine learning
    applications, providing fast tokenization and modern text generation
    capabilities. It combines simplicity for common use cases with flexibility
    for advanced workflows.

    {1 Library Overview}

    Saga consists of three main components:
    - {!section-tokenization}: Fast tokenization with BPE, WordPiece, and custom
      methods
    - {!section-io}: Efficient file I/O utilities for large text corpora
    - {!section-sampling}: Advanced text generation with composable processors

    All components work together seamlessly but can be used independently.

    {1 Quick Start}

    {b Note:} The [~sequence] parameter in tokenization functions accepts
    [Either.Left string] for raw text input or [Either.Right string list] for
    pretokenized input (where text is already split into words or tokens).

    {2 Advanced text generation}
    {[
      (* Create a model function (typically a neural network) *)
      let model_fn token_ids =
        (* Your neural network forward pass *)
        Array.make 50000 0.0 (* Example: uniform logits *)

      (* Build a vocabulary and tokenizer *)
      let tok = Tokenizer.create ~model:(Models.chars ())

      (* Configure generation with custom processors *)
      let config =
        Sampler.default
        |> Sampler.with_temperature 0.9
        |> Sampler.with_top_k 40
        |> Sampler.with_repetition_penalty 1.1

      (* Generate with fine-grained control *)
      let result =
        Sampler.generate ~model:model_fn ~tokenizer:tok ~prompt:"Hello"
          ~generation_config:config ()
    ]}

    {1 Performance Tips}

    - Use {!read_lines_lazy} for very large files to avoid memory issues
    - BPE and WordPiece tokenizers handle out-of-vocabulary words better than
      simple word splitting
    - Batch encoding with padding is more efficient than encoding sequences one
      at a time *)

(** {1:section-tokenization Tokenization}

    Fast and flexible tokenization supporting multiple algorithms and custom
    patterns. Handles everything from simple word splitting to advanced subword
    tokenization. *)

include
  module type of Saga_tokenizers
    with type Tokenizer.t = Saga_tokenizers.Tokenizer.t
(** @inline *)

(** {1:section-io File I/O}

    Efficient file I/O utilities optimized for large text corpora and ML
    workflows. *)

include module type of Io
(** @inline *)

(** {1:section-sampling Advanced Text Generation}

    Modern text generation with composable processors and fine-grained control,
    designed for integration with neural language models. *)

module Sampler = Sampler
(** Advanced text generation and sampling utilities.

    Provides Transformers-style generation with:
    - Composable logits processors (temperature, top-k, top-p, repetition
      penalties)
    - Flexible stopping criteria (length, time, custom strings)
    - Configuration builders and presets
    - Full compatibility with neural language models

    Integrates with neural models via simple function interface:
    [token_ids -> logits].

    Use this for production text generation systems or when you need fine
    control over the generation process. *)

(** {1 Examples}

    {2 Quick tokenization}
    {[
      open Saga

      (* Character tokenization *)
      let tok = Tokenizer.create ~model:(Models.chars ())
      let enc = Tokenizer.encode tok ~sequence:(Either.Left "Hello world!") ()
      let ids = Encoding.get_ids enc
      let text = Tokenizer.decode tok (Array.to_list ids) ()

      (* BPE tokenization with batch processing *)
      let tok = Tokenizer.from_file "tokenizer.json"

      let batch_enc =
        Tokenizer.encode_batch tok
          ~sequences:[ Either.Left "Hello"; Either.Left "World" ]
          ~padding:true ()
    ]}

    {2 Neural Model Integration}
    {[
      (* Wraps a neural model for Sampler integration. Example - illustrative
         pseudocode, adapt to your model API. *)
      let setup_neural_generation neural_model =
        let tok = Tokenizer.from_file "tokenizer.json" in

        (* Model function: token_ids -> logits *)
        let model_fn token_ids =
          (* Convert to your model's input format *)
          let input_tensor = your_tensor_creation_fn token_ids in
          let output_tensor = neural_model input_tensor in
          (* Convert output to float array *)
          your_tensor_to_array_fn output_tensor
        in

        (* Configure generation with custom processors *)
        let config =
          Sampler.creative_writing
          |> Sampler.with_max_new_tokens 200
          |> Sampler.with_repetition_penalty 1.15
        in

        (* Generate text *)
        Sampler.generate_text ~model:model_fn ~tokenizer:tok ~prompt:"Hello"
          ~generation_config:config ()
    ]} *)
