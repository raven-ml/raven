(** Saga - Fast tokenization and text processing for ML in OCaml.

    Saga is a comprehensive text processing library for machine learning
    applications, providing fast tokenization, statistical language models, and
    modern text generation capabilities. It combines simplicity for common use
    cases with flexibility for advanced workflows.

    {1 Library Overview}

    Saga consists of four main components:
    - {!section-tokenization}: Fast tokenization with BPE, WordPiece, and custom
      methods
    - {!section-io}: Efficient file I/O utilities for large text corpora
    - {!section-lm}: High-level statistical language models (n-grams)
    - {!section-sampling}: Advanced text generation with composable processors

    All components work together seamlessly but can be used independently.

    {1 Quick Start}

    {2 Simple tokenization and text processing}
    {[
      open Saga

      (* Basic word tokenization *)
      let tokens = tokenize "Hello, world! How are you?"
      (* Returns: ["Hello"; ","; "world"; "!"; "How"; "are"; "you"; "?"] *)

      (* Character-level tokenization *)
      let chars = tokenize ~method_:`Chars "Hello"
      (* Returns: ["H"; "e"; "l"; "l"; "o"] *)

      (* Batch processing with padding *)
      let batch_ids = encode_batch [ "Hello world"; "Hi there" ] ~pad:true
      (* Returns: padded tensor of token IDs *)
    ]}

    {2 Training a language model}
    {[
      (* Load training data *)
      let texts = read_lines "training_data.txt"

      (* Create and train a bigram model *)
      let model =
        LM.ngram ~n:2 ~tokenizer:(tokenizer `Words) () |> LM.train texts

      (* Generate new text *)
      let generated =
        LM.generate model ~num_tokens:20 ~temperature:0.8 () Printf.printf
          "Generated: %s\n" generated

      (* Evaluate on test data *)
      let test_perplexity =
        LM.perplexity model "the quick brown fox" Printf.printf
          "Test perplexity: %.2f\n" test_perplexity
    ]}

    {2 Advanced text generation}
    {[
      (* Create a model function (typically a neural network) *)
      let model_fn token_ids =
        (* Your neural network forward pass *)
        Array.make 50000 0.0 (* Example: uniform logits *)

      (* Configure generation with custom processors *)
      let config =
        Sampler.default
        |> Sampler.with_temperature 0.9
        |> Sampler.with_top_k 40
        |> Sampler.with_repetition_penalty 1.1

      (* Generate with fine-grained control *)
      let result =
        Sampler.generate_text ~model:model_fn
          ~tokenizer:(encode ~vocab:(vocab [ "hello"; "world" ]))
          ~decoder:(decode (vocab [ "hello"; "world" ]))
          ~prompt:"Hello" ~generation_config:config ()
    ]}

    {1 Common Patterns}

    {2 Text preprocessing pipeline}
    {[
      let preprocess_texts texts =
        texts
        |> List.map (normalize ~lowercase:true ~collapse_whitespace:true)
        |> List.filter (fun s -> String.length s > 10) (* Filter short texts *)
        |> List.map (tokenize ~method_:`Words)
    ]}

    {2 Model comparison and evaluation}
    {[
      let compare_models texts test_texts =
        let models =
          [
            ("unigram", LM.ngram ~n:1 ());
            ("bigram", LM.ngram ~n:2 ());
            ("trigram", LM.ngram ~n:3 ~smoothing:0.1 ());
          ]
        in
        List.map
          (fun (name, model) ->
            let trained = LM.train model texts in
            let avg_perp =
              List.map (LM.perplexity trained) test_texts
              |> List.fold_left ( +. ) 0.
              |> fun sum -> sum /. float_of_int (List.length test_texts)
            in
            (name, avg_perp))
          models
    ]}

    {2 Custom tokenization workflows}
    {[
      let create_code_tokenizer () =
        tokenizer (`Regex {|[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[(){}[\].,;]|\S|})
        |> Tokenizer.with_normalizer (normalize ~collapse_whitespace:true)

      let process_code_files filenames =
        filenames |> List.map read_lines |> List.flatten
        |> List.map (Tokenizer.run (create_code_tokenizer ()))
    ]}

    {1 Performance Tips}

    - Use {!read_lines_lazy} for very large files to avoid memory issues
    - Character-level models work well for small vocabularies (names, short
      sequences)
    - Word-level models are better for natural language with large vocabularies
    - Higher n-gram orders need exponentially more training data
    - BPE and WordPiece tokenizers handle out-of-vocabulary words better than
      simple word splitting

    {1 Integration with Other Libraries}

    Saga integrates well with:
    - {{:https://github.com/janestreet/base}Base/Core} for functional
      programming utilities
    - {{:https://nx.ocaml.org}Nx} for tensor operations and neural networks
    - {{:https://dune.build}Dune} for build system integration
    - Standard CSV/JSON libraries for data loading *)

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

(** {1:section-lm Language Models}

    High-level statistical language models with simple training and generation
    APIs. *)

(* Re-export LM high-level API at the top level. *)
include module type of Lm
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

    Use this for production text generation systems or when you need fine
    control over the generation process. For simple statistical models,
    {!LM.generate} may be more convenient.

    See {!Sampler} for detailed documentation and examples. *)

(** {1 Examples}

    {2 Quick tokenization}
    {[
      open Saga

      (* Simple char tokenization *)
      let tok = Tokenizer.create ~model:(Models.chars ())
      let enc = Tokenizer.encode tok ~sequence:(Either.Left "Hello world!") ()
      let ids = Encoding.get_ids enc
      let text = Tokenizer.decode tok (Array.to_list ids) ()

      (* BPE tokenization *)
      let tok = tokenizer (`BPE ("vocab.json", "merges.txt"))
      let batch = encode_batch tok [ "Hello"; "World" ] ~padding:true
    ]}

    {2 Training a language model}
    {[
      (* Train a bigram model *)
      let texts = [ "The cat sat"; "The dog ran"; "The cat ran" ]
      let tok = tokenizer `Words
      let model = LM.train_ngram ~n:2 tok texts

      (* Generate text *)
      let generated =
        LM.generate model ~max_tokens:50 ~temperature:0.8 tok print_endline
          generated
    ]}

    {2 Custom tokenizer}
    {[
      (* Tokenizer with normalization *)
      let tok =
        tokenizer `Words
        |> Tokenizer.with_normalizer (normalize ~lowercase:true)

      (* Regex tokenizer for code *)
      let code_tok = tokenizer (`Regex {|[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|.|})
    ]}

    {2 End-to-End Name Generator}
    {[
      let build_name_generator training_file =
        (* Load and train character-level model *)
        let names = read_lines training_file in
        let model =
          LM.ngram ~n:3 ~tokenizer:(tokenizer `Chars) ~smoothing:0.1 ()
          |> LM.train names in

        (* Return generator function *)
        fun ?(temperature=0.8) ?(max_len=12) () ->
          LM.generate model ~num_tokens:max_len ~temperature ()

      (* Generate 10 new names *)
      let gen = build_name_generator "names.txt" in
      List.init 10 (fun _ -> gen ()) |> List.iter (Printf.printf "%s\n")
    ]}

    {2 Advanced Neural Model Integration}
    {[
      let setup_neural_generation neural_model vocab_file =
        let vocab = vocab_load vocab_file in
        let tokenize_fn text = encode ~vocab text in
        let decode_fn ids = decode vocab ids in

        (* Wrap neural model for Sampler API *)
        let model_fn token_ids =
          let tensor = Nx.of_array1 (Array.of_list token_ids) in
          let logits = neural_model tensor in
          Nx.to_array1 logits
        in

        (* Creative writing configuration *)
        let config =
          Sampler.creative_writing
          |> Sampler.with_max_new_tokens 200
          |> Sampler.with_repetition_penalty 1.15
        in

        let processors =
          [
            Sampler.temperature_warper ~temperature:1.1;
            Sampler.top_p_warper ~p:0.9;
            Sampler.no_repeat_ngram ~ngram_size:3;
          ]
        in

        (* Generate with stopping criteria *)
        Sampler.generate_text ~model:model_fn ~tokenizer:tokenize_fn
          ~decoder:decode_fn ~generation_config:config
          ~logits_processor:processors
    ]}

    {2 Batch Processing Pipeline}
    {[
      let process_corpus_directory input_dir output_file =
        (* Custom preprocessing pipeline *)
        let preprocess text =
          text |> normalize ~lowercase:true ~collapse_whitespace:true
          |> fun s -> if String.length s > 20 then Some s else None
        in

        Sys.readdir input_dir |> Array.to_list
        |> List.filter (fun f -> Filename.extension f = ".txt")
        |> List.map (fun f -> Filename.concat input_dir f)
        |> List.map read_lines_lazy |> Seq.concat
        |> Seq.filter_map preprocess
        |> List.of_seq |> write_lines output_file
    ]} *)
