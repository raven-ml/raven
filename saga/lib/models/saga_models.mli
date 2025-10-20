(** Statistical language models and sequence models.

    This module provides classical statistical models for natural language
    processing and sequence analysis:
    - {!Ngram}: N-gram language models with smoothing and perplexity evaluation
    - {!Hmm}: Hidden Markov Models for sequence tagging and unsupervised
      learning
    - {!Pcfg}: Probabilistic Context-Free Grammars for parsing and tree
      generation

    All models support both training and inference, with EM-based parameter
    estimation where applicable.

    {1 Overview}

    These models operate on integer sequences representing tokenized text or
    observations. Use Saga's tokenization utilities to convert text into
    appropriate integer arrays before training or evaluation.

    {1 Quick Example}

    Training an n-gram model:
    {[
      open Saga.Models

      (* Tokenize text to integer arrays *)
      let vocab = Hashtbl.create 100 in
      let tokenize text =
        (* ... convert text to int array using vocab ... *)
        [||]

      (* Train a trigram model *)
      let sequences = List.map tokenize ["the cat sat"; "the dog ran"] in
      let model = Ngram.of_sequences ~order:3 sequences in

      (* Evaluate perplexity *)
      let test_seq = tokenize "the cat ran" in
      let ppl = Ngram.perplexity model test_seq
    ]}

    {1 Model Selection}

    - Use {!Ngram} for language modeling, text generation, and perplexity
      evaluation
    - Use {!Hmm} for sequence tagging tasks like POS tagging or speech
      recognition
    - Use {!Pcfg} for syntactic parsing and hierarchical structure learning *)

module Ngram = Ngram
(** Low-level n-gram language models.

    Provides n-gram models with configurable smoothing strategies for handling
    unseen contexts. Operates on pre-tokenized integer sequences for maximum
    efficiency. *)

module Hmm = Hmm
(** Hidden Markov Models for sequence analysis.

    Implements forward-backward algorithms, Viterbi decoding, and Baum-Welch
    training for discrete observation HMMs. *)

module Pcfg = Pcfg
(** Probabilistic Context-Free Grammars.

    Supports CKY parsing with inside-outside algorithms for training. Useful for
    syntactic analysis and tree-structured data. *)
