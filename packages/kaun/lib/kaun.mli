(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Neural networks as typed parameter records.

    Kaun has no layer or trainer abstraction. A layer is a record of parameters
    with a payload hole and an [apply] function; a model is a record of layers
    with one-line traversals (the {!Nx.Ptree.Uniform} contract, hand-written or
    derived with [ppx_ptree]; see {!Linear} for the pattern). [Nx.Ptree.instantiate]
    instantiates the model at its tensor type, a training step composes
    {!Rune.value_and_grad} with a [Vega] optimizer update, and the training
    loop is ordinary [Seq] iteration over {!Data} minibatches.

    {[
    let model = Nx.Ptree.instantiate (module Model)

    let step (params, ostate) (x, y) =
      let loss p = Loss.softmax_cross_entropy_sparse (Model.apply p x) y in
      let l, grads = Rune.value_and_grad model loss params in
      let params, ostate =
        Vega.adamw_step model ~lr:1e-3 ostate ~params ~grads
      in
      ((params, ostate), Nx.item [] l)
    ]}

    Random initialization and shuffling draw from the ambient {!Nx.Rng} scope;
    open one with {!Nx.Rng.with_key} for reproducibility. Under {!Rune.val-jit}
    what matters is the key the scope is rooted at, not whether a draw named
    one: root it at a key threaded through the step's inputs and the keyless
    draws inside compile, exactly as an explicit key ({!Dropout}) would. *)

(** {1:layers Layers}

    Parameterized building blocks. Each module pairs a parameter record with
    constructors ({!Linear.init}, {!Linear.make}), an [apply] function, and the
    traversals that make it compose into differentiable, checkpointable models.
*)

module Linear = Linear
(** Dense (fully connected) layers, and the model-as-record pattern. *)

module Embedding = Embedding
(** Token-id to dense-vector lookup tables. *)

module Conv = Conv
(** 2-D convolution layers (NCHW). *)

module Attention = Attention
(** Multi-head self-attention, and the pure
    {!Attention.scaled_dot_product_attention} core. *)

module Layer_norm = Layer_norm
(** Layer normalization over the feature axis. *)

module Batch_norm = Batch_norm
(** Batch normalization, with running statistics as explicit state. *)

(** {1:functions Stateless functions}

    Pure operations with no parameters: activations, pooling, dropout. *)

module Fn = Fn
(** Activation functions ([relu], [gelu], [softmax], ...). *)

module Pool = Pool
(** 2-D max and average pooling. *)

module Dropout = Dropout
(** Dropout with an explicit [~training] flag and optional {!Nx.Rng} key. *)

module Init = Init
(** Weight initializers (Glorot, He, LeCun, variance scaling). *)

(** {1:training Training} *)

module Loss = Loss
(** Scalar training objectives: regression and classification losses. *)

module Data = Data
(** Minibatch [Seq.t]s over in-memory tensors, with shuffling. *)

module Metric = Metric
(** Evaluation metrics: accuracy, precision/recall/F1, AUC-ROC. *)

(** {1:persistence Persistence} *)

module Checkpoint = Checkpoint
(** Save and load parameter structures as safetensors checkpoints. *)
