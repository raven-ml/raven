(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Neural network layer constructors.

    This module provides functional layer constructors for building neural
    networks. Each function creates a layer configuration that returns a
    {!module_}, which encapsulates parameter initialization and forward
    computation. Layers can be composed using {!sequential} to build complex
    architectures.

    All layers follow a consistent pattern: they take architecture parameters
    (dimensions, hyperparameters) and optional initialization strategies,
    returning a module that can be initialized with random number generators and
    applied to input tensors.

    {1 Usage Overview}

    Create layers by calling constructor functions:
    {[
      let dense = Layer.linear ~in_features:784 ~out_features:128 () in
      let activation = Layer.relu () in
    ]}

    Compose layers into networks:
    {[
      let network = Layer.sequential [
        Layer.linear ~in_features:784 ~out_features:128 ();
        Layer.relu ();
        Layer.dropout ~rate:0.2 ();
        Layer.linear ~in_features:128 ~out_features:10 ();
      ] in
    ]}

    Initialize and apply:
    {[
      let params = Kaun.init network ~rngs ~dtype in
      let output = Kaun.apply network params ~training:true input in
    ]} *)

type module_ = {
  init :
    'layout. rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> Ptree.t;
      (** [init ~rngs ~dtype] initializes module parameters.

          Creates a parameter tree containing all trainable parameters for this
          module. The function is polymorphic over layout and device to support
          different tensor backends and memory layouts.

          @param rngs
            Random number generator key for deterministic initialization
          @param device Target device (CPU, CUDA, etc.) for parameter allocation
          @param dtype Data type specification, typically [Rune.float32]

          The RNG key should be split appropriately for modules with multiple
          parameters to ensure independent initialization. *)
  apply :
    'layout.
    Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t;
      (** [apply params ~training ?rngs input] performs forward computation.

          Executes the module's forward pass using the provided parameters and
          input tensor.

          @param params Parameter tree from [init] function
          @param training
            Whether module is in training mode (affects dropout, batch norm,
            etc.)
          @param rngs Optional RNG key for stochastic operations (dropout, etc.)
          @param input Input tensor to transform

          The training flag enables different behaviors:
          - Dropout: Applied only when [training=true]
          - Batch normalization: Uses batch statistics when [training=true]
          - Other regularization: Activated based on training mode

          RNG is required for stochastic operations during training. Operations
          needing randomness will fail if [rngs] is [None] when [training=true].
      *)
}

(** {1 Convolutional Layers} *)

val conv1d :
  in_channels:int ->
  out_channels:int ->
  ?kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding:[ `Same | `Valid | `Causal ] ->
  unit ->
  module_
(** [conv1d ~in_channels ~out_channels ?kernel_size ?stride ?dilation ?padding
     ()] creates a 1D convolutional layer over inputs of shape
    [batch; in_channels; length]. Supports `Same, `Valid, and `Causal padding.
    Default: kernel_size=3, stride=1, dilation=1, padding=`Same. *)

val conv2d :
  in_channels:int ->
  out_channels:int ->
  ?kernel_size:int * int ->
  unit ->
  module_
(** [conv2d ~in_channels ~out_channels ?kernel_size ()] creates a 2D
    convolutional layer.

    Performs 2D convolution over 4D input tensors of shape
    [batch_size, in_channels, height, width]. The layer maintains learnable
    weight and bias parameters.

    @param in_channels Number of input channels
    @param out_channels Number of output filters
    @param kernel_size Filter dimensions as [(height, width)]. Default: [(3, 3)]

    The weight tensor has shape
    [out_channels, in_channels, kernel_height, kernel_width] and is initialized
    using Glorot uniform initialization. The bias tensor has shape
    [out_channels] and is zero-initialized.

    {4 Example}
    {[
      let conv = Layer.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:(5, 5) () in
      (* Processes RGB images (3 channels) to produce 64 feature maps with 5x5 filters *)
    ]} *)

(** {1 Dense Layers} *)

val linear :
  in_features:int ->
  out_features:int ->
  ?weight_init:Initializers.t ->
  ?bias_init:Initializers.t ->
  unit ->
  module_
(** [linear ~in_features ~out_features ?weight_init ?bias_init ()] creates a
    fully connected layer.

    Applies linear transformation [y = xW^T + b] where [x] is input, [W] is
    weight matrix, and [b] is bias vector. Accepts inputs of any shape with last
    dimension matching [in_features].

    @param in_features Size of input feature dimension
    @param out_features Size of output feature dimension
    @param weight_init
      Weight initialization strategy. Default: {!Initializers.glorot_uniform}
    @param bias_init
      Bias initialization strategy. Default: {!Initializers.zeros}

    The weight tensor has shape [out_features, in_features] and bias has shape
    [out_features].

    {4 Examples}
    {[
      let classifier = Layer.linear ~in_features:512 ~out_features:10 () in
      (* Maps 512-dimensional features to 10 class logits *)

      let custom_init = Layer.linear
        ~in_features:256 ~out_features:128
        ~weight_init:(Initializers.he_normal ())
        ~bias_init:(Initializers.constant 0.1) () in
    ]} *)

(** {1 Regularization Layers} *)

val dropout : rate:float -> unit -> module_
(** [dropout ~rate ()] creates a dropout layer for regularization.

    During training, randomly sets elements to zero with probability [rate] and
    scales remaining elements by [1 / (1 - rate)] to maintain expected values.
    During evaluation, applies identity transformation.

    @param rate Dropout probability in range [0.0, 1.0]

    Requires random number generator during training. No learnable parameters.

    {4 Example}
    {[
      let drop = Layer.dropout ~rate:0.5 () in
      (* Randomly zeros 50% of activations during training *)
    ]} *)

val batch_norm : num_features:int -> unit -> module_
(** [batch_norm ~num_features ()] creates a batch normalization layer.

    Normalizes inputs across the batch dimension, learning scale and shift
    parameters. Applies transformation [y = γ((x - μ) / σ) + β] where μ and σ
    are batch statistics, and γ, β are learnable parameters.

    @param num_features
      Number of features to normalize (typically channel dimension)

    Maintains running statistics for evaluation mode. Parameters include scale
    (γ), bias (β), running mean, and running variance. *)

(** {1 Pooling Layers} *)

val max_pool2d : kernel_size:int * int -> ?stride:int * int -> unit -> module_
(** [max_pool2d ~kernel_size ?stride ()] creates a 2D max pooling layer.

    Applies maximum operation over spatial windows, reducing spatial dimensions
    while preserving channel dimension.

    @param kernel_size Pooling window size as [(height, width)]
    @param stride
      Pooling stride as [(height, width)]. Default: same as [kernel_size]

    No learnable parameters. *)

val avg_pool2d : kernel_size:int * int -> ?stride:int * int -> unit -> module_
(** [avg_pool2d ~kernel_size ?stride ()] creates a 2D average pooling layer.

    Applies average operation over spatial windows, providing smoother
    downsampling compared to max pooling.

    @param kernel_size Pooling window size as [(height, width)]
    @param stride
      Pooling stride as [(height, width)]. Default: same as [kernel_size]

    No learnable parameters. *)

(** {1 Reshape Layers} *)

val flatten : unit -> module_
(** [flatten ()] creates a flatten layer that reshapes multidimensional inputs
    to 2D.

    Preserves batch dimension while flattening all other dimensions. Transforms
    shape [batch_size, d1, d2, ..., dn] to [batch_size, d1 * d2 * ... * dn].

    Commonly used before dense layers in CNN architectures. No learnable
    parameters. *)

(** {1 Activation Functions} *)

val relu : unit -> module_
(** [relu ()] creates a ReLU activation layer applying [max(0, x)] elementwise.

    Most common activation for hidden layers. Computationally efficient with
    good gradient flow for positive inputs. No learnable parameters. *)

val sigmoid : unit -> module_
(** [sigmoid ()] creates a sigmoid activation layer applying [1 / (1 + exp(-x))]
    elementwise.

    Maps inputs to range (0, 1). Commonly used for binary classification and
    gating mechanisms. No learnable parameters. *)

val tanh : unit -> module_
(** [tanh ()] creates a hyperbolic tangent activation layer applying [tanh(x)]
    elementwise.

    Maps inputs to range (-1, 1). Provides stronger gradients than sigmoid but
    can suffer from vanishing gradients. No learnable parameters. *)

val gelu : unit -> module_
(** [gelu ()] creates a GELU activation layer.

    Applies Gaussian Error Linear Unit activation, popular in transformer
    architectures. Smoother alternative to ReLU with better gradient properties.
    No learnable parameters. *)

val swish : unit -> module_
(** [swish ()] creates a Swish activation layer applying [x * sigmoid(x)]
    elementwise.

    Self-gated activation function that can outperform ReLU in deep networks. No
    learnable parameters. *)

(** {1 Composition} *)

val sequential : module_ list -> module_
(** [sequential layers] creates a sequential composition of layers.

    Applies layers in order, threading output of each layer as input to the
    next. The resulting module's parameters are the union of all component layer
    parameters.

    @param layers List of layers to compose

    {4 Example}
    {[
      let mlp = Layer.sequential [
        Layer.linear ~in_features:784 ~out_features:256 ();
        Layer.relu ();
        Layer.dropout ~rate:0.3 ();
        Layer.linear ~in_features:256 ~out_features:10 ();
      ] in
    ]} *)

(** {1 Advanced Layers} *)

val einsum :
  einsum_str:string ->
  shape:int array ->
  ?kernel_init:Initializers.t ->
  unit ->
  module_
(** [einsum ~einsum_str ~shape ?kernel_init ()] creates a parameterized Einstein
    summation layer.

    Implements learnable tensor contractions specified by Einstein notation.
    Useful for implementing custom linear transformations and attention
    mechanisms.

    @param einsum_str Einstein summation string describing the contraction
    @param shape Shape of the learnable kernel parameter
    @param kernel_init
      Kernel initialization strategy. Default: {!Initializers.glorot_uniform} *)

(** {1 Normalization Layers} *)

val rms_norm :
  dim:int -> ?eps:float -> ?scale_init:Initializers.t -> unit -> module_
(** [rms_norm ~dim ?eps ?scale_init ()] creates a Root Mean Square normalization
    layer.

    Applies RMS normalization with learnable scaling. Normalizes by the RMS of
    activations rather than full statistics like batch normalization.

    @param dim Dimension to normalize over
    @param eps Small constant for numerical stability. Default: [1e-6]
    @param scale_init
      Scale parameter initialization. Default: {!Initializers.ones} *)

val layer_norm :
  dim:int -> ?eps:float -> ?elementwise_affine:bool -> unit -> module_
(** [layer_norm ~dim ?eps ?elementwise_affine ()] creates a layer normalization
    layer.

    Normalizes activations across the feature dimension within each sample.
    Popular in transformer architectures for stable training.

    @param dim Dimension to normalize over
    @param eps Small constant for numerical stability. Default: [1e-6]
    @param elementwise_affine
      Whether to learn scale and shift parameters. Default: [true] *)

(** {1 Embedding Layers} *)

val embedding :
  vocab_size:int ->
  embed_dim:int ->
  ?scale:bool ->
  ?embedding_init:Initializers.t ->
  unit ->
  module_
(** [embedding ~vocab_size ~embed_dim ?scale ?embedding_init ()] creates an
    embedding lookup layer.

    Maps discrete tokens (integers) to dense vectors. Commonly used as the first
    layer in NLP models to convert token IDs to continuous representations.

    @param vocab_size Size of the vocabulary (number of possible tokens)
    @param embed_dim Dimensionality of embedding vectors
    @param scale
      Whether to scale embeddings by [sqrt(embed_dim)]. Default: [false]
    @param embedding_init
      Embedding matrix initialization. Default: {!Initializers.normal}

    The embedding matrix has shape [vocab_size, embed_dim]. *)

(** {1 Attention and Position Encoding} *)

val mlp :
  in_features:int ->
  hidden_features:int ->
  out_features:int ->
  ?activation:[ `relu | `gelu | `swish ] ->
  ?dropout:float ->
  unit ->
  module_
(** [mlp ~in_features ~hidden_features ~out_features ...] creates a multi-layer
    perceptron (feed-forward network).

    Standard MLP architecture: Linear -> Activation -> Dropout -> Linear ->
    Dropout Commonly used in transformers and other architectures.

    @param in_features Input dimension
    @param hidden_features Hidden layer dimension
    @param out_features Output dimension
    @param activation Activation function. Default: [`gelu]
    @param dropout Dropout probability. Default: [0.0] *)

(** {1 Recurrent Layers} *)

val rnn :
  input_size:int ->
  hidden_size:int ->
  ?return_sequences:bool ->
  ?learned_init:bool ->
  unit ->
  module_
(** Simple tanh RNN over a sequence. Input [batch; seq; input_size], output
    [batch; hidden_size] (last hidden state). *)

val gru :
  input_size:int ->
  hidden_size:int ->
  ?return_sequences:bool ->
  ?learned_init:bool ->
  unit ->
  module_
(** GRU over a sequence. Input/output like rnn. *)

val lstm :
  input_size:int ->
  hidden_size:int ->
  ?return_sequences:bool ->
  ?learned_init:bool ->
  unit ->
  module_
(** LSTM over a sequence. Input/output like rnn. *)

(** {1 Positional Encodings} *)

val positional_embedding_learned :
  max_len:int -> embed_dim:int -> unit -> module_
(** Adds learned positional embeddings to input [batch; seq; embed_dim]. *)

val positional_encoding_sinusoidal_table :
  max_len:int ->
  embed_dim:int ->
  dtype:(float, 'layout) Rune.dtype ->
  (float, 'layout) Rune.t
(** Create a [max_len; embed_dim] sinusoidal positional encoding table (not
    trainable). Can be added to token embeddings. *)
