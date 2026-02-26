(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Composable neural network layers.

    A {!type:t} pairs parameter/state initialization with a forward computation.
    Layers compose with {!compose} for heterogeneous pipelines (for example
    embeddings to dense layers) and with {!sequential} for homogeneous float
    pipelines. *)

(** {1:types Types} *)

type 'layout vars
(** The type for model variables.

    [params] are trainable variables consumed by {!Optim}. [state] is
    non-trainable mutable state updated by forward passes (for example running
    statistics in {!batch_norm}). *)

val params : 'layout vars -> Ptree.t
(** [params v] is [v]'s trainable parameter tree. *)

val state : 'layout vars -> Ptree.t
(** [state v] is [v]'s non-trainable mutable state tree. *)

val dtype : 'layout vars -> (float, 'layout) Rune.dtype
(** [dtype v] is [v]'s floating dtype witness. *)

val with_params : 'layout vars -> Ptree.t -> 'layout vars
(** [with_params v params] is [v] with replaced trainable parameters. *)

val with_state : 'layout vars -> Ptree.t -> 'layout vars
(** [with_state v state] is [v] with replaced non-trainable state. *)

val make_vars :
  params:Ptree.t ->
  state:Ptree.t ->
  dtype:(float, 'layout) Rune.dtype ->
  'layout vars
(** [make_vars ~params ~state ~dtype] builds model variables.

    This is mainly useful for layer constructors implemented outside the
    {!Layer} module. *)

type ('input, 'output) t = {
  init :
    'layout.
    rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> 'layout vars;
  apply :
    'layout 'in_elt.
    params:Ptree.t ->
    state:Ptree.t ->
    dtype:(float, 'layout) Rune.dtype ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ?ctx:Context.t ->
    ('input, 'in_elt) Rune.t ->
    ('output, 'layout) Rune.t * Ptree.t;
}
(** The type for layers.

    [init] creates fresh [params] and [state]. [apply] computes a forward pass
    and returns updated [state].

    The input tensor's dtype witness ['in_elt] is independent of the model's
    float dtype witness ['layout]. This allows layers like {!embedding} to
    accept [int32_elt] indices while the model parameters use [float32_elt].
    Float-consuming layers (e.g. {!linear}) require the input dtype to match the
    model dtype exactly and raise [Invalid_argument] on mismatch.

    [ctx] carries per-call auxiliary data (attention masks, position ids,
    encoder memory). Most layers ignore it; transformer layers read from it
    using well-known key names. See {!Context}. *)

val init :
  ('a, 'b) t ->
  rngs:Rune.Rng.key ->
  dtype:(float, 'layout) Rune.dtype ->
  'layout vars
(** [init m ~rngs ~dtype] is [m]'s fresh variables.

    The RNG key is split internally by composite layers such as {!compose} and
    {!sequential}. *)

val apply :
  ('a, 'b) t ->
  'layout vars ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  ?ctx:Context.t ->
  ('a, 'in_elt) Rune.t ->
  ('b, 'layout) Rune.t * 'layout vars
(** [apply m vars ~training ?rngs ?ctx x] is the forward pass of [m].

    Returns [(y, vars')] where [params vars' = params vars] and [state vars'] is
    the updated state from the forward pass.

    The input tensor's dtype witness ['in_elt] is independent of the model's
    float dtype witness ['layout]. For float-consuming layers, the input must
    have the same dtype as the model; a mismatch raises [Invalid_argument].

    [training] controls stochastic/stateful behavior. For example, {!dropout}
    uses dropout masks only when [training = true], and {!batch_norm} updates
    running statistics only when [training = true].

    [ctx] carries per-call auxiliary data such as attention masks. See
    {!Context}. *)

(** {1:compose Composition} *)

val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
(** [compose left right] applies [left] then [right].

    Parameters and state are stored as {!Ptree.Dict} nodes with keys ["left"]
    and ["right"]. The RNG key is split between both layers during
    initialization and forward pass. *)

val sequential : (float, float) t list -> (float, float) t
(** [sequential layers] applies [layers] in order.

    Parameters and state are stored as {!Ptree.List} nodes with one entry per
    layer. The RNG key is split per layer during initialization and forward
    pass.

    Raises [Invalid_argument] if runtime parameter/state list lengths do not
    match [layers]. *)

(** {1:dense Dense} *)

val linear :
  in_features:int ->
  out_features:int ->
  ?weight_init:Init.t ->
  ?bias_init:Init.t ->
  unit ->
  (float, float) t
(** [linear ~in_features ~out_features ?weight_init ?bias_init ()] is the fully
    connected map [xW + b].

    [weight_init] defaults to {!Init.glorot_uniform ()}. [bias_init] defaults to
    {!Init.zeros}.

    Parameters:
    - [weight] with shape [[in_features; out_features]].
    - [bias] with shape [[out_features]]. *)

(** {1:conv Convolution} *)

val conv1d :
  in_channels:int ->
  out_channels:int ->
  ?kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding:[ `Same | `Valid | `Causal ] ->
  unit ->
  (float, float) t
(** [conv1d ~in_channels ~out_channels ?kernel_size ?stride ?dilation ?padding
     ()] is 1D convolution over inputs shaped [[batch; in_channels; length]].

    [kernel_size] defaults to [3]. [stride] defaults to [1]. [dilation] defaults
    to [1]. [padding] defaults to [`Same].

    Parameters:
    - [weight] with shape [[out_channels; in_channels; kernel_size]].
    - [bias] with shape [[out_channels]]. *)

val conv2d :
  in_channels:int ->
  out_channels:int ->
  ?kernel_size:int * int ->
  unit ->
  (float, float) t
(** [conv2d ~in_channels ~out_channels ?kernel_size ()] is 2D convolution over
    inputs shaped [[batch; in_channels; height; width]].

    [kernel_size] defaults to [(3, 3)]. Stride is fixed to [(1, 1)] and padding
    mode is [`Same].

    Parameters:
    - [weight] with shape [[out_channels; in_channels; kh; kw]].
    - [bias] with shape [[out_channels]]. *)

(** {1:norm Normalization} *)

val layer_norm : dim:int -> ?eps:float -> unit -> (float, float) t
(** [layer_norm ~dim ?eps ()] is layer normalization with learnable affine
    parameters.

    [eps] defaults to [1e-5].

    Parameters:
    - [gamma] with shape [[dim]].
    - [beta] with shape [[dim]]. *)

val rms_norm : dim:int -> ?eps:float -> unit -> (float, float) t
(** [rms_norm ~dim ?eps ()] is RMS normalization with learnable scale.

    [eps] defaults to [1e-6].

    Parameters:
    - [scale] with shape [[dim]]. *)

val batch_norm : num_features:int -> unit -> (float, float) t
(** [batch_norm ~num_features ()] is stateful batch normalization.

    During training, batch statistics are used and running statistics are
    updated. During evaluation, running statistics are used and preserved.

    Normalization axes are inferred from rank:
    - rank 2 uses [[0]].
    - rank 3 uses [[0; 2]].
    - rank 4 uses [[0; 2; 3]].
    - other ranks use [[0]].

    Parameters:
    - [scale] with shape [[num_features]].
    - [bias] with shape [[num_features]].

    State:
    - [running_mean] with shape [[num_features]].
    - [running_var] with shape [[num_features]]. *)

(** {1:embed Embedding} *)

val embedding :
  vocab_size:int -> embed_dim:int -> ?scale:bool -> unit -> (int32, float) t
(** [embedding ~vocab_size ~embed_dim ?scale ()] is an embedding lookup layer.

    Inputs are int32 token indices. Output shape is
    [indices_shape ++ [embed_dim]].

    [scale] defaults to [true]. When [true], output vectors are multiplied by
    [sqrt embed_dim].

    Parameters:
    - [embedding] with shape [[vocab_size; embed_dim]]. *)

(** {1:reg Regularization} *)

val dropout : rate:float -> unit -> (float, float) t
(** [dropout ~rate ()] is elementwise dropout.

    When [training = false], it is identity. When [training = true], [~rngs] is
    required.

    Raises [Invalid_argument] if [rate] is outside [0.0 <= rate < 1.0]. Raises
    [Invalid_argument] if [~rngs] is missing during training when [rate > 0.0].
*)

(** {1:act Activation Layers} *)

val relu : unit -> (float, float) t
(** [relu ()] is [max(0, x)]. No parameters. *)

val gelu : unit -> (float, float) t
(** [gelu ()] is the Gaussian error linear unit. No parameters. *)

val silu : unit -> (float, float) t
(** [silu ()] is [x * sigmoid(x)]. No parameters. *)

val tanh : unit -> (float, float) t
(** [tanh ()] is hyperbolic tangent. No parameters. *)

val sigmoid : unit -> (float, float) t
(** [sigmoid ()] is the logistic function. No parameters. *)

(** {1:pool Pooling} *)

val max_pool2d :
  kernel_size:int * int -> ?stride:int * int -> unit -> (float, float) t
(** [max_pool2d ~kernel_size ?stride ()] is 2D max pooling.

    [stride] defaults to [kernel_size]. No parameters. *)

val avg_pool2d :
  kernel_size:int * int -> ?stride:int * int -> unit -> (float, float) t
(** [avg_pool2d ~kernel_size ?stride ()] is 2D average pooling.

    [stride] defaults to [kernel_size]. No parameters. *)

(** {1:reshape Reshape} *)

val flatten : unit -> (float, float) t
(** [flatten ()] flattens all dimensions after the batch dimension.

    [[batch; d1; ...; dn]] becomes [[batch; d1 * ... * dn]].

    Raises [Invalid_argument] if the input rank is [0]. *)
