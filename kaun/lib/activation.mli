(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Activation functions for neural networks.

    All functions are differentiable through Rune's autodiff. The standard
    activations {!relu}, {!sigmoid} and {!tanh} are re-exported from {!Rune} for
    convenience. *)

(** {1:standard Standard activations} *)

val relu : ('a, 'b) Rune.t -> ('a, 'b) Rune.t
(** [relu x] is [max(x, 0)]. *)

val sigmoid : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [sigmoid x] is [1 / (1 + exp(-x))]. *)

val tanh : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [tanh x] is the hyperbolic tangent of [x]. *)

val relu6 : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [relu6 x] is [min(max(x, 0), 6)]. *)

val leaky_relu :
  ?negative_slope:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [leaky_relu x] is [max(x, negative_slope * x)].

    [negative_slope] defaults to [0.01]. *)

val hard_tanh : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [hard_tanh x] is [max(-1, min(1, x))]. *)

val hard_sigmoid :
  ?alpha:float -> ?beta:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [hard_sigmoid x] is [min(1, max(0, alpha * x + beta))].

    [alpha] defaults to [1/6]. [beta] defaults to [0.5]. *)

val prelu : alpha:(float, 'b) Rune.t -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [prelu ~alpha x] is [max(0, x) + alpha * min(0, x)].

    [alpha] is a learnable tensor, broadcast against [x]. *)

(** {1:exponential Exponential family} *)

val elu : ?alpha:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [elu x] is [x] when [x >= 0] and [alpha * (exp(x) - 1)] otherwise.

    [alpha] defaults to [1.0]. *)

val selu : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [selu x] is [lambda * elu(x, alpha)] with self-normalizing constants. *)

val celu : ?alpha:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [celu x] is [max(0, x) + min(0, alpha * (exp(x/alpha) - 1))].

    [alpha] defaults to [1.0]. *)

(** {1:smooth Smooth activations} *)

val gelu : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [gelu x] is [0.5 * x * (1 + erf(x / sqrt(2)))]. *)

val gelu_approx : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [gelu_approx x] is the tanh-based approximation of {!gelu}. *)

val silu : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [silu x] is [x * sigmoid(x)] (also known as Swish). *)

val swish : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [swish x] is {!silu}. *)

val hard_silu : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [hard_silu x] is [x * hard_sigmoid(x)]. *)

val hard_swish : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [hard_swish x] is {!hard_silu}. *)

val mish : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [mish x] is [x * tanh(softplus(x))]. *)

val softplus : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [softplus x] is [log(1 + exp(x))]. *)

val softsign : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [softsign x] is [x / (abs(x) + 1)]. *)

val squareplus : ?b:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [squareplus x] is [0.5 * (x + sqrt(x^2 + b))].

    [b] defaults to [4.0]. *)

val log_sigmoid : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [log_sigmoid x] is [log(sigmoid(x))], computed in a numerically stable way
    by branching on the sign of [x]. *)

(** {1:gating Gating} *)

val glu : ?axis:int -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [glu x] splits [x] in half along [axis] and returns [left * sigmoid(right)].

    [axis] defaults to [-1].

    Raises [Invalid_argument] if the split does not produce two partitions. *)

(** {1:sparse Sparse activations} *)

val sparse_plus : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [sparse_plus x] is [x] when [x >= 1], [0] when [x <= -1], and
    [0.25 * (x + 1)^2] otherwise. *)

val sparse_sigmoid : (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [sparse_sigmoid x] is [1] when [x >= 1], [0] when [x <= -1], and
    [0.5 * (x + 1)] otherwise. *)
