(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Activation functions for neural networks.

    All functions are differentiable through Rune's autodiff. The standard
    activations {!relu}, {!sigmoid} and {!tanh} are re-exported from {!Nx} for
    convenience. *)

(** {1:standard Standard activations} *)

val relu : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [relu x] is [max(x, 0)]. *)

val sigmoid : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [sigmoid x] is [1 / (1 + exp(-x))]. *)

val tanh : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [tanh x] is the hyperbolic tangent of [x]. *)

val relu6 : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [relu6 x] is [min(max(x, 0), 6)]. *)

val leaky_relu : ?negative_slope:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [leaky_relu x] is [max(x, negative_slope * x)].

    [negative_slope] defaults to [0.01]. *)

val hard_tanh : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [hard_tanh x] is [max(-1, min(1, x))]. *)

val hard_sigmoid :
  ?alpha:float -> ?beta:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [hard_sigmoid x] is [min(1, max(0, alpha * x + beta))].

    [alpha] defaults to [1/6]. [beta] defaults to [0.5]. *)

val prelu : alpha:(float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [prelu ~alpha x] is [max(0, x) + alpha * min(0, x)].

    [alpha] is a learnable tensor, broadcast against [x]. *)

(** {1:exponential Exponential family} *)

val elu : ?alpha:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [elu x] is [x] when [x >= 0] and [alpha * (exp(x) - 1)] otherwise.

    [alpha] defaults to [1.0]. *)

val selu : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [selu x] is [lambda * elu(x, alpha)] with self-normalizing constants. *)

val celu : ?alpha:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [celu x] is [max(0, x) + min(0, alpha * (exp(x/alpha) - 1))].

    [alpha] defaults to [1.0]. *)

(** {1:smooth Smooth activations} *)

val gelu : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [gelu x] is [0.5 * x * (1 + erf(x / sqrt(2)))]. *)

val gelu_approx : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [gelu_approx x] is the tanh-based approximation of {!gelu}. *)

val silu : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [silu x] is [x * sigmoid(x)] (also known as Swish). *)

val swish : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [swish x] is {!silu}. *)

val hard_silu : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [hard_silu x] is [x * hard_sigmoid(x)]. *)

val hard_swish : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [hard_swish x] is {!hard_silu}. *)

val mish : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [mish x] is [x * tanh(softplus(x))]. *)

val softplus : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [softplus x] is [log(1 + exp(x))]. *)

val softsign : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [softsign x] is [x / (abs(x) + 1)]. *)

val squareplus : ?b:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [squareplus x] is [0.5 * (x + sqrt(x^2 + b))].

    [b] defaults to [4.0]. *)

val log_sigmoid : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [log_sigmoid x] is [log(sigmoid(x))], computed in a numerically stable way
    by branching on the sign of [x]. *)

(** {1:gating Gating} *)

val glu : ?axis:int -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [glu x] splits [x] in half along [axis] and returns [left * sigmoid(right)].

    [axis] defaults to [-1].

    Raises [Invalid_argument] if the split does not produce two partitions. *)

(** {1:sparse Sparse activations} *)

val sparse_plus : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [sparse_plus x] is [x] when [x >= 1], [0] when [x <= -1], and
    [0.25 * (x + 1)^2] otherwise. *)

val sparse_sigmoid : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [sparse_sigmoid x] is [1] when [x >= 1], [0] when [x <= -1], and
    [0.5 * (x + 1)] otherwise. *)
