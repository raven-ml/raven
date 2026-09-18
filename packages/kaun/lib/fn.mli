(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Activation functions.

    Pure, stateless nonlinearities for building models: apply them between
    parameterized layers. Every function preserves the shape and dtype of its
    argument, is meant for floating-point tensors, and is differentiable through
    Rune in both reverse and forward mode. {!relu}, {!sigmoid} and {!tanh} equal
    their {!Nx} counterparts and are included so that [Fn] is a complete
    activation vocabulary.

    Everything is element-wise except {!softmax} and {!log_softmax}, which
    normalize along one axis. *)

(** {1:elementwise Element-wise activations} *)

val relu : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [relu x] is [max(x, 0)]. *)

val leaky_relu : ?negative_slope:float -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [leaky_relu x] is [x] where [x > 0] and [negative_slope * x] elsewhere.

    [negative_slope] defaults to [0.01]. *)

val sigmoid : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [sigmoid x] is [1 / (1 + exp(-x))]. Values lie in \[[0];[1]\]; the
    computation saturates without overflowing for large [|x|]. *)

val tanh : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [tanh x] is the hyperbolic tangent of [x]. *)

val gelu : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [gelu x] is the exact Gaussian error linear unit [x * Φ(x)], computed as
    [0.5 * x * (1 + erf(x / sqrt 2))] where [Φ] is the standard normal CDF.

    See {!gelu_approx} for the cheaper tanh approximation. *)

val gelu_approx : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [gelu_approx x] is the tanh approximation of {!gelu}:
    [0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))]. It agrees with
    {!gelu} to about [1e-3] absolute error; use it to match models trained with
    the approximation (GPT-2 style). *)

val silu : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [silu x] is [x * sigmoid(x)], also known as Swish. *)

val softplus : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [softplus x] is [log(1 + exp(x))], a smooth approximation of {!relu}.
    Computed as [max(x, 0) + log(1 + exp(-|x|))], which does not overflow for
    large [x]. *)

(** {1:normalizing Normalizing activations} *)

val softmax : ?axis:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [softmax ?axis x] is [exp(x) / sum(exp(x))] along [axis]. Values along
    [axis] are positive and sum to [1]. The maximum along [axis] is subtracted
    before exponentiating, so arbitrarily large inputs do not overflow.

    [axis] defaults to [-1]; negative values count from the last axis.

    Raises [Invalid_argument] if [axis] is out of bounds for [x]. *)

val log_softmax : ?axis:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [log_softmax ?axis x] is [log(softmax x)] along [axis], computed as
    [x - max(x) - log(sum(exp(x - max(x))))]: unlike composing [log] with
    {!softmax} it does not lose precision or produce [-inf] for entries far
    below the maximum. Same default and errors as {!softmax}. *)

(** {1:sampling Sampling masks}

    A sampling policy is a pipeline over next-token logits: divide by a
    temperature ({!Nx.div}), mask with {!top_k} and {!top_p}, and draw with
    {!Nx.Rng.categorical}, or take {!Nx.argmax} for greedy decoding. The masks
    set the entries they remove to negative infinity and keep the shape, so they
    compose in any order and trace under {!Rune.jit}.

    Their parameters are tensors, a scalar or one entry per row, so a batch can
    mix requests with different settings. Under {!Rune.jit} pass them as inputs
    of the step: like any captured value, a captured parameter is a constant of
    the compiled program.

    Pass float32 logits. Casting the selected position's logits up costs nothing
    and keeps the cumulative sum of {!top_p} and the noise of the draw out of
    half precision, where neither has the resolution a vocabulary needs. *)

val top_k : k:(int32, Nx.int32_elt) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [top_k ~k logits] is [logits] with every entry below the [k]-th largest of
    its row set to negative infinity; the last axis is the vocabulary. Entries
    equal to the [k]-th largest are all kept, so ties can leave more than [k].
    [k] is clamped to the vocabulary: [k <= 1] keeps the maximum, [k >= vocab]
    keeps everything.

    Raises [Invalid_argument] if [logits] is a scalar or [k] is neither a scalar
    nor of [logits]'s leading shape. *)

val top_p : p:(float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [top_p ~p logits] is [logits] with the least likely entries of each row set
    to negative infinity, keeping the fewest entries whose softmax probabilities
    sum to at least [p] (nucleus sampling). The most likely entry always stays,
    so [p <= 0] is greedy; [p >= 1] keeps everything. Ties at the threshold are
    kept.

    Raises [Invalid_argument] if [logits] is a scalar or [p] is neither a scalar
    nor of [logits]'s leading shape. *)
