(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Activation functions.

    The activations are pure, element-wise nonlinearities for building models:
    apply them between parameterized layers. Each preserves the shape and dtype
    of its argument, is meant for floating-point tensors, and is differentiable
    through Rune in both reverse and forward mode. The general functions models
    also use are in {!Nx}: {!Nx.sigmoid}, {!Nx.tanh}, {!Nx.softmax} and
    {!Nx.log_softmax}. *)

(** {1:elementwise Element-wise activations} *)

val relu : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [relu x] is zero where [x <= 0] and [x] elsewhere: [-0.] maps to [0.] and
    NaN stays NaN. Its derivative at [0] is [0]. *)

val leaky_relu : ?negative_slope:float -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [leaky_relu x] is [x] where [x > 0] and [negative_slope * x] elsewhere.

    [negative_slope] defaults to [0.01]. *)

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
(** [silu x] is [x * Nx.sigmoid x], also known as Swish. *)

val softplus : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [softplus x] is [log(1 + exp(x))], a smooth approximation of {!relu}.
    Computed as [max(x, 0) + log(1 + exp(-|x|))], which does not overflow for
    large [x]. *)

(** {1:sampling Sampling masks}

    A sampling policy is a pipeline over next-token logits: divide by a
    temperature ({!Nx.div}), mask with {!keep_top_k} and {!keep_top_p}, and draw
    with {!Nx.Rng.categorical}, or take {!Nx.argmax} for greedy decoding. The
    masks set the entries they remove to negative infinity and keep the shape,
    so they compose in any order and trace under {!Rune.jit}.

    Their parameters are tensors, a scalar or one entry per row, so a batch can
    mix requests with different settings. Under {!Rune.jit} pass them as inputs
    of the step: like any captured value, a captured parameter is a constant of
    the compiled program.

    Pass float32 logits. Casting the selected position's logits up costs nothing
    and keeps the cumulative sum of {!keep_top_p} and the noise of the draw out
    of half precision, where neither has the resolution a vocabulary needs. *)

val keep_top_k : k:Nx.int64_t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [keep_top_k ~k logits] is [logits] with every entry below the [k]-th largest
    of its row set to negative infinity; the last axis is the vocabulary.
    Entries equal to the [k]-th largest are all kept, so ties can leave more
    than [k]. [k] is clamped to the vocabulary: [k <= 1] keeps the maximum,
    [k >= vocab] keeps everything.

    Raises [Invalid_argument] if [logits] is a scalar or [k] is neither a scalar
    nor of [logits]'s leading shape. *)

val keep_top_p : p:(float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [keep_top_p ~p logits] is [logits] with the least likely entries of each row
    set to negative infinity, keeping the fewest entries whose softmax
    probabilities sum to at least [p] (nucleus sampling). The most likely entry
    always stays, so [p <= 0] is greedy; [p >= 1] keeps everything. Ties at the
    threshold are kept.

    Raises [Invalid_argument] if [logits] is a scalar or [p] is neither a scalar
    nor of [logits]'s leading shape. *)
