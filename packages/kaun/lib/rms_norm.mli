(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Root mean square normalization (Zhang and Sennrich, 2019).

    RMS norm divides each feature vector — the last axis of the input — by its
    root mean square, then applies a learned per-feature scale ([gamma]). It is
    layer normalization without the centering and without a shift: one reduction
    instead of two, and the normalization Llama-class models use. Construct
    parameters with {!init} or {!make} and normalize with {!apply}. *)

(** {1:types Types} *)

type 'a t = { gamma : 'a }
(** The type for RMS-norm parameters over payload ['a]. At tensor payloads
    [gamma] (the scale) has shape [[| dim |]], one entry per feature. *)

(** {1:constructors Constructors} *)

val make : dim:int -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
(** [make ~dim dtype] is a fresh normalization over [dim] features with [gamma]
    all ones.

    Raises [Invalid_argument] if [dim] is not positive. *)

val init : dim:int -> Nx.float32_t t
(** [init ~dim] is [make ~dim Nx.float32]. *)

(** {1:applying Applying} *)

val apply :
  ?eps:float -> (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply p x] scales each vector along [x]'s last axis to unit root mean
    square and rescales it:

    {v x / sqrt (mean (x * x) + eps) * gamma v}

    where [mean] is taken along the last axis; every other axis is a batch axis.
    [eps] keeps the division finite for a zero vector and defaults to [1e-6].
    The result has [x]'s shape. Differentiable through Rune.

    For half and quarter precision inputs (float16, bfloat16, float8) the mean
    square and the division run in a float32 island and the normalized values
    are cast back to [x]'s dtype before the [gamma] scale, as in
    {!Layer_norm.apply}.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis does not
    have size [dim], or if [eps] is negative. *)

(** {1:structure Structure} *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
(** [walk c p] walks [p]'s parameters, [gamma], at that path: the layer's
    {!Nx.Ptree.S} instance. [Nx.Ptree.instantiate (module Rms_norm)] is the
    layer at one dtype, and [Nx.Ptree.cast (module Rms_norm) dtype p] converts
    its precision. *)
