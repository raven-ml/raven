(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Layer normalization (Ba, Kiros and Hinton, 2016).

    Layer norm standardizes each feature vector — the last axis of the input —
    to zero mean and unit variance, then applies a learned per-feature scale
    ([gamma]) and, unless the layer is made without one, a learned shift
    ([beta]). Unlike batch normalization it is stateless and
    independent of the batch: the same function at training and inference time.
    Construct parameters with {!init} or {!make} and normalize with {!apply};
    {!walk} makes it a structure. Normalization without learned parameters is
    {!Nx.standardize} alone. *)

(** {1:types Types} *)

type ('a, _) t = { gamma : 'a; beta : 'a option }
(** The type for layer-norm parameters over payload ['a]. At tensor payloads,
    [gamma] (the scale) and [beta] (the shift), when present, have shape
    [[| dim |]], one entry per normalized feature. [beta] is [None] for a
    layer made without a shift ({!make}[ ~shift:false]): it has no shift
    parameter, so {!walk} skips it and {!apply} adds nothing. *)

(** {1:constructors Constructors} *)

val make :
  ?shift:bool ->
  dim:int ->
  (float, 'b) Nx.dtype ->
  'd Nx.Placement.t ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~dim dtype at] is a fresh identity normalization over [dim] features
    at [at]: [gamma] all ones and, with [shift] (defaults to [true]), [beta]
    all zeros. [~shift:false] sets [beta] to [None].

    Raises [Invalid_argument] if [dim] is not positive. *)

val init :
  ?shift:bool -> dim:int -> 'd Nx.Placement.t -> ('d Nx.float32_t, 'd) t
(** [init ?shift ~dim at] is [make ?shift ~dim Nx.float32 at]. *)

(** {1:applying Applying} *)

val apply :
  ?eps:float ->
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply p x] normalizes each vector along [x]'s last axis and rescales it:

    {v (x - mean(x)) / sqrt (var(x) + eps) * gamma + beta v}

    without the [+ beta] when [beta] is [None], where [mean] and [var] (the
    biased variance) are taken along the last axis; every other axis is
    treated as a batch axis. [eps] keeps the division finite for constant
    vectors and defaults to [1e-5]. The result has [x]'s shape.
    Differentiable through Rune.

    It is [Nx.standardize ~eps ~scale:gamma ?shift:beta x], whose doc states
    its algorithm and error bound: the affine transform is applied before the
    one rounding to [x]'s dtype, so a narrow [x] (float16, bfloat16, float8,
    float4) is normalized and rescaled at float32 and rounded once, and a
    constant vector of up to [2{^17}] features maps to [beta], or to zero
    without it.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis does not
    have size [dim], or if [eps] is negative or NaN. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [p]'s parameters: [gamma] at ["gamma"], then [beta] at
    ["beta"], reporting whether [beta] is present, so a layer with a shift and
    one without are distinct structures, and a checkpoint of one does not load
    into the other ([Nx_io.Archive.to_value]). It is the layer's
    {!Nx.Ptree.S} instance. [Nx.Ptree.instantiate (module Layer_norm)]
    is the layer at one dtype, and [Nx.Ptree.cast (module Layer_norm) dtype p]
    converts its precision. *)
