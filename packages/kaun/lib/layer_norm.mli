(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Layer normalization (Ba, Kiros and Hinton, 2016).

    Layer norm standardizes each feature vector — the last axis of the input —
    to zero mean and unit variance, then applies a learned per-feature scale
    ([gamma]) and shift ([beta]). Unlike batch normalization it is stateless and
    independent of the batch: the same function at training and inference time.
    Construct parameters with {!init} or {!make} and normalize with {!apply};
    the traversals supply the {!Nx.Ptree.Uniform} and checkpoint plumbing. *)

(** {1:types Types} *)

type 'a t = { gamma : 'a; beta : 'a }
(** The type for layer-norm parameters over payload ['a]. At tensor payloads,
    [gamma] (the scale) and [beta] (the shift) both have shape [[| dim |]], one
    entry per normalized feature. *)

(** {1:constructors Constructors} *)

val make : dim:int -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t t
(** [make ~dim dtype] is a fresh identity normalization over [dim] features:
    [gamma] all ones, [beta] all zeros.

    Raises [Invalid_argument] if [dim] is not positive. *)

val init : dim:int -> Nx.float32_t t
(** [init ~dim] is [make ~dim Nx.float32]. *)

(** {1:applying Applying} *)

val apply : ?eps:float -> (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply p x] normalizes each vector along [x]'s last axis and rescales it:

    {v (x - mean(x)) / sqrt (var(x) + eps) * gamma + beta v}

    where [mean] and [var] (the biased variance) are taken along the last axis;
    every other axis is treated as a batch axis. [eps] keeps the division finite
    for constant vectors — a constant vector maps to [beta] — and defaults to
    [1e-5]. The result has [x]'s shape. Differentiable through Rune.

    For half and quarter precision inputs (float16, bfloat16, float8) the
    statistics and normalization are computed in a float32 island: [x] is
    upcast, normalized at float32, and the normalized values are cast back to
    [x]'s dtype before the [gamma]/[beta] affine transform. Float32 and float64
    inputs use their own dtype throughout, exactly as if the island were absent.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis does not
    have size [dim], or if [eps] is negative. *)

(** {1:traversals Traversals}

    Payload traversals in the order [gamma] then [beta], satisfying the
    {!Nx.Ptree.Uniform} contract. Leaf paths are ["gamma"] and ["beta"]. *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f p] is [p] with [f] applied to every payload leaf. [map (Nx.cast dt)]
    converts a layer's precision; the cast is differentiable through Rune. *)

val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
(** [map2 f p q] combines [p] and [q] leafwise with [f]. *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f p] applies [f] to every payload leaf of [p]. *)

val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f acc p] reduces [p] leafwise, threading each leaf's path. *)

val fold2 : (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
(** [fold2 f acc p q] is like {!fold} across two layers. *)

val names : 'a t -> string t
(** [names p] is [{ gamma = "gamma"; beta = "beta" }]. *)
