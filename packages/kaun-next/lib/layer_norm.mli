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
    {!map}, {!map2}, {!iter} and {!names} supply the {!Nx.Ptree.S} and
    checkpoint plumbing. *)

(** {1:types Types} *)

type 'b params = { gamma : (float, 'b) Nx.t; beta : (float, 'b) Nx.t }
(** The type for layer-norm parameters with float dtype layout ['b]. [gamma]
    (the scale) and [beta] (the shift) both have shape [[| dim |]], one entry
    per normalized feature. *)

type t = Nx.float32_elt params
(** The type for single-precision layer norms, the common case. *)

(** {1:constructors Constructors} *)

val make : dim:int -> (float, 'b) Nx.dtype -> 'b params
(** [make ~dim dtype] is a fresh identity normalization over [dim] features:
    [gamma] all ones, [beta] all zeros.

    Raises [Invalid_argument] if [dim] is not positive. *)

val init : dim:int -> t
(** [init ~dim] is [make ~dim Nx.float32]. *)

(** {1:applying Applying} *)

val apply : ?eps:float -> 'b params -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply p x] normalizes each vector along [x]'s last axis and rescales it:

    {v (x - mean(x)) / sqrt (var(x) + eps) * gamma + beta v}

    where [mean] and [var] (the biased variance) are taken along the last axis;
    every other axis is treated as a batch axis. [eps] keeps the division finite
    for constant vectors — a constant vector maps to [beta] — and defaults to
    [1e-5]. The result has [x]'s shape. Differentiable through Rune-next.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis does not
    have size [dim], or if [eps] is negative. *)

(** {1:traversals Traversals}

    Plain traversals over the parameter leaves, in the order [gamma] then
    [beta]. They satisfy the {!Nx.Ptree.S} contract at any fixed ['b]. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to every parameter leaf. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p q] combines [p] and [q] leafwise with [f]. *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to every parameter leaf of [p]. *)

val names : 'b params -> string list
(** [names p] is [["gamma"; "beta"]], the checkpoint names of the parameter
    leaves in traversal order. See {!Checkpoint.Named}. *)
