(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dense (fully connected) layers.

    A linear layer is a plain record of parameters: a weight matrix and an
    optional bias. Construct one with {!init} or {!make}, transform inputs with
    {!apply}, and compose layers into models by putting records inside records —
    the {!map}, {!map2}, {!iter} traversals and {!names} give any such model a
    one-line {!Nx.Ptree.S} (and checkpoint-ready) instance:

    {[
    module Mlp = struct
      type t = { l1 : Linear.t; l2 : Linear.t }

      let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { l1; l2 } =
        { l1 = Linear.map f l1; l2 = Linear.map f l2 }
      (* map2, iter, names: same one-liners over the fields. *)

      let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
    end
    ]} *)

(** {1:types Types} *)

type 'b params = { w : (float, 'b) Nx.t; b : (float, 'b) Nx.t option }
(** The type for linear-layer parameters with float dtype layout ['b].

    [w] has shape [[| inputs; outputs |]] and [b], when present, shape
    [[| outputs |]]. [b] is [None] for layers built without a bias
    ({!make}[ ~bias:false]); such layers have no bias parameter at all, so
    traversals skip it and {!apply} performs no shift. *)

type t = Nx.float32_elt params
(** The type for single-precision linear layers, the common case. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:'b Init.t ->
  ?bias_init:'b Init.t ->
  ?bias:bool ->
  inputs:int ->
  outputs:int ->
  (float, 'b) Nx.dtype ->
  'b params
(** [make ~inputs ~outputs dtype] is a fresh layer mapping [inputs] features to
    [outputs] features, with:

    - [w_init], the weight initializer, applied with [~fan_in:inputs] and
      [~fan_out:outputs]. Defaults to {!Init.glorot_uniform}.
    - [bias_init], the bias initializer, applied with the same fans. Defaults to
      {!Init.zeros}.
    - [bias], whether the layer has a bias parameter. Defaults to [true];
      [false] sets [b] to [None] and ignores [bias_init].

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if [inputs] or [outputs] is not positive. *)

val init : inputs:int -> outputs:int -> t
(** [init ~inputs ~outputs] is [make ~inputs ~outputs Nx.float32]:
    Glorot-uniform weights, zero bias. *)

(** {1:applying Applying} *)

val apply : 'b params -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply p x] is [x @ p.w + p.b] (the shift is omitted when [p.b] is [None]).
    [x]'s last axis must have size [inputs]; leading axes are treated as batch
    axes, so the result has [x]'s shape with the last axis replaced by
    [outputs]. Differentiable through Rune.

    Raises [Invalid_argument] if [x]'s last axis does not have size [inputs]. *)

(** {1:traversals Traversals}

    Plain traversals over the parameter leaves, in the order [w] then [b]. They
    satisfy the {!Nx.Ptree.S} contract at any fixed ['b]. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to every parameter leaf. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p q] combines [p] and [q] leafwise with [f].

    Raises [Invalid_argument] if one of [p] and [q] has a bias and the other
    does not. *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to every parameter leaf of [p]. *)

val astype : (float, 'c) Nx.dtype -> 'b params -> 'c params
(** [astype dt p] is [p] with every parameter leaf cast to [dt]. Differentiable
    through Rune: gradients flow back at each original leaf's dtype, so an
    astype of float32 parameters inside a loss function yields float32
    gradients. *)

val names : 'b params -> string list
(** [names p] is the checkpoint name of each parameter leaf of [p], in traversal
    order: [["w"; "b"]], or [["w"]] when [p] has no bias. See
    {!Checkpoint.Named}. *)
