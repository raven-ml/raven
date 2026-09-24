(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dense (fully connected) layers.

    A linear layer is a record of parameters with a payload hole. Filled with
    tensors it is the layer itself; filled with floats, a per-leaf learning
    rate. Construct one with {!init} or {!make}, transform inputs with {!apply},
    and compose layers into models by nesting records. A model is a structure
    with one {!walk}, one line per field, and [Nx.Ptree.instantiate] turns it
    into the {!Nx.Ptree.t} the transformations take:

    {[
    module Mlp = struct
      type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

      let walk c { l1; l2 } =
        let open Nx.Ptree.Walk in
        let l1 = field c "l1" Linear.walk l1 in
        let l2 = field c "l2" Linear.walk l2 in
        { l1; l2 }

      let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
    end

    let mlp = Nx.Ptree.instantiate (module Mlp)
    ]}

    The paths [walk] gives each leaf, here [l1.w], [l1.b], [l2.w] and [l2.b],
    are the leaf's checkpoint names. *)

(** {1:types Types} *)

type 'a t = { w : 'a; b : 'a option }
(** The type for linear-layer parameters over payload ['a].

    At tensor payloads — [(float, 'b) Nx.t t] — [w] has shape
    [[| inputs; outputs |]] and [b], when present, shape [[| outputs |]]. [b] is
    [None] for layers built without a bias ({!make}[ ~bias:false]); such layers
    have no bias parameter at all, so {!walk} skips it and {!apply} performs no
    shift. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:'b Init.t ->
  ?bias_init:'b Init.t ->
  ?bias:bool ->
  inputs:int ->
  outputs:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t t
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

val init : inputs:int -> outputs:int -> Nx.float32_t t
(** [init ~inputs ~outputs] is [make ~inputs ~outputs Nx.float32]:
    Glorot-uniform weights, zero bias. *)

(** {1:applying Applying} *)

val apply : (float, 'b) Nx.t t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply p x] is [x @ p.w + p.b] (the shift is omitted when [p.b] is [None]).
    [x]'s last axis must have size [inputs]; leading axes are treated as batch
    axes, so the result has [x]'s shape with the last axis replaced by
    [outputs]. Differentiable through Rune.

    Raises [Invalid_argument] if [x]'s last axis does not have size [inputs]. *)

(** {1:structure Structure} *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
(** [walk c p] walks [p]'s parameters: [w] at ["w"], then [b] at ["b"],
    reporting whether [b] is present, so a layer with a bias and one without are
    distinct structures. It is the layer's {!Nx.Ptree.S} instance.
    [Nx.Ptree.instantiate (module Linear)] is the layer at one dtype, and
    [Nx.Ptree.cast (module Linear) dtype p] converts its precision. *)
