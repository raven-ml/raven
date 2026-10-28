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
      type ('a, 'd) t = { l1 : ('a, 'd) Linear.t; l2 : ('a, 'd) Linear.t }

      let walk c { l1; l2 } =
        let open Nx.Ptree.Walk in
        let l1 = field c "l1" Linear.walk l1 in
        let l2 = field c "l2" Linear.walk l2 in
        { l1; l2 }

      let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
    end

    let mlp = Nx.Ptree.instantiate (module Mlp)

    let params =
      let k = Nx.Rng.split (Nx.Rng.key 0) in
      {
        Mlp.l1 = Linear.init ~key:k.(0) ~inputs:4 ~outputs:8;
        l2 = Linear.init ~key:k.(1) ~inputs:8 ~outputs:2;
      }
    ]}

    The paths [walk] gives each leaf, here [l1.w], [l1.b], [l2.w] and [l2.b],
    name the leaves in a saved file. *)

(** {1:types Types} *)

type ('a, _) t = { w : 'a; b : 'a option }
(** The type for linear-layer parameters over payload ['a].

    At tensor payloads — [((float, 'b, 'd) Nx.t, 'd) t] — [w] has shape
    [[| inputs; outputs |]] and [b], when present, shape [[| outputs |]]. [b] is
    [None] for layers built without a bias ({!make}[ ~bias:false]); such layers
    have no bias parameter at all, so {!walk} skips it and {!apply} performs no
    shift. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:('b, 'd) Init.t ->
  ?bias_init:('b, 'd) Init.t ->
  ?bias:bool ->
  key:'d Nx.Rng.key ->
  inputs:int ->
  outputs:int ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~key ~inputs ~outputs dtype] is a fresh layer mapping [inputs]
    features to [outputs] features, drawn from [key] and lying where it lies,
    with:

    - [w_init], the weight initializer, applied with [~fan_in:inputs] and
      [~fan_out:outputs]. Defaults to {!Init.glorot_uniform}.
    - [bias_init], the bias initializer, applied with the same fans. Defaults to
      {!Init.zeros}.
    - [bias], whether the layer has a bias parameter. Defaults to [true];
      [false] sets [b] to [None] and ignores [bias_init].

    The weight and the bias are drawn from two keys {!Nx.Rng.split} makes of
    [key].

    Raises [Invalid_argument] if [inputs] or [outputs] is not positive. *)

val init :
  key:'d Nx.Rng.key -> inputs:int -> outputs:int -> ('d Nx.float32_t, 'd) t
(** [init ~key ~inputs ~outputs] is [make ~key ~inputs ~outputs Nx.float32]:
    Glorot-uniform weights, zero bias. *)

(** {1:applying Applying} *)

val apply :
  ((float, 'b, 'd) Nx.t, 'd) t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [apply p x] is [x @ p.w + p.b] (the shift is omitted when [p.b] is [None]).
    [x]'s last axis must have size [inputs]; leading axes are treated as batch
    axes, so the result has [x]'s shape with the last axis replaced by
    [outputs]. Differentiable through Rune.

    Raises [Invalid_argument] if [x]'s last axis does not have size [inputs]. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [p]'s parameters: [w] at ["w"], then [b] at ["b"],
    reporting whether [b] is present, so a layer with a bias and one without are
    distinct structures. It is the layer's {!Nx.Ptree.S} instance.
    [Nx.Ptree.instantiate (module Linear)] is the layer at one dtype, and
    [Nx.Ptree.cast (module Linear) dtype p] converts its precision. *)
