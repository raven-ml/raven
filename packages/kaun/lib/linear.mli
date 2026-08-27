(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dense (fully connected) layers.

    A linear layer is a record of parameters with a payload hole. Filled with
    tensors it is the layer itself; filled with floats, a per-leaf learning
    rate; filled with strings, the checkpoint names. Construct one with {!init}
    or {!make}, transform inputs with {!apply}, and compose layers into models
    by nesting records — the traversals give any such model a one-line
    {!Nx.Ptree.Uniform} instance, and [Nx.Ptree.instantiate] turns that into the
    {!Nx.Ptree.S} the transformations take:

    {[
    module Mlp = struct
      type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

      let map f { l1; l2 } =
        let l1 = Linear.map f l1 in
        let l2 = Linear.map f l2 in
        { l1; l2 }
      (* map2, iter, fold, fold2, names: same one-liners over the fields,
         or [@@deriving ptree]. *)

      let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
    end
    ]} *)

(** {1:types Types} *)

type 'a t = { w : 'a; b : 'a option }
(** The type for linear-layer parameters over payload ['a].

    At tensor payloads — [(float, 'b) Nx.t t] — [w] has shape
    [[| inputs; outputs |]] and [b], when present, shape [[| outputs |]]. [b]
    is [None] for layers built without a bias ({!make}[ ~bias:false]); such
    layers have no bias parameter at all, so traversals skip it and {!apply}
    performs no shift. *)

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

(** {1:traversals Traversals}

    Payload traversals in the order [w] then [b], satisfying the
    {!Nx.Ptree.Uniform} contract. Leaf paths are ["w"] and ["b"]. *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f p] is [p] with [f] applied to every payload leaf.
    [map (Nx.cast dt) p] converts a layer's precision; the cast is
    differentiable through Rune. *)

val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
(** [map2 f p q] combines [p] and [q] leafwise with [f].

    Raises [Invalid_argument] if one of [p] and [q] has a bias and the other
    does not. *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f p] applies [f] to every payload leaf of [p]. *)

val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f acc p] reduces [p] leafwise, threading each leaf's path. *)

val fold2 : (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
(** [fold2 f acc p q] is like {!fold} across two structurally equal layers.

    Raises [Invalid_argument] if one of [p] and [q] has a bias and the other
    does not. *)

val names : 'a t -> string t
(** [names p] is [p] with every payload replaced by its path. *)
