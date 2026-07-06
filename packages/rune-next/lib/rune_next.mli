(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functional transformations over structured tensor collections.

    Rune-next differentiates functions over any user-defined parameter
    structure. A structure declares how to traverse its tensor leaves by
    implementing {!Differentiable}; every transformation then works on it
    directly, preserving its type. Leaves may have different dtypes: a single
    forward and backward pass produces gradients for all of them.

    {[
    type params = { w : Nx.float32_t; b : Nx.float32_t }

    module Params = struct
      type t = params

      let map f { w; b } = { w = f w; b = f b }
      let map2 f a b = { w = f a.w b.w; b = f a.b b.b }

      let iter f { w; b } =
        f w;
        f b
    end

    let grads = Rune_next.grad (module Params) loss params
    ]}

    Use {!Ptree} when the parameter structure is only known at runtime. *)

(** {1:differentiable Differentiable structures} *)

(** Structures whose tensor leaves transformations can traverse.

    Implementations must satisfy:
    - [map f t] applies [f] to every tensor leaf of [t] exactly once and
      preserves the structure of [t].
    - [map2 f a b] applies [f] to corresponding leaves of [a] and [b], which
      must be structurally equal.
    - [iter f t] applies [f] to every tensor leaf of [t] exactly once.
    - [iter] visits leaves in a stable order; {!vmap} pairs its [in_axes]
      entries with leaves in that order. *)
module type Differentiable = sig
  type t
  (** The structure type. *)

  val map : ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t
  (** [map f t] is [t] with [f] applied to every tensor leaf. *)

  val map2 :
    ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t -> t
  (** [map2 f a b] is [a] and [b] combined leafwise with [f].

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)

  val iter : ('a 'b. ('a, 'b) Nx.t -> unit) -> t -> unit
  (** [iter f t] applies [f] to every tensor leaf of [t]. *)
end

module Ptree = Ptree
(** Dynamically-typed parameter trees, a stock {!Differentiable} instance for
    structures only known at runtime. *)

(** {1:reverse Reverse-mode differentiation} *)

val grad : (module P : Differentiable) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> P.t
(** [grad (module P) f params] is the gradient of [f] at [params], with the same
    structure and leaf types as [params]. Leaves of [params] that do not
    contribute to the result have all-zero gradients.

    Raises [Invalid_argument] if [f params] is not a scalar tensor; use {!vjp}
    to differentiate non-scalar outputs against an explicit cotangent. *)

val value_and_grad :
  (module P : Differentiable) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t * P.t
(** [value_and_grad (module P) f params] is
    [(f params, grad (module P) f params)], computed in a single forward and
    backward pass. *)

val value_and_grad_aux :
  (module P : Differentiable) ->
  (P.t -> ('c, 'd) Nx.t * 'aux) -> P.t -> ('c, 'd) Nx.t * P.t * 'aux
(** [value_and_grad_aux (module P) f params] is like {!value_and_grad} for an
    objective returning auxiliary data alongside its result. The auxiliary value
    is returned as-is and does not contribute to the gradient. *)

val vjp :
  (module P : Differentiable) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t -> ('c, 'd) Nx.t * P.t
(** [vjp (module P) f params cotangent] is [(f params, pullback)] where
    [pullback] is the vector-Jacobian product of [f] at [params] against
    [cotangent]. *)

(** {1:forward Forward-mode differentiation} *)

val jvp :
  (module P : Differentiable) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> P.t -> ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp (module P) f params tangents] is [(f params, df)] where [df] is the
    Jacobian-vector product of [f] at [params] against [tangents], computed in a
    single forward pass. [tangents] must be structurally equal to [params]; each
    tangent leaf must have its parameter leaf's shape. The output may have any
    shape.

    Raises [Invalid_argument] if a tangent leaf's shape does not match its
    parameter leaf's shape. *)

val jvp_aux :
  (module P : Differentiable) ->
  (P.t -> ('c, 'd) Nx.t * 'aux) ->
  P.t ->
  P.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t * 'aux
(** [jvp_aux (module P) f params tangents] is like {!jvp} for an objective
    returning auxiliary data alongside its result. The auxiliary value is
    returned as-is and does not contribute to the tangent. *)

(** {1:vmap Vectorizing maps} *)

val vmap :
  ?in_axes:int option list ->
  ?out_axis:int ->
  (module P : Differentiable) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t
(** [vmap ?in_axes ?out_axis (module P) f params] maps [f] over the tensor
    leaves of [params]. [f] is written for unbatched values: it observes each
    mapped leaf without its mapped axis, and its result gains a batch axis at
    [out_axis] (default [0]). Values [f] closes over are constants of the map,
    and a result that does not depend on the mapped inputs is broadcast along
    the batch axis.

    [in_axes] gives the mapped axis of each leaf, paired with leaves in the
    structure's traversal order: [Some i] maps axis [i] (negative from the end),
    [None] passes the leaf whole as a constant. It defaults to mapping axis [0]
    of every leaf. Mapped axes must agree on their size.

    Composes with the other transformations: [vmap] of {!grad} computes
    per-example gradients, and {!grad} of [vmap] differentiates through the map.

    Raises [Invalid_argument] if [in_axes] does not have one entry per leaf,
    maps no leaf, names an axis out of bounds, or if the mapped axis sizes
    disagree. *)

val vmap' :
  ?in_axis:int ->
  ?out_axis:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [vmap' ?in_axis ?out_axis f x] is like {!vmap} for a function of a single
    tensor, mapping over [x]'s axis [in_axis] and placing the batch axis of the
    result at [out_axis]. Both default to [0]. *)

(** {1:custom Custom differentiation rules} *)

val custom_vjp :
  (module P : Differentiable) ->
  fwd:(P.t -> ('c, 'd) Nx.t * 'res) ->
  bwd:('res -> ('c, 'd) Nx.t -> P.t) ->
  P.t ->
  ('c, 'd) Nx.t
(** [custom_vjp (module P) ~fwd ~bwd params] is [fst (fwd params)], with a
    user-defined reverse rule. Under the innermost reverse-mode transformation,
    [fwd]'s internal operations are not differentiated; instead
    [bwd residual cotangent] provides the parameter gradients, with each leaf
    matching its parameter leaf's shape and dtype. [residual] is whatever [fwd]
    returned alongside its result. Enclosing transformations (an outer {!grad},
    {!vmap}) see the forward computation itself.

    Raises [Invalid_argument] if the call is differentiated in forward mode;
    define a {!custom_jvp} rule for that. *)

val custom_jvp :
  (module P : Differentiable) ->
  f:(P.t -> ('c, 'd) Nx.t) ->
  jvp:(P.t -> P.t -> ('c, 'd) Nx.t * ('c, 'd) Nx.t) ->
  P.t ->
  ('c, 'd) Nx.t
(** [custom_jvp (module P) ~f ~jvp params] is [f params], with a user-defined
    forward rule. Under the innermost forward-mode transformation,
    [jvp params tangents] provides both the result and its tangent, replacing
    [f]'s internal operations.

    Raises [Invalid_argument] if the call is differentiated in reverse mode;
    define a {!custom_vjp} rule for that. *)

(** {1:tensor Single-tensor variants} *)

val grad' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad' f x] is like {!grad} for a function of a single tensor. *)

val value_and_grad' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad' f x] is like {!value_and_grad} for a function of a single
    tensor. *)

val vjp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp' f x cotangent] is like {!vjp} for a function of a single tensor. *)

val jvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp' f x tangent] is like {!jvp} for a function of a single tensor. *)

(** {1:control Autodiff control} *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach t] is a copy of [t] through which gradients do not flow. Use it to
    hold a value constant inside a differentiated function, including as input
    to an operation whose gradient is not implemented. *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] runs [f] with gradient tracking disabled: tensors it produces
    are constants of the surrounding differentiation. *)
