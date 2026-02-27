(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {2 Automatic Differentiation}

    Functions for automatic differentiation and gradient computation. *)

val grad : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad f x] is the gradient of [f] with respect to [x]. *)

val grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list
(** [grads f xs] is the list of gradients of [f] with respect to each tensor in
    [xs]. *)

val value_and_grad :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad f x] is [(f x, grad f x)]. *)

val value_and_grad_aux :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t * 'e
(** [value_and_grad_aux f x] is [(y, grad, aux)] where [(y, aux) = f x] and
    [grad] is the gradient of the scalar output with respect to [x]. The
    auxiliary output [aux] is not differentiated. *)

val value_and_grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [value_and_grads f xs] is [(f xs, grads f xs)]. *)

val value_and_grads_aux :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list * 'e
(** [value_and_grads_aux f xs] is [(y, grads, aux)] where [(y, aux) = f xs] and
    [grads] is the list of gradients. The auxiliary output [aux] is not
    differentiated. *)

val jvp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp f primals tangents] is [(y, t)] where [y = f primals] and
    [t = Jf(primals) · tangents] (Jacobian-vector product, forward-mode AD). *)

val jvp_aux :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t * 'e
(** [jvp_aux f primals tangents] is like {!jvp} but for functions with auxiliary
    output. Returns [(primal_out, tangent_out, aux)]. *)

val jvps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvps f primals tangents] is {!jvp} for functions with multiple inputs. *)

val vjp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp f primal cotangent] is [(y, g)] where [y = f primal] and
    [g = cotangent · Jf(primal)] (vector-Jacobian product, reverse-mode AD). *)

val vjps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [vjps f primals cotangent] is {!vjp} for functions with multiple inputs. *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] evaluates [f ()] without recording operations for automatic
    differentiation. All tensors produced within [f] are treated as constants
    for subsequent gradient calculations. *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach t] is a copy of [t] that is treated as a constant with respect to
    automatic differentiation. *)

(** {2 Gradient Checking} *)

type method_ = [ `Central | `Forward | `Backward ]
(** Finite difference method:
    - [`Central]: [(f(x+h) - f(x-h)) / 2h] (most accurate)
    - [`Forward]: [(f(x+h) - f(x)) / h]
    - [`Backward]: [(f(x) - f(x-h)) / h] *)

val finite_diff :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [finite_diff ?eps ?method_ f x] is the gradient of scalar-valued [f] at [x]
    computed via finite differences. [eps] defaults to [1e-5]. [method_]
    defaults to [`Central]. *)

val finite_diff_jacobian :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [finite_diff_jacobian ?eps ?method_ f x] is the Jacobian matrix of [f] at
    [x] computed via finite differences. *)

type gradient_check_result = {
  max_abs_error : float;
  max_rel_error : float;
  mean_abs_error : float;
  mean_rel_error : float;
  failed_indices : (int array * float * float * float) list;
  passed : bool;
  num_checked : int;
  num_failed : int;
}

val check_gradient :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?check_indices:int list option ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a) Nx.t -> ('b, 'c) Nx.t) ->
  (float, 'a) Nx.t ->
  [ `Pass of gradient_check_result | `Fail of gradient_check_result ]
(** [check_gradient ?eps ?rtol ?atol ?verbose ?check_indices ?method_ f x]
    compares the gradient of [f] at [x] computed via automatic differentiation
    against finite differences. *)

val check_gradients :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a) Nx.t list -> ('b, 'c) Nx.t) ->
  (float, 'a) Nx.t list ->
  [ `Pass of gradient_check_result list | `Fail of gradient_check_result list ]
(** [check_gradients ?eps ?rtol ?atol ?verbose ?method_ f xs] compares the
    gradients of [f] with respect to each input in [xs] computed via automatic
    differentiation against finite differences. *)

(** {2 Vectorizing Map (vmap)} *)

type axis_spec = Vmap.axis_spec =
  | Map of int  (** Map over this axis index. *)
  | NoMap  (** Don't map this axis. *)

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

val vmap :
  ?in_axes:'a in_axes_spec ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) Nx.t -> ('e, 'f) Nx.t) ->
  ('c, 'd) Nx.t ->
  ('e, 'f) Nx.t
(** [vmap ?in_axes ?out_axes ?axis_name ?axis_size f] is a vectorized version of
    [f]. *)

val vmaps :
  ?in_axes:Vmap.axis_spec list ->
  ?out_axes:'b Vmap.out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) Nx.t list -> ('e, 'f) Nx.t) ->
  ('c, 'd) Nx.t list ->
  ('e, 'f) Nx.t
(** [vmaps ?in_axes ?out_axes ?axis_name ?axis_size f] is {!vmap} for functions
    with multiple tensor arguments. *)

(** {2 Debugging} *)

val debug : ('a -> 'b) -> 'a -> 'b
(** [debug f x] applies [f] to [x] and prints debug information about every
    tensor operation. *)

(** {2 Submodules} *)

module Finite_diff = Finite_diff
module Gradcheck = Gradcheck
