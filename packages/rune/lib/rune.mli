(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {2 Automatic Differentiation}

    Functions for automatic differentiation and gradient computation. *)

val grad : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad f t] computes the gradient of [f] with respect to [t].

    Returns a tensor of the same shape as [t] containing the gradient values. *)

val grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list
(** [grads f ts] computes gradients of [f] with respect to each tensor in [ts].

    Returns a list of gradients, one for each input tensor. *)

val value_and_grad :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad f t] computes both the value of [f] and the gradient with
    respect to [t].

    Returns a tuple of the function value and the gradient tensor. *)

val value_and_grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [value_and_grads f ts] computes both the value of [f] and the gradients with
    respect to each tensor in [ts].

    Returns a tuple of the function value and a list of gradient tensors. *)

val jvp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp f primals tangents] computes a Jacobian-vector product (forward-mode
    AD).

    Returns a tuple of (primal_output, tangent_output) where:
    - primal_output = f(primals)
    - tangent_output = Jf(primals) · tangents *)

val jvp_aux :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t * 'e
(** [jvp_aux f primals tangents] like [jvp] but for functions with auxiliary
    output.

    Returns (primal_output, tangent_output, aux) where aux is the auxiliary
    data. *)

val jvps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvps f primals tangents] computes JVP for functions with multiple inputs.

    Returns (primal_output, tangent_output) for the list of inputs. *)

val vjp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp f primal cotangent] computes a vector-Jacobian product (reverse-mode
    AD).

    Returns a tuple of (primal_output, gradient) where:
    - primal_output = f(primal)
    - gradient = cotangent · Jf(primal) *)

val vjps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [vjps f primals cotangent] computes VJP for functions with multiple inputs.

    Returns (primal_output, gradients) for the list of inputs. *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] evaluates [f ()] without recording operations for automatic
    differentiation. This mirrors JAX's [lax.stop_gradient] semantics when
    applied to a computation block: all tensors produced within [f] are treated
    as constants for subsequent gradient calculations. *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach t] returns a tensor with the same value as [t] but which is treated
    as a constant with respect to automatic differentiation. Equivalent to JAX's
    [lax.stop_gradient] on a single tensor. *)

(** {2 Gradient Checking} *)

type method_ = [ `Central | `Forward | `Backward ]
(** Finite difference method to use:
    - [`Central]: (f(x+h) - f(x-h)) / 2h (most accurate)
    - [`Forward]: (f(x+h) - f(x)) / h
    - [`Backward]: (f(x) - f(x-h)) / h *)

val finite_diff :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [finite_diff ?eps ?method_ f x] computes the gradient of scalar-valued
    function [f] with respect to input [x] using finite differences. The
    function [f] must return a scalar tensor. *)

val finite_diff_jacobian :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [finite_diff_jacobian ?eps ?method_ f x] computes the Jacobian matrix of
    function [f] with respect to input [x] using finite differences. *)

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

(** {2 Vectorizing Map (vmap)}

    Functions for mapping computations over batch dimensions. *)

type axis_spec = Vmap.axis_spec =
  | Map of int  (** Map over this axis index *)
  | NoMap  (** Don't map this axis *)

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
(** [vmap ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f]. *)

val vmaps :
  ?in_axes:Vmap.axis_spec list ->
  ?out_axes:'b Vmap.out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) Nx.t list -> ('e, 'f) Nx.t) ->
  ('c, 'd) Nx.t list ->
  ('e, 'f) Nx.t
(** [vmaps ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f] that takes multiple tensor arguments. *)

(** {2 Debugging} *)

val debug : ('a -> 'b) -> 'a -> 'b
(** [debug f x] applies [f] to [x] and prints debug information. *)

val debug_with_context : string -> (unit -> 'a) -> 'a
(** [debug_with_context context f] runs [f] with a debug context. *)

val debug_push_context : string -> unit
(** [debug_push_context context] pushes a new debug context. *)

val debug_pop_context : unit -> unit
(** [debug_pop_context ()] pops the last debug context. *)

(** {2 Submodules} *)

module Finite_diff = Finite_diff
module Gradcheck = Gradcheck
