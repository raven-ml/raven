(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functional transformations for {!Nx} tensors.

    Rune provides automatic differentiation (forward and reverse mode),
    vectorising maps, and gradient checking. It operates by intercepting {!Nx}
    tensor operations via OCaml 5 effect handlers — no special tensor type is
    needed.

    {b Terminology.}
    - {e Primal}: the input value at which a derivative is evaluated.
    - {e Tangent}: the directional derivative seed (forward mode).
    - {e Cotangent}: the adjoint seed propagated backward (reverse mode).
    - {e JVP}: Jacobian-vector product (forward-mode AD).
    - {e VJP}: vector-Jacobian product (reverse-mode AD). *)

(** {1:reverse Reverse-mode AD}

    Compute gradients of scalar-valued functions via reverse-mode
    (backpropagation). The function [f] must return a scalar tensor; the
    gradient has the same shape as the input. *)

val grad : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad f x] is the gradient of scalar-valued [f] at [x].

    Equivalent to [snd (value_and_grad f x)].

    See also {!grads}, {!value_and_grad}. *)

val grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list
(** [grads f xs] is the list of gradients of scalar-valued [f] with respect to
    each tensor in [xs]. The {e i}-th element of the result has the same shape
    as the {e i}-th element of [xs].

    See also {!grad}, {!value_and_grads}. *)

val value_and_grad :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad f x] is [(f x, grad f x)], computed in a single
    forward-backward pass.

    See also {!value_and_grad_aux}. *)

val value_and_grad_aux :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t * 'e
(** [value_and_grad_aux f x] is [(y, g, aux)] where [(y, aux) = f x] and [g] is
    the gradient of [y] with respect to [x]. The auxiliary output [aux] is
    carried through but not differentiated.

    See also {!value_and_grads_aux}. *)

val value_and_grads :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [value_and_grads f xs] is [(f xs, grads f xs)], computed in a single
    forward-backward pass.

    See also {!value_and_grads_aux}. *)

val value_and_grads_aux :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list * 'e
(** [value_and_grads_aux f xs] is [(y, gs, aux)] where [(y, aux) = f xs] and
    [gs] is the list of gradients of [y] with respect to each tensor in [xs].
    The auxiliary output [aux] is carried through but not differentiated.

    See also {!value_and_grad_aux}. *)

val vjp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp f x v] is [(y, g)] where [y = f x] and [g = v{^T} J{_f}(x)]
    (vector-Jacobian product). Unlike {!grad}, [f] need not return a scalar —
    the cotangent [v] must have the same shape as [y].

    See also {!vjps}. *)

val vjps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t list
(** [vjps f xs v] is like {!vjp} for functions with multiple inputs. Returns
    [(y, gs)] where each gradient in [gs] corresponds to one input in [xs]. *)

(** {1:forward Forward-mode AD}

    Compute Jacobian-vector products by propagating tangent vectors alongside
    primal values. Forward mode is efficient when the number of inputs is small
    relative to the number of outputs. *)

val jvp :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp f x v] is [(y, t)] where [y = f x] and [t = J{_f}(x) v]
    (Jacobian-vector product). The tangent [v] must have the same shape as [x].

    See also {!jvps}, {!jvp_aux}. *)

val jvp_aux :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t * 'e) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t * 'e
(** [jvp_aux f x v] is like {!jvp} but for functions with auxiliary output.
    Returns [(y, t, aux)] where [aux] is carried through but not differentiated.
*)

val jvps :
  (('a, 'b) Nx.t list -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t list ->
  ('a, 'b) Nx.t list ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvps f xs vs] is like {!jvp} for functions with multiple inputs. Each
    tangent in [vs] must have the same shape as the corresponding primal in
    [xs]. *)

(** {1:stop Stopping gradients} *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] evaluates [f ()] without recording operations for automatic
    differentiation. All tensors produced inside [f] are treated as constants by
    enclosing gradient computations. *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach x] is a copy of [x] that is treated as a constant with respect to
    automatic differentiation.

    See also {!no_grad}. *)

(** {1:gradcheck Gradient checking}

    Compare autodiff gradients against finite-difference approximations. Useful
    for testing custom operations. *)

type method_ = [ `Central | `Forward | `Backward ]
(** The type for finite difference methods.
    - [`Central] — [(f(x+h) - f(x-h)) / 2h]. Most accurate, requires two
      evaluations per element.
    - [`Forward] — [(f(x+h) - f(x)) / h].
    - [`Backward] — [(f(x) - f(x-h)) / h]. *)

val finite_diff :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [finite_diff f x] is the gradient of scalar-valued [f] at [x] approximated
    by finite differences.

    [eps] defaults to [1e-4]. [method_] defaults to [`Central]. *)

val finite_diff_jacobian :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [finite_diff_jacobian f x] is the Jacobian of [f] at [x] approximated by
    finite differences.

    [eps] defaults to [1e-4]. [method_] defaults to [`Central]. *)

type gradient_check_result = {
  max_abs_error : float;  (** Largest absolute error across all elements. *)
  max_rel_error : float;  (** Largest relative error across all elements. *)
  mean_abs_error : float;  (** Mean absolute error. *)
  mean_rel_error : float;  (** Mean relative error. *)
  failed_indices : (int array * float * float * float) list;
      (** [(index, autodiff, finite_diff, abs_error)] for each failed element.
      *)
  passed : bool;  (** [true] iff no element exceeded the tolerances. *)
  num_checked : int;  (** Number of elements checked. *)
  num_failed : int;  (** Number of elements that exceeded tolerances. *)
}
(** The type for gradient check results. *)

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
(** [check_gradient f x] compares the autodiff gradient of [f] at [x] against a
    finite-difference approximation.

    An element passes when [abs_error <= atol] or [rel_error <= rtol].

    - [eps] defaults to [1e-4].
    - [rtol] defaults to [2e-3].
    - [atol] defaults to [2e-3].
    - [verbose] defaults to [false]. When [true], prints per-element failures
      and a summary to standard output.
    - [check_indices] defaults to [None] (check all elements). When
      [Some indices], only the listed flat indices are checked.
    - [method_] defaults to [`Central].

    See also {!check_gradients}. *)

val check_gradients :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a) Nx.t list -> ('b, 'c) Nx.t) ->
  (float, 'a) Nx.t list ->
  [ `Pass of gradient_check_result list | `Fail of gradient_check_result list ]
(** [check_gradients f xs] is like {!check_gradient} for functions with multiple
    inputs. Returns one {!gradient_check_result} per input tensor.

    Optional parameters have the same defaults as {!check_gradient}. *)

(** {1:vmap Vectorising map}

    Map a computation over a batch dimension. [vmap] transforms a function that
    operates on single examples into one that operates on batches, without the
    user writing explicit batch loops. *)

(** The type for per-input axis specifications. *)
type axis_spec = Vmap.axis_spec =
  | Map of int  (** Map over the axis at this index. *)
  | NoMap  (** Do not map; broadcast the input as-is. *)

(** The type for input axis specifications. *)
type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec  (** Apply to all inputs. *)
  | Container of 'a  (** Per-input specifications. *)

(** The type for output axis specifications. *)
type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
      (** Stack outputs along this axis ([None] to discard). *)
  | OutContainer of 'a  (** Per-output specifications. *)

val vmap :
  ?in_axes:'a in_axes_spec ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) Nx.t -> ('e, 'f) Nx.t) ->
  ('c, 'd) Nx.t ->
  ('e, 'f) Nx.t
(** [vmap f x] is a vectorised version of [f] applied to [x].

    - [in_axes] defaults to [Single (Map 0)].
    - [out_axes] defaults to [OutSingle (Some 0)].
    - [axis_name] is an optional label for the mapped axis (used in error
      messages).
    - [axis_size] overrides the batch size inferred from the input shape.
      Required when all inputs use {!NoMap}.

    See also {!vmaps}. *)

val vmaps :
  ?in_axes:Vmap.axis_spec list ->
  ?out_axes:'b Vmap.out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) Nx.t list -> ('e, 'f) Nx.t) ->
  ('c, 'd) Nx.t list ->
  ('e, 'f) Nx.t
(** [vmaps f xs] is like {!vmap} for functions with multiple inputs. Each
    element of [in_axes] corresponds to one input in [xs].

    [in_axes] defaults to [Map 0] for every input. *)

(** {1:debug Debugging} *)

val debug : ('a -> 'b) -> 'a -> 'b
(** [debug f x] applies [f] to [x] under a tracing handler that prints every
    tensor operation, its inputs, and its outputs to standard output. *)
