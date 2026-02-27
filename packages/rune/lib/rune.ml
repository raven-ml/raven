(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reverse-mode *)

let grad = Vjp.grad
let grads = Vjp.grads
let value_and_grad = Vjp.value_and_grad
let value_and_grad_aux = Vjp.value_and_grad_aux
let value_and_grads = Vjp.value_and_grads
let value_and_grads_aux = Vjp.value_and_grads_aux
let vjp = Vjp.vjp
let vjps = Vjp.vjps
let no_grad = Vjp.no_grad
let detach = Vjp.detach

(* Forward-mode *)

let jvp = Jvp.jvp
let jvp_aux = Jvp.jvp_aux
let jvps = Jvp.jvps

(* Gradient checking *)

type method_ = Finite_diff.method_

type gradient_check_result = Gradcheck.gradient_check_result = {
  max_abs_error : float;
  max_rel_error : float;
  mean_abs_error : float;
  mean_rel_error : float;
  failed_indices : (int array * float * float * float) list;
  passed : bool;
  num_checked : int;
  num_failed : int;
}

let finite_diff = Finite_diff.finite_diff
let finite_diff_jacobian = Finite_diff.finite_diff_jacobian
let check_gradient = Gradcheck.check_gradient
let check_gradients = Gradcheck.check_gradients

(* Vmap *)

type axis_spec = Vmap.axis_spec = Map of int | NoMap

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

let vmap = Vmap.vmap
let vmaps = Vmap.vmaps

(* Debugging *)

let debug = Debug.debug
