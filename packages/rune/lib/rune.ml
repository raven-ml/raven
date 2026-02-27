(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Autodiff ───── *)

let vjp = Autodiff.vjp
let vjps = Autodiff.vjps
let grad = Autodiff.grad
let grads = Autodiff.grads
let value_and_grad = Autodiff.value_and_grad
let value_and_grads = Autodiff.value_and_grads
let jvp = Autodiff.jvp
let jvp_aux = Autodiff.jvp_aux
let jvps = Autodiff.jvps
let no_grad = Autodiff.no_grad
let detach = Autodiff.detach

(* ───── Gradient Checking ───── *)

module Finite_diff = Finite_diff
module Gradcheck = Gradcheck

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

(* ───── Vmap ───── *)

type axis_spec = Vmap.axis_spec = Map of int | NoMap

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

let vmap = Vmap.vmap
let vmaps = Vmap.vmaps

(* ───── Debugging ───── *)

let debug = Debug.debug
let debug_with_context = Debug.with_context
let debug_push_context = Debug.push_context
let debug_pop_context = Debug.pop_context
