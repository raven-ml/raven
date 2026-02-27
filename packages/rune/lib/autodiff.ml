(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared infrastructure for automatic differentiation. Contains the
   identity-based hash table, derivative rules, and the autodiff-enabled flag
   used by both JVP (forward-mode) and VJP (reverse-mode) handlers. *)

open Nx_core
module T = Nx

(* Physical identity table *)

module Physical_tbl = struct
  module Tbl = Hashtbl.Make (struct
    type t = Obj.t

    let equal = ( == )

    let hash obj =
      (* Hash on the object's address, not its contents. Obj.magic to nativeint
         extracts the pointer value. *)
      Hashtbl.hash (Obj.magic obj : nativeint)
  end)

  type ('k, 'v) t = 'v Tbl.t

  let create n = Tbl.create n
  let find t key = Tbl.find_opt t (Obj.repr key)
  let add t key value = Tbl.replace t (Obj.repr key) value
end

(* Autodiff gate *)

let autodiff_enabled = ref true

let without_autodiff f =
  let prev = !autodiff_enabled in
  autodiff_enabled := false;
  Fun.protect f ~finally:(fun () -> autodiff_enabled := prev)

(* Derivative rules *)

let ln2 = 0.693147180559945309417
let two_over_sqrt_pi = 1.12837916709551257390

let float_scalar_like (type a b) (x : (a, b) T.t) (v : float) : (a, b) T.t =
  Obj.magic (T.full (T.dtype x) [||] (Obj.magic v : a))

(* d/dx sin(x) = cos(x) *)
let deriv_sin (type a b) (x : (a, b) T.t) : (a, b) T.t =
  Obj.magic (T.cos (Obj.magic x))

(* d/dx sqrt(x) = 1 / (2 * sqrt(x)) *)
let deriv_sqrt (type a b) (sqrt_x : (a, b) T.t) : (a, b) T.t =
  T.div (T.ones_like sqrt_x) (T.mul (float_scalar_like sqrt_x 2.0) sqrt_x)

(* d/dx (1/x) = -1/x^2 *)
let deriv_recip (type a b) (x : (a, b) T.t) : (a, b) T.t =
  T.neg (T.recip (T.mul x x))

(* d/dx tan(x) = 1/cos^2(x) *)
let deriv_tan (type a b) (x : (a, b) T.t) : (a, b) T.t =
  let cos_x = Obj.magic (T.cos (Obj.magic x)) in
  T.recip (T.mul cos_x cos_x)

(* d/dx asin(x) = 1/sqrt(1 - x^2) *)
let deriv_asin (type a b) (x : (a, b) T.t) : (a, b) T.t =
  let one = T.ones_like x in
  T.recip (T.sqrt (T.sub one (T.mul x x)))

(* d/dx acos(x) = -1/sqrt(1 - x^2) *)
let deriv_acos (type a b) (x : (a, b) T.t) : (a, b) T.t = T.neg (deriv_asin x)

(* d/dx atan(x) = 1/(1 + x^2) *)
let deriv_atan (type a b) (x : (a, b) T.t) : (a, b) T.t =
  let one = T.ones_like x in
  T.recip (T.add one (T.mul x x))

(* d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2) *)
let deriv_erf (type a b) (x : (a, b) T.t) : (a, b) T.t =
  let coeff = float_scalar_like x two_over_sqrt_pi in
  T.mul coeff (T.exp (T.neg (T.mul x x)))

(* d/da (a^b) = b * a^(b-1) *)
let deriv_pow_wrt_base (type a b) (base : (a, b) T.t) (exp : (a, b) T.t) :
    (a, b) T.t =
  T.mul exp (T.pow base (T.sub exp (T.ones_like exp)))

(* d/db (a^b) = a^b * ln(a) = a^b * log2(a) * ln(2) *)
let deriv_pow_wrt_exp (type a b) (base : (a, b) T.t) (result : (a, b) T.t) :
    (a, b) T.t =
  let ln_base =
    T.mul (Obj.magic (T.log2 (Obj.magic base))) (float_scalar_like base ln2)
  in
  T.mul result ln_base

(* Reduce gradient to match source shape (for broadcasting). *)
let unbroadcast_grad (type a b) (g : (a, b) T.t) (src_shape : int array) :
    (a, b) T.t =
  let dst_shape = T.shape g in
  if src_shape = dst_shape then g
  else
    let src_rank = Array.length src_shape in
    let dst_rank = Array.length dst_shape in
    let axes = ref [] in
    for i = 0 to dst_rank - src_rank - 1 do
      axes := i :: !axes
    done;
    for i = 0 to src_rank - 1 do
      if src_shape.(i) = 1 && dst_shape.(i + (dst_rank - src_rank)) > 1 then
        axes := (i + (dst_rank - src_rank)) :: !axes
    done;
    match !axes with
    | [] -> g
    | ax ->
        let summed = T.sum g ~axes:ax ~keepdims:true in
        if T.shape summed <> src_shape then T.reshape src_shape summed
        else summed
