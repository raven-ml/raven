(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transformation semantics over parameter-tree structures: grad,
   value_and_grad, value_and_grad_aux, and vjp on user-defined records. *)

open Windtrap
open Rune_test_support.Support

(* loss p = sum (w * w) + 3 * sum b, ignoring scale. d/dw = 2w, d/db = 3,
   d/dscale = 0. *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.0)

let test_grad_record_analytic () =
  let g = Rune.grad (module Params) quadratic (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"db" [| 3.0 |] g.b

let test_grad_unused_leaf_zero () =
  let g = Rune.grad (module Params) quadratic (params ()) in
  check_arr ~msg:"dscale" [| 0.0 |] g.scale

let test_grad_preserves_structure () =
  let p = params () in
  let g = Rune.grad (module Params) quadratic p in
  equal ~msg:"w shape" (array int) (Nx.shape p.w) (Nx.shape g.w);
  equal ~msg:"b shape" (array int) (Nx.shape p.b) (Nx.shape g.b);
  equal ~msg:"scale shape" (array int) (Nx.shape p.scale) (Nx.shape g.scale)

let test_value_and_grad_value () =
  let v, _ = Rune.value_and_grad (module Params) quadratic (params ()) in
  (* 1 + 4 + 9 + 3 * 0.5 = 15.5 *)
  check_arr ~msg:"value" [| 15.5 |] v

let test_value_and_grad_aux () =
  let f p = (quadratic p, "aux") in
  let v, g, aux = Rune.value_and_grad_aux (module Params) f (params ()) in
  check_arr ~msg:"value" [| 15.5 |] v;
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  equal ~msg:"aux" string "aux" aux

let test_vjp_cotangent_scales () =
  let _, g =
    Rune.vjp (module Params) quadratic (params ()) (Nx.scalar f32 2.0)
  in
  check_arr ~msg:"dw scaled" [| 4.0; -8.0; 12.0 |] g.w

let test_vjp_non_scalar_output () =
  (* vjp accepts non-scalar outputs: for f(w) = w*w and cotangent ct, the
     pulled-back cotangent is 2*w*ct. *)
  let p = params () in
  let f p = Nx.mul p.w p.w in
  let _, g = Rune.vjp (module Params) f p (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"dw" [| 2.0; -8.0; 18.0 |] g.w

let test_mixed_dtype_single_pass () =
  (* loss p = sum (w * w) + sum (scale * scale), via a cast to float64. d/dw =
     2w (float32), d/dscale = 2 scale (float64). *)
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  let g = Rune.grad (module Params) f (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"dscale" [| 4.0 |] g.scale

let test_gradient_descent_converges () =
  (* Minimize sum (w * w) + sum (scale * scale): both go to zero. *)
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  let step p =
    let g = Rune.grad (module Params) f p in
    Params.map2 (fun p g -> Nx.sub p (Nx.mul g (scalar_like g 0.1))) p g
  in
  let p = ref (params ()) in
  for _ = 1 to 100 do
    p := step !p
  done;
  check_arr ~msg:"w" [| 0.0; 0.0; 0.0 |] !p.w;
  check_arr ~msg:"scale" [| 0.0 |] !p.scale

let test_grad_single_tensor () =
  let f x = Nx.sum (Nx.mul x x) in
  let g = Rune.grad' f (vec32 [| 1.0; -2.0; 3.0 |]) in
  check_arr ~msg:"dx" [| 2.0; -4.0; 6.0 |] g

let test_vjp_single_tensor () =
  let f x = Nx.mul x x in
  let _, g = Rune.vjp' f (vec32 [| 1.0; 2.0 |]) (vec32 [| 10.0; 1.0 |]) in
  check_arr ~msg:"dx" [| 20.0; 4.0 |] g

let test_vjp2_structured_output () =
  (* vjp2 against cotangents equals the gradient of <cotangents, f>. *)
  let a = vec64 [| 0.7; -1.3; 2.1 |] and b = vec64 [| 1.9; 0.8; -0.6 |] in
  let ca = vec64 [| 1.0; -2.0; 0.5 |] and cb = vec64 [| 0.3; 1.1; -0.7 |] in
  let f p = { fst = Nx.mul p.fst p.snd; snd = Nx.add p.fst p.snd } in
  let _, g =
    Rune.vjp2
      (module Pair)
      (module Pair)
      f { fst = a; snd = b } { fst = ca; snd = cb }
  in
  let dotted p =
    let y = f p in
    Nx.add (Nx.sum (Nx.mul y.fst ca)) (Nx.sum (Nx.mul y.snd cb))
  in
  let expected = Rune.grad (module Pair) dotted { fst = a; snd = b } in
  check_arr ~msg:"da" (to_arr expected.fst) g.fst;
  check_arr ~msg:"db" (to_arr expected.snd) g.snd

let test_vjp2_cotangent_shape_mismatch () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vjp2
           (module Pair)
           (module Pair)
           (fun p -> p)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 3.0 |] }
           { fst = vec64 [| 1.0 |]; snd = vec64 [| 1.0 |] }))

let test_remat_same_gradient () =
  (* remat changes memory behavior, never values or gradients. *)
  let f p = Nx.sum (Nx.mul (Nx.exp p.fst) (Nx.sin p.snd)) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  let g = Rune.grad (module Pair) f params in
  let g' =
    Rune.grad (module Pair) (fun p -> Rune.remat (module Pair) f p) params
  in
  check_arr ~msg:"d fst" (to_arr g.fst) g'.fst;
  check_arr ~msg:"d snd" (to_arr g.snd) g'.snd

let test_remat_value () =
  let f p = Nx.sum (Nx.mul p.fst p.snd) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  check_arr ~msg:"value" (to_arr (f params)) (Rune.remat (module Pair) f params)

(* A single tensor has nowhere to carry a non-differentiable value, so an
   integer argument there is simply the wrong dtype. *)
let test_grad_rejects_integer_leaves () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.grad'
           (fun x -> Nx.sum x)
           (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |])))

(* A structure does have somewhere. The canonical case is an RNG key, which
   must sit in the structure to reach a compiled step as an input, but is not
   something to differentiate. Such a leaf is carried: zero in the gradient,
   untouched by the optimizer, while the real parameters update. Before this,
   grad refused the whole structure and the key had to be captured in a closure
   and the parameters kept in a second structure. *)
module Stepper = struct
  type t = { w : Nx.float32_t; key : Nx.Rng.key }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { w = f t.w; key = f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    f t.w;
    f t.key
end

let stepper () =
  Stepper.{ w = Nx.create f32 [| 3 |] [| 1.0; -2.0; 3.0 |]; key = Nx.Rng.key 7 }

(* The draw is a constant of the tape, so the loss below has the same gradient
   it would without the mask, scaled by it. *)
let test_grad_carries_a_key_leaf () =
  let p = stepper () in
  let g =
    Rune.grad
      (module Stepper)
      (fun p -> Nx.sum (Nx.mul p.Stepper.w p.Stepper.w))
      p
  in
  check_arr ~msg:"the float leaf differentiates" [| 2.0; -4.0; 6.0 |] g.Stepper.w;
  equal ~msg:"the key's gradient is zero" (array int32)
    [| 0l; 0l |]
    (Nx.to_array g.Stepper.key)

let tests =
  [
    group "grad over records"
      [
        test "matches the analytic gradient" test_grad_record_analytic;
        test "unused leaf has zero gradient" test_grad_unused_leaf_zero;
        test "preserves structure and shapes" test_grad_preserves_structure;
        test "value_and_grad returns the value" test_value_and_grad_value;
        test "value_and_grad_aux returns auxiliary data" test_value_and_grad_aux;
        test "mixed dtypes differentiate in one pass"
          test_mixed_dtype_single_pass;
        test "gradient descent converges" test_gradient_descent_converges;
        test "rejects an integer single-tensor argument"
          test_grad_rejects_integer_leaves;
        test "carries a non-differentiable leaf" test_grad_carries_a_key_leaf;
      ];
    group "vjp"
      [
        test "scales by the cotangent" test_vjp_cotangent_scales;
        test "accepts non-scalar outputs" test_vjp_non_scalar_output;
        test "vjp2 pulls back structured cotangents" test_vjp2_structured_output;
        test "vjp2 rejects cotangent shape mismatch"
          test_vjp2_cotangent_shape_mismatch;
      ];
    group "remat"
      [
        test "gradients are unchanged" test_remat_same_gradient;
        test "values are unchanged" test_remat_value;
      ];
    group "single-tensor variants"
      [
        test "grad' matches the analytic gradient" test_grad_single_tensor;
        test "vjp' pulls back the cotangent" test_vjp_single_tensor;
      ];
  ]

let () = run "rune grad" tests
