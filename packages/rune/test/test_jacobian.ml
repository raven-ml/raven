(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reusable pullbacks and the Jacobian family. The two modes are independent
   implementations, so their agreement on full Jacobians is itself an oracle. *)

open Windtrap
open Rune_test_support.Support

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]
let v3_f32 () = vec32 [| 0.7; -1.3; 2.1 |]

(* f : R^3 -> R^2, non-linear with cross terms. *)
let jacobian_fn x =
  let x0 = Nx.slice [ Nx.R (0, 1) ] x
  and x1 = Nx.slice [ Nx.R (1, 2) ] x
  and x2 = Nx.slice [ Nx.R (2, 3) ] x in
  Nx.concatenate ~axis:0 [ Nx.mul x0 x1; Nx.mul (Nx.sin x2) x0 ]

let test_pullback_reusable () =
  let f p = Nx.mul p.fst p.snd in
  let a = v3 () and b = vec64 [| 1.9; 0.8; -0.6 |] in
  let _, pullback = Rune.vjp_fun (module Pair) f { fst = a; snd = b } in
  let ct1 = vec64 [| 1.0; 0.0; 0.0 |] and ct2 = vec64 [| 0.0; 2.0; 0.0 |] in
  let g1 = pullback ct1 in
  let g2 = pullback ct2 in
  let _, e1 = Rune.vjp (module Pair) f { fst = a; snd = b } ct1 in
  let _, e2 = Rune.vjp (module Pair) f { fst = a; snd = b } ct2 in
  check_arr ~msg:"first call" (to_arr e1.fst) g1.fst;
  check_arr ~msg:"second call" (to_arr e2.fst) g2.fst;
  check_arr ~msg:"second call snd" (to_arr e2.snd) g2.snd

let test_pullback_shape_mismatch () =
  let _, pullback =
    Rune.vjp_fun
      (module Pair)
      (fun p -> Nx.add p.fst p.snd)
      { fst = v3 (); snd = v3 () }
  in
  raises_invalid_arg (fun () -> ignore (pullback (vec64 [| 1.0 |])))

let dtype_name x = Nx_core.Dtype.to_string (Nx.dtype x)

let test_jacobians_float64 () =
  let jf = Rune.jacfwd' jacobian_fn (v3 ()) in
  let jr = Rune.jacrev' jacobian_fn (v3 ()) in
  equal ~msg:"jacfwd dtype" string "float64" (dtype_name jf);
  equal ~msg:"jacrev dtype" string "float64" (dtype_name jr);
  equal ~msg:"shape" (array int) [| 2; 3 |] (Nx.shape jf);
  check_arr ~msg:"agree" (to_arr jf) jr

let test_jacobian_analytic () =
  (* d(x0*x1)/dx = [x1; x0; 0]; d(sin x2 * x0)/dx = [sin x2; 0; x0 cos x2]. *)
  let x = v3 () in
  let j = Rune.jacrev' jacobian_fn x in
  check_arr ~msg:"jacobian"
    [| -1.3; 0.7; 0.0; Stdlib.sin 2.1; 0.0; 0.7 *. Stdlib.cos 2.1 |]
    j

let test_jacobians_float32 () =
  let x = v3_f32 () in
  let jf = Rune.jacfwd' jacobian_fn x in
  let jr = Rune.jacrev' jacobian_fn x in
  equal ~msg:"jacfwd dtype" string "float32" (dtype_name jf);
  equal ~msg:"jacrev dtype" string "float32" (dtype_name jr);
  check_arr ~msg:"agree" ~eps:1e-5 (to_arr jf) jr

let test_jacobians_mixed_dtypes () =
  let x = v3_f32 () in
  let f x = Nx.cast Nx.float64 (Nx.mul x x) in
  let jf = Rune.jacfwd' f x in
  let jr = Rune.jacrev' f x in
  equal ~msg:"jacfwd follows output" string "float64" (dtype_name jf);
  equal ~msg:"jacrev follows input" string "float32" (dtype_name jr);
  let expected = [| 1.4; 0.0; 0.0; 0.0; -2.6; 0.0; 0.0; 0.0; 4.2 |] in
  check_arr ~msg:"jacfwd" ~eps:1e-5 expected jf;
  check_arr ~msg:"jacrev" ~eps:1e-5 expected jr

let test_jacobians_restore_tensor_shapes () =
  let x = Nx.reshape [| 1; 3 |] (v3 ()) in
  let f x = Nx.reshape [| 2; 1 |] (jacobian_fn (Nx.reshape [| 3 |] x)) in
  let jf = Rune.jacfwd' f x in
  let jr = Rune.jacrev' f x in
  equal ~msg:"jacfwd shape" (array int) [| 2; 1; 1; 3 |] (Nx.shape jf);
  equal ~msg:"jacrev shape" (array int) [| 2; 1; 1; 3 |] (Nx.shape jr);
  check_arr ~msg:"agree" (to_arr jf) jr

let test_jacobians_evaluate_function_once () =
  let calls = ref 0 in
  let f x =
    incr calls;
    jacobian_fn x
  in
  ignore (Rune.jacfwd' f (v3 ()));
  equal ~msg:"jacfwd calls" int 1 !calls;
  calls := 0;
  ignore (Rune.jacrev' f (v3 ()));
  equal ~msg:"jacrev calls" int 1 !calls

let test_hessian_analytic () =
  (* Hessian of sum(x³) is diag(6x). *)
  let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let h = Rune.hessian' cube (v3 ()) in
  equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape h);
  check_arr ~msg:"hessian" [| 4.2; 0.0; 0.0; 0.0; -7.8; 0.0; 0.0; 0.0; 12.6 |] h

let test_hvp_matches_hessian () =
  let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let x = v3 () in
  let v = vec64 [| 1.0; 0.5; -1.0 |] in
  let hv = Rune.hvp' cube x v in
  let expected = Nx.matmul (Rune.hessian' cube x) v in
  check_arr ~msg:"hvp" (to_arr expected) hv

let test_hvp_structured () =
  (* f p = sum(fst²) + sum(fst ⊙ snd): H·v computable analytically: d/dfst = 2
     fst + snd, d/dsnd = fst; H·(vf, vs) = (2 vf + vs, vf). *)
  let f p =
    Nx.add (Nx.sum (Nx.mul p.fst p.fst)) (Nx.sum (Nx.mul p.fst p.snd))
  in
  let params = { fst = v3 (); snd = vec64 [| 1.9; 0.8; -0.6 |] } in
  let v =
    { fst = vec64 [| 1.0; 0.0; 2.0 |]; snd = vec64 [| 0.5; -1.0; 0.0 |] }
  in
  let hv = Rune.hvp (module Pair) f params v in
  check_arr ~msg:"d fst" [| 2.5; -1.0; 4.0 |] hv.fst;
  check_arr ~msg:"d snd" [| 1.0; 0.0; 2.0 |] hv.snd

module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let test_check_grads_accepts_correct () =
  let f p = Nx.sum (Nx.mul (Nx.exp p.fst) (Nx.sin p.snd)) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  match Rune.check_grads (module Pair) f params with
  | Ok () -> ()
  | Error msg -> fail msg

let test_check_grads_catches_wrong_rule () =
  (* A custom rule with a wrong bwd must be flagged. *)
  let broken x =
    Rune.custom_vjp
      (module Single)
      ~fwd:(fun x -> (Nx.sin x, x))
      ~bwd:(fun x ct -> Nx.mul ct (Nx.mul_s (Nx.cos x) 2.0))
      x
  in
  match
    Rune.check_grads (module Single) (fun x -> Nx.sum (broken x)) (v3 ())
  with
  | Ok () -> fail "check_grads accepted a wrong gradient"
  | Error _ -> ()

let tests =
  [
    group "pullbacks"
      [
        test "pullback is reusable across cotangents" test_pullback_reusable;
        test "pullback rejects a cotangent shape mismatch"
          test_pullback_shape_mismatch;
      ];
    group "gradient checking"
      [
        test "accepts correct gradients" test_check_grads_accepts_correct;
        test "catches a wrong custom rule" test_check_grads_catches_wrong_rule;
      ];
    group "jacobians"
      [
        test "jacobians preserve float64" test_jacobians_float64;
        test "jacobian matches the analytic matrix" test_jacobian_analytic;
        test "jacobians preserve float32" test_jacobians_float32;
        test "mixed-dtype jacobians follow tangent space dtypes"
          test_jacobians_mixed_dtypes;
        test "jacobians restore input and output shapes"
          test_jacobians_restore_tensor_shapes;
        test "jacobians evaluate the function once"
          test_jacobians_evaluate_function_once;
        test "hessian matches the analytic matrix" test_hessian_analytic;
        test "hvp agrees with the materialized hessian" test_hvp_matches_hessian;
        test "structured hvp matches analytic" test_hvp_structured;
      ];
  ]

let () = run "rune jacobian" tests
