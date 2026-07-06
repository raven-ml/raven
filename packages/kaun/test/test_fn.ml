(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Fn = Kaun.Fn

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_arr ?(eps = 1e-9) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

(* Single-tensor Ptree.S instance for Rune.check_grads. *)
module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* Analytic values *)

let test_relu () =
  check_arr ~msg:"relu" [| 0.; 0.; 0.; 1.; 2. |]
    (Fn.relu (vec [| -2.; -1.; 0.; 1.; 2. |]))

let test_leaky_relu () =
  check_arr ~msg:"default slope" [| -0.02; 0.; 3. |]
    (Fn.leaky_relu (vec [| -2.; 0.; 3. |]));
  check_arr ~msg:"custom slope" [| -0.4; 2. |]
    (Fn.leaky_relu ~negative_slope:0.2 (vec [| -2.; 2. |]))

let test_sigmoid () =
  check_arr ~msg:"sigmoid"
    [| 0.5; 0.7310585786300049; 0.2689414213699951 |]
    (Fn.sigmoid (vec [| 0.; 1.; -1. |]))

let test_tanh () =
  check_arr ~msg:"tanh"
    [| 0.; 0.7615941559557649; -0.7615941559557649 |]
    (Fn.tanh (vec [| 0.; 1.; -1. |]))

let test_gelu () =
  check_arr ~msg:"gelu"
    [| 0.; 0.8413447460685429; -0.15865525393145707; 1.9544997361036416 |]
    (Fn.gelu (vec [| 0.; 1.; -1.; 2. |]))

let test_gelu_approx () =
  check_arr ~msg:"gelu_approx"
    [| 0.; 0.8411919906082768; -0.15880800939172324; 1.954597694087775 |]
    (Fn.gelu_approx (vec [| 0.; 1.; -1.; 2. |]))

let test_gelu_approx_close_to_gelu () =
  (* The documented contract: about 1e-3 absolute error. *)
  let x = vec (Array.init 33 (fun i -> -4. +. (0.25 *. float_of_int i))) in
  let diff = Nx.max (Nx.abs (Nx.sub (Fn.gelu x) (Fn.gelu_approx x))) in
  is_true ~msg:"within 2e-3 of exact gelu" (Nx.item [] diff < 2e-3)

let test_silu () =
  check_arr ~msg:"silu"
    [| 0.; 0.7310585786300049; -0.2689414213699951; 1.7615941559557646 |]
    (Fn.silu (vec [| 0.; 1.; -1.; 2. |]))

let test_softplus () =
  check_arr ~msg:"softplus"
    [| 0.6931471805599453; 1.3132616875182228; 0.31326168751822286 |]
    (Fn.softplus (vec [| 0.; 1.; -1. |]))

let softmax_123 =
  [| 0.09003057317038046; 0.24472847105479764; 0.6652409557748218 |]

let test_softmax () =
  check_arr ~msg:"softmax [1;2;3]" softmax_123
    (Fn.softmax (vec [| 1.; 2.; 3. |]))

let test_softmax_axis () =
  let x = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_arr ~msg:"rows sum to 1 along the default last axis" [| 1.; 1. |]
    (Nx.sum ~axes:[ 1 ] (Fn.softmax x));
  check_arr ~msg:"columns sum to 1 along axis 0" [| 1.; 1.; 1. |]
    (Nx.sum ~axes:[ 0 ] (Fn.softmax ~axis:0 x));
  (* The rows differ by a shift, so their softmax is identical. *)
  check_arr ~msg:"shift invariance"
    (Array.append softmax_123 softmax_123)
    (Fn.softmax x)

let test_log_softmax () =
  check_arr ~msg:"log_softmax [1;2;3]"
    [| -2.4076059644443806; -1.4076059644443804; -0.4076059644443804 |]
    (Fn.log_softmax (vec [| 1.; 2.; 3. |]));
  let x = vec [| 0.3; -1.2; 0.8 |] in
  check_arr ~msg:"agrees with log of softmax at moderate logits"
    (to_arr (Nx.log (Fn.softmax x)))
    (Fn.log_softmax x)

(* Numerical stability *)

let test_softmax_large_logits () =
  check_arr ~msg:"large logits equal shifted logits" softmax_123
    (Fn.softmax (vec [| 1001.; 1002.; 1003. |]))

let test_log_softmax_extreme_logits () =
  check_arr ~msg:"extreme spread stays finite" [| -2000.; -1000.; 0. |]
    (Fn.log_softmax (vec [| -1000.; 0.; 1000. |]))

let test_softplus_saturates () =
  check_arr ~msg:"softplus at large |x|" [| 1000.; 0. |]
    (Fn.softplus (vec [| 1000.; -1000. |]))

let test_sigmoid_saturates () =
  check_arr ~msg:"sigmoid at large |x|" [| 1.; 0. |]
    (Fn.sigmoid (vec [| 1000.; -1000. |]))

(* Gradients: analytic derivatives via Rune.grad' on float64. *)

let grad_of f x = Rune.grad' (fun x -> Nx.sum (f x)) x

let test_grad_relu () =
  let x = vec [| -1.5; 0.5; 2.0 |] in
  check_arr ~msg:"relu'" [| 0.; 1.; 1. |] (grad_of Fn.relu x);
  check_arr ~msg:"leaky_relu'" [| 0.01; 1.; 1. |]
    (grad_of (fun x -> Fn.leaky_relu x) x);
  check_arr ~msg:"leaky_relu' custom slope" [| 0.2; 1.; 1. |]
    (grad_of (Fn.leaky_relu ~negative_slope:0.2) x)

let test_grad_sigmoid_tanh_softplus () =
  let x = vec [| 0.9; -1.7 |] in
  (* sigmoid' = s * (1 - s); softplus' = sigmoid; tanh' = 1 - tanh². *)
  check_arr ~msg:"sigmoid'"
    [| 0.2055003073422635; 0.13060574696620805 |]
    (grad_of Fn.sigmoid x);
  check_arr ~msg:"tanh'"
    [| 0.4869173611483415; 0.1250098706334466 |]
    (grad_of Fn.tanh x);
  check_arr ~msg:"softplus'"
    [| 0.7109495026250039; 0.1544652650835347 |]
    (grad_of Fn.softplus x)

let test_grad_gelu_silu () =
  let x = vec [| 1.0; -1.0; 0.5 |] in
  (* gelu' = Φ(x) + x φ(x); silu' = s(x) (1 + x (1 - s(x))). *)
  check_arr ~msg:"gelu'"
    [| 1.0833154705876864; -0.08331547058768629; 0.8674951246561629 |]
    (grad_of Fn.gelu x);
  check_arr ~msg:"silu'"
    [| 0.9276705118714869; 0.07232948812851325; 0.7399611873026519 |]
    (grad_of Fn.silu x)

let test_grad_softmax () =
  let x = vec [| 1.; 2.; 3. |] in
  (* sum(softmax x) is constantly 1, so its gradient vanishes. *)
  check_arr ~msg:"softmax rows are on the simplex" [| 0.; 0.; 0. |]
    (grad_of (fun x -> Fn.softmax x) x);
  (* d/dx_i sum_j log_softmax(x)_j = 1 - n * softmax(x)_i. *)
  check_arr ~msg:"log_softmax'"
    (Array.map (fun s -> 1. -. (3. *. s)) softmax_123)
    (grad_of (fun x -> Fn.log_softmax x) x)

(* Gradients: finite-difference checks on points away from the relu kink. *)

let grad_check_tests =
  let x () = vec [| 0.9; -1.7; 0.3; 2.4; -0.6 |] in
  let w () = vec [| 0.7; -0.3; 1.1; 0.2; -0.9 |] in
  let sum f x = Nx.sum (f x) in
  (* Weight softmax and log_softmax so the objective is not constant. *)
  let weighted f x = Nx.sum (Nx.mul (w ()) (f x)) in
  List.map
    (fun (name, objective) ->
      test (name ^ " gradient matches finite differences") (fun () ->
          match Rune.check_grads (module Single) objective (x ()) with
          | Ok () -> ()
          | Error msg -> fail msg))
    [
      ("relu", sum Fn.relu);
      ("leaky_relu", sum (fun x -> Fn.leaky_relu x));
      ("sigmoid", sum Fn.sigmoid);
      ("tanh", sum Fn.tanh);
      ("gelu", sum Fn.gelu);
      ("gelu_approx", sum Fn.gelu_approx);
      ("silu", sum Fn.silu);
      ("softplus", sum Fn.softplus);
      ("softmax", weighted (fun x -> Fn.softmax x));
      ("log_softmax", weighted (fun x -> Fn.log_softmax x));
    ]

let tests =
  [
    group "values"
      [
        test "relu clamps negatives to zero" test_relu;
        test "leaky_relu scales negatives by the slope" test_leaky_relu;
        test "sigmoid matches the logistic function" test_sigmoid;
        test "tanh matches the hyperbolic tangent" test_tanh;
        test "gelu matches the exact erf form" test_gelu;
        test "gelu_approx matches the tanh form" test_gelu_approx;
        test "gelu_approx stays within 2e-3 of gelu"
          test_gelu_approx_close_to_gelu;
        test "silu is x times sigmoid" test_silu;
        test "softplus matches log(1 + exp x)" test_softplus;
        test "softmax normalizes exponentials" test_softmax;
        test "softmax normalizes along the requested axis" test_softmax_axis;
        test "log_softmax is the log of softmax" test_log_softmax;
      ];
    group "numerical stability"
      [
        test "softmax survives large logits" test_softmax_large_logits;
        test "log_softmax survives extreme logits"
          test_log_softmax_extreme_logits;
        test "softplus does not overflow" test_softplus_saturates;
        test "sigmoid saturates cleanly" test_sigmoid_saturates;
      ];
    group "gradients"
      ([
         test "relu family has piecewise-constant gradients" test_grad_relu;
         test "sigmoid, tanh and softplus have analytic gradients"
           test_grad_sigmoid_tanh_softplus;
         test "gelu and silu have analytic gradients" test_grad_gelu_silu;
         test "softmax and log_softmax have analytic gradients"
           test_grad_softmax;
       ]
      @ grad_check_tests);
  ]

let () = run "kaun fn" tests
