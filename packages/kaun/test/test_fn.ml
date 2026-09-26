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
  let t = if eps = 0. then float_exact else float eps in
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) t e actual.(i))
    expected

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
          match Rune.check_grads Nx.Ptree.tensor objective (x ()) with
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

(* Sampling masks *)

let ninf = Float.neg_infinity
let k_of n = Nx.scalar Nx.int32 (Int32.of_int n)
let p_of v = Nx.scalar f64 v

let masked ~msg expected t =
  equal ~msg (array float_exact) expected (Nx.to_array t)

let test_top_k () =
  let logits = vec [| 1.0; 4.0; 2.0; 3.0 |] in
  masked ~msg:"the two largest stay in place" [| ninf; 4.0; ninf; 3.0 |]
    (Fn.keep_top_k ~k:(k_of 2) logits);
  masked ~msg:"k = 1 is the maximum"
    [| ninf; 4.0; ninf; ninf |]
    (Fn.keep_top_k ~k:(k_of 1) logits);
  masked ~msg:"k below 1 clamps"
    [| ninf; 4.0; ninf; ninf |]
    (Fn.keep_top_k ~k:(k_of 0) logits);
  masked ~msg:"k past the vocabulary keeps everything" [| 1.0; 4.0; 2.0; 3.0 |]
    (Fn.keep_top_k ~k:(k_of 9) logits);
  masked ~msg:"ties at the threshold are kept" [| 2.0; 2.0; ninf; 5.0 |]
    (Fn.keep_top_k ~k:(k_of 2) (vec [| 2.0; 2.0; 1.0; 5.0 |]))

let test_top_k_per_row () =
  let logits = Nx.create f64 [| 2; 3 |] [| 1.0; 3.0; 2.0; 6.0; 4.0; 5.0 |] in
  let k = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
  masked ~msg:"each row has its own k"
    [| ninf; 3.0; ninf; 6.0; ninf; 5.0 |]
    (Fn.keep_top_k ~k logits)

let test_top_p () =
  (* Probabilities 0.5, 0.3, 0.15, 0.05, given out of order. *)
  let logits = Nx.log (vec [| 0.15; 0.5; 0.05; 0.3 |]) in
  let kept p =
    Array.map
      (fun v -> v > ninf)
      (Nx.to_array (Fn.keep_top_p ~p:(p_of p) logits))
  in
  equal ~msg:"0.5 is reached by the first entry" (array bool)
    [| false; true; false; false |]
    (kept 0.5);
  equal ~msg:"0.7 needs two entries" (array bool)
    [| false; true; false; true |]
    (kept 0.7);
  equal ~msg:"0.9 needs three" (array bool)
    [| true; true; false; true |]
    (kept 0.9);
  equal ~msg:"p = 0 is greedy" (array bool)
    [| false; true; false; false |]
    (kept 0.0);
  equal ~msg:"p = 1 keeps everything" (array bool)
    [| true; true; true; true |]
    (kept 1.0);
  (* A confident row: its cumulative sum rounds to one before the tail. *)
  let peaked = Nx.create Nx.float32 [| 1; 4 |] [| 20.0; 0.0; 0.0; 0.0 |] in
  masked ~msg:"p = 1 keeps the tail of a confident row"
    [| 20.0; 0.0; 0.0; 0.0 |]
    (Fn.keep_top_p ~p:(Nx.scalar Nx.float32 1.0) peaked);
  masked ~msg:"an all-equal row keeps everything by the tie rule"
    [| 1.0; 1.0; 1.0 |]
    (Fn.keep_top_p ~p:(p_of 0.1) (vec [| 1.0; 1.0; 1.0 |]));
  masked ~msg:"entries already removed stay removed" [| ninf; 2.0; ninf |]
    (Fn.keep_top_p ~p:(p_of 0.5) (vec [| ninf; 2.0; 1.0 |]))

let test_top_p_per_row () =
  (* Rank 3, one p per leading position, over rows of probabilities 0.5, 0.3 and
     0.2. *)
  let row = [| log 0.5; log 0.3; log 0.2 |] in
  let logits = Nx.create f64 [| 2; 1; 3 |] (Array.append row row) in
  let p = Nx.create f64 [| 2; 1 |] [| 0.6; 0.95 |] in
  equal ~msg:"each row has its own p" (array bool)
    [| true; true; false; true; true; true |]
    (Array.map (fun v -> v > ninf) (Nx.to_array (Fn.keep_top_p ~p logits)))

let test_masks_compose_and_sample () =
  let logits = Nx.log (vec [| 0.15; 0.5; 0.05; 0.3 |]) in
  let policy logits =
    Fn.keep_top_p ~p:(p_of 0.7)
      (Fn.keep_top_k ~k:(k_of 3) (Nx.div_s logits 0.8))
  in
  let draws =
    List.init 40 (fun i ->
        Int32.to_int
          (Nx.item [ 0 ]
             (Nx.Rng.categorical (Nx.Rng.key i)
                (Nx.reshape [| 1; 4 |] (policy logits)))))
  in
  is_true ~msg:"only surviving tokens are drawn"
    (List.for_all (fun t -> t = 1 || t = 3) draws);
  is_true ~msg:"both survivors are drawn" (List.mem 1 draws && List.mem 3 draws)

let test_masks_jit () =
  let logits =
    Nx.create Nx.float32 [| 2; 4 |] [| 1.0; 4.0; 2.0; 3.0; 0.5; 0.1; 0.9; 0.3 |]
  in
  let p = Nx.scalar Nx.float32 0.8 in
  let policy l = Fn.keep_top_p ~p (Fn.keep_top_k ~k:(k_of 3) l) in
  masked ~msg:"compiled masks equal eager masks"
    (Nx.to_array (policy logits))
    (Rune.jit' policy logits)

let test_masks_reject_bad_shapes () =
  raises
    (Invalid_argument
       "Fn.keep_top_k: the parameter must be a scalar or have the logits' \
        leading shape") (fun () ->
      Fn.keep_top_k
        ~k:(Nx.create Nx.int32 [| 3 |] [| 1l; 1l; 1l |])
        (Nx.zeros f64 [| 2; 4 |]));
  raises (Invalid_argument "Fn.keep_top_p: logits must not be a scalar")
    (fun () -> Fn.keep_top_p ~p:(p_of 0.5) (Nx.scalar f64 1.0))

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
    group "sampling masks"
      [
        test "keep_top_k keeps the k largest of a row" test_top_k;
        test "keep_top_k takes one k per row" test_top_k_per_row;
        test "keep_top_p keeps the fewest entries reaching p" test_top_p;
        test "keep_top_p takes one p per row" test_top_p_per_row;
        test "masks compose into a sampling policy"
          test_masks_compose_and_sample;
        test "masks compile" test_masks_jit;
        test "bad parameter shapes are rejected" test_masks_reject_bad_shapes;
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

let () = exit (run "kaun fn" tests)
