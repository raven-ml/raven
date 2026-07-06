(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun_next

(* A bare float64 tensor as a differentiable structure, for gradient checks. *)
module Tensor = struct
  type t = (float, Nx.float64_elt) Nx.t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (t : t) : t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (a : t)
      (b : t) : t =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (t : t) = f t
end

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let mat rows cols xs = Nx.create Nx.float64 [| rows; cols |] xs
let labels ls = Nx.create Nx.int32 [| Array.length ls |] ls
let close ?(eps = 1e-9) expected l = equal (float eps) expected (Nx.item [] l)

let close_grad ?(eps = 1e-9) expected g =
  equal (array (float eps)) expected (Nx.to_array g)

let grads_ok ?msg f x =
  match Rune_next.check_grads (module Tensor) f x with
  | Ok () -> ()
  | Error e -> fail (match msg with Some m -> m ^ ": " ^ e | None -> e)

(* Regression *)

let mse_tests =
  [
    test "matches the analytic mean" (fun () ->
        close (14. /. 3.)
          (Loss.mse (vec [| 1.; 2.; 3. |]) (vec [| 0.; 0.; 0. |])));
    test "sum reduction adds instead of averaging" (fun () ->
        close 14.
          (Loss.mse ~reduction:`Sum
             (vec [| 1.; 2.; 3. |])
             (vec [| 0.; 0.; 0. |])));
    test "is zero when predictions equal targets" (fun () ->
        close 0. (Loss.mse (vec [| 1.; -2. |]) (vec [| 1.; -2. |])));
  ]

let mae_tests =
  [
    test "matches the analytic mean" (fun () ->
        close 2. (Loss.mae (vec [| 1.; -2.; 3. |]) (vec [| 0.; 0.; 0. |])));
    test "sum reduction adds instead of averaging" (fun () ->
        close 6.
          (Loss.mae ~reduction:`Sum
             (vec [| 1.; -2.; 3. |])
             (vec [| 0.; 0.; 0. |])));
  ]

let huber_tests =
  [
    test "is half the squared error inside delta" (fun () ->
        close 0.125 (Loss.huber (vec [| 0.5; -0.5 |]) (vec [| 0.; 0. |])));
    test "is linear beyond delta" (fun () ->
        close 1.5 (Loss.huber (vec [| 2.; -2. |]) (vec [| 0.; 0. |])));
    test "agrees with both branches at the crossover" (fun () ->
        close 0.5 (Loss.huber (vec [| 1. |]) (vec [| 0. |])));
    test "honors a custom delta" (fun () ->
        close 4. (Loss.huber ~delta:2. (vec [| 3. |]) (vec [| 0. |])));
    test "sum reduction adds instead of averaging" (fun () ->
        close 1.625
          (Loss.huber ~reduction:`Sum (vec [| 0.5; 2. |]) (vec [| 0.; 0. |])));
    test "rejects a non-positive delta" (fun () ->
        raises_invalid_arg "Loss.huber: delta must be positive (got 0)"
          (fun () -> Loss.huber ~delta:0. (vec [| 1. |]) (vec [| 0. |])));
  ]

(* Classification *)

let sigmoid_bce_tests =
  [
    test "is log 2 at zero logits" (fun () ->
        close (log 2.) (Loss.sigmoid_bce (vec [| 0.; 0. |]) (vec [| 0.; 1. |])));
    test "matches the analytic value" (fun () ->
        (* loss(1, 1) = loss(-1, 0) = log (1 + e^-1) *)
        close
          (log (1. +. exp (-1.)))
          (Loss.sigmoid_bce (vec [| 1.; -1. |]) (vec [| 1.; 0. |])));
    test "is zero at extreme correct logits" (fun () ->
        close 0. (Loss.sigmoid_bce (vec [| 1000.; -1000. |]) (vec [| 1.; 0. |])));
    test "is finite at extreme incorrect logits" (fun () ->
        close 1000.
          (Loss.sigmoid_bce (vec [| 1000.; -1000. |]) (vec [| 0.; 1. |])));
    test "sum reduction adds instead of averaging" (fun () ->
        close
          (2. *. log 2.)
          (Loss.sigmoid_bce ~reduction:`Sum
             (vec [| 0.; 0. |])
             (vec [| 0.; 1. |])));
  ]

let softmax_ce_tests =
  [
    test "is log n for uniform logits and one-hot targets" (fun () ->
        close (log 2.)
          (Loss.softmax_cross_entropy
             (mat 1 2 [| 0.; 0. |])
             (mat 1 2 [| 1.; 0. |])));
    test "matches the analytic value" (fun () ->
        (* -log_softmax([1;2;3]).(2) = log (1 + e^-1 + e^-2) *)
        close
          (log (1. +. exp (-1.) +. exp (-2.)))
          (Loss.softmax_cross_entropy
             (mat 1 3 [| 1.; 2.; 3. |])
             (mat 1 3 [| 0.; 0.; 1. |])));
    test "handles soft targets" (fun () ->
        close (log 2.)
          (Loss.softmax_cross_entropy
             (mat 1 2 [| 0.; 0. |])
             (mat 1 2 [| 0.5; 0.5 |])));
    test "averages over the batch" (fun () ->
        close (log 2.)
          (Loss.softmax_cross_entropy
             (mat 2 2 [| 0.; 0.; 0.; 0. |])
             (mat 2 2 [| 1.; 0.; 0.; 1. |])));
    test "sum reduction adds per-example losses" (fun () ->
        close
          (2. *. log 2.)
          (Loss.softmax_cross_entropy ~reduction:`Sum
             (mat 2 2 [| 0.; 0.; 0.; 0. |])
             (mat 2 2 [| 1.; 0.; 0.; 1. |])));
    test "is zero at extreme correct logits" (fun () ->
        close 0.
          (Loss.softmax_cross_entropy
             (mat 1 2 [| 1000.; 0. |])
             (mat 1 2 [| 1.; 0. |])));
    test "is finite at extreme incorrect logits" (fun () ->
        close 1000.
          (Loss.softmax_cross_entropy
             (mat 1 2 [| 1000.; 0. |])
             (mat 1 2 [| 0.; 1. |])));
    test "rejects mismatched target shapes" (fun () ->
        raises_invalid_arg
          "Loss.softmax_cross_entropy: targets shape [2; 2] does not match \
           logits shape [2; 3]" (fun () ->
            Loss.softmax_cross_entropy
              (mat 2 3 (Array.make 6 0.))
              (mat 2 2 (Array.make 4 0.))));
    test "rejects rank-0 logits" (fun () ->
        raises_invalid_arg
          "Loss.softmax_cross_entropy: logits must have rank >= 1" (fun () ->
            Loss.softmax_cross_entropy (Nx.scalar Nx.float64 0.)
              (Nx.scalar Nx.float64 1.)));
  ]

let logits_and_labels =
  Testable.with_gen
    Gen.(
      pair
        (list_size (pure 6) (float_range (-10.) 10.))
        (list_size (pure 2) (int_range 0 2)))
    (pair (list (float 1e-12)) (list int))

let softmax_ce_sparse_tests =
  [
    test "matches the analytic value" (fun () ->
        close (log 2.)
          (Loss.softmax_cross_entropy_sparse
             (mat 1 2 [| 0.; 0. |])
             (labels [| 0l |])));
    prop' "matches the dense loss on one-hot targets" logits_and_labels
      (fun (xs, ls) ->
        let logits = mat 2 3 (Array.of_list xs) in
        let ls = labels (Array.of_list (List.map Int32.of_int ls)) in
        let one_hot = Nx.cast Nx.float64 (Nx.one_hot ~num_classes:3 ls) in
        close ~eps:1e-9
          (Nx.item [] (Loss.softmax_cross_entropy logits one_hot))
          (Loss.softmax_cross_entropy_sparse logits ls));
    test "accepts a single unbatched example" (fun () ->
        close (log 2.)
          (Loss.softmax_cross_entropy_sparse
             (vec [| 0.; 0. |])
             (Nx.scalar Nx.int32 1l)));
    test "is finite at extreme incorrect logits" (fun () ->
        close 1000.
          (Loss.softmax_cross_entropy_sparse
             (mat 1 2 [| 1000.; 0. |])
             (labels [| 1l |])));
    test "sum reduction adds per-example losses" (fun () ->
        close
          (2. *. log 2.)
          (Loss.softmax_cross_entropy_sparse ~reduction:`Sum
             (mat 2 2 [| 0.; 0.; 0.; 0. |])
             (labels [| 0l; 1l |])));
    test "rejects labels that keep the class axis" (fun () ->
        raises_invalid_arg
          "Loss.softmax_cross_entropy_sparse: labels shape [2; 3] does not \
           match logits batch shape [2]" (fun () ->
            Loss.softmax_cross_entropy_sparse
              (mat 2 3 (Array.make 6 0.))
              (Nx.create Nx.int32 [| 2; 3 |] (Array.make 6 0l))));
  ]

(* Gradients. Rune_next.check_grads compares reverse-mode gradients against
   central differences; float64 keeps the comparison reliable. Points are chosen
   away from the kinks of mae and huber. *)

let grad_tests =
  [
    test "mse gradient matches finite differences" (fun () ->
        grads_ok
          (fun p -> Loss.mse p (vec [| 0.5; -1.; 2. |]))
          (vec [| 1.; 2.; -3. |]));
    test "mse gradient is 2 (p - t) / n" (fun () ->
        let g =
          Rune_next.grad'
            (fun p -> Loss.mse p (vec [| 0.; 0. |]))
            (vec [| 1.; -2. |])
        in
        close_grad [| 1.; -2. |] g);
    test "mse sum-reduction gradient matches finite differences" (fun () ->
        grads_ok
          (fun p -> Loss.mse ~reduction:`Sum p (vec [| 0.5; -1. |]))
          (vec [| 1.; 2. |]));
    test "mae gradient matches finite differences away from zero" (fun () ->
        grads_ok
          (fun p -> Loss.mae p (vec [| 0.; 0.; 0. |]))
          (vec [| 1.; -2.; 3. |]));
    test "huber gradient matches finite differences in both regions" (fun () ->
        grads_ok
          (fun p -> Loss.huber p (vec [| 0.; 0.; 0.; 0. |]))
          (vec [| 0.5; -0.25; 2.; -3. |]));
    test "huber gradient is d inside delta and +/- delta beyond" (fun () ->
        let g =
          Rune_next.grad'
            (fun p -> Loss.huber ~reduction:`Sum p (vec [| 0.; 0. |]))
            (vec [| 0.5; -3. |])
        in
        close_grad [| 0.5; -1. |] g);
    test "sigmoid_bce gradient matches finite differences" (fun () ->
        grads_ok
          (fun z -> Loss.sigmoid_bce z (vec [| 1.; 0.; 0.5 |]))
          (vec [| 0.3; -1.2; 2. |]));
    test "sigmoid_bce gradient is finite at extreme logits" (fun () ->
        (* Analytically (sigmoid z - y) / n. *)
        let g =
          Rune_next.grad'
            (fun z -> Loss.sigmoid_bce z (vec [| 1.; 1. |]))
            (vec [| 1000.; -1000. |])
        in
        close_grad [| 0.; -0.5 |] g);
    test "softmax_cross_entropy gradient matches finite differences" (fun () ->
        grads_ok
          (fun z ->
            Loss.softmax_cross_entropy z
              (mat 2 3 [| 1.; 0.; 0.; 0.; 0.5; 0.5 |]))
          (mat 2 3 [| 0.1; -0.4; 1.2; 2.; 0.; -1. |]));
    test "softmax_cross_entropy gradient is softmax minus targets" (fun () ->
        let g =
          Rune_next.grad'
            (fun z ->
              Loss.softmax_cross_entropy ~reduction:`Sum z
                (mat 1 3 [| 0.; 0.; 1. |]))
            (mat 1 3 [| 1.; 2.; 3. |])
        in
        let s = exp 1. +. exp 2. +. exp 3. in
        close_grad ~eps:1e-9
          [| exp 1. /. s; exp 2. /. s; (exp 3. /. s) -. 1. |]
          g);
    test "softmax_cross_entropy gradient is finite at extreme logits" (fun () ->
        let g =
          Rune_next.grad'
            (fun z -> Loss.softmax_cross_entropy z (mat 1 2 [| 1.; 0. |]))
            (mat 1 2 [| 1000.; 0. |])
        in
        close_grad [| 0.; 0. |] g);
    test "softmax_cross_entropy_sparse gradient matches finite differences"
      (fun () ->
        grads_ok
          (fun z -> Loss.softmax_cross_entropy_sparse z (labels [| 2l; 0l |]))
          (mat 2 3 [| 0.1; -0.4; 1.2; 2.; 0.; -1. |]));
  ]

let tests =
  [
    group "mse" mse_tests;
    group "mae" mae_tests;
    group "huber" huber_tests;
    group "sigmoid_bce" sigmoid_bce_tests;
    group "softmax_cross_entropy" softmax_ce_tests;
    group "softmax_cross_entropy_sparse" softmax_ce_sparse_tests;
    group "gradients" grad_tests;
  ]

let () = run "kaun-next loss" tests
