(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Engine invariants: what is tracked, how gradient flow is controlled, how the
   engine fails, and that differentiation composes with itself. *)

open Windtrap
open Rune_next_test_support.Support

(* Higher-order differentiation *)

let test_second_derivative () =
  (* d²/dx² of sum(x³) = 6x. *)
  let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let ddf = Rune_next.grad' (fun x -> Nx.sum (Rune_next.grad' cube x)) in
  check_arr ~msg:"d2x" [| 6.0; -12.0; 18.0 |] (ddf (vec32 [| 1.0; -2.0; 3.0 |]))

let test_third_derivative () =
  (* d³/dx³ of x⁴ = 24x, at x = 2: 48. *)
  let quart x = Nx.sum (Nx.mul (Nx.mul x x) (Nx.mul x x)) in
  let d1 x = Nx.sum (Rune_next.grad' quart x) in
  let d2 x = Nx.sum (Rune_next.grad' d1 x) in
  let d3 = Rune_next.grad' d2 in
  check_arr ~msg:"d3x" [| 48.0 |] (d3 (vec32 [| 2.0 |]))

(* Gradient-flow control *)

let test_detach_stops_gradient () =
  let x = vec32 [| 3.0 |] in
  let f x = Nx.sum (Nx.mul x (Rune_next.detach x)) in
  (* d/dx (x * detach x) = detach x, not 2x. *)
  check_arr ~msg:"dx" [| 3.0 |] (Rune_next.grad' f x)

let test_no_grad_region_is_constant () =
  let x = vec32 [| 3.0 |] in
  let f x =
    let c = Rune_next.no_grad (fun () -> Nx.mul x x) in
    Nx.sum (Nx.mul x c)
  in
  (* c = x² is a constant 9, so d/dx (x * c) = 9. *)
  check_arr ~msg:"dx" [| 9.0 |] (Rune_next.grad' f x)

let test_constants_are_not_differentiated () =
  (* A computation on tensors unrelated to the parameters contributes nothing,
     even through operations without gradient rules. *)
  let x = vec32 [| 1.0; 2.0 |] in
  let c = Nx.create f32 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  let f x =
    let _, s, _ = Nx.svd c in
    let first = Nx.reshape [||] (Nx.shrink [| (0, 1) |] s) in
    Nx.mul (Nx.sum (Nx.mul x x)) (Nx.astype f32 first)
  in
  ignore (Rune_next.grad' f x)

(* Error contracts *)

let test_unsupported_op_raises_when_tracked () =
  let x = Nx.create f32 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  raises_invalid_arg (fun () ->
      ignore
        (Rune_next.grad'
           (fun x ->
             let _, s, _ = Nx.svd x in
             Nx.sum (Nx.astype f32 s))
           x))

let test_grad_requires_scalar () =
  raises_invalid_arg (fun () ->
      ignore (Rune_next.grad' (fun x -> Nx.mul x x) (vec32 [| 1.0; 2.0 |])))

let test_mutation_raises () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune_next.grad'
           (fun x ->
             Nx.set_item [ 0 ] 1.0 x;
             Nx.sum x)
           (vec32 [| 1.0; 2.0 |])))

(* Statefulness *)

let test_reads_are_transparent_to_grad () =
  (* Reading a tracked tensor's value inside the objective neither raises nor
     perturbs the gradient. *)
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  let f x =
    let (_ : float) = Nx.item [ 0 ] x in
    Nx.sum (Nx.mul x x)
  in
  check_arr ~msg:"dx" [| 2.0; -4.0; 6.0 |] (Rune_next.grad' f x)

let test_grad_is_repeatable () =
  (* Differentiating twice with the same inputs gives the same result: no state
     leaks between tapes. *)
  let f x = Nx.sum (Nx.mul x x) in
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  let g1 = to_arr (Rune_next.grad' f x) in
  let g2 = to_arr (Rune_next.grad' f x) in
  equal ~msg:"same gradient" (array (float 0.0)) g1 g2

let test_engine_fixes =
  [
    test "pad keeps its fill value under grad" (fun () ->
        (* The padded positions carry the requested fill value even while
           differentiating, and the gradient flows only to the original
           extent. *)
        let x = vec32 [| 1.0; 2.0 |] in
        let padded = ref None in
        let f x =
          let p = Nx.pad [| (1, 1) |] 5.0 x in
          padded := Some (to_arr p);
          Nx.sum (Nx.mul p p)
        in
        let g = Rune_next.grad' f x in
        check_arr ~msg:"dx" [| 2.0; 4.0 |] g;
        check_arr ~msg:"padded values" [| 5.0; 1.0; 2.0; 5.0 |]
          (vec32 (Option.get !padded)));
    test "sort routes gradient through the permutation" (fun () ->
        let x = vec32 [| 3.0; 1.0; 2.0 |] in
        let f x =
          Nx.sum (Nx.mul (fst (Nx.sort ~axis:0 x)) (vec32 [| 10.; 20.; 30. |]))
        in
        check_arr ~msg:"dsort" [| 30.0; 10.0; 20.0 |] (Rune_next.grad' f x));
  ]

let tests =
  [
    group "higher order"
      [
        test "second derivative composes" test_second_derivative;
        test "third derivative composes" test_third_derivative;
      ];
    group "gradient flow"
      [
        test "detach stops the gradient" test_detach_stops_gradient;
        test "no_grad region is constant" test_no_grad_region_is_constant;
        test "constants pass through unsupported ops"
          test_constants_are_not_differentiated;
      ];
    group "error contracts"
      [
        test "unsupported op raises when its input is tracked"
          test_unsupported_op_raises_when_tracked;
        test "grad requires a scalar objective" test_grad_requires_scalar;
        test "in-place mutation raises" test_mutation_raises;
      ];
    group "statefulness"
      [
        test "grad is repeatable" test_grad_is_repeatable;
        test "value reads are transparent" test_reads_are_transparent_to_grad;
      ];
    group "regressions" test_engine_fixes;
  ]

let () = run "rune-next engine" tests
