(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Engine invariants: what is tracked, how gradient flow is controlled, how the
   engine fails, and that differentiation composes with itself. *)

open Windtrap
open Rune_test_support.Support

(* Higher-order differentiation *)

let test_second_derivative () =
  (* d²/dx² of sum(x³) = 6x. *)
  let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let ddf = Rune.grad' (fun x -> Nx.sum (Rune.grad' cube x)) in
  check_arr ~msg:"d2x" [| 6.0; -12.0; 18.0 |] (ddf (vec32 [| 1.0; -2.0; 3.0 |]))

let test_third_derivative () =
  (* d³/dx³ of x⁴ = 24x, at x = 2: 48. *)
  let quart x = Nx.sum (Nx.mul (Nx.mul x x) (Nx.mul x x)) in
  let d1 x = Nx.sum (Rune.grad' quart x) in
  let d2 x = Nx.sum (Rune.grad' d1 x) in
  let d3 = Rune.grad' d2 in
  check_arr ~msg:"d3x" [| 48.0 |] (d3 (vec32 [| 2.0 |]))

(* Gradient-flow control *)

let test_detach_stops_gradient () =
  let x = vec32 [| 3.0 |] in
  let f x = Nx.sum (Nx.mul x (Rune.detach x)) in
  (* d/dx (x * detach x) = detach x, not 2x. *)
  check_arr ~msg:"dx" [| 3.0 |] (Rune.grad' f x)

let test_no_grad_region_is_constant () =
  let x = vec32 [| 3.0 |] in
  let f x =
    let c = Rune.no_grad (fun () -> Nx.mul x x) in
    Nx.sum (Nx.mul x c)
  in
  (* c = x² is a constant 9, so d/dx (x * c) = 9. *)
  check_arr ~msg:"dx" [| 9.0 |] (Rune.grad' f x)

let test_constants_are_not_differentiated () =
  (* A computation on tensors unrelated to the parameters contributes nothing,
     even through operations without gradient rules. *)
  let x = vec32 [| 1.0; 2.0 |] in
  let c = Nx.create f32 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  let f x =
    let _, s, _ = Nx.svd c in
    let first = Nx.reshape [||] (Nx.shrink [| (0, 1) |] s) in
    Nx.mul (Nx.sum (Nx.mul x x)) (Nx.cast f32 first)
  in
  ignore (Rune.grad' f x)

(* Error contracts *)

let test_unsupported_op_raises_when_tracked () =
  let x = Nx.create f32 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.grad'
           (fun x ->
             let _, s, _ = Nx.svd x in
             Nx.sum (Nx.cast f32 s))
           x))

let test_grad_requires_scalar () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rune.grad' (fun x -> Nx.mul x x) (vec32 [| 1.0; 2.0 |])))

(* Statefulness *)

let test_reads_are_transparent_to_grad () =
  (* Reading a tracked tensor's value inside the objective neither raises nor
     perturbs the gradient. *)
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  let f x =
    let (_ : float) = Nx.item [ 0 ] x in
    Nx.sum (Nx.mul x x)
  in
  check_arr ~msg:"dx" [| 2.0; -4.0; 6.0 |] (Rune.grad' f x)

let test_grad_is_repeatable () =
  (* Differentiating twice with the same inputs gives the same result: no state
     leaks between tapes. *)
  let f x = Nx.sum (Nx.mul x x) in
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  let g1 = to_arr (Rune.grad' f x) in
  let g2 = to_arr (Rune.grad' f x) in
  equal ~msg:"same gradient" (array float_exact) g1 g2

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
        let g = Rune.grad' f x in
        check_arr ~msg:"dx" [| 2.0; 4.0 |] g;
        check_arr ~msg:"padded values" [| 5.0; 1.0; 2.0; 5.0 |]
          (vec32 (Option.get !padded)));
    test "sort routes gradient through the permutation" (fun () ->
        let x = vec32 [| 3.0; 1.0; 2.0 |] in
        let f x =
          Nx.sum (Nx.mul (fst (Nx.sort ~axis:0 x)) (vec32 [| 10.; 20.; 30. |]))
        in
        check_arr ~msg:"dsort" [| 30.0; 10.0; 20.0 |] (Rune.grad' f x));
  ]

let test_with_debug_logs_and_preserves () =
  let buf = Buffer.create 256 in
  let ppf = Format.formatter_of_buffer buf in
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  let g =
    Rune.with_debug ~ppf (fun () -> Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x)
  in
  Format.pp_print_flush ppf ();
  check_arr ~msg:"gradient unchanged" [| 2.0; -4.0; 6.0 |] g;
  let log = Buffer.contents buf in
  let contains sub =
    let n = String.length sub and m = String.length log in
    let rec go i = i + n <= m && (String.sub log i n = sub || go (i + 1)) in
    go 0
  in
  is_true ~msg:"logs mul" (contains "mul ->");
  is_true ~msg:"logs reduce_sum" (contains "reduce_sum ->")

(* The backward pass holds cotangents as the lazy views its pulls produce and
   materializes where a reshape needs it and where a gradient leaves the tape. A
   pair of transposes that cancel must then cost the gradient four permutes and
   nothing else: no copy, so nothing a compiler has to run. The graph is the one
   a grouped attention layer differentiates. *)
let test_lazy_cotangents () =
  let q = Nx.ones Nx.float32 [| 2; 2; 2; 3; 4 |] in
  let attention_like k =
    let scores = Nx.matmul q (Nx.swapaxes 3 4 (Nx.unsqueeze ~axes:[ 2 ] k)) in
    Nx.sum (Nx.mul scores scores)
  in
  let heads x = Nx.swapaxes 1 2 (Nx.reshape [| 2; 3; 2; 4 |] x) in
  let pair t = Nx.swapaxes 1 2 (Nx.swapaxes 1 2 t) in
  let x =
    Nx.create Nx.float32 [| 2; 3; 8 |]
      (Array.init 48 (fun i -> float_of_int (i mod 7)))
  in
  let ops f =
    let buf = Buffer.create 256 in
    let ppf = Format.formatter_of_buffer buf in
    let g = Rune.with_debug ~ppf (fun () -> Rune.grad' f x) in
    Format.pp_print_flush ppf ();
    let names =
      List.filter_map
        (fun line ->
          match String.index_opt line ' ' with
          | Some i -> Some (String.sub line 0 i)
          | None -> None)
        (String.split_on_char '\n' (Buffer.contents buf))
    in
    (g, List.sort compare names)
  in
  let g, plain = ops (fun x -> attention_like (heads x)) in
  let g', paired = ops (fun x -> attention_like (pair (heads x))) in
  check_arr ~msg:"same gradient"
    (Nx.to_array (Nx.reshape [| 48 |] g))
    (Nx.reshape [| 48 |] g');
  let without name = List.filter (fun n -> n <> name) in
  equal ~msg:"the pair adds permutes only" (list string)
    (without "permute" plain) (without "permute" paired);
  let count name l = List.length (List.filter (fun n -> n = name) l) in
  equal ~msg:"two forward, two backward" int
    (count "permute" plain + 4)
    (count "permute" paired);
  equal ~msg:"copies: the two reshape pulls and the gradient leaving" int 3
    (count "contiguous" paired)

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
      ];
    group "statefulness"
      [
        test "grad is repeatable" test_grad_is_repeatable;
        test "value reads are transparent" test_reads_are_transparent_to_grad;
      ];
    group "regressions" test_engine_fixes;
    group "backward pass"
      [
        test "cotangents stay lazy views until a reshape or the result"
          test_lazy_cotangents;
      ];
    group "debugging"
      [
        test "with_debug logs ops and preserves results"
          test_with_debug_logs_and_preserves;
      ];
  ]

let () = exit (run "rune engine" tests)
