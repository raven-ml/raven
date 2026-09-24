(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Control-flow combinators: eager semantics, and composition with grad and
   vmap. *)

open Windtrap
open Rune_test_support.Support

let v4 () = vec64 [| 0.5; -1.2; 2.1; 0.8 |]

(* scan with a running-sum carry is a cumulative sum. *)
let cumsum_scan xs =
  snd
    (Rune.scan'
       ~f:(fun c x ->
         let c = Nx.add c x in
         (c, c))
       ~init:(Nx.scalar f64 0.0) xs)

let test_scan_is_cumsum () =
  check_arr ~msg:"scan cumsum"
    (to_arr (Nx.cumsum (v4 ())))
    (cumsum_scan (v4 ()))

let test_scan_final_carry () =
  let carry, _ =
    Rune.scan' ~f:(fun c x -> (Nx.add c x, c)) ~init:(Nx.scalar f64 0.0) (v4 ())
  in
  check_arr ~msg:"carry" [| 0.5 -. 1.2 +. 2.1 +. 0.8 |] carry

let test_grad_through_scan () =
  (* Gradients through the scan equal gradients through the primitive. *)
  let f xs = Nx.sum (Nx.mul (cumsum_scan xs) (cumsum_scan xs)) in
  let g xs = Nx.sum (Nx.mul (Nx.cumsum xs) (Nx.cumsum xs)) in
  check_arr ~msg:"d scan" (to_arr (Rune.grad' g (v4 ()))) (Rune.grad' f (v4 ()))

let test_vmap_of_scan () =
  let x =
    Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]
  in
  let y = Rune.vmap' cumsum_scan x in
  check_arr ~msg:"vmap scan" (to_arr (Nx.cumsum ~axis:1 x)) y

let test_scan_rejects_scalar () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (cumsum_scan (Nx.scalar f64 1.0)))

(* Structured scans: a carry of two tensors, rows of a pair, outputs that are a
   list, and a fold that emits nothing. *)
let test_scan_structures () =
  let xs = (v4 (), Nx.mul_s (v4 ()) 2.0) in
  let (sum, count), ys =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.(list tensor)
      ~f:(fun (sum, count) (a, b) ->
        let sum = Nx.add sum (Nx.add a b) in
        ((sum, Nx.add_s count 1.0), [ sum; a ]))
      ~init:(Nx.scalar f64 0.0, Nx.scalar f64 0.0)
      xs
  in
  check_arr ~msg:"sum" [| 3.0 *. (0.5 -. 1.2 +. 2.1 +. 0.8) |] sum;
  check_arr ~msg:"count" [| 4.0 |] count;
  (match ys with
  | [ sums; rows ] ->
      check_arr ~msg:"sums" (to_arr (Nx.cumsum (Nx.mul_s (v4 ()) 3.0))) sums;
      check_arr ~msg:"rows" (to_arr (v4 ())) rows
  | _ -> fail "the outputs are a list of two tensors");
  let total, () =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun c x -> (Nx.add c x, ()))
      ~init:(Nx.scalar f64 0.0) (v4 ())
  in
  check_arr ~msg:"nothing to emit" [| 0.5 -. 1.2 +. 2.1 +. 0.8 |] total

(* A body that returns a carry of another structure than it received raises,
   naming the path where they differ. *)
let grow c x = (c @ [ x ], ())
let carry = Nx.Ptree.(list tensor)

let test_scan_rejects_a_changed_carry () =
  raises
    (Invalid_argument
       "Rune.scan: the root: length 2 in the carry the body returned, length 1 \
        in the carry it received") (fun () ->
      ignore
        (Rune.scan carry Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
           ~init:[ Nx.scalar f64 0.0 ]
           (v4 ())))

let test_staged_scan_rejects_a_changed_carry () =
  let f xs =
    let c, () =
      Rune.scan carry Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
        ~init:[ Nx.scalar f64 0.0 ]
        xs
    in
    List.hd c
  in
  raises
    (Invalid_argument
       "Rune.scan: the root: length 2 in the carry the body returned, length 1 \
        in the carry it received") (fun () -> ignore (Rune.jit' f (v4 ())))

(* Every step's outputs have the first step's visits. *)
(* A carry that changes dtype is named by the scan. The carry is a tensor of
   any dtype, so that the body can change it. *)
module Packed = struct
  type _ t = Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

let test_scan_rejects_a_changed_dtype () =
  raises
    (Invalid_argument
       "Rune.scan: the root: float32 in the carry the body returned, float64 \
        in the carry it received") (fun () ->
      ignore
        (Rune.scan
           (Nx.Ptree.instantiate (module Packed))
           Nx.Ptree.tensor Nx.Ptree.unit
           ~f:(fun (Nx.P c) _ -> (Nx.P (Nx.cast Nx.float32 c), ()))
           ~init:(Nx.P (Nx.scalar f64 0.0))
           (v4 ())))

let test_scan_rejects_changed_outputs () =
  let f c x =
    let c = Nx.add_s c 1.0 in
    (c, if Nx.item [] c > 1.5 then Some x else None)
  in
  raises
    (Invalid_argument
       "Rune.scan: the root: Some in a step's outputs, None in the first \
        step's outputs") (fun () ->
      ignore
        (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor
           Nx.Ptree.(option tensor)
           ~f ~init:(Nx.scalar f64 0.0) (v4 ())))

(* Compositions where another transformation claims the scan below any stager:
   reverse-mode must fold eagerly so every step lands on its tape. Each case
   compares against the identical computation over the cumsum primitive. *)

let quad_scan xs = Nx.sum (Nx.mul (cumsum_scan xs) (cumsum_scan xs))
let quad_prim xs = Nx.sum (Nx.mul (Nx.cumsum xs) (Nx.cumsum xs))

let test_vmap_of_grad_through_scan () =
  let x =
    Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]
  in
  check_arr ~msg:"per-sample grads"
    (to_arr (Rune.vmap' (Rune.grad' quad_prim) x))
    (Rune.vmap' (Rune.grad' quad_scan) x)

let test_grad_of_grad_through_scan () =
  let second f xs = Rune.grad' (fun xs -> Nx.sum (Rune.grad' f xs)) xs in
  check_arr ~msg:"second order"
    (to_arr (second quad_prim (v4 ())))
    (second quad_scan (v4 ()))

let test_hvp_through_scan () =
  (* Forward-over-reverse. *)
  let v = vec64 [| 1.0; 0.0; -1.0; 0.5 |] in
  check_arr ~msg:"hvp"
    (to_arr (Rune.hvp' quad_prim (v4 ()) v))
    (Rune.hvp' quad_scan (v4 ()) v)

let test_cond_branches () =
  let branch x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then" [| 0.25 +. 4.0 |] (branch (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else" [| -2.5 |] (branch (vec64 [| -0.5; -2.0 |]))

let test_grad_through_cond () =
  (* The taken branch is what gets differentiated. *)
  let f x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then grad" [| 1.0; 4.0 |]
    (Rune.grad' f (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else grad" [| 1.0; 1.0 |]
    (Rune.grad' f (vec64 [| -0.5; -2.0 |]))

let test_while_loop () =
  (* Double until the sum exceeds 10: 1.5 -> 3 -> 6 -> 12. *)
  let y =
    Rune.while_loop
      ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
      ~body:(fun c -> Nx.mul_s c 2.0)
      (vec64 [| 1.0; 0.5 |])
  in
  check_arr ~msg:"final" [| 8.0; 4.0 |] y

let test_grad_through_while_loop () =
  (* Each x doubles k times before the loop exits; d/dx sum = 2^k. *)
  let f x =
    Nx.sum
      (Rune.while_loop
         ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
         ~body:(fun c -> Nx.mul_s c 2.0)
         x)
  in
  check_arr ~msg:"d while" [| 8.0; 8.0 |] (Rune.grad' f (vec64 [| 1.0; 0.5 |]))

let tests =
  [
    group "scan"
      [
        test "running-sum scan is cumsum" test_scan_is_cumsum;
        test "returns the final carry" test_scan_final_carry;
        test "differentiates like the primitive" test_grad_through_scan;
        test "vectorizes over the batch" test_vmap_of_scan;
        test "rejects a scalar input" test_scan_rejects_scalar;
        test "folds structures" test_scan_structures;
        test "rejects a changed carry" test_scan_rejects_a_changed_carry;
        test "rejects a changed carry under jit"
          test_staged_scan_rejects_a_changed_carry;
        test "rejects changed outputs" test_scan_rejects_changed_outputs;
        test "rejects a carry of another dtype"
          test_scan_rejects_a_changed_dtype;
        test "per-sample gradients (vmap of grad)"
          test_vmap_of_grad_through_scan;
        test "second-order gradients (grad of grad)"
          test_grad_of_grad_through_scan;
        test "hessian-vector product (jvp of grad)" test_hvp_through_scan;
      ];
    group "cond"
      [
        test "selects the branch by predicate" test_cond_branches;
        test "differentiates the taken branch" test_grad_through_cond;
      ];
    group "while_loop"
      [
        test "iterates until the predicate fails" test_while_loop;
        test "differentiates the taken iterations" test_grad_through_while_loop;
      ];
  ]

let () = run "rune control" tests
