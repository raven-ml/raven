(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let f64 = Nx.float64

(* Test: Jacobian of linear function f(x) = A*x + b is exactly A *)
let test_jacfwd_linear () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = Nx.create f64 [| 2 |] [| 10.0; 20.0 |] in
  let f x =
    Nx.add (Nx.reshape [| 2 |] (Nx.matmul a (Nx.reshape [| 3; 1 |] x))) b
  in
  let x = Nx.create f64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let j = Rune.jacfwd f x in
  for i = 0 to 1 do
    for k = 0 to 2 do
      is_true
        ~msg:(Printf.sprintf "jacfwd J[%d,%d] = A[%d,%d]" i k i k)
        (Float.abs (Nx.item [ i; k ] j -. Nx.item [ i; k ] a) < 1e-10)
    done
  done

let test_jacrev_linear () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = Nx.create f64 [| 2 |] [| 10.0; 20.0 |] in
  let f x =
    Nx.add (Nx.reshape [| 2 |] (Nx.matmul a (Nx.reshape [| 3; 1 |] x))) b
  in
  let x = Nx.create f64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let j = Rune.jacrev f x in
  for i = 0 to 1 do
    for k = 0 to 2 do
      is_true
        ~msg:(Printf.sprintf "jacrev J[%d,%d] = A[%d,%d]" i k i k)
        (Float.abs (Nx.item [ i; k ] j -. Nx.item [ i; k ] a) < 1e-10)
    done
  done

(* Test: jacfwd and jacrev produce the same result on nonlinear f *)
let test_jacfwd_jacrev_agree () =
  let f x =
    let x0 = Nx.slice [ I 0 ] x in
    let x1 = Nx.slice [ I 1 ] x in
    Nx.stack ~axis:0
      [ Nx.mul x0 x1; Nx.add (Nx.square x0) (Nx.sin x1); Nx.exp x1 ]
  in
  let x = Nx.create f64 [| 2 |] [| 1.5; 0.7 |] in
  let j_fwd = Rune.jacfwd f x in
  let j_rev = Rune.jacrev f x in
  let shape_fwd = Nx.shape j_fwd in
  let shape_rev = Nx.shape j_rev in
  is_true ~msg:"same shape[0]" (shape_fwd.(0) = shape_rev.(0));
  is_true ~msg:"same shape[1]" (shape_fwd.(1) = shape_rev.(1));
  for i = 0 to shape_fwd.(0) - 1 do
    for k = 0 to shape_fwd.(1) - 1 do
      is_true
        ~msg:(Printf.sprintf "jacfwd[%d,%d] = jacrev[%d,%d]" i k i k)
        (Float.abs (Nx.item [ i; k ] j_fwd -. Nx.item [ i; k ] j_rev) < 1e-10)
    done
  done

let () =
  run "Jacobian"
    [
      test "jacfwd: linear function" test_jacfwd_linear;
      test "jacrev: linear function" test_jacrev_linear;
      test "jacfwd and jacrev agree" test_jacfwd_jacrev_agree;
    ]
