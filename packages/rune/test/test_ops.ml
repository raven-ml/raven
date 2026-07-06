(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Per-operation gradient rules, validated against the finite-difference oracle
   in Support. Every differentiable primitive the engine implements should have
   a case here; a new rule in reverse.ml comes with a new case.

   Inputs are chosen inside each operation's smooth domain (away from branch
   points such as 0 for [abs], ties for [max]/[sort], and poles). *)

open Windtrap
open Rune_test_support.Support

(* Fixture tensors, reused across cases. Values are distinct and away from
   non-smooth points. *)

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]
let v3_pos () = vec64 [| 0.7; 1.3; 2.1 |]
let v3_unit () = vec64 [| 0.3; -0.6; 0.8 |]
let m23 () = mat64 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let m23_pos () = mat64 2 3 [| 0.5; 1.2; 2.1; 1.7; 0.4; 0.9 |]

(* Unary operations *)

let unary_cases =
  [
    ("neg", Nx.neg, v3);
    ("exp", Nx.exp, v3);
    ("log", Nx.log, v3_pos);
    ("sqrt", Nx.sqrt, v3_pos);
    ("recip", Nx.recip, v3);
    ("sin", Nx.sin, v3);
    ("cos", Nx.cos, v3);
    ("tan", Nx.tan, v3_unit);
    ("asin", Nx.asin, v3_unit);
    ("acos", Nx.acos, v3_unit);
    ("atan", Nx.atan, v3);
    ("sinh", Nx.sinh, v3);
    ("cosh", Nx.cosh, v3);
    ("tanh", Nx.tanh, v3);
    ("abs", Nx.abs, v3);
    ("erf", Nx.erf, v3);
  ]

let unary_tests =
  List.map
    (fun (name, op, x) -> test name (fun () -> check_grad ~msg:name op (x ())))
    unary_cases

(* Binary operations, including broadcast shapes. *)

let b3 () = vec64 [| 1.9; 0.8; -0.6 |]
let b3_pos () = vec64 [| 1.9; 0.8; 0.6 |]

let binary_cases =
  [
    ("add", Nx.add, v3, b3);
    ("sub", Nx.sub, v3, b3);
    ("mul", Nx.mul, v3, b3);
    ("div", Nx.div, v3, b3_pos);
    ("pow", Nx.pow, v3_pos, b3);
    ("maximum", Nx.maximum, v3, b3);
    ("minimum", Nx.minimum, v3, b3);
    ("atan2", Nx.atan2, v3, b3_pos);
  ]

let binary_tests =
  List.map
    (fun (name, op, a, b) ->
      test name (fun () -> check_grad2 ~msg:name op (a ()) (b ())))
    binary_cases

let broadcast_tests =
  [
    test "add broadcasts a row" (fun () ->
        check_grad2 ~msg:"add [2x3]+[3]" Nx.add (m23 ()) (b3 ()));
    test "mul broadcasts a column" (fun () ->
        check_grad2 ~msg:"mul [2x3]*[2x1]" Nx.mul (m23 ())
          (mat64 2 1 [| 1.4; -0.7 |]));
    test "sub broadcasts a scalar" (fun () ->
        check_grad2 ~msg:"sub [2x3]-[]" Nx.sub (m23 ()) (Nx.scalar f64 0.8));
  ]

(* Reductions *)

let reduction_tests =
  [
    test "sum over all axes" (fun () ->
        check_grad ~msg:"sum" (fun x -> Nx.sum x) (m23 ()));
    test "sum over one axis" (fun () ->
        check_grad ~msg:"sum axis0" (Nx.sum ~axes:[ 0 ]) (m23 ()));
    test "sum keepdims" (fun () ->
        check_grad ~msg:"sum keepdims"
          (Nx.sum ~axes:[ 1 ] ~keepdims:true)
          (m23 ()));
    test "prod over one axis" (fun () ->
        check_grad ~msg:"prod axis1" (Nx.prod ~axes:[ 1 ]) (m23_pos ()));
    test "max over one axis" (fun () ->
        check_grad ~msg:"max axis0" (Nx.max ~axes:[ 0 ]) (m23 ()));
    test "max keepdims" (fun () ->
        check_grad ~msg:"max keepdims"
          (Nx.max ~axes:[ 1 ] ~keepdims:true)
          (m23 ()));
    test "min over one axis" (fun () ->
        check_grad ~msg:"min axis1" (Nx.min ~axes:[ 1 ]) (m23 ()));
    test "mean over one axis" (fun () ->
        check_grad ~msg:"mean axis0" (Nx.mean ~axes:[ 0 ]) (m23 ()));
  ]

(* Movement operations *)

let movement_tests =
  [
    test "reshape" (fun () ->
        check_grad ~msg:"reshape" (Nx.reshape [| 3; 2 |]) (m23 ()));
    test "transpose" (fun () ->
        check_grad ~msg:"transpose" (fun x -> Nx.transpose x) (m23 ()));
    test "broadcast_to" (fun () ->
        check_grad ~msg:"broadcast_to"
          (Nx.broadcast_to [| 2; 3 |])
          (mat64 1 3 [| 0.5; -1.2; 2.1 |]));
    test "pad" (fun () ->
        check_grad ~msg:"pad" (Nx.pad [| (1, 1); (0, 2) |] 5.0) (m23 ()));
    test "shrink" (fun () ->
        check_grad ~msg:"shrink" (Nx.shrink [| (0, 2); (1, 3) |]) (m23 ()));
    test "flip" (fun () ->
        check_grad ~msg:"flip" (Nx.flip ~axes:[ 1 ]) (m23 ()));
    test "concatenate" (fun () ->
        check_grad2 ~msg:"concatenate"
          (fun a b -> Nx.concatenate ~axis:0 [ a; b ])
          (m23 ())
          (mat64 1 3 [| 0.3; 0.9; -1.1 |]));
    test "slice" (fun () ->
        check_grad ~msg:"slice"
          (fun x -> Nx.slice [ Nx.R (0, 2); Nx.I 1 ] x)
          (m23 ()));
    test "tril" (fun () ->
        check_grad ~msg:"tril" Nx.tril (mat64 2 2 [| 0.5; -1.2; 2.1; 1.7 |]));
  ]

(* Selection and indexing *)

let selection_tests =
  [
    test "where" (fun () ->
        let cond = Nx.greater (m23 ()) (Nx.zeros_like (m23 ())) in
        check_grad2 ~msg:"where"
          (fun a b -> Nx.where cond a b)
          (m23 ())
          (mat64 2 3 [| 0.3; 0.9; -1.1; 0.2; -0.5; 1.3 |]));
    test "take_along_axis" (fun () ->
        let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 2l |] in
        check_grad ~msg:"take_along_axis"
          (fun x -> Nx.take_along_axis ~axis:1 idx x)
          (m23 ()));
    test "sort" (fun () ->
        check_grad ~msg:"sort" (fun x -> fst (Nx.sort ~axis:1 x)) (m23 ()));
  ]

(* Scans *)

let scan_tests =
  [
    test "cumsum" (fun () ->
        check_grad ~msg:"cumsum" (Nx.cumsum ~axis:1) (m23 ()));
    test "cumprod" (fun () ->
        check_grad ~msg:"cumprod" (Nx.cumprod ~axis:1) (m23_pos ()));
  ]

(* Matrix multiplication: the four batching branches of the rule. *)

let a2 () = mat64 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let b2 () = mat64 3 2 [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5 |]

let a3 () =
  Nx.create f64 [| 2; 2; 3 |]
    [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let b3t () =
  Nx.create f64 [| 2; 3; 2 |]
    [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.9; -0.2; 0.7; 1.4; -0.3; 0.5 |]

let matmul_tests =
  [
    test "2d x 2d" (fun () ->
        check_grad2 ~msg:"matmul 2x2d" Nx.matmul (a2 ()) (b2 ()));
    test "batched x batched" (fun () ->
        check_grad2 ~msg:"matmul 3x3d" Nx.matmul (a3 ()) (b3t ()));
    test "2d x batched" (fun () ->
        check_grad2 ~msg:"matmul 2x3d" Nx.matmul (a2 ()) (b3t ()));
    test "batched x 2d" (fun () ->
        check_grad2 ~msg:"matmul 3x2d" Nx.matmul (a3 ()) (b2 ()));
  ]

(* Linear algebra. Inputs are conditioned so the operations are smooth: cholesky
   gets a positive-definite matrix built from the input. *)

let linalg_tests =
  [
    test "cholesky" (fun () ->
        check_grad ~msg:"cholesky" ~tol:5e-3
          (fun x ->
            let xxt = Nx.matmul x (Nx.transpose x) in
            let spd = Nx.add xxt (Nx.mul_s (Nx.eye f64 2) 3.0) in
            Nx.cholesky spd)
          (mat64 2 2 [| 0.9; -0.4; 0.3; 1.2 |]));
    test "qr (reduced)" (fun () ->
        (* The rule requires a square R, i.e. reduced mode. *)
        check_grad ~msg:"qr" ~tol:5e-3
          (fun x ->
            let q, r = Nx.qr ~mode:`Reduced x in
            Nx.add (Nx.sum q) (Nx.sum r))
          (mat64 3 2 [| 1.3; 0.4; -0.6; 1.8; 0.2; -1.1 |]));
  ]

(* Composite functions: several rules interacting in one graph. *)

let composite_tests =
  [
    test "softmax cross-entropy shaped loss" (fun () ->
        check_grad ~msg:"softmax"
          (fun x ->
            let e = Nx.exp x in
            Nx.log (Nx.div e (Nx.sum e ~keepdims:true)))
          (v3 ()));
    test "layer-norm shaped function" (fun () ->
        check_grad ~msg:"layernorm"
          (fun x ->
            let mu = Nx.mean x ~keepdims:true in
            let centered = Nx.sub x mu in
            let var = Nx.mean (Nx.mul centered centered) ~keepdims:true in
            Nx.div centered (Nx.sqrt (Nx.add var (Nx.scalar f64 1e-5))))
          (v3 ()));
  ]

let tests =
  [
    group "unary rules" unary_tests;
    group "binary rules" binary_tests;
    group "broadcasting" broadcast_tests;
    group "reduction rules" reduction_tests;
    group "movement rules" movement_tests;
    group "selection rules" selection_tests;
    group "scan rules" scan_tests;
    group "matmul rules" matmul_tests;
    group "linalg rules" linalg_tests;
    group "composites" composite_tests;
  ]

let () = run "rune ops" tests
