(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward-mode differentiation: jvp semantics over structures, per-op tangent
   rules against the directional finite-difference oracle, and composition with
   reverse mode. *)

open Windtrap
open Rune_test_support.Support

(* Fixture tensors, matching test_ops: distinct values inside each operation's
   smooth domain. *)

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]
let v3_pos () = vec64 [| 0.7; 1.3; 2.1 |]
let v3_unit () = vec64 [| 0.3; -0.6; 0.8 |]
let b3 () = vec64 [| 1.9; 0.8; -0.6 |]
let b3_pos () = vec64 [| 1.9; 0.8; 0.6 |]
let m23 () = mat64 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let m23_pos () = mat64 2 3 [| 0.5; 1.2; 2.1; 1.7; 0.4; 0.9 |]

(* Structural semantics *)

let test_jvp_record_analytic () =
  (* f p = sum (w * w) with tangent v: df = 2 <w, v>. *)
  let p = params () in
  let t =
    {
      w = vec32 [| 1.0; 1.0; 1.0 |];
      b = vec32 [| 0.0 |];
      scale = vec64 [| 0.0 |];
    }
  in
  let f p = Nx.sum (Nx.mul p.w p.w) in
  let _, dy = Rune.jvp (module Params) f p t in
  (* 2 * (1 - 2 + 3) = 4 *)
  check_arr ~msg:"df" [| 4.0 |] dy

let test_jvp_mixed_dtype () =
  let p = params () in
  let t =
    {
      w = vec32 [| 1.0; 0.0; 0.0 |];
      b = vec32 [| 0.0 |];
      scale = vec64 [| 1.0 |];
    }
  in
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  (* df = 2*w0*t_w0 + 2*scale*t_scale = 2*1 + 2*2 = 6 *)
  let _, dy = Rune.jvp (module Params) f p t in
  check_arr ~msg:"df" [| 6.0 |] dy

let test_jvp_constant_function () =
  (* The output does not depend on the input: the tangent is zero. *)
  let f _ = Nx.scalar f64 42.0 in
  let _, dy = Rune.jvp' f (v3 ()) (tangent_like (v3 ())) in
  check_arr ~msg:"df" [| 0.0 |] dy

let test_jvp_aux () =
  let p = params () in
  let t =
    {
      w = vec32 [| 1.0; 1.0; 1.0 |];
      b = vec32 [| 0.0 |];
      scale = vec64 [| 0.0 |];
    }
  in
  let f p = (Nx.sum (Nx.mul p.w p.w), "aux") in
  let y, dy, aux = Rune.jvp_aux (module Params) f p t in
  (* value = 1 + 4 + 9 = 14; df = 2 * (1 - 2 + 3) = 4 *)
  check_arr ~msg:"value" [| 14.0 |] y;
  check_arr ~msg:"df" [| 4.0 |] dy;
  equal ~msg:"aux" string "aux" aux

let test_jvp_tangent_shape_mismatch () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.jvp' (fun x -> Nx.sum x) (vec64 [| 1.0; 2.0 |]) (vec64 [| 1.0 |])))

let test_jvp_matches_grad () =
  (* For a scalar objective, the jvp along v equals <grad, v>. *)
  let f x = Nx.sum (Nx.mul (Nx.sin x) (Nx.exp x)) in
  let x = v3 () in
  let v = tangent_like x in
  let _, dy = Rune.jvp' f x v in
  let g = Rune.grad' f x in
  check_arr ~msg:"jvp = <grad, v>"
    (to_arr (Nx.sum (Nx.mul g v)))
    (Nx.reshape [| 1 |] dy)

let test_jvp2_structured_output () =
  (* Each output leaf's tangent matches the component-wise jvp. *)
  let a = v3 () and b = b3 () in
  let va = tangent_like (v3 ()) and vb = tangent_like (b3 ()) in
  let f p = { fst = Nx.mul p.fst p.snd; snd = Nx.add p.fst p.snd } in
  let _, dy =
    Rune.jvp2
      (module Pair)
      (module Pair)
      f { fst = a; snd = b } { fst = va; snd = vb }
  in
  let _, d_fst =
    Rune.jvp
      (module Pair)
      (fun p -> Nx.mul p.fst p.snd)
      { fst = a; snd = b } { fst = va; snd = vb }
  in
  let _, d_snd =
    Rune.jvp
      (module Pair)
      (fun p -> Nx.add p.fst p.snd)
      { fst = a; snd = b } { fst = va; snd = vb }
  in
  check_arr ~msg:"d fst" (to_arr d_fst) dy.fst;
  check_arr ~msg:"d snd" (to_arr d_snd) dy.snd

(* Per-op tangent rules against the directional oracle. *)

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
    (fun (name, op, x) -> test name (fun () -> check_jvp ~msg:name op (x ())))
    unary_cases

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
      test name (fun () -> check_jvp2 ~msg:name op (a ()) (b ())))
    binary_cases

let broadcast_tests =
  [
    test "add broadcasts a row" (fun () ->
        check_jvp2 ~msg:"add [2x3]+[3]" Nx.add (m23 ()) (b3 ()));
    test "mul broadcasts a column" (fun () ->
        check_jvp2 ~msg:"mul [2x3]*[2x1]" Nx.mul (m23 ())
          (mat64 2 1 [| 1.4; -0.7 |]));
  ]

let reduction_tests =
  [
    test "sum over one axis" (fun () ->
        check_jvp ~msg:"sum axis0" (Nx.sum ~axes:[ 0 ]) (m23 ()));
    test "sum keepdims" (fun () ->
        check_jvp ~msg:"sum keepdims"
          (Nx.sum ~axes:[ 1 ] ~keepdims:true)
          (m23 ()));
    test "prod over one axis" (fun () ->
        check_jvp ~msg:"prod axis1" (Nx.prod ~axes:[ 1 ]) (m23_pos ()));
    test "max over one axis" (fun () ->
        check_jvp ~msg:"max axis0" (Nx.max ~axes:[ 0 ]) (m23 ()));
    test "min over one axis" (fun () ->
        check_jvp ~msg:"min axis1" (Nx.min ~axes:[ 1 ]) (m23 ()));
    test "mean over one axis" (fun () ->
        check_jvp ~msg:"mean axis0" (Nx.mean ~axes:[ 0 ]) (m23 ()));
  ]

let movement_tests =
  [
    test "reshape" (fun () ->
        check_jvp ~msg:"reshape" (Nx.reshape [| 3; 2 |]) (m23 ()));
    test "transpose" (fun () ->
        check_jvp ~msg:"transpose" (fun x -> Nx.transpose x) (m23 ()));
    test "pad" (fun () ->
        check_jvp ~msg:"pad" (Nx.pad [| (1, 1); (0, 2) |] 5.0) (m23 ()));
    test "shrink" (fun () ->
        check_jvp ~msg:"shrink" (Nx.shrink [| (0, 2); (1, 3) |]) (m23 ()));
    test "flip" (fun () -> check_jvp ~msg:"flip" (Nx.flip ~axes:[ 1 ]) (m23 ()));
    test "concatenate" (fun () ->
        check_jvp2 ~msg:"concatenate"
          (fun a b -> Nx.concatenate ~axis:0 [ a; b ])
          (m23 ())
          (mat64 1 3 [| 0.3; 0.9; -1.1 |]));
    test "slice" (fun () ->
        check_jvp ~msg:"slice"
          (fun x -> Nx.slice [ Nx.R (0, 2); Nx.I 1 ] x)
          (m23 ()));
  ]

let selection_tests =
  [
    test "where" (fun () ->
        let cond = Nx.greater (m23 ()) (Nx.zeros_like (m23 ())) in
        check_jvp2 ~msg:"where"
          (fun a b -> Nx.where cond a b)
          (m23 ())
          (mat64 2 3 [| 0.3; 0.9; -1.1; 0.2; -0.5; 1.3 |]));
    test "take_along_axis" (fun () ->
        let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 2l |] in
        check_jvp ~msg:"take_along_axis"
          (fun x -> Nx.take_along_axis ~axis:1 idx x)
          (m23 ()));
    test "sort" (fun () ->
        check_jvp ~msg:"sort" (fun x -> fst (Nx.sort ~axis:1 x)) (m23 ()));
  ]

let scan_tests =
  [
    test "cumsum" (fun () ->
        check_jvp ~msg:"cumsum" (Nx.cumsum ~axis:1) (m23 ()));
    test "cumprod" (fun () ->
        check_jvp ~msg:"cumprod" (Nx.cumprod ~axis:1) (m23_pos ()));
  ]

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
        check_jvp2 ~msg:"matmul 2x2d" Nx.matmul (a2 ()) (b2 ()));
    test "batched x batched" (fun () ->
        check_jvp2 ~msg:"matmul 3x3d" Nx.matmul (a3 ()) (b3t ()));
    test "2d x batched" (fun () ->
        check_jvp2 ~msg:"matmul 2x3d" Nx.matmul (a2 ()) (b3t ()));
    test "batched x 2d" (fun () ->
        check_jvp2 ~msg:"matmul 3x2d" Nx.matmul (a3 ()) (b2 ()));
  ]

let linalg_tests =
  [
    test "cholesky" (fun () ->
        check_jvp ~msg:"cholesky" ~tol:5e-3
          (fun x ->
            let xxt = Nx.matmul x (Nx.transpose x) in
            let spd = Nx.add xxt (Nx.mul_s (Nx.eye f64 2) 3.0) in
            Nx.cholesky spd)
          (mat64 2 2 [| 0.9; -0.4; 0.3; 1.2 |]));
  ]

let composite_tests =
  [
    test "softmax cross-entropy shaped function" (fun () ->
        check_jvp ~msg:"softmax"
          (fun x ->
            let e = Nx.exp x in
            Nx.log (Nx.div e (Nx.sum e ~keepdims:true)))
          (v3 ()));
  ]

(* Composition with reverse mode *)

let test_hessian_vector_product () =
  (* f(x) = sum(x³): H = diag(6x), so hvp(x, v) = 6 x v. Forward over reverse:
     jvp of grad. *)
  let f x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let x = vec64 [| 1.0; -2.0; 3.0 |] in
  let v = vec64 [| 1.0; 0.5; -1.0 |] in
  let _, hv = Rune.jvp' (Rune.grad' f) x v in
  check_arr ~msg:"hvp" [| 6.0; -6.0; -18.0 |] hv

let test_grad_of_jvp () =
  (* Reverse over forward: d/dx of jvp(f, x, v) for f = sum(x²), v fixed: jvp =
     2<x, v>, gradient is 2v. *)
  let v = vec64 [| 1.0; 0.5; -1.0 |] in
  let f x = Nx.sum (Nx.mul x x) in
  let outer x = snd (Rune.jvp' f x v) in
  let g = Rune.grad' outer (vec64 [| 1.0; -2.0; 3.0 |]) in
  check_arr ~msg:"d jvp" [| 2.0; 1.0; -2.0 |] g

let test_nested_jvp () =
  (* Second directional derivative via nested jvp: f = sum(x³), direction v:
     d²f[v,v] = 6 <x, v²> with elementwise square. *)
  let f x = Nx.sum (Nx.mul x (Nx.mul x x)) in
  let v = vec64 [| 1.0; 0.5; -1.0 |] in
  let inner x = snd (Rune.jvp' f x v) in
  let _, ddy = Rune.jvp' inner (vec64 [| 1.0; -2.0; 3.0 |]) v in
  (* 6 * (1*1 + (-2)*0.25 + 3*1) = 6 * 3.5 = 21 *)
  check_arr ~msg:"d2f" [| 21.0 |] ddy

(* Gates and error contracts *)

let test_no_grad_stops_tangents () =
  let x = vec64 [| 3.0 |] in
  let f x =
    let c = Rune.no_grad (fun () -> Nx.mul x x) in
    Nx.mul x c
  in
  (* c is constant 9, so df = 9 * v. *)
  let _, dy = Rune.jvp' f x (vec64 [| 1.0 |]) in
  check_arr ~msg:"df" [| 9.0 |] dy

let test_detach_stops_tangents () =
  let x = vec64 [| 3.0 |] in
  let f x = Nx.mul x (Rune.detach x) in
  (* detach x is a constant 3, so df = 3 * v, not 2 x v. *)
  let _, dy = Rune.jvp' f x (vec64 [| 1.0 |]) in
  check_arr ~msg:"df" [| 3.0 |] dy

let test_jvp_structural_shape_mismatch () =
  (* The per-leaf shape check also guards the structural entry point. *)
  raises_invalid_arg (fun () ->
      ignore
        (Rune.jvp
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 3.0; 4.0 |] }
           { fst = vec64 [| 1.0 |]; snd = vec64 [| 0.0; 0.0 |] }))

let test_unsupported_op_raises_when_active () =
  let x = Nx.create f64 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  raises_invalid_arg (fun () ->
      ignore
        (Rune.jvp'
           (fun x ->
             let _, s, _ = Nx.svd x in
             s)
           x (tangent_like x)))

let test_mutation_raises () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.jvp'
           (fun x ->
             Nx.set_item [ 0 ] 1.0 x;
             Nx.sum x)
           (vec64 [| 1.0; 2.0 |])
           (vec64 [| 1.0; 0.0 |])))

let tests =
  [
    group "jvp over records"
      [
        test "matches the analytic tangent" test_jvp_record_analytic;
        test "mixed dtypes propagate in one pass" test_jvp_mixed_dtype;
        test "constant function has zero tangent" test_jvp_constant_function;
        test "jvp_aux returns auxiliary data" test_jvp_aux;
        test "rejects tangent shape mismatch" test_jvp_tangent_shape_mismatch;
        test "agrees with grad on scalar objectives" test_jvp_matches_grad;
        test "jvp2 gives per-leaf output tangents" test_jvp2_structured_output;
      ];
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
    group "composition"
      [
        test "hessian-vector product (forward over reverse)"
          test_hessian_vector_product;
        test "grad of jvp (reverse over forward)" test_grad_of_jvp;
        test "nested jvp" test_nested_jvp;
      ];
    group "gates and errors"
      [
        test "no_grad stops tangents" test_no_grad_stops_tangents;
        test "detach stops tangents" test_detach_stops_tangents;
        test "rejects a leaf tangent shape mismatch"
          test_jvp_structural_shape_mismatch;
        test "unsupported op raises when input is active"
          test_unsupported_op_raises_when_active;
        test "in-place mutation raises" test_mutation_raises;
      ];
  ]

let () = run "rune jvp" tests
