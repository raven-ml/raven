(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Batched forward-mode differentiation: jvp_k semantics over structures, the
   per-op tangent rules against the vmap∘jvp reference (one single-tangent jvp
   per lane, batched by vmap — the same mathematical object, computed the slow
   way), the degenerate single-lane case against jvp, composition with vmap,
   grad and jit, and the bounded-memory property of the ephemeron-keyed tangent
   store. *)

open Windtrap
open Rune_test_support.Support

(* Fixture tensors, matching test_jvp: distinct values inside each operation's
   smooth domain. *)

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]
let v3_pos () = vec64 [| 0.7; 1.3; 2.1 |]
let v3_unit () = vec64 [| 0.3; -0.6; 0.8 |]
let b3 () = vec64 [| 1.9; 0.8; -0.6 |]
let b3_pos () = vec64 [| 1.9; 0.8; 0.6 |]
let m23 () = mat64 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let m23_pos () = mat64 2 3 [| 0.5; 1.2; 2.1; 1.7; 0.4; 0.9 |]
let v7 () = vec64 [| 0.7; -1.3; 2.1; 0.4; -0.9; 1.6; -0.2 |]

(* A deterministic, lane-distinct tangent batch [k; shape t]: lane i carries a
   different pattern, so lane mix-ups and permutation mistakes cannot cancel
   out. Built in float64 and cast to the leaf's dtype. *)
let lane_batch ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
        let lane = i / n and j = i mod n in
        float_of_int ((((j * 7) + (3 * lane)) mod 11) - 5) /. 4.0)
  in
  Nx.cast (Nx.dtype t) (Nx.create f64 (Array.append [| k |] s) data)

(* The complex counterpart, for differentiating complex inputs directly. *)
let lane_batch_c ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
        let lane = i / n and j = i mod n in
        Complex.
          {
            re = float_of_int ((((j * 3) + lane) mod 5) - 2) /. 2.0;
            im = float_of_int (((j + (2 * lane)) mod 4) - 1) /. 3.0;
          })
  in
  Nx.create c128 (Array.append [| k |] s) data

(* The reference implementation: one single-tangent jvp per lane, batched by
   vmap. It composes the existing transformations rather than calling jvp_k, so
   agreement is evidence about the batched rules, not a tautology. *)
let reference_k (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t)
    (x : (a, b) Nx.t) (thetas : (a, b) Nx.t) : (c, d) Nx.t =
  Rune.vmap' (fun v -> snd (Rune.jvp' f x v)) thetas

(* The Pair counterpart, feeding lane batches to both arguments. *)
let reference_k2 (f : pair -> Nx.float64_t) (p : pair) (thetas : pair) :
    Nx.float64_t =
  Rune.vmap (module Pair) (fun th -> snd (Rune.jvp (module Pair) f p th)) thetas

(* [check_jvp_k ~msg f x] compares [jvp_k' f x] against the reference on a
   deterministic tangent batch, primals included. *)
let check_jvp_k ?(k = 3) ?(tol = 1e-5) ~msg (f : Nx.float64_t -> Nx.float64_t)
    (x : Nx.float64_t) =
  let thetas = lane_batch ~k x in
  let y, dy = Rune.jvp_k' f x thetas in
  check_arr ~msg:"primal" (to_arr (f x)) y;
  check_arr ~msg ~eps:tol (to_arr (reference_k f x thetas)) dy

let check_jvp_k2 ?(k = 3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let ta = lane_batch ~k a and tb = lane_batch ~k b in
  let f' p = f p.fst p.snd in
  let p = { fst = a; snd = b } in
  let thetas = { fst = ta; snd = tb } in
  let y, dy = Rune.jvp_k (module Pair) f' p thetas in
  check_arr ~msg:"primal" (to_arr (f a b)) y;
  check_arr ~msg (to_arr (reference_k2 f' p thetas)) dy

(* [check_jvp_k2_lanes ~msg f a b] is {!check_jvp_k2} with a per-lane reference:
   one single-tangent jvp per lane, stacked. Used where the vmap∘jvp reference
   cannot run — vmap passes matmul through untranslated, so a batched operand
   against a constant operand with its own leading batch dimensions misaligns
   them. *)
let check_jvp_k2_lanes ?(k = 3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let ta = lane_batch ~k a and tb = lane_batch ~k b in
  let f' p = f p.fst p.snd in
  let y, dy =
    Rune.jvp_k (module Pair) f' { fst = a; snd = b } { fst = ta; snd = tb }
  in
  check_arr ~msg:"primal" (to_arr (f a b)) y;
  let per_lane =
    Nx.stack ~axis:0
      (List.init k (fun i ->
           let va = Nx.slice [ Nx.I i ] ta and vb = Nx.slice [ Nx.I i ] tb in
           snd
             (Rune.jvp
                (module Pair)
                f' { fst = a; snd = b } { fst = va; snd = vb })))
  in
  check_arr ~msg (to_arr per_lane) dy

(* Structural semantics *)

let test_record_mixed_dtype () =
  (* One pass over a mixed-dtype record: f32 and f64 leaves batch together. *)
  let p = params () in
  let k = 4 in
  let thetas =
    {
      w = lane_batch ~k p.w;
      b = lane_batch ~k p.b;
      scale = lane_batch ~k p.scale;
    }
  in
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  let y, dy = Rune.jvp_k (module Params) f p thetas in
  check_arr ~msg:"primal" (to_arr (f p)) y;
  let dy_ref =
    Rune.vmap
      (module Params)
      (fun th -> snd (Rune.jvp (module Params) f p th))
      thetas
  in
  check_arr ~msg:"tangent" (to_arr dy_ref) dy

let test_jvp_k2_structured_output () =
  let a = v3 () and b = b3 () in
  let ta = lane_batch ~k:3 a and tb = lane_batch ~k:3 b in
  let f p = { fst = Nx.mul p.fst p.snd; snd = Nx.add p.fst p.snd } in
  let p = { fst = a; snd = b } in
  let thetas = { fst = ta; snd = tb } in
  let _, dy = Rune.jvp_k2 (module Pair) (module Pair) f p thetas in
  (* Each output leaf's tangent batch matches the component-wise reference. *)
  let d_fst =
    Rune.vmap
      (module Pair)
      (fun th ->
        snd (Rune.jvp (module Pair) (fun p -> Nx.mul p.fst p.snd) p th))
      thetas
  in
  let d_snd =
    Rune.vmap
      (module Pair)
      (fun th ->
        snd (Rune.jvp (module Pair) (fun p -> Nx.add p.fst p.snd) p th))
      thetas
  in
  check_arr ~msg:"d fst" (to_arr d_fst) dy.fst;
  check_arr ~msg:"d snd" (to_arr d_snd) dy.snd

let test_jvp_k_aux () =
  let p = params () in
  let thetas =
    {
      w = lane_batch ~k:3 p.w;
      b = lane_batch ~k:3 p.b;
      scale = lane_batch ~k:3 p.scale;
    }
  in
  let f p = (Nx.sum (Nx.mul p.w p.w), "aux") in
  let _, dy, aux = Rune.jvp_k_aux (module Params) f p thetas in
  equal ~msg:"aux" string "aux" aux;
  check_arr ~msg:"tangent"
    (to_arr
       (Rune.vmap
          (module Params)
          (fun th -> snd (Rune.jvp (module Params) (fun p -> fst (f p)) p th))
          thetas))
    dy

let test_constant_function () =
  (* The output does not depend on the input: every lane's tangent is zero. *)
  let f _ = Nx.scalar f64 42.0 in
  let x = v3 () in
  let _, dy = Rune.jvp_k' f x (lane_batch ~k:3 x) in
  check_arr ~msg:"zero tangent" (Array.make 3 0.0) dy

let test_k1_matches_jvp () =
  (* One lane is jvp up to the leading unit axis. *)
  let f x = Nx.sum (Nx.mul (Nx.sin x) (Nx.exp x)) in
  let x = v3 () in
  let _, dy_k = Rune.jvp_k' f x (lane_batch ~k:1 x) in
  let _, dy = Rune.jvp' f x (tangent_like x) in
  check_arr ~msg:"k=1 lane equals jvp" (to_arr dy)
    (Nx.reshape (Nx.shape dy) dy_k)

let test_lane_counts_agree () =
  (* The same function under several lane counts, each against its own
     reference. *)
  let f x = Nx.sum (Nx.mul x x) in
  let x = v3 () in
  List.iter
    (fun k -> check_jvp_k ~k ~msg:(Printf.sprintf "k=%d" k) f x)
    [ 1; 2; 5 ]

(* Per-op tangent rules against the vmap∘jvp reference. The case lists mirror
   test_jvp so the two rule tables stay pinned to the same fixtures. *)

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
    (fun (name, op, x) -> test name (fun () -> check_jvp_k ~msg:name op (x ())))
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
      test name (fun () -> check_jvp_k2 ~msg:name op (a ()) (b ())))
    binary_cases

(* Broadcasting is where the lane axis needs the view lift: a rank-deficient
   operand would otherwise broadcast its lane axis against the other operand's
   first dimension. *)
let broadcast_tests =
  [
    test "add broadcasts a row" (fun () ->
        check_jvp_k2 ~msg:"add [2x3]+[3]" Nx.add (m23 ()) (b3 ()));
    test "add broadcasts a column" (fun () ->
        check_jvp_k2 ~msg:"add [2x3]+[2x1]" Nx.add (m23 ())
          (mat64 2 1 [| 1.4; -0.7 |]));
    test "mul broadcasts a column" (fun () ->
        check_jvp_k2 ~msg:"mul [2x3]*[2x1]" Nx.mul (m23 ())
          (mat64 2 1 [| 1.4; -0.7 |]));
    test "mul broadcasts a scalar" (fun () ->
        check_jvp_k2 ~msg:"mul [2x3]*scalar" Nx.mul (m23 ()) (Nx.scalar f64 1.5));
    test "div broadcasts a row" (fun () ->
        check_jvp_k2 ~msg:"div [2x3]/[3]" Nx.div (m23 ()) (b3_pos ()));
    test "a vector against a matrix" (fun () ->
        check_jvp_k2 ~msg:"[3]+[2x3]" Nx.add (v3 ()) (m23 ()));
    test "tanh of a rank-lifted broadcast chain" (fun () ->
        (* mul broadcasts the row, then tanh keeps the lifted shape. *)
        check_jvp_k ~msg:"tanh([2x3]*[3])"
          (fun x -> Nx.tanh (Nx.mul x (b3 ())))
          (m23 ()));
  ]

let reduction_tests =
  [
    test "sum over one axis" (fun () ->
        check_jvp_k ~msg:"sum axis0" (Nx.sum ~axes:[ 0 ]) (m23 ()));
    test "sum keepdims" (fun () ->
        check_jvp_k ~msg:"sum keepdims"
          (Nx.sum ~axes:[ 1 ] ~keepdims:true)
          (m23 ()));
    test "sum over everything gives a lane vector" (fun () ->
        (* The total loss's tangent is the [k] vector of lane derivatives. *)
        let f = Nx.sum in
        let x = m23 () in
        let thetas = lane_batch ~k:3 x in
        let _, dy = Rune.jvp_k' f x thetas in
        check_arr ~msg:"scalar tangent lanes"
          (to_arr (reference_k f x thetas))
          dy;
        equal ~msg:"lane count" int 3 (Array.length (to_arr dy)));
    test "prod over one axis" (fun () ->
        check_jvp_k ~msg:"prod axis1" (Nx.prod ~axes:[ 1 ]) (m23_pos ()));
    test "max over one axis" (fun () ->
        check_jvp_k ~msg:"max axis0" (Nx.max ~axes:[ 0 ]) (m23 ()));
    test "min over one axis" (fun () ->
        check_jvp_k ~msg:"min axis1" (Nx.min ~axes:[ 1 ]) (m23 ()));
    test "mean over one axis" (fun () ->
        check_jvp_k ~msg:"mean axis0" (Nx.mean ~axes:[ 0 ]) (m23 ()));
  ]

let movement_tests =
  [
    test "reshape" (fun () ->
        check_jvp_k ~msg:"reshape" (Nx.reshape [| 3; 2 |]) (m23 ()));
    test "reshape to scalar" (fun () ->
        check_jvp_k ~msg:"reshape []" (Nx.reshape [||]) (mat64 1 1 [| 0.6 |]));
    test "transpose" (fun () ->
        check_jvp_k ~msg:"transpose" (Nx.transpose ~axes:[ 1; 0 ]) (m23 ()));
    test "expand" (fun () ->
        check_jvp_k ~msg:"expand"
          (Nx.expand [| 2; 3; -1 |])
          (Nx.reshape [| 1; 3 |] (v3 ())));
    test "pad" (fun () ->
        check_jvp_k ~msg:"pad" (Nx.pad [| (1, 1); (0, 2) |] 5.0) (m23 ()));
    test "shrink" (fun () ->
        check_jvp_k ~msg:"shrink" (Nx.shrink [| (0, 2); (1, 3) |]) (m23 ()));
    test "flip" (fun () ->
        check_jvp_k ~msg:"flip" (Nx.flip ~axes:[ 1 ]) (m23 ()));
    test "sliding window" (fun () ->
        check_jvp_k ~msg:"sliding window"
          (sliding_window ~axis:0 ~window:3 ~step:2)
          (v7 ()));
    test "sliding window tangent is the windowed tangent" (fun () ->
        (* Linearity per lane: lane i's tangent maps through the operation
           itself, so the tangent batch is the per-lane application stacked. *)
        let f = sliding_window ~axis:0 ~window:2 ~step:3 in
        let x = v7 () in
        let k = 2 in
        let thetas = lane_batch ~k x in
        let _, dy = Rune.jvp_k' f x thetas in
        let per_lane =
          Nx.stack ~axis:0
            (List.init k (fun i -> f (Nx.slice [ Nx.I i ] thetas)))
        in
        check_arr ~msg:"linearity" (to_arr per_lane) dy);
    test "concatenate" (fun () ->
        check_jvp_k2 ~msg:"concatenate"
          (fun a b -> Nx.concatenate ~axis:0 [ a; b ])
          (m23 ())
          (mat64 1 3 [| 0.3; 0.9; -1.1 |]));
    test "concatenate along a middle axis" (fun () ->
        check_jvp_k2 ~msg:"concatenate axis1"
          (fun a b -> Nx.concatenate ~axis:1 [ a; b ])
          (m23 ())
          (mat64 2 1 [| 0.3; 0.9 |]));
    test "slice" (fun () ->
        check_jvp_k ~msg:"slice"
          (fun x -> Nx.slice [ Nx.R (0, 2); Nx.I 1 ] x)
          (m23 ()));
  ]

let selection_tests =
  [
    test "where" (fun () ->
        let cond = Nx.greater (m23 ()) (Nx.zeros_like (m23 ())) in
        check_jvp_k2 ~msg:"where"
          (fun a b -> Nx.where cond a b)
          (m23 ())
          (mat64 2 3 [| 0.3; 0.9; -1.1; 0.2; -0.5; 1.3 |]));
    test "where with a broadcast condition" (fun () ->
        (* The condition is a row: the mask broadcasts under the lane axis. *)
        let cond = Nx.greater (b3 ()) (Nx.zeros_like (b3 ())) in
        check_jvp_k2 ~msg:"where [3] over [2x3]"
          (fun a b -> Nx.where cond a b)
          (m23 ())
          (mat64 2 3 [| 0.3; 0.9; -1.1; 0.2; -0.5; 1.3 |]));
    test "take_along_axis" (fun () ->
        let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 2l |] in
        check_jvp_k ~msg:"take_along_axis"
          (fun x -> Nx.take_along_axis ~axis:1 ~indices:idx x)
          (m23 ()));
    test "sort" (fun () ->
        check_jvp_k ~msg:"sort" (fun x -> fst (Nx.sort ~axis:1 x)) (m23 ()));
  ]

let scan_tests =
  [
    test "cumsum" (fun () ->
        check_jvp_k ~msg:"cumsum" (Nx.cumsum ~axis:1) (m23 ()));
    test "cumprod" (fun () ->
        check_jvp_k ~msg:"cumprod" (Nx.cumprod ~axis:1) (m23_pos ()));
    test "cummax" (fun () ->
        check_jvp_k ~msg:"cummax" (Nx.cummax ~axis:1) (m23 ()));
    test "cummin" (fun () ->
        check_jvp_k ~msg:"cummin" (Nx.cummin ~axis:1) (m23 ()));
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
        check_jvp_k2 ~msg:"matmul 2x2d" Nx.matmul (a2 ()) (b2 ()));
    test "batched x batched" (fun () ->
        check_jvp_k2 ~msg:"matmul 3x3d" Nx.matmul (a3 ()) (b3t ()));
    test "2d x batched" (fun () ->
        check_jvp_k2_lanes ~msg:"matmul 2x3d" Nx.matmul (a2 ()) (b3t ()));
    test "batched x 2d" (fun () ->
        check_jvp_k2_lanes ~msg:"matmul 3x2d" Nx.matmul (a3 ()) (b2 ()));
    test "vector x matrix" (fun () ->
        check_jvp_k2 ~msg:"matmul [3]x[3;2]" Nx.matmul (v3 ()) (b2 ()));
    test "matrix x vector" (fun () ->
        check_jvp_k2 ~msg:"matmul [2;3]x[3]" Nx.matmul (a2 ()) (v3 ()));
  ]

let complex_tests =
  [
    (* Assembling a complex tensor and reading a component back keeps the input
       and the output real, so the reference covers the whole complex path. *)
    test "magnitude of an assembled complex tensor" (fun () ->
        let f x =
          Nx.magnitude f64 (Nx.complex Nx.complex128 ~re:x ~im:(Nx.mul_s x 2.0))
        in
        let x = v3_pos () in
        let thetas = lane_batch ~k:3 x in
        let _, dy = Rune.jvp_k' f x thetas in
        check_arr ~msg:"magnitude" (to_arr (reference_k f x thetas)) dy);
    test "a complex-to-complex function" (fun () ->
        (* The input itself is complex, so the tangent batches are too. *)
        let f (x : Nx.complex128_t) = Nx.mul (Nx.exp x) x in
        let x = cvec [| (0.3, 0.2); (-1.1, 0.4); (0.7, -0.5) |] in
        let thetas = lane_batch_c ~k:2 x in
        let _, dy = Rune.jvp_k' f x thetas in
        let dy_ref = Rune.vmap' (fun v -> snd (Rune.jvp' f x v)) thetas in
        check_carr ~msg:"complex exp mul" (to_carr dy_ref) dy);
  ]

(* Composition *)

let test_batched_hvp () =
  (* jvp_k of grad is k Hessian-vector products in one pass: f(x) = sum(x⁴), H =
     diag(12x²), one lane per direction. *)
  let f x = Nx.sum (Nx.mul (Nx.mul x x) (Nx.mul x x)) in
  let x = vec64 [| 1.0; -2.0; 3.0 |] in
  let k = 4 in
  let thetas = lane_batch ~k x in
  let _, hv = Rune.jvp_k' (Rune.grad' f) x thetas in
  let per_lane =
    Nx.stack ~axis:0
      (List.init k (fun i ->
           let v = Nx.slice [ Nx.I i ] thetas in
           Rune.hvp' f x v))
  in
  check_arr ~msg:"batched hvp" (to_arr per_lane) hv

let test_grad_of_jvp_k () =
  (* Reverse over forward: the gradient of the summed lanes' tangent, against
     the same construction over the reference composition. *)
  let f x = Nx.sum (Nx.mul (Nx.sin x) x) in
  let thetas = lane_batch ~k:3 (vec64 [| 0.5; 0.25; -0.75 |]) in
  let g x = Nx.sum (snd (Rune.jvp_k' f x thetas)) in
  let g_ref x = Nx.sum (Rune.vmap' (fun v -> snd (Rune.jvp' f x v)) thetas) in
  let x = vec64 [| 1.0; -2.0; 3.0 |] in
  check_arr ~msg:"grad of jvp_k" (to_arr (Rune.grad' g_ref x)) (Rune.grad' g x)

let test_vmap_inside () =
  (* The natural per-trial style: vmap batches trials inside the batched forward
     scope, so the trial axis lands inside the tangent axis. *)
  let w = mat64 3 2 [| 0.4; -0.2; 0.9; 0.3; -0.5; 0.1 |] in
  let f x = Nx.sum (Rune.vmap' (fun v -> Nx.tanh (Nx.matmul v w)) x) in
  let x =
    mat64 5 3
      [|
        0.3;
        -1.1;
        0.7;
        0.9;
        0.2;
        -0.4;
        1.3;
        -0.8;
        0.5;
        0.1;
        1.0;
        -0.3;
        0.6;
        0.4;
        0.2;
      |]
  in
  check_jvp_k ~msg:"vmap of a per-trial matmul" f x

let test_vmap_inside_with_mean () =
  (* A mapped loss reduced to a scalar inside the same scope. *)
  let f x = Nx.mean (Rune.vmap' (fun v -> Nx.sum (Nx.mul v v)) x) in
  let x =
    mat64 4 3
      [| 0.3; -1.1; 0.7; 0.9; 0.2; -0.4; 1.3; -0.8; 0.5; 0.1; 1.0; -0.3 |]
  in
  check_jvp_k ~msg:"vmap mean" f x

(* A recurrence written with scan: the eager fold runs under the handler step by
   step, so the tangent batches track through time. *)
module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let test_scan_inside () =
  let gain = vec64 [| 0.8; -0.3 |] in
  let f x =
    let xs =
      Nx.reshape [| 6; 2 |]
        (Nx.create f64 [| 12 |]
           [| 0.2; -0.4; 0.6; 0.1; -0.7; 0.5; 0.3; 0.9; -0.2; 0.8; 0.4; -0.1 |])
    in
    let c, ys =
      Rune.scan
        (module Single)
        ~f:(fun c x ->
          let c' = Nx.tanh (Nx.add (Nx.mul c gain) x) in
          (c', c'))
        ~init:(Nx.reshape [| 2 |] x) xs
    in
    Nx.add (Nx.sum (Nx.mul ys ys)) (Nx.sum (Nx.mul c c))
  in
  let x = vec64 [| 0.5; -1.0 |] in
  check_jvp_k ~msg:"scan recurrence" f x

let test_jit_inside_degrades () =
  (* jit steps aside under a transformation and runs eagerly; results are
     unchanged. *)
  let f x = Nx.sum (Rune.jit' (fun z -> Nx.mul (Nx.sin z) z) x) in
  let x = v3 () in
  check_jvp_k ~msg:"jit inside jvp_k" f x

let test_jit_outside_stages () =
  (* jit outermost: the batched forward-mode handler's operations are traced
     into the compiled program, as the reverse engine's are, so a compiled jvp_k
     agrees with eager execution (jit fuses kernels, so up to floating point)
     and replays repeatably. Note that a scan inside the forward-mode scope
     unrolls into the trace: jit stages a scan as a loop only when no forward
     handler claims it. *)
  let w = mat64 2 2 [| 0.6; -0.3; 0.4; 0.7 |] in
  let xs =
    Nx.reshape [| 6; 1 |]
      (Nx.create f64 [| 6 |] [| 0.2; -0.4; 0.6; 0.1; -0.7; 0.5 |])
  in
  let f x =
    let c, ys =
      Rune.scan
        (module Single)
        ~f:(fun c x ->
          let c' = Nx.tanh (Nx.add (Nx.matmul c w) (Nx.mul_s x 0.1)) in
          (c', c'))
        ~init:(Nx.reshape [| 2 |] x) xs
    in
    Nx.add (Nx.sum (Nx.mul ys ys)) (Nx.sum (Nx.mul c c))
  in
  let x = vec64 [| 0.5; -1.0 |] in
  let thetas = lane_batch ~k:2 x in
  let eager_y, eager_dy = Rune.jvp_k' f x thetas in
  let jitted =
    Rune.jit2
      (module Single)
      (module Pair)
      (fun x ->
        let y, dy = Rune.jvp_k' f x thetas in
        { fst = y; snd = dy })
  in
  let out = jitted x in
  check_arr ~msg:"jit primal" (to_arr eager_y) out.fst;
  check_arr ~msg:"jit tangent" (to_arr eager_dy) out.snd;
  (* Replay on another input: no re-tracing, same computation. *)
  let x2 = vec64 [| -0.25; 0.75 |] in
  let eager2_y, eager2_dy = Rune.jvp_k' f x2 thetas in
  let out2 = jitted x2 in
  check_arr ~msg:"replay primal" (to_arr eager2_y) out2.fst;
  check_arr ~msg:"replay tangent" (to_arr eager2_dy) out2.snd

let test_no_grad_stops_lanes () =
  let x = vec64 [| 3.0 |] in
  let f x =
    let c = Rune.no_grad (fun () -> Nx.mul x x) in
    Nx.mul x c
  in
  (* c is constant 9, so every lane's derivative is 9 * its direction: the [k]
     tangent of the scalar output. *)
  let k = 2 in
  let thetas = lane_batch ~k x in
  let _, dy = Rune.jvp_k' f x thetas in
  let expected =
    Array.map (fun v -> 9.0 *. v) (to_arr (Nx.reshape [| k |] thetas))
  in
  check_arr ~msg:"gated lanes" expected dy

let test_detach_stops_lanes () =
  let x = vec64 [| 3.0 |] in
  let f x = Nx.mul x (Rune.detach x) in
  let k = 2 in
  let thetas = lane_batch ~k x in
  let _, dy = Rune.jvp_k' f x thetas in
  (* detach x is a constant 3, so every lane's derivative is 3 * its
     direction. *)
  let expected =
    Array.map (fun v -> 3.0 *. v) (to_arr (Nx.reshape [| k |] thetas))
  in
  check_arr ~msg:"detached lanes" expected dy

(* Custom rules: a rule is written for a single tangent, so the handler
   vectorizes it over the lane axis. *)

let my_sin_fwd x =
  Rune.custom_jvp
    (module Single)
    ~f:Nx.sin
    ~jvp:(fun x dx -> (Nx.sin x, Nx.mul dx (Nx.cos x)))
    x

let test_custom_jvp_rule_matches_autodiff () =
  let x = v3 () in
  let thetas = lane_batch ~k:3 x in
  let y, dy = Rune.jvp_k' my_sin_fwd x thetas in
  check_arr ~msg:"value" (to_arr (Nx.sin x)) y;
  check_arr ~msg:"tangent" (to_arr (reference_k my_sin_fwd x thetas)) dy

let test_custom_jvp_rule_is_used () =
  (* The rule replaces autodiff: every lane reports 100 * its direction. *)
  let x = v3 () in
  let thetas = lane_batch ~k:3 x in
  let fake x =
    Rune.custom_jvp
      (module Single)
      ~f:Nx.sin
      ~jvp:(fun x dx -> (Nx.sin x, Nx.mul_s dx 100.0))
      x
  in
  let _, dy = Rune.jvp_k' fake x thetas in
  check_arr ~msg:"fake rule" (to_arr (Nx.mul_s thetas 100.0)) dy

let test_custom_vjp_raises () =
  let x = v3 () in
  let f x =
    Rune.custom_vjp
      (module Single)
      ~fwd:(fun x -> (Nx.sin x, x))
      ~bwd:(fun x ct -> Nx.mul ct (Nx.cos x))
      x
  in
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rune.jvp_k' f x (lane_batch ~k:2 x)))

(* Excluded operations and mutation *)

let test_svd_raises_when_active () =
  let x = Nx.create f64 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k'
           (fun x ->
             let _, s, _ = Nx.svd x in
             s)
           x (lane_batch ~k:2 x)))

let test_cholesky_raises_when_active () =
  (* jvp has a rule for cholesky; the batched handler does not yet, because its
     backend solve wants exactly matching leading batch dimensions. *)
  let x = mat64 2 2 [| 0.9; -0.4; 0.3; 1.2 |] in
  let f x =
    Nx.cholesky
      (Nx.add (Nx.matmul x (Nx.transpose x)) (Nx.mul_s (Nx.eye f64 2) 3.0))
  in
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rune.jvp_k' f x (lane_batch ~k:2 x)))

let test_mutation_raises () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k'
           (fun x ->
             Nx.set_item [ 0 ] 1.0 x;
             Nx.sum x)
           (vec64 [| 1.0; 2.0 |])
           (lane_batch ~k:2 (vec64 [| 1.0; 2.0 |]))))

(* Lane-structure errors *)

let test_rejects_lane_shape_mismatch () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k'
           (fun x -> Nx.sum x)
           (vec64 [| 1.0; 2.0 |])
           (vec64 [| 1.0; 2.0 |])))

let test_rejects_disagreeing_lane_counts () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 3.0 |] }
           {
             fst = lane_batch ~k:3 (vec64 [| 1.0; 2.0 |]);
             snd = lane_batch ~k:2 (vec64 [| 3.0 |]);
           }))

let test_rejects_zero_lanes () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k'
           (fun x -> Nx.sum x)
           (vec64 [| 1.0; 2.0 |])
           (Nx.create f64 [| 0; 2 |] [| 0.0; 0.0 |])))

let test_rejects_scalar_tangent () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.jvp_k'
           (fun x -> Nx.sum x)
           (vec64 [| 1.0; 2.0 |])
           (Nx.scalar f64 1.0)))

(* An enclosing vmap whose lanes the tangents depend on batches around the
   tangent axis and must fail with the lane-invariant error rather than compute
   wrong shapes. *)
let test_vmap_outside_raises () =
  let p = { fst = v3 (); snd = b3 () } in
  let f p = Nx.sum (Nx.mul p.fst p.snd) in
  (* Per-data-lane tangent structures, stacked for vmap: the tangent leaves
     carry a data axis outside the lane axis. *)
  let data_thetas =
    List.init 2 (fun m ->
        {
          fst =
            Nx.create f64 [| 2; 3 |]
              (Array.init 6 (fun i -> float_of_int ((i + m) mod 5) /. 3.0));
          snd =
            Nx.create f64 [| 2; 3 |]
              (Array.init 6 (fun i -> float_of_int ((i + m + 2) mod 4) /. 2.0));
        })
  in
  let stacked =
    {
      fst = Nx.stack ~axis:0 (List.map (fun t -> t.fst) data_thetas);
      snd = Nx.stack ~axis:0 (List.map (fun t -> t.snd) data_thetas);
    }
  in
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.vmap
           (module Pair)
           (fun th ->
             let _, dy = Rune.jvp_k (module Pair) f p th in
             Nx.sum dy)
           stacked))

(* An exception raised in an enclosing effect handler is re-raised in the
   handler's fiber, abandoning the transformation's fiber without unwinding it,
   so a transformation's installation cannot be tracked in a global counter with
   [Fun.protect]: the cleanup is skipped and every later [jit] steps aside for
   the rest of the process. The gate is asked as an effect instead, and these
   are the cases that used to leak. *)

type _ Effect.t += Boom : unit Effect.t

let raising_handler : type r. (r, r) Effect.Deep.handler =
  {
    retc = Fun.id;
    exnc = raise;
    effc =
      (fun (type c) (eff : c Effect.t) ->
        match eff with Boom -> Some (fun _k -> failwith "boom") | _ -> None);
  }

(* [jit_is_active ()] is [true] when a [jit] trace still refuses to read a
   traced value — that is, when no transformation is (wrongly) claiming to be
   installed. *)
let jit_is_active () =
  let f = Rune.jit' (fun x -> Nx.scalar f64 (Nx.item [ 0 ] x)) in
  match f (vec64 [| 1.0; 2.0 |]) with
  | _ -> false
  | exception Rune.Jit_error _ -> true

let test_raising_handler_keeps_jit_active () =
  equal ~msg:"jit is active to begin with" bool true (jit_is_active ());
  let raises f =
    match f () with
    | _ -> fail "the enclosing handler should have raised"
    | exception Failure _ -> ()
  in
  (* reverse mode *)
  raises (fun () ->
      Effect.Deep.match_with
        (fun () ->
          Rune.grad'
            (fun x ->
              Effect.perform Boom;
              Nx.sum x)
            (vec64 [| 1.0; 2.0 |]))
        () raising_handler);
  equal ~msg:"grad did not close the gate" bool true (jit_is_active ());
  (* batched forward mode with an inner vmap: the abandoned fiber is the
     map's *)
  raises (fun () ->
      Effect.Deep.match_with
        (fun () ->
          Rune.jvp_k'
            (fun x ->
              Nx.sum
                (Rune.vmap'
                   (fun y ->
                     Effect.perform Boom;
                     Nx.sum y)
                   x))
            (vec64 [| 1.0; 2.0 |])
            (lane_batch ~k:1 (vec64 [| 1.0; 2.0 |])))
        () raising_handler);
  equal ~msg:"jvp_k of vmap did not close the gate" bool true (jit_is_active ())

(* Memory: the ephemeron-keyed store *)

let test_live_tangents_bounded_across_loop () =
  (* A recurrence stepped in plain OCaml: each step's intermediates die with the
     step, so the store must not retain them. A strong-keyed store would hold
     one binding per intermediate — two per step here, plus the loop's carries —
     for hundreds of entries at this horizon. *)
  let w =
    mat64 4 4
      [|
        0.3;
        -0.1;
        0.2;
        0.05;
        -0.15;
        0.4;
        0.1;
        -0.2;
        0.25;
        0.05;
        -0.3;
        0.15;
        0.1;
        0.2;
        -0.05;
        0.35;
      |]
  in
  let x0 = vec64 [| 0.5; -0.2; 0.8; 0.1 |] in
  let step c = Nx.tanh (Nx.matmul c w) in
  let horizon = 400 in
  let probe = ref 0 in
  let f x =
    let rec go i c = if i = horizon then c else go (i + 1) (step c) in
    let c = go 0 x in
    (* Force a major collection so the measurement does not depend on collector
       timing. *)
    Gc.compact ();
    Gc.compact ();
    probe := Rune.live_tangent_entries ();
    Nx.sum c
  in
  let _, dy = Rune.jvp_k' f x0 (lane_batch ~k:3 x0) in
  if !probe > 40 then
    fail
      (Printf.sprintf
         "the tangent store holds %d live bindings after %d steps; the \
          ephemerons should have collected the dead steps' tangents"
         !probe horizon);
  (* The measurement must also have produced a real tangent: the final sum's
     tangent is the [k] vector of directional derivatives of the loop's
     endpoint. *)
  equal ~msg:"endpoint lanes" int 3 (Array.length (to_arr dy))

let test_live_tangent_entries_outside_scope () =
  equal ~msg:"no store outside a jvp_k" int 0 (Rune.live_tangent_entries ())

let tests =
  [
    group "jvp_k over structures"
      [
        test "mixed dtypes propagate in one pass" test_record_mixed_dtype;
        test "jvp_k2 gives per-leaf output tangent batches"
          test_jvp_k2_structured_output;
        test "jvp_k_aux returns auxiliary data" test_jvp_k_aux;
        test "constant function has zero tangent batches" test_constant_function;
        test "k=1 matches jvp" test_k1_matches_jvp;
        test "several lane counts agree with the reference"
          test_lane_counts_agree;
      ];
    group "unary rules" unary_tests;
    group "binary rules" binary_tests;
    group "broadcasting" broadcast_tests;
    group "reduction rules" reduction_tests;
    group "movement rules" movement_tests;
    group "selection rules" selection_tests;
    group "scan rules" scan_tests;
    group "matmul rules" matmul_tests;
    group "complex dtypes" complex_tests;
    group "composition"
      [
        test "batched Hessian-vector products in one pass" test_batched_hvp;
        test "grad of jvp_k (reverse over forward)" test_grad_of_jvp_k;
        test "vmap inside: trials inside the tangent axis" test_vmap_inside;
        test "vmap inside with a mean reduction" test_vmap_inside_with_mean;
        test "scan inside: recurrence over time" test_scan_inside;
        test "jit inside degrades to eager execution" test_jit_inside_degrades;
        test "jit outside stages the batched tangents" test_jit_outside_stages;
        test "no_grad stops lanes" test_no_grad_stops_lanes;
        test "detach stops lanes" test_detach_stops_lanes;
      ];
    group "custom rules and excluded operations"
      [
        test "custom_jvp rule matches autodiff"
          test_custom_jvp_rule_matches_autodiff;
        test "custom_jvp rule replaces autodiff" test_custom_jvp_rule_is_used;
        test "custom_vjp raises" test_custom_vjp_raises;
        test "svd raises when its input is active" test_svd_raises_when_active;
        test "cholesky raises when its input is active"
          test_cholesky_raises_when_active;
        test "in-place mutation raises" test_mutation_raises;
      ];
    group "lane structure errors"
      [
        test "rejects a tangent without a lane axis"
          test_rejects_lane_shape_mismatch;
        test "rejects disagreeing lane counts"
          test_rejects_disagreeing_lane_counts;
        test "rejects zero lanes" test_rejects_zero_lanes;
        test "rejects a scalar tangent" test_rejects_scalar_tangent;
        test "vmap outside raises the lane-invariant error"
          test_vmap_outside_raises;
      ];
    group "the transformation gate"
      [
        test "a raising handler keeps jit active"
          test_raising_handler_keeps_jit_active;
      ];
    group "memory"
      [
        test "live tangents stay bounded across a loop"
          test_live_tangents_bounded_across_loop;
        test "live count is zero outside a jvp_k"
          test_live_tangent_entries_outside_scope;
      ];
  ]

let () = run "rune jvp_k" tests
