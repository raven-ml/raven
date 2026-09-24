(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Just-in-time compilation: trace/compile/replay correctness, signature
   retracing, composition with the other transformations, in-place state, and
   trace-time failure modes. *)

open Windtrap
open Rune_test_support.Support

(* loss p = sum (w * w) + 3 * sum b. d/dw = 2w, d/db = 3, d/dscale = 0. *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.0)

(* Basics *)

let test_elementwise_matches_eager () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_replay_reads_fresh_inputs () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  ignore (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"fresh data" [| 4.0; 9.0; 16.0 |]
    (g (vec32 [| 2.0; 3.0; 4.0 |]))

let test_retrace_on_new_shape () =
  let g = Rune.jit' (fun x -> Nx.sum x) in
  check_arr ~msg:"vector" [| 6.0 |] (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"matrix" [| 10.0 |]
    (g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]))

(* An output with no elements has no buffer to schedule; every call returns an
   empty tensor of its dtype and shape. *)
let test_zero_size_outputs () =
  let check name f x =
    let g = Rune.jit' f in
    for call = 1 to 2 do
      let msg = Printf.sprintf "%s, call %d" name call in
      let y = g x in
      equal ~msg (array int) (Nx.shape (f x)) (Nx.shape y);
      let dtype t = Format.asprintf "%a" Nx.pp_dtype (Nx.dtype t) in
      equal ~msg string (dtype (f x)) (dtype y)
    done
  in
  check "int8 cumsum" (Nx.cumsum ~axis:1) (Nx.zeros Nx.int8 [| 2; 0 |]);
  check "float32 add of a cumsum"
    (fun x -> Nx.add (Nx.cumsum x) x)
    (Nx.zeros f32 [| 0 |])

let test_closure_matmul () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.matmul x w in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"matmul" (to_arr (f x)) (g x)

let test_jit2_structured_output () =
  let f p =
    { w = Nx.mul p.w p.w; b = Nx.add p.b p.b; scale = Nx.mul_s p.scale 2.0 }
  in
  let g = Rune.jit2 (module Params) (module Params) f in
  let p = params () in
  let r = g p in
  let e = f p in
  check_arr ~msg:"w" (to_arr e.w) r.w;
  check_arr ~msg:"b" (to_arr e.b) r.b;
  check_arr ~msg:"scale (float64)" (to_arr e.scale) r.scale

(* Composition *)

let test_grad_inside_jit () =
  let step =
    Rune.jit2
      (module Params)
      (module Params)
      (fun p -> Rune.grad (module Params) quadratic p)
  in
  let g = step (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"db" [| 3.0 |] g.b;
  check_arr ~msg:"dscale" [| 0.0 |] g.scale;
  (* Replay computes gradients at the new point. *)
  let p2 = { (params ()) with w = vec32 [| 4.0; 5.0; 6.0 |] } in
  let g2 = step p2 in
  check_arr ~msg:"dw at new point" [| 8.0; 10.0; 12.0 |] g2.w

let test_jit_under_grad_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  let dx = Rune.grad' (fun x -> Nx.sum (g x)) (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"d(sum x^2)" [| 2.0; 4.0; 6.0 |] dx

let test_jit_under_vmap_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y = Rune.vmap' g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]) in
  check_arr ~msg:"vmap over jit" [| 2.0; 4.0; 6.0; 8.0 |] y

(* Linear algebra

   QR and triangular solves compile through trace-time unrolling: the tracer
   writes them as ordinary Tolk compositions (one Householder reflector or
   substitution step per matrix dimension), and the lowering compiles the whole
   factorization. Every case compares against the eager C kernels, which fixes
   the LAPACK conventions: the reflector sign, and a column with a zero tail
   taking no reflector at all. *)

(* A single-tensor structure: jit2's input or output. *)
module Csingle = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) x = f x

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) x = f x
end

(* QR returns two tensors, so the jitted form needs a pair output. *)
module Pair32 = struct
  type t = Nx.float32_t * Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (q, r) = (f q, f r)

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (q1, r1)
      (q2, r2) =
    (f q1 q2, f r1 r2)

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (q, r) =
    f q;
    f r
end

let test_qr_reduced_matches_eager () =
  let a =
    Nx.create f32 [| 4; 4 |]
      [|
        2.0;
        1.0;
        1.0;
        0.5;
        1.0;
        3.0;
        2.0;
        1.0;
        1.5;
        2.0;
        4.0;
        0.25;
        0.5;
        1.0;
        0.5;
        5.0;
      |]
  in
  let jq, jr =
    Rune.jit2
      (module Csingle)
      (module Pair32)
      (fun m -> Nx.qr ~mode:`Reduced m)
      a
  in
  let q, r = Nx.qr ~mode:`Reduced a in
  check_arr ~msg:"Q" (to_arr q) jq;
  check_arr ~msg:"R" (to_arr r) jr

(* The second column's tail is zero, so it takes no reflector (tau = 0, R[1][1]
   keeps alpha); the third is full. Both paths must match eager. *)
let test_qr_zero_tail_matches_eager () =
  let a =
    Nx.create f32 [| 3; 3 |] [| 1.0; 0.0; 2.0; 0.0; 2.0; 3.0; 0.0; 0.0; 4.0 |]
  in
  let jq, jr =
    Rune.jit2
      (module Csingle)
      (module Pair32)
      (fun m -> Nx.qr ~mode:`Reduced m)
      a
  in
  let q, r = Nx.qr ~mode:`Reduced a in
  check_arr ~msg:"Q" (to_arr q) jq;
  check_arr ~msg:"R" (to_arr r) jr

let test_solve_triangular_flags_match_eager () =
  let a =
    Nx.create f32 [| 3; 3 |] [| 4.0; 1.0; 2.0; 1.0; 5.0; 3.0; 2.0; 3.0; 6.0 |]
  in
  let b = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  List.iter
    (fun (upper, transpose, unit_diag) ->
      let msg =
        Printf.sprintf "upper %b transpose %b unit_diag %b" upper transpose
          unit_diag
      in
      let x =
        Rune.jit2
          (module Pair32)
          (module Csingle)
          (fun (a, b) -> Nx.solve_triangular ~upper ~transpose ~unit_diag a b)
          (a, b)
      in
      check_arr ~msg
        (to_arr (Nx.solve_triangular ~upper ~transpose ~unit_diag a b))
        x)
    [
      (false, false, false);
      (true, false, false);
      (false, true, false);
      (true, true, false);
      (false, false, true);
      (true, false, true);
      (false, true, true);
      (true, true, true);
    ]

let test_solve_triangular_vector_rhs () =
  let a =
    Nx.create f32 [| 3; 3 |] [| 2.0; 1.0; 0.0; 1.0; 3.0; 1.0; 0.0; 1.0; 4.0 |]
  in
  let b = Nx.create f32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let x =
    Rune.jit2
      (module Pair32)
      (module Csingle)
      (fun (a, b) ->
        Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:false a b)
      (a, b)
  in
  check_arr ~msg:"vector right-hand side"
    (to_arr
       (Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:false a b))
    x

let test_solve_triangular_batched () =
  let a =
    Nx.create f32 [| 2; 3; 3 |]
      [|
        4.0;
        1.0;
        2.0;
        0.0;
        5.0;
        3.0;
        0.0;
        0.0;
        6.0;
        2.0;
        1.0;
        0.0;
        0.0;
        3.0;
        1.0;
        0.0;
        0.0;
        4.0;
      |]
  in
  let b = Nx.create f32 [| 2; 3; 1 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let x =
    Rune.jit2
      (module Pair32)
      (module Csingle)
      (fun (a, b) ->
        Nx.solve_triangular ~upper:true ~transpose:false ~unit_diag:false a b)
      (a, b)
  in
  check_arr ~msg:"batched upper"
    (to_arr
       (Nx.solve_triangular ~upper:true ~transpose:false ~unit_diag:false a b))
    x

(* Wide right-hand sides take the blocked path: rows are partitioned into 32-row
   blocks, each solved with one GEMM against the rows solved so far after its
   diagonal block has been inverted once. This size spans several blocks plus a
   partial trailing block; check the residual, which is independent of the
   solver. *)
let test_solve_triangular_blocked () =
  let n = 80 in
  let a =
    Nx.init Nx.float64 [| n; n |] (fun idx ->
        let i, j = (idx.(0), idx.(1)) in
        if i > j then Float.of_int ((((i * 37) + (j * 11)) mod 13) - 6) /. 8.0
        else if i = j then 2.0
        else 0.0)
  in
  let b =
    Nx.init Nx.float64 [| n; n |] (fun idx ->
        Float.of_int ((((idx.(0) * 5) + idx.(1)) mod 7) - 3))
  in
  let resid =
    Rune.jit'
      (fun m ->
        let x =
          Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:false m b
        in
        Nx.max (Nx.abs (Nx.sub (Nx.matmul m x) b)))
      a
  in
  check_close ~tol:1e-9 ~msg:"blocked triangular solve residual" [| 0.0 |]
    (to_arr resid);
  (* The flags compose with blocking. ~upper reads the strict upper triangle of
     a matrix whose stored diagonal is garbage (never read under ~unit_diag),
     and ~transpose solves the transposed system — so the effective system is [I
     + strict_upper(m)]ᵀ. *)
  let au =
    Nx.add
      (Nx.mul_s (Nx.triu ~k:1 (Nx.transpose ~axes:[ 1; 0 ] a)) 0.125)
      (Nx.mul_s (Nx.eye Nx.float64 n) 7.0)
  in
  let resid_flags =
    Rune.jit'
      (fun m ->
        let x =
          Nx.solve_triangular ~upper:true ~transpose:true ~unit_diag:true m b
        in
        let e =
          Nx.add (Nx.eye Nx.float64 n)
            (Nx.transpose ~axes:[ 1; 0 ] (Nx.triu ~k:1 m))
        in
        Nx.max (Nx.abs (Nx.sub (Nx.matmul e x) b)))
      au
  in
  check_close ~tol:1e-9 ~msg:"blocked flags residual" [| 0.0 |]
    (to_arr resid_flags)

(* Differentiating a QR-using loss inside jit: the forward factorization
   compiles via [Tolk_frontend.Linalg], and the reverse pullback (recorded on
   the tape by the nested grad) traces too — its matmuls and triangular solve
   are ordinary graph ops, so the whole backward pass ends up in the compiled
   program. Compare against the eager gradient. *)
let test_qr_gradient_compiles () =
  let loss m =
    let q, r = Nx.qr ~mode:`Reduced m in
    let lq = Nx.mul q (Nx.tril ~k:0 (Nx.full Nx.float64 (Nx.shape q) 1.0)) in
    let lr = Nx.mul r (Nx.triu ~k:0 (Nx.full Nx.float64 (Nx.shape r) 1.0)) in
    Nx.add (Nx.sum (Nx.mul lq lq)) (Nx.sum (Nx.mul lr lr))
  in
  let a =
    mat64 4 4
      [|
        12.0;
        1.0;
        3.0;
        0.5;
        1.0;
        13.0;
        2.0;
        1.0;
        3.0;
        2.0;
        14.0;
        0.25;
        0.5;
        1.0;
        0.5;
        15.0;
      |]
  in
  let compiled = Rune.jit' (fun m -> Rune.grad' loss m) a in
  check_close ~tol:1e-10 ~msg:"grad through compiled QR"
    (to_arr (Rune.grad' loss a))
    (to_arr compiled)

(* Only the lower triangle is read, in both triangles' factors: the upper
   triangle holds garbage here, and the compiled program must ignore it as the
   eager kernel does. *)
let test_cholesky_matches_eager () =
  let a =
    Nx.create f32 [| 3; 3 |] [| 4.0; 9.0; 9.0; 1.0; 5.0; 9.0; 2.0; 3.0; 6.0 |]
  in
  let l = Rune.jit' (fun m -> Nx.cholesky m) a in
  check_arr ~msg:"lower" (to_arr (Nx.cholesky a)) l;
  let u = Rune.jit' (fun m -> Nx.cholesky ~upper:true m) a in
  check_arr ~msg:"upper" (to_arr (Nx.cholesky ~upper:true a)) u

(* The Cholesky pullback is jit-safe (its diagonal terms and triangular solves
   are graph ops), so differentiating a Cholesky-using loss inside jit compiles
   the whole backward pass. *)
let test_cholesky_gradient_compiles () =
  let loss m = Nx.sum (Nx.mul (Nx.cholesky m) (Nx.cholesky m)) in
  let a = mat64 3 3 [| 4.0; 1.0; 2.0; 1.0; 5.0; 3.0; 2.0; 3.0; 6.0 |] in
  let compiled = Rune.jit' (fun m -> Rune.grad' loss m) a in
  check_close ~tol:1e-10 ~msg:"grad through compiled Cholesky"
    (to_arr (Rune.grad' loss a))
    (to_arr compiled)

(* [Nx.solve] compiles: its singularity check lives in the graph, so the QR and
   the triangular solve trace as one program, and [Nx.inv] follows. *)
let test_solve_matches_eager () =
  let solve (a, b) = Nx.solve a b in
  let a =
    Nx.create f32 [| 3; 3 |] [| 4.0; 1.0; 2.0; 1.0; 5.0; 3.0; 2.0; 3.0; 6.0 |]
  in
  let b = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let g = Rune.jit2 (module Pair32) (module Csingle) solve in
  check_arr ~msg:"first call" (to_arr (Nx.solve a b)) (g (a, b));
  (* Replay solves a different system with the same compiled program. *)
  let a2 = Nx.mul_s a 1.5 in
  let b2 = Nx.add_s b 2.0 in
  check_arr ~msg:"replay" (to_arr (Nx.solve a2 b2)) (g (a2, b2));
  check_arr ~msg:"inv" (to_arr (Nx.inv a)) (Rune.jit' (fun m -> Nx.inv m) a)

(* Staged scans: under jit a [Rune.scan] compiles the fold step once and runs it
   as a loop in the compiled program, and [grad] through it compiles a reversed
   loop over the body's pullback. Every case compares against the eager
   (unrolled) scan and the eager gradient. *)

let cumsum xs =
  Rune.scan'
    ~f:(fun c x ->
      let c = Nx.add c x in
      (c, c))
    ~init:(Nx.scalar f32 0.0) xs

(* A two-tensor carry. *)
module Pair = struct
  type t = { u : Nx.float32_t; v : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { u; v } =
    { u = f u; v = f v }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { u = f p.u q.u; v = f p.v q.v }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { u; v } =
    f u;
    f v
end

type pair = Pair.t

let test_scan_matches_eager () =
  let g = Rune.jit' (fun xs -> snd (cumsum xs)) in
  let xs = vec32 [| 1.0; 2.0; 3.0 |] in
  check_arr ~msg:"cumulative sum" (to_arr (snd (cumsum xs))) (g xs);
  (* Replay computes on fresh data. *)
  check_arr ~msg:"replay" [| 0.5; 2.5; 5.5 |] (g (vec32 [| 0.5; 2.0; 3.0 |]))

let test_grad_through_scan_matches_eager () =
  (* c' = tanh (c + x): the pullback reads the carry stack the forward loop
     records. *)
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.tanh (Nx.add c x) in
          (c, c))
        ~init:(Nx.scalar f32 0.0) xs
    in
    Nx.add (Nx.reshape [||] c) (Nx.sum ys)
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  let g = Rune.jit' (fun xs -> Rune.grad' loss xs) in
  check_arr ~msg:"tanh recurrence" (to_arr (Rune.grad' loss xs)) (g xs);
  check_arr ~msg:"replay with fresh data"
    (to_arr (Rune.grad' loss (vec32 [| 0.25; -1.0; 1.5; 0.75 |])))
    (g (vec32 [| 0.25; -1.0; 1.5; 0.75 |]));
  (* n = 1 *)
  check_arr ~msg:"single step"
    (to_arr (Rune.grad' loss (vec32 [| 2.0 |])))
    (g (vec32 [| 2.0 |]))

let test_grad_through_scan_ys_only () =
  (* The loss reads only the stacked outputs: the final carry's cotangent is
     zero. *)
  let loss xs =
    let _c, ys = cumsum xs in
    Nx.sum ys
  in
  let xs = vec32 [| 1.0; 2.0; 3.0 |] in
  check_arr ~msg:"ys only"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_carry_only () =
  (* The loss reads only the final carry: the stacked outputs' cotangent is
     zero. *)
  let loss xs =
    let c, _ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.mul c x in
          (c, c))
        ~init:(Nx.scalar f32 1.0) xs
    in
    Nx.reshape [||] c
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  check_arr ~msg:"final carry only"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_multi_leaf () =
  let loss xs =
    let p, ys =
      Rune.scan
        (module Pair)
        Nx.Ptree.leaf Nx.Ptree.leaf
        ~f:(fun p x ->
          let u = Nx.add p.u x and v = Nx.mul p.v x in
          ({ u; v }, Nx.mul u v))
        ~init:{ u = Nx.scalar f32 0.0; v = Nx.scalar f32 1.0 }
        xs
    in
    Nx.add (Nx.add (Nx.reshape [||] p.u) (Nx.reshape [||] p.v)) (Nx.sum ys)
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  check_arr ~msg:"pair carry"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_asymmetric_pair () =
  (* Same-shaped leaves entering the loss with different weights: pairing a
     final-carry buffer (or an init-carry cotangent) with the wrong leaf changes
     the result instead of cancelling out. *)
  let loss xs =
    let p, ys =
      Rune.scan
        (module Pair)
        Nx.Ptree.leaf Nx.Ptree.leaf
        ~f:(fun p x ->
          let u = Nx.tanh (Nx.add p.u x) and v = Nx.mul p.v (Nx.add_s x 0.5) in
          ({ u; v }, Nx.add (Nx.mul_s u 2.0) v))
        ~init:{ u = Nx.scalar f32 0.1; v = Nx.scalar f32 1.0 }
        xs
    in
    Nx.add
      (Nx.add (Nx.mul_s (Nx.reshape [||] p.u) 3.0) (Nx.reshape [||] p.v))
      (Nx.sum ys)
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  check_arr ~msg:"asymmetric pair forward"
    (to_arr (loss xs))
    (Rune.jit' loss xs);
  check_arr ~msg:"asymmetric pair grad"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_scan_shape_unstable_carry_unrolls () =
  (* The carry grows a slot per step, so no single compiled body can stand for
     every iteration: the jit declines staging and the fold unrolls into the
     trace, forward and under grad. *)
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          (Nx.concatenate ~axis:0 [ c; Nx.reshape [| 1 |] x ], Nx.sum c))
        ~init:(Nx.zeros f32 [| 1 |]) xs
    in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  check_arr ~msg:"unstable forward" (to_arr (loss xs)) (Rune.jit' loss xs);
  check_arr ~msg:"unstable grad"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_nested () =
  (* The body itself scans (over the elements of a vector x). *)
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let ci, inner =
            Rune.scan'
              ~f:(fun ci xi ->
                let ci = Nx.add ci xi in
                (ci, Nx.mul ci xi))
              ~init:c x
          in
          let c = Nx.add ci (Nx.sum inner) in
          (c, c))
        ~init:(Nx.zeros f32 [| 2 |]) xs
    in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let xs = Nx.create f32 [| 3; 2 |] [| 1.0; 0.5; -1.0; 2.0; 0.25; 1.0 |] in
  check_arr ~msg:"forward" (to_arr (loss xs)) (Rune.jit' loss xs);
  check_arr ~msg:"grad"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_captured_weight () =
  (* The body reads a closure capture (a compile-time constant). *)
  let w = vec32 [| 2.0 |] in
  let loss xs =
    let _c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.add c (Nx.mul x (Nx.reshape [||] w)) in
          (c, Nx.mul c c))
        ~init:(Nx.scalar f32 0.0) xs
    in
    Nx.sum ys
  in
  let xs = vec32 [| 1.0; 2.0; 3.0; 0.5 |] in
  check_arr ~msg:"captured weight"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_vector_carry () =
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.tanh (Nx.add c x) in
          (c, Nx.mul c c))
        ~init:(Nx.zeros f32 [| 2 |]) xs
    in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let xs = Nx.create f32 [| 3; 2 |] [| 1.0; 0.5; -1.0; 2.0; 0.25; 1.0 |] in
  check_arr ~msg:"vector carry"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

(* A loop steps through its stacked rows at a stride padded to 16 bytes, so rows
   that fall short of it (five halves, three floats) read and write only their
   own elements, forward and backward. *)
(* Rows and outputs are structures: a stack of per-step weights and a mixed
   float/int row, and two outputs. The cotangent of the rows is stacked like
   them, row i from step i; the integer row gets none. *)
module Rows = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t; step : Nx.int32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) r =
    { w = f r.w; b = f r.b; step = f r.step }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) r r' =
    { w = f r.w r'.w; b = f r.b r'.b; step = f r.step r'.step }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) r =
    f r.w;
    f r.b;
    f r.step
end

let layers xs =
  Rune.scan Nx.Ptree.leaf
    (module Rows)
    (module Pair)
    ~f:(fun h { Rows.w; b; step } ->
      let h =
        Nx.tanh
          (Nx.add
             (Nx.reshape [| 3 |] (Nx.matmul (Nx.reshape [| 1; 3 |] h) w))
             b)
      in
      (h, { Pair.u = h; v = Nx.mul_s (Nx.cast f32 step) 2.0 }))
    ~init:(vec32 [| 0.5; -0.25; 1.0 |])
    xs

let rows () =
  {
    Rows.w =
      Nx.create f32 [| 4; 3; 3 |]
        (Array.init 36 (fun i -> (Float.of_int (i * 5 mod 7) /. 7.0) -. 0.4));
    b =
      Nx.create f32 [| 4; 3 |] (Array.init 12 (fun i -> Float.of_int i /. 12.0));
    step = Nx.arange Nx.int32 0 4 1;
  }

let test_scan_over_structured_rows () =
  let xs = rows () in
  let h, ys = layers xs in
  let g = Rune.jit2 (module Rows) (module Pair) (fun xs -> snd (layers xs)) in
  let ys' = g xs in
  check_arr ~msg:"first output" (to_arr ys.Pair.u) ys'.Pair.u;
  check_arr ~msg:"second output" (to_arr ys.Pair.v) ys'.Pair.v;
  check_arr ~msg:"final carry" (to_arr h)
    (Rune.jit2 (module Rows) (module Csingle) (fun xs -> fst (layers xs)) xs);
  let loss xs =
    let h, ys = layers xs in
    Nx.add (Nx.sum h) (Nx.sum (Nx.mul ys.Pair.u ys.Pair.u))
  in
  let expected = Rune.grad (module Rows) loss xs in
  let actual =
    Rune.jit2
      (module Rows)
      (module Rows)
      (fun xs -> Rune.grad (module Rows) loss xs)
      xs
  in
  check_arr ~msg:"stacked weights' cotangent" (to_arr expected.Rows.w)
    actual.Rows.w;
  check_arr ~msg:"stacked biases' cotangent" (to_arr expected.Rows.b)
    actual.Rows.b;
  equal ~msg:"the integer row gets a zero cotangent" (array int32)
    [| 0l; 0l; 0l; 0l |]
    (Nx.to_array actual.Rows.step)

let test_scan_rejects_ragged_rows () =
  let bad xs =
    ignore
      (Rune.scan Nx.Ptree.leaf
         (module Rows)
         Nx.Ptree.leaf
         ~f:(fun c _ -> (c, c))
         ~init:(vec32 [| 0.0 |]) xs)
  in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> bad { (rows ()) with Rows.step = Nx.arange Nx.int32 0 3 1 });
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> bad { (rows ()) with Rows.step = Nx.scalar Nx.int32 0l })

(* A carry updated in place: a body that writes one row of a stacked cache per
   step moves that row, not the cache. A body that reads the old cache after
   writing the new one still sees the old values: the write lands in a copy. *)
module Cache_carry = struct
  type t = { h : Nx.float32_t; cache : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) c =
    { h = f c.h; cache = f c.cache }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) c c' =
    { h = f c.h c'.h; cache = f c.cache c'.cache }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) c =
    f c.h;
    f c.cache
end

module Carry_and_sums = struct
  type t = Cache_carry.t * Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (c, y) =
    (Cache_carry.map f c, f y)

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (c, y)
      (c', y') =
    (Cache_carry.map2 f c c', f y y')

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (c, y) =
    Cache_carry.iter f c;
    f y
end

let test_scan_carry_written_in_place () =
  let layers = 4 and slots = 64 and d = 16 in
  let fold ~read_old (c : Cache_carry.t) =
    Rune.scan
      (module Cache_carry)
      Nx.Ptree.leaf Nx.Ptree.leaf
      ~f:(fun (c : Cache_carry.t) l ->
        let h = Nx.tanh (Nx.add_s c.h 0.25) in
        let cache =
          Nx.set
            [ D (l, 1); D (Nx.mul_s l 3l, 1) ]
            (Nx.reshape [| 1; 1; d |] h)
            c.cache
        in
        let y = if read_old then Nx.sum c.cache else Nx.sum h in
        ({ Cache_carry.h; cache }, y))
      ~init:c
      (Nx.arange Nx.int32 0 layers 1)
  in
  let c0 =
    {
      Cache_carry.h =
        Nx.create f32 [| d |] (Array.init d (fun i -> Float.of_int i /. 16.0));
      cache = Nx.zeros f32 [| layers; slots; d |];
    }
  in
  List.iter
    (fun read_old ->
      let expected_c, expected_ys = fold ~read_old c0 in
      let g =
        Rune.jit2 (module Cache_carry) (module Carry_and_sums) (fold ~read_old)
      in
      ignore (g c0);
      let before = !Tolk.Helpers.Global_counters.global_mem in
      let c, ys = g c0 in
      let bytes = !Tolk.Helpers.Global_counters.global_mem - before in
      let msg what = Printf.sprintf "%s (read_old %b)" what read_old in
      check_arr ~msg:(msg "cache") (to_arr expected_c.cache) c.cache;
      check_arr ~msg:(msg "state") (to_arr expected_c.h) c.h;
      check_arr ~msg:(msg "outputs") (to_arr expected_ys) ys;
      if not read_old then
        is_true ~msg:"a replay moves less than three caches' worth of bytes"
          (bytes < 3 * layers * slots * d * 4))
    [ false; true ]

(* The layers of a stack passed as rows are read in place: a replay launches the
   kernels of the same body reading one captured layer, and nothing copies the
   row the step reads. The body gathers two of the layer's eight rows and
   broadcasts them into a matrix product. *)
let test_scan_reads_rows_in_place () =
  let stack =
    Nx.create f32 [| 4; 8; 4; 4 |]
      (Array.init 512 (fun i -> Float.of_int (i * 7 mod 13) /. 13.0))
  in
  let rows = Nx.create Nx.int32 [| 2 |] [| 1l; 5l |] in
  let step x w =
    let w = Nx.take ~axis:0 ~indices:rows w in
    let y = Nx.sum ~axes:[ 0 ] (Nx.matmul w (Nx.reshape [| 4; 1 |] x)) in
    (Nx.tanh (Nx.reshape [| 4 |] y), Nx.zeros f32 [||])
  in
  let over_rows x0 = fst (Rune.scan' ~f:step ~init:x0 stack) in
  let layer = Nx.slice [ I 2 ] stack in
  let over_capture x0 =
    fst (Rune.scan' ~f:(fun x _ -> step x layer) ~init:x0 stack)
  in
  let x0 = vec32 [| 0.5; -1.0; 0.25; 2.0 |] in
  let kernels_per_replay f =
    let g = Rune.jit' f in
    ignore (g x0);
    let before = !Tolk.Helpers.Global_counters.kernel_count in
    let y = g x0 in
    (y, !Tolk.Helpers.Global_counters.kernel_count - before)
  in
  let y, from_rows = kernels_per_replay over_rows in
  let _, from_capture = kernels_per_replay over_capture in
  check_arr ~msg:"matches the eager fold" (to_arr (over_rows x0)) y;
  equal ~msg:"no copy of the row" int from_capture from_rows

let test_scan_rows_short_of_16_bytes () =
  let fold xs =
    Rune.scan'
      ~f:(fun c x ->
        let c = Nx.add (Nx.mul_s c 0.5) x in
        (c, Nx.mul c x))
      ~init:(Nx.zeros Nx.float16 [| 5 |])
      xs
  in
  let xs =
    Nx.cast Nx.float16
      (Nx.create f32 [| 4; 5 |]
         (Array.init 20 (fun i -> Float.of_int (i - 7) /. 8.0)))
  in
  let to_f32 t = to_arr (Nx.cast f32 t) in
  let c, ys = fold xs in
  check_arr ~msg:"half carry" (to_f32 c)
    (Nx.cast f32 (Rune.jit' (fun xs -> fst (fold xs)) xs));
  check_arr ~msg:"half rows" (to_f32 ys)
    (Nx.cast f32 (Rune.jit' (fun xs -> snd (fold xs)) xs));
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.tanh (Nx.add c x) in
          (c, Nx.mul c x))
        ~init:(Nx.zeros f32 [| 3 |]) xs
    in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let xs =
    Nx.create f32 [| 4; 3 |] (Array.init 12 (fun i -> Float.of_int i /. 6.0))
  in
  check_arr ~msg:"three-float rows, gradient"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

let test_grad_through_scan_external_input () =
  (* The body closes over a *differentiated* input that is neither the carry nor
     the scanned sequence: an external co-tangent the backward loop must total
     across the steps. *)
  let loss (p : pair) =
    let w = p.Pair.u and xs = p.Pair.v in
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.tanh (Nx.add (Nx.mul c (Nx.reshape [||] w)) x) in
          (c, c))
        ~init:(Nx.scalar f32 0.0) xs
    in
    Nx.add (Nx.reshape [||] c) (Nx.sum ys)
  in
  let p = { Pair.u = Nx.scalar f32 0.5; v = vec32 [| 1.0; 2.0; 3.0; 0.5 |] } in
  let expected = Rune.grad (module Pair) loss p in
  let g =
    Rune.jit2
      (module Pair)
      (module Pair)
      (fun p -> Rune.grad (module Pair) loss p)
  in
  let actual = g p in
  check_arr ~msg:"external weight" (to_arr expected.Pair.u) actual.Pair.u;
  check_arr ~msg:"scanned input" (to_arr expected.Pair.v) actual.Pair.v;
  (* Replay computes the totals on fresh data. *)
  let p2 =
    { Pair.u = Nx.scalar f32 (-0.25); v = vec32 [| 0.25; -1.0; 1.5; 0.75 |] }
  in
  let expected2 = Rune.grad (module Pair) loss p2 in
  let actual2 = g p2 in
  check_arr ~msg:"external weight, replay" (to_arr expected2.Pair.u)
    actual2.Pair.u;
  check_arr ~msg:"scanned input, replay" (to_arr expected2.Pair.v)
    actual2.Pair.v

(* A three-leaf input structure. *)
(* A tensor with a run-time window start: the shape of every decode step. *)
type windowed = { x : Nx.float32_t; pos : Nx.int32_t }

module Windowed = struct
  type t = windowed

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { x; pos } =
    { x = f x; pos = f pos }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { x = f p.x q.x; pos = f p.pos q.pos }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { x; pos } =
    f x;
    f pos
end

let pos_at i = Nx.scalar Nx.int32 (Int32.of_int i)

(* One compiled program serves every window position: the start is read on every
   call, so the second call must write where its own [pos] says, not where the
   trace was taken. *)
let test_set_traced_window_replays_position () =
  let v = vec32 [| 9.0; 8.0 |] in
  let f { x; pos } = Nx.set [ Nx.D (pos, 2) ] v x in
  let g = Rune.jit2 (module Windowed) (module Csingle) f in
  let x = vec32 [| 0.0; 1.0; 2.0; 3.0; 4.0 |] in
  let at i = { x; pos = pos_at i } in
  check_arr ~msg:"first position" (to_arr (f (at 1))) (g (at 1));
  check_arr ~msg:"second position, same program" (to_arr (f (at 3))) (g (at 3));
  check_arr ~msg:"clamped start" (to_arr (f (at 9))) (g (at 9));
  check_arr ~msg:"the input is a value" [| 0.0; 1.0; 2.0; 3.0; 4.0 |] x

(* A window over two axes: the compiled write addresses the window's elements at
   their flat positions in [x]. *)
let test_set_traced_window_over_two_axes () =
  let v = Nx.create f32 [| 2; 3 |] [| 9.0; 8.0; 7.0; 6.0; 5.0; 4.0 |] in
  let f { x; pos } = Nx.set [ Nx.D (pos, 2); Nx.D (pos, 3) ] v x in
  let g = Rune.jit2 (module Windowed) (module Csingle) f in
  let x = Nx.create f32 [| 4; 6 |] (Array.init 24 float_of_int) in
  let at i = { x; pos = pos_at i } in
  List.iter
    (fun i ->
      let msg = Printf.sprintf "corner %d" i in
      check_arr ~msg (to_arr (f (at i))) (g (at i)))
    [ 0; 1; 2; 7 ]

let test_set_static_window_matches_eager () =
  let v = vec32 [| 9.0; 8.0 |] in
  let f x = Nx.set [ Nx.R (1, 3) ] v x in
  let x = vec32 [| 0.0; 1.0; 2.0; 3.0; 4.0 |] in
  check_arr ~msg:"window" (to_arr (f x)) (Rune.jit' f x);
  let h x = Nx.set [ Nx.L [ 0; 3 ] ] v x in
  check_arr ~msg:"gather" (to_arr (h x)) (Rune.jit' h x);
  let m = Nx.create Nx.bool [| 5 |] [| true; false; true; false; false |] in
  let k x = Nx.set [ Nx.M m ] (Nx.scalar f32 7.0) x in
  check_arr ~msg:"mask" (to_arr (k x)) (Rune.jit' k x)

let test_slice_traced_window_replays_position () =
  let f { x; pos } = Nx.slice [ Nx.D (pos, 2) ] x in
  let g = Rune.jit2 (module Windowed) (module Csingle) f in
  let x = vec32 [| 0.0; 1.0; 2.0; 3.0; 4.0 |] in
  let at i = { x; pos = pos_at i } in
  check_arr ~msg:"first position" [| 1.0; 2.0 |] (g (at 1));
  check_arr ~msg:"second position, same program" [| 3.0; 4.0 |] (g (at 3))

module Trio = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t; xs : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b; xs } =
    { a = f a; b = f b; xs = f xs }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b; xs = f p.xs q.xs }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b; xs } =
    f a;
    f b;
    f xs
end

let test_grad_through_scan_external_matrices () =
  (* An RNN step: the recurrence and input matrices are external inputs of the
     loop, each earning a cotangent contribution per step. *)
  let loss (p : Trio.t) =
    let step x ut =
      let x = Nx.tanh (Nx.add (Nx.matmul x p.Trio.a) (Nx.matmul ut p.Trio.b)) in
      (x, x)
    in
    let _, ys = Rune.scan' ~f:step ~init:(Nx.zeros f32 [| 2 |]) p.Trio.xs in
    Nx.sum ys
  in
  let p =
    Trio.
      {
        a = Nx.create f32 [| 2; 2 |] [| 0.5; 0.25; 0.0; 0.75 |];
        b = Nx.create f32 [| 2; 2 |] [| 1.0; -0.5; 0.25; 0.5 |];
        xs = Nx.create f32 [| 3; 2 |] [| 1.0; 0.5; -1.0; 2.0; 0.25; 1.0 |];
      }
  in
  let expected = Rune.grad (module Trio) loss p in
  let g =
    Rune.jit2
      (module Trio)
      (module Trio)
      (fun p -> Rune.grad (module Trio) loss p)
  in
  let actual = g p in
  check_arr ~msg:"recurrence matrix" (to_arr expected.Trio.a) actual.Trio.a;
  check_arr ~msg:"input matrix" (to_arr expected.Trio.b) actual.Trio.b;
  check_arr ~msg:"scanned input" (to_arr expected.Trio.xs) actual.Trio.xs

let test_grad_through_scan_matrix_carry () =
  (* A 2-D carry and per-step output: the loop's flat slot buffers store them
     flattened. *)
  let loss xs =
    let c, ys =
      Rune.scan'
        ~f:(fun c x ->
          let c = Nx.tanh (Nx.add c x) in
          (c, Nx.mul c c))
        ~init:(Nx.zeros f32 [| 2; 2 |])
        xs
    in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let xs =
    Nx.create f32 [| 3; 2; 2 |]
      [| 1.0; 0.5; -1.0; 2.0; 0.25; 1.0; 0.75; -0.5; -0.25; 0.0; 1.5; 0.5 |]
  in
  check_arr ~msg:"matrix carry forward" (to_arr (loss xs)) (Rune.jit' loss xs);
  check_arr ~msg:"matrix carry grad"
    (to_arr (Rune.grad' loss xs))
    (Rune.jit' (fun xs -> Rune.grad' loss xs) xs)

(* Buffer sharing: strided leaves must fall back to copies, views with an offset
   must read the right span, and each call must return tensors with their own
   storage. *)

let test_non_contiguous_input_matches_eager () =
  let f x = Nx.add (Nx.mul x x) x in
  let g = Rune.jit' f in
  let x =
    Nx.transpose (Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
  in
  check_arr ~msg:"transposed input" (to_arr (f x)) (g x)

let test_offset_view_input_matches_eager () =
  let f x = Nx.mul_s x 3.0 in
  let g = Rune.jit' f in
  let x =
    Nx.get [ 1 ] (Nx.create f32 [| 3; 4 |] (Array.init 12 float_of_int))
  in
  check_arr ~msg:"offset row view" (to_arr (f x)) (g x)

let test_outputs_have_their_own_storage () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y1 = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  let _y2 = g (vec32 [| 10.0; 20.0; 30.0 |]) in
  check_arr ~msg:"first result unchanged by the second call" [| 2.0; 4.0; 6.0 |]
    y1

(* Sliding windows *)

(* An asymmetric configuration so any axis-ordering mistake shows up: distinct
   kernel, stride, dilation, and padding per spatial dimension. *)
let window_config =
  ( [| 2; 3 |] (* kernel *),
    [| 2; 1 |] (* stride *),
    [| 1; 2 |] (* dilation *),
    [| (1, 0); (2, 1) |] (* padding *) )

let window_input () =
  Nx.create f32 [| 2; 3; 5; 6 |]
    (Array.init (2 * 3 * 5 * 6) (fun i -> float_of_int (i mod 17) -. 8.0))

let test_unfold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let f x = Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x in
  let g = Rune.jit' f in
  let x = window_input () in
  equal ~msg:"shape" (array int) (Nx.shape (f x)) (Nx.shape (g x));
  check_arr ~msg:"unfold" (to_arr (f x)) (g x)

let test_fold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let output_size = [| 5; 6 |] in
  let f x =
    Nx.combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding
      (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x)
  in
  let g = Rune.jit' f in
  let x = window_input () in
  check_arr ~msg:"fold of unfold" (to_arr (f x)) (g x)

let test_sliding_window_matches_eager () =
  let f x = sliding_window ~axis:1 ~window:3 ~step:2 x in
  let g = Rune.jit' f in
  let x =
    Nx.create f32 [| 2; 8 |] (Array.init 16 (fun i -> float_of_int i -. 7.5))
  in
  equal ~msg:"shape" (array int) (Nx.shape (f x)) (Nx.shape (g x));
  check_arr ~msg:"sliding windows" (to_arr (f x)) (g x)

let test_correlate_matches_eager () =
  let kernel =
    Nx.create f32 [| 3; 3 |]
      [| 1.0; 0.0; -1.0; 2.0; 0.5; -2.0; 1.0; 0.0; -1.0 |]
  in
  let f x = Nx.correlate ~padding:`Same x kernel in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 6; 7 |] (Array.init 42 (fun i -> float_of_int i)) in
  check_arr ~msg:"correlate same" (to_arr (f x)) (g x)

(* Cumulative reductions *)

(* A sum over int8 or int16 accumulates in int32; the compiled scan hands back
   the input's dtype, so its values wrap as eager's do. Compacting between calls
   exposes a result written past a buffer sized for the input dtype. *)
let test_small_int_scans_keep_dtype () =
  let check (type b) name (dtype : (int, b) Nx.dtype) values =
    let x = Nx.create dtype [| Array.length values |] values in
    let g = Rune.jit' (Nx.cumsum ~axis:0) in
    let expected = Nx.to_array (Nx.cumsum ~axis:0 x) in
    for call = 1 to 20 do
      equal
        ~msg:(Printf.sprintf "%s, call %d" name call)
        (array int) expected
        (Nx.to_array (g x));
      Gc.compact ()
    done
  in
  check "int8" Nx.int8 [| 100; 100; 100; 1 |];
  check "int8, 64" Nx.int8 (Array.make 64 1);
  check "int8, 600" Nx.int8 (Array.make 600 1);
  check "int16" Nx.int16 [| 1; 2; 3 |];
  check "int16, 600" Nx.int16 (Array.init 600 (fun i -> i * 50))

(* An axis longer than 512 scans in chunks; 1000 leaves a partial chunk. *)
let test_long_scans_match_eager () =
  let input f = Nx.create f32 [| 1000; 2 |] (Array.init 2000 f) in
  let values = input (fun i -> float_of_int ((i * 7 mod 11) - 5)) in
  let signs = input (fun i -> if i mod 97 = 0 then -1.0 else 1.0) in
  let check name f x =
    let g = Rune.jit' f in
    check_arr ~msg:name (to_arr (f x)) (g x)
  in
  check "cumsum" (Nx.cumsum ~axis:0) values;
  check "cummax" (Nx.cummax ~axis:0) values;
  check "cumprod" (Nx.cumprod ~axis:0) signs

(* Indexed access *)

(* Row 1 repeats an index so duplicate handling is pinned under jit: [`Set]
   keeps the last update, [`Add] accumulates both on top of [x]'s value. *)
let test_scatter_matches_eager () =
  let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 1l |] in
  let f mode x =
    Nx.scatter ~mode ~axis:1 ~indices:idx
      ~values:(Nx.slice [ Nx.A; Nx.R (0, 2) ] x)
      x
  in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let g_set = Rune.jit' (f `Set) and g_add = Rune.jit' (f `Add) in
  check_arr ~msg:"set" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"set replay" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"add" (to_arr (f `Add x)) (g_add x)

(* The compiled scatter ranges over the updates, not over the destination. Each
   case is held to the eager result. *)

let i32 shape xs = Nx.create Nx.int32 shape (Array.map Int32.of_int xs)

let iota shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape (Array.init n (fun i -> float_of_int (i + 1)))

let check_scatter ~msg ?unique_indices ~axis ~indices ~values t =
  List.iter
    (fun (name, mode) ->
      let f t = Nx.scatter ~mode ?unique_indices ~axis ~indices ~values t in
      let g = Rune.jit' f in
      check_arr ~msg:(msg ^ ", " ^ name) (to_arr (f t)) (g t);
      check_arr ~msg:(msg ^ ", " ^ name ^ ", replay") (to_arr (f t)) (g t))
    [ ("set", `Set); ("add", `Add) ]

let test_scatter_duplicates () =
  check_scatter ~msg:"rows aimed at one row twice" ~axis:0
    ~indices:(i32 [| 3; 3 |] [| 2; 0; 1; 2; 3; 1; 0; 0; 1 |])
    ~values:(iota [| 3; 3 |])
    (iota [| 4; 3 |]);
  check_scatter ~msg:"every update of a lane aims at one cell" ~axis:1
    ~indices:(i32 [| 2; 3 |] [| 1; 1; 1; 3; 3; 3 |])
    ~values:(iota [| 2; 3 |])
    (iota [| 2; 4 |])

let test_scatter_middle_axis () =
  check_scatter ~msg:"middle axis" ~axis:1
    ~indices:(i32 [| 2; 2; 3 |] [| 3; 0; 1; 3; 2; 1; 0; 0; 0; 1; 2; 3 |])
    ~values:(iota [| 2; 2; 3 |])
    (iota [| 2; 4; 3 |])

let test_scatter_unique_indices () =
  check_scatter ~msg:"unique" ~unique_indices:true ~axis:0
    ~indices:(i32 [| 2; 2 |] [| 3; 0; 1; 2 |])
    ~values:(iota [| 2; 2 |])
    (iota [| 4; 2 |])

(* The promise of unique indices broken at one row, the way a cache write aims
   every token it does not store at a scratch row: eager and compiled, every
   other row is exact and each element of the repeated row is one of the updates
   aimed at it. *)
let test_scatter_unique_indices_broken_at_one_row () =
  let rows = 6 and width = 8 and scratch = 5 in
  let targets = [| 2; scratch; 0; scratch; scratch; 3 |] in
  let indices = Nx.broadcast_to [| rows; width |] (i32 [| rows; 1 |] targets) in
  let values = iota [| rows; width |] in
  let f t = Nx.scatter ~unique_indices:true ~axis:0 ~indices ~values t in
  let t = Nx.zeros f32 [| rows; width |] in
  let check name got =
    Array.iteri
      (fun k target ->
        if target <> scratch then
          for j = 0 to width - 1 do
            equal
              ~msg:(Printf.sprintf "%s, row %d, element %d" name target j)
              float_exact
              (float_of_int ((k * width) + j + 1))
              got.((target * width) + j)
          done)
      targets;
    for j = 0 to width - 1 do
      let v = got.((scratch * width) + j) in
      let aimed k = v = float_of_int ((k * width) + j + 1) in
      is_true
        ~msg:
          (Printf.sprintf "%s, the repeated row holds an update at %d" name j)
        (aimed 1 || aimed 3 || aimed 4)
    done;
    for j = 0 to width - 1 do
      equal
        ~msg:(name ^ ", an untouched row")
        float_exact 0.0
        got.((1 * width) + j)
    done
  in
  check "eager" (to_arr (f t));
  let g = Rune.jit' f in
  check "compiled" (to_arr (g t));
  check "replay" (to_arr (g t))

(* Eager raises on an index outside the axis; compiled, the update is dropped.
   -1 is the address a slot map gives a token that is not written. *)
let test_scatter_out_of_range_writes_nothing () =
  let indices = i32 [| 4; 2 |] [| -1; 4; 1; -7; 2; 1; 5; -1 |] in
  let values = iota [| 4; 2 |] in
  let f mode t = Nx.scatter ~mode ~axis:0 ~indices ~values t in
  check_arr ~msg:"set"
    [| 0.0; 0.0; 3.0; 6.0; 5.0; 0.0; 0.0; 0.0 |]
    (Rune.jit' (f `Set) (Nx.zeros f32 [| 4; 2 |]));
  check_arr ~msg:"add"
    [| 1.0; 2.0; 6.0; 10.0; 10.0; 6.0; 7.0; 8.0 |]
    (Rune.jit' (f `Add) (iota [| 4; 2 |]))

let test_scatter_payload_dtypes () =
  let indices = i32 [| 4 |] [| 2; 0; 2; 1 |] in
  List.iter
    (fun mode ->
      let values = i32 [| 4 |] [| 5; 7; 9; 11 |] in
      let ints t = Nx.scatter ~mode ~axis:0 ~indices ~values t in
      let t = i32 [| 3 |] [| 100; 200; 300 |] in
      check_arr ~msg:"int32"
        (to_arr (Nx.cast f32 (ints t)))
        (Nx.cast f32 (Rune.jit' ints t));
      let halves t =
        Nx.scatter ~mode ~axis:0 ~indices
          ~values:(Nx.cast Nx.bfloat16 (vec32 [| 0.5; 1.5; 2.5; 4.0 |]))
          t
      in
      let t = Nx.cast Nx.bfloat16 (vec32 [| 8.0; 16.0; 32.0 |]) in
      check_arr ~msg:"bfloat16"
        (to_arr (Nx.cast f32 (halves t)))
        (Nx.cast f32 (Rune.jit' halves t)))
    [ `Set; `Add ]

let test_scatter_under_vmap () =
  let indices = i32 [| 3; 2 |] [| 1; 0; 1; 2; 0; 0 |] in
  let values = iota [| 3; 2 |] and t = iota [| 3; 2 |] in
  let batch x = Nx.stack ~axis:0 [ x; Nx.add x x ] in
  let check ~msg f x =
    check_arr ~msg (to_arr (Rune.vmap' f x)) (Rune.jit' (Rune.vmap' f) x)
  in
  List.iter
    (fun (name, mode) ->
      check
        ~msg:("over the destination, " ^ name)
        (fun t -> Nx.scatter ~mode ~axis:0 ~indices ~values t)
        (batch t);
      check
        ~msg:("over the values, " ^ name)
        (fun values -> Nx.scatter ~mode ~axis:0 ~indices ~values t)
        (batch values))
    [ ("set", `Set); ("add", `Add) ];
  let rows = i32 [| 2; 3; 2 |] [| 1; 0; 1; 2; 0; 0; 2; 2; 2; 1; 0; 1 |] in
  let f indices = Nx.scatter ~mode:`Add ~axis:0 ~indices ~values t in
  check_arr ~msg:"over the indices"
    (to_arr (Rune.vmap' f rows))
    (Rune.jit' (Rune.vmap' f) rows)

(* The pullback of [take] accumulates a row's cotangent once per occurrence of
   its token. *)
let test_grad_of_take_with_repeated_tokens () =
  let indices = i32 [| 6 |] [| 3; 1; 3; 3; 0; 1 |] in
  let weights = iota [| 6; 2 |] in
  let loss table = Nx.sum (Nx.mul weights (Nx.take ~axis:0 ~indices table)) in
  let table = iota [| 5; 2 |] in
  check_arr ~msg:"embedding gradient"
    (to_arr (Rune.grad' loss table))
    (Rune.jit' (Rune.grad' loss) table)

(* A table past the reduce-split threshold: the row count is where a split
   one-hot reduce used to cost a pass over the table. *)
let test_take_large_table_matches_eager () =
  let rows = 65_536 in
  let table =
    Nx.create f32 [| rows; 2 |]
      (Array.init (rows * 2) (fun i -> float_of_int (i mod 1000)))
  in
  let indices =
    Nx.create Nx.int32 [| 4 |] [| 0l; 65_535l; 40_000l; 32_768l |]
  in
  let f table = Nx.take ~axis:0 ~indices table in
  check_arr ~msg:"take" (to_arr (f table)) (Rune.jit' f table)

(* Either side of [Nx.top_k]'s switch from selection rounds to a sort, with
   repeated scores in every row. *)
let test_top_k_matches_eager () =
  let scores =
    Nx.create f32 [| 3; 24 |]
      (Array.init 72 (fun i -> float_of_int (i * 7 mod 11)))
  in
  List.iter
    (fun k ->
      let values x = fst (Nx.top_k ~k x) in
      let indices x = Nx.cast f32 (snd (Nx.top_k ~k x)) in
      check_arr
        ~msg:(Printf.sprintf "top %d values" k)
        (to_arr (values scores))
        (Rune.jit' values scores);
      check_arr
        ~msg:(Printf.sprintf "top %d indices" k)
        (to_arr (indices scores))
        (Rune.jit' indices scores))
    [ 2; 17 ];
  let along_rows x = Nx.cast f32 (snd (Nx.top_k ~k:2 ~axis:0 x)) in
  check_arr ~msg:"top 2 along axis 0"
    (to_arr (along_rows scores))
    (Rune.jit' along_rows scores)

let test_grad_of_top_k () =
  let scores =
    Nx.create f32 [| 2; 5 |] [| 3.; 9.; 1.; 7.; 5.; 4.; 2.; 8.; 6.; 0. |]
  in
  let weights = Nx.create f32 [| 1; 2 |] [| 1.; 2. |] in
  let loss x = Nx.sum (Nx.mul weights (fst (Nx.top_k ~k:2 x))) in
  check_arr ~msg:"the gradient lands on the chosen entries"
    [| 0.; 1.; 0.; 2.; 0.; 0.; 0.; 1.; 2.; 0. |]
    (Rune.grad' loss scores);
  check_arr ~msg:"compiled gradient"
    (to_arr (Rune.grad' loss scores))
    (Rune.jit' (Rune.grad' loss) scores)

(* [Nx.diag] is traceable in both directions: extraction gathers, construction
   scatters into a zero template. *)
let test_diag_matches_eager () =
  let m =
    Nx.create f32 [| 3; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0 |]
  in
  let g = Rune.jit' (Nx.diag ~k:(-1)) in
  check_arr ~msg:"extract" (to_arr (Nx.diag ~k:(-1) m)) (g m);
  let v = vec32 [| 2.0; 3.0; 4.0 |] in
  let g2 = Rune.jit' (Nx.diag ~k:1) in
  check_arr ~msg:"construct" (to_arr (Nx.diag ~k:1 v)) (g2 v);
  check_arr ~msg:"construct replay" (to_arr (Nx.diag ~k:1 v)) (g2 v)

(* Training-step integration: a two-layer MLP trained by a jitted step must
   follow the eager trajectory exactly. *)

type mlp = { w1 : Nx.float32_t; b1 : Nx.float32_t; w2 : Nx.float32_t }

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w1; b1; w2 } =
    { w1 = f w1; b1 = f b1; w2 = f w2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w1 = f p.w1 q.w1; b1 = f p.b1 q.b1; w2 = f p.w2 q.w2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w1; b1; w2 } =
    f w1;
    f b1;
    f w2
end

let test_jitted_training_matches_eager () =
  let xs =
    Nx.create f32 [| 8; 4 |]
      (Array.init 32 (fun i -> float_of_int (i mod 7) /. 7.0))
  in
  let ys =
    Nx.create f32 [| 8; 1 |]
      (Array.init 8 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  let init () =
    {
      w1 =
        Nx.create f32 [| 4; 5 |]
          (Array.init 20 (fun i -> (0.1 *. float_of_int (i mod 5)) -. 0.2));
      b1 = Nx.zeros f32 [| 5 |];
      w2 =
        Nx.create f32 [| 5; 1 |]
          (Array.init 5 (fun i -> 0.3 -. (0.1 *. float_of_int i)));
    }
  in
  let loss p =
    let h = Nx.tanh (Nx.add (Nx.matmul xs p.w1) p.b1) in
    let d = Nx.sub (Nx.matmul h p.w2) ys in
    Nx.mean (Nx.mul d d)
  in
  let update p =
    let g = Rune.grad (module Mlp) loss p in
    Mlp.map2 (fun w dw -> Nx.sub w (Nx.mul (scalar_like dw 0.1) dw)) p g
  in
  let step = Rune.jit2 (module Mlp) (module Mlp) update in
  let rec train f p n = if n = 0 then p else train f (f p) (n - 1) in
  let jitted = train step (init ()) 5 in
  let eager = train update (init ()) 5 in
  check_arr ~msg:"w1" (to_arr eager.w1) jitted.w1;
  check_arr ~msg:"b1" (to_arr eager.b1) jitted.b1;
  check_arr ~msg:"w2" (to_arr eager.w2) jitted.w2;
  let l0 = scalar (loss (init ())) and l5 = scalar (loss jitted) in
  is_true ~msg:"loss decreased" (l5 < l0)

(* Device residency. CPU:1 is a device with storage of its own, which takes the
   staged-copy path used by CUDA and Metal: outputs are placed values that stay
   on the device until read, and a placed value fed back into a compiled call
   seeds its input buffer directly. The transfer counters make the no-copy
   claims observable. *)

let cpu1 = Rune.device "CPU:1"
let place x = Nx.place (Nx.Placement.device cpu1) x

(* Run [f] and return its result with the bytes moved to and from the device
   during the run. *)
let delta f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  let s1 = Rune.jit_stats () in
  ( r,
    s1.bytes_to_device - s0.bytes_to_device,
    s1.bytes_from_device - s0.bytes_from_device )

(* Placement. [Nx.place] makes a value resident without a compiled call. *)

let resident () = (Rune.jit_stats ()).resident_bytes

let test_place_equals_its_argument () =
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let base = resident () in
  let p, up, down = delta (fun () -> place x) in
  equal ~msg:"placing uploads the value once" int 24 up;
  equal ~msg:"placing reads nothing back" int 0 down;
  equal ~msg:"shape" (array int) [| 2; 3 |] (Nx.shape p);
  is_true ~msg:"dtype" (Nx.dtype p = f32);
  equal ~msg:"the placed value is resident" int 24 (resident () - base);
  let (), up, down = delta (fun () -> check_arr ~msg:"value" (to_arr x) p) in
  equal ~msg:"the first read copies it back" int 24 down;
  equal ~msg:"and uploads nothing" int 0 up;
  equal ~msg:"a read leaves the value placed" int 24 (resident () - base);
  check_arr ~msg:"the argument is untouched"
    [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    x

let test_place_strided_and_offset () =
  let m = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let t = Nx.matrix_transpose m in
  check_arr ~msg:"strided" (to_arr t) (place t);
  let s = Nx.slice [ Nx.R (1, 2) ] m in
  check_arr ~msg:"offset" [| 4.0; 5.0; 6.0 |] (place s)

let test_place_feeds_inputs_without_transfer () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  ignore (g (vec32 [| 0.0; 0.0 |]));
  let p = place (vec32 [| 1.0; 2.0 |]) in
  let y, up, down = delta (fun () -> g p) in
  equal ~msg:"no upload" int 0 up;
  equal ~msg:"no read-back" int 0 down;
  check_arr ~msg:"result" [| 2.0; 4.0 |] y;
  check_arr ~msg:"the input is still readable" [| 1.0; 2.0 |] p

let test_place_resident_value_is_returned () =
  let p = place (vec32 [| 1.0; 2.0 |]) in
  let q, up, _ = delta (fun () -> place p) in
  is_true ~msg:"the same value" (p == q);
  equal ~msg:"no upload" int 0 up;
  let h = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) p in
  is_true ~msg:"an unread output too" (place h == h)

let test_place_on_the_host_device () =
  let host = Nx.Placement.device (Rune.device "CPU") in
  let x = vec32 [| 1.0; 2.0; 3.0 |] in
  is_true ~msg:"a host value placed on the host is itself" (Nx.place host x == x);
  let t =
    Nx.matrix_transpose (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |])
  in
  is_true ~msg:"a strided one too" (Nx.place host t == t)

let test_place_is_the_identity_under_transformations () =
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  let loss place x =
    let x = place x in
    Nx.sum (Nx.mul x (Nx.mul x x))
  in
  let plain = loss Fun.id and placed = loss (fun x -> place x) in
  let g = Rune.grad' placed x in
  check_arr ~msg:"grad, eagerly" (to_arr (Rune.grad' plain x)) g;
  is_true ~msg:"the cotangent comes back to its primal's placement"
    (Nx.Placement.equal Nx.Placement.host (Nx.placement g));
  check_arr ~msg:"grad, compiled"
    (to_arr (Rune.grad' plain x))
    (Rune.jit' ~device:"CPU:1" (Rune.grad' placed) x);
  let tangent = vec32 [| 1.0; 1.0; 1.0 |] in
  check_arr ~msg:"jvp"
    (to_arr (snd (Rune.jvp' plain x tangent)))
    (snd (Rune.jvp' placed x tangent));
  let rows = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let double place x = Nx.mul_s (place x) 2.0 in
  check_arr ~msg:"vmap"
    (to_arr (Rune.vmap' (double Fun.id) rows))
    (Rune.vmap' (double (fun x -> place x)) rows);
  let (_ : Nx.float32_t), up, _ =
    delta (fun () ->
        Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s (place x) 2.0) x)
  in
  equal ~msg:"inside jit, placing where the program runs moves nothing" int 12
    up

(* Devices have one name and one value each; CPU is the host. *)
let test_one_device_per_name () =
  is_true ~msg:"CPU is the host" (Rune.device "CPU" == Nx.Device.host);
  is_true ~msg:"index 0 is the device itself"
    (Rune.device "cpu:0" == Nx.Device.host);
  let d = Rune.device "CPU:1" in
  is_true ~msg:"one value per name" (Rune.device "cpu:1" == d);
  is_true ~msg:"whatever the index's spelling" (Rune.device "CPU:01" == d);
  equal ~msg:"its name" string "CPU:1" (Nx.Device.name d);
  is_true ~msg:"the host backend's devices"
    (List.equal ( == ) [ Nx.Device.host ] (Rune.devices "CPU"));
  is_true ~msg:"DEV=CPU makes the host the default"
    (Rune.default_device () == Nx.Device.host);
  let invalid f =
    raises_match (function Invalid_argument _ -> true | _ -> false) f
  in
  invalid (fun () -> Rune.device "TPU");
  invalid (fun () -> Rune.device "CPU:-1");
  invalid (fun () -> Rune.devices "CPU:1")

(* A program on the host computes on host values: placing on the host inside it
   is the identity, and its outputs are on the host. *)
let test_host_program_is_on_the_host () =
  let x = vec32 [| 1.0; 2.0 |] in
  let f x = Nx.mul_s (Nx.place Nx.Placement.host x) 2.0 in
  let y = Rune.jit' ~device:"CPU" f x in
  check_arr ~msg:"placing on the host is the identity" [| 2.0; 4.0 |] y;
  is_true ~msg:"the output is on the host"
    (Nx.Placement.equal Nx.Placement.host (Nx.placement y));
  check_arr ~msg:"and so is its gradient's" [| 2.0; 2.0 |]
    (Rune.jit' ~device:"CPU" (Rune.grad' (fun x -> Nx.sum (f x))) x)

(* A compiled function runs where its placed inputs and captures live, and
   refuses a value placed elsewhere before it runs. *)
let invalid_starting prefix f =
  raises_match
    (function
      | Invalid_argument msg -> String.starts_with ~prefix msg | _ -> false)
    f

let test_runs_where_its_inputs_live () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let x = place (vec32 [| 1.0; 2.0 |]) in
  let y, up, _ = delta (fun () -> g x) in
  is_true ~msg:"the output is on the input's device"
    (Nx.Placement.equal (Nx.Placement.device cpu1) (Nx.placement y));
  equal ~msg:"nothing is uploaded" int 0 up;
  check_arr ~msg:"value" [| 2.0; 4.0 |] y;
  is_true ~msg:"a host input runs on the default device"
    (Nx.Placement.equal Nx.Placement.host
       (Nx.placement (g (vec32 [| 1.0; 2.0 |]))))

let test_leaves_elsewhere_raise () =
  let x = place (vec32 [| 1.0; 2.0 |]) in
  let y = Nx.place (Nx.Placement.device (Rune.device "CPU:2")) x in
  let (), up, _ =
    delta (fun () ->
        invalid_starting
          "Rune.jit: input leaf 0 is on CPU:1 and ~device names CPU:2"
          (fun () -> Rune.jit' ~device:"CPU:2" (fun x -> Nx.mul_s x 2.0) x);
        invalid_starting
          "Rune.jit: input leaf 0 is on CPU:1 and ~device names CPU" (fun () ->
            Rune.jit' ~device:"CPU" (fun x -> Nx.mul_s x 2.0) x);
        invalid_starting "Rune.jit: input leaves 0 and 1 are on CPU:1 and CPU:2"
          (fun () ->
            Rune.jit
              (module Pair)
              (fun p -> Nx.add p.Pair.u p.v)
              { Pair.u = x; v = y }))
  in
  equal ~msg:"nothing is uploaded" int 0 up

(* A capture decides where a function whose inputs are on the host runs, at the
   first trace that meets it; later calls run there without tracing. *)
let test_capture_decides_the_device () =
  let w = place (vec32 [| 1.0; 2.0 |]) in
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.mul x w)
  in
  let y = g (vec32 [| 3.0; 4.0 |]) in
  is_true ~msg:"the output is on the capture's device"
    (Nx.Placement.equal (Nx.Placement.device cpu1) (Nx.placement y));
  check_arr ~msg:"value" [| 3.0; 8.0 |] y;
  let traced = !traces in
  let y, up, _ = delta (fun () -> g (vec32 [| 1.0; 1.0 |])) in
  equal ~msg:"a later call does not trace" int traced !traces;
  equal ~msg:"and uploads only its input" int 8 up;
  check_arr ~msg:"value" [| 1.0; 2.0 |] y;
  invalid_starting
    "Rune.jit: a captured value is on CPU:1 and the program runs on CPU:2"
    (fun () ->
      Rune.jit' ~device:"CPU:2" (fun x -> Nx.mul x w) (vec32 [| 1.0; 1.0 |]))

(* A program that binds a capture makes its device the closure's: a later call
   from the host runs there with no new trace. *)
let test_capture_device_is_remembered () =
  let w = place (vec32 [| 1.0; 2.0 |]) in
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.mul x w)
  in
  check_arr ~msg:"a placed input" [| 3.0; 8.0 |]
    (g (place (vec32 [| 3.0; 4.0 |])));
  let traced = !traces in
  let y = g (vec32 [| 1.0; 1.0 |]) in
  check_arr ~msg:"a host input" [| 1.0; 2.0 |] y;
  equal ~msg:"shares the program" int traced !traces;
  is_true ~msg:"on the capture's device"
    (Nx.Placement.equal (Nx.Placement.device cpu1) (Nx.placement y))

(* Placed views. A view of part of a placed storage binds the storage it
   reaches, with no copy; a strided view is movement in the program. *)
let m34 () = place (Nx.create f32 [| 3; 4 |] (Array.init 12 float_of_int))

let check_bound ~msg f x =
  let expected = to_arr (f (Nx.place Nx.Placement.host x)) in
  let y, up, _ = delta (fun () -> Rune.jit' f x) in
  check_arr ~msg expected y;
  equal ~msg:(msg ^ ": nothing is uploaded") int 0 up

let test_views_bind_without_a_copy () =
  let p = m34 () in
  let f x = Nx.add_s (Nx.mul_s x 2.0) 1.0 in
  check_bound ~msg:"a C-order window" f (Nx.slice [ Nx.R (1, 3) ] p);
  check_bound ~msg:"a transpose" f (Nx.matrix_transpose p);
  check_bound ~msg:"a column cut" f (Nx.slice [ Nx.A; Nx.R (1, 3) ] p);
  check_bound ~msg:"a flip" f (Nx.flip ~axes:[ 1 ] p);
  check_bound ~msg:"a broadcast" f
    (Nx.broadcast_to [| 3; 2; 4 |] (Nx.slice [ Nx.R (1, 3) ] p));
  check_bound ~msg:"a reduction over a cut" Nx.sum
    (Nx.slice [ Nx.Rs (0, 3, 2); Nx.R (1, 4) ] p)

(* A C-order window shares the program of a value covering its storage, and two
   strided views whose offsets differ by a multiple of 16 bytes share one. *)
let test_views_share_programs () =
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.mul_s x 2.0)
  in
  let p = m34 () in
  let covering = place (Nx.zeros f32 [| 2; 4 |]) in
  let (_ : Nx.float32_t), up, _ = delta (fun () -> g covering) in
  equal ~msg:"a covering value uploads nothing" int 0 up;
  let n = !traces in
  let y, up, _ = delta (fun () -> g (Nx.slice [ Nx.R (1, 3) ] p)) in
  check_arr ~msg:"a window" [| 8.; 10.; 12.; 14.; 16.; 18.; 20.; 22. |] y;
  equal ~msg:"a window uploads nothing" int 0 up;
  equal ~msg:"shares the covering value's program" int n !traces;
  let wide = place (Nx.create f32 [| 3; 8 |] (Array.init 24 float_of_int)) in
  ignore (g (Nx.slice [ Nx.A; Nx.R (0, 2) ] wide));
  let n = !traces in
  let y, up, _ = delta (fun () -> g (Nx.slice [ Nx.A; Nx.R (4, 6) ] wide)) in
  check_arr ~msg:"another offset" [| 8.; 10.; 24.; 26.; 40.; 42. |] y;
  equal ~msg:"a strided view uploads nothing" int 0 up;
  equal ~msg:"shares the strided program" int n !traces

(* A range is bound from a 16-byte boundary and the program skips the elements
   before it, so windows whose offsets differ by four float32 share a program,
   and the others each have one. *)
let test_windows_bind_from_aligned_offsets () =
  let n = 1024 in
  let x = place (Nx.create f32 [| n + 4 |] (Array.init (n + 4) float_of_int)) in
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.add_s x 1.0)
  in
  for k = 0 to 4 do
    let y, up, _ = delta (fun () -> g (Nx.slice [ Nx.R (k, k + n) ] x)) in
    let msg = Printf.sprintf "offset %d" k in
    check_arr ~msg (Array.init n (fun i -> float_of_int (k + i + 1))) y;
    equal ~msg:(msg ^ ": nothing is uploaded") int 0 up
  done;
  equal ~msg:"offsets 0 and 4 share a program" int 4 !traces

(* An output that is a movement of an input is copied out of it: an unaligned
   window returned as it is, a slice of an input. *)
let test_views_of_inputs_as_outputs () =
  let x = place (Nx.create f32 [| 8 |] (Array.init 8 float_of_int)) in
  let w = Nx.slice [ Nx.R (1, 5) ] x in
  check_arr ~msg:"an unaligned window returned" [| 1.; 2.; 3.; 4. |]
    (Rune.jit' (fun v -> v) w);
  check_arr ~msg:"the window is still readable" [| 1.; 2.; 3.; 4. |] w;
  let slice v = Nx.slice [ Nx.R (1, 3) ] v in
  check_arr ~msg:"a slice of a host input" [| 1.; 2. |]
    (Rune.jit' slice (Nx.create f32 [| 4 |] [| 0.; 1.; 2.; 3. |]));
  check_arr ~msg:"on the host" [| 1.; 2. |]
    (Rune.jit' ~device:"CPU" slice (Nx.create f32 [| 4 |] [| 0.; 1.; 2.; 3. |]));
  let step = Rune.jit_step (module Nx.Ptree) (module Csingle) (fun _ v -> v) in
  check_arr ~msg:"a window of the state, read" [| 1.; 2.; 3.; 4. |]
    (step (Nx.Ptree.list [ Nx.Ptree.tensor x ]) w)

(* Views of one shape with other strides are other programs. *)
let test_strides_key_programs () =
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.mul_s x 1.0)
  in
  let a = place (Nx.create f32 [| 3; 4 |] (Array.init 12 float_of_int)) in
  let b = place (Nx.create f32 [| 4; 6 |] (Array.init 24 float_of_int)) in
  let cut = Nx.slice [ Nx.A; Nx.R (1, 4) ] b in
  List.iter
    (fun (msg, v) ->
      check_arr ~msg (to_arr (Nx.place Nx.Placement.host v)) (g v))
    [
      ("a transpose", Nx.matrix_transpose a);
      ("a column cut", cut);
      ("its flip", Nx.flip ~axes:[ 1 ] cut);
      ("a stepped cut", Nx.slice [ Nx.A; Nx.Rs (0, 6, 2) ] b);
    ];
  equal ~msg:"four programs" int 4 !traces

(* Overlapping windows do not nest: they are copied, and read correctly. *)
let test_overlapping_views_are_copied () =
  let p = place (Nx.create f32 [| 6 |] (Array.init 6 float_of_int)) in
  let w = Nx.sliding_window ~window:3 p in
  let y, up, _ = delta (fun () -> Rune.jit' (fun x -> Nx.mul_s x 2.0) w) in
  check_arr ~msg:"value"
    (to_arr
       (Nx.mul_s
          (Nx.sliding_window ~window:3 (Nx.place Nx.Placement.host p))
          2.0))
    y;
  is_true ~msg:"uploaded" (up > 0)

let test_captured_views_bind () =
  let p = m34 () in
  let w = Nx.matrix_transpose (Nx.slice [ Nx.R (1, 3) ] p) in
  let g = Rune.jit' (fun x -> Nx.matmul x w) in
  let x = Nx.ones f32 [| 1; 4 |] in
  let y, up, _ = delta (fun () -> g x) in
  check_arr ~msg:"value" (to_arr (Nx.matmul x (Nx.place Nx.Placement.host w))) y;
  equal ~msg:"only the input is uploaded" int (Nx.nbytes x) up;
  let (_ : Nx.float32_t), up, _ = delta (fun () -> g x) in
  equal ~msg:"again" int (Nx.nbytes x) up

(* Reads and moves keep a placed value where it is (RFC 0005, Laws 3 and 4). *)
let test_item_reads_one_element () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let y = g (Nx.create f32 [| 64; 64 |] (Array.init 4096 float_of_int)) in
  let v, up, down = delta (fun () -> Nx.item [ 1; 2 ] y) in
  equal ~msg:"the element" float_exact 132.0 v;
  equal ~msg:"one element moves" int 4 down;
  equal ~msg:"nothing is uploaded" int 0 up;
  is_true ~msg:"the value stays resident" (bound_by 0 y);
  let (_ : Nx.float32_t), up, _ = delta (fun () -> g y) in
  equal ~msg:"and feeds a call with no upload" int 0 up

let test_move_to_host_keeps_its_source () =
  let p = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let h = Nx.place Nx.Placement.host p in
  is_true ~msg:"a host copy"
    (Nx.Placement.equal Nx.Placement.host (Nx.placement h));
  check_arr ~msg:"its elements" [| 1.0; 2.0; 3.0 |] h;
  check_arr ~msg:"the source is still readable" [| 1.0; 2.0; 3.0 |] p;
  is_true ~msg:"and resident" (bound_by 0 p)

let test_mixed_placements_raise () =
  let p = place (vec32 [| 1.0; 2.0 |]) in
  let q =
    Rune.jit' ~device:"CPU:2" (fun x -> Nx.mul_s x 1.0) (vec32 [| 3.0; 4.0 |])
  in
  raises_match
    (function
      | Invalid_argument msg -> String.ends_with ~suffix:"place one of them" msg
      | _ -> false)
    (fun () -> Nx.add p q);
  check_arr ~msg:"a host operand joins" [| 2.0; 3.0 |]
    (Nx.add p (vec32 [| 1.0; 1.0 |]))

let test_placing_elsewhere_inside_jit_raises () =
  let other = Nx.Placement.device (Rune.device "CPU:2") in
  raises_jit_error (fun () ->
      Rune.jit' ~device:"CPU:1" (fun x -> Nx.place other x) (vec32 [| 1.0 |]))

(* Chunked transfers. A copy between host and device moves 64 MiB at a time, so
   these values are larger than that, with a last chunk shorter than the
   others. *)

let chunk = 64 * 1024 * 1024

(* [f] on CPU:1 against [f] eagerly: every element uploaded, computed and read
   back in its place. *)
let check_transfers ~msg f x =
  let (y : Nx.int32_t), up, _ =
    delta (fun () -> Rune.jit' ~device:"CPU:1" f x)
  in
  let worst, _, down =
    delta (fun () ->
        let y = Nx.place Nx.Placement.host y in
        Nx.item [] (Nx.max (Nx.abs (Nx.sub y (f x)))))
  in
  equal ~msg:(msg ^ ": difference from eager") int32 0l worst;
  is_true ~msg:(msg ^ ": larger than a chunk") (Nx.nbytes x > chunk);
  equal ~msg:(msg ^ ": bytes uploaded") int (Nx.nbytes x) up;
  equal ~msg:(msg ^ ": bytes read back") int (Nx.nbytes x) down

let test_chunked_contiguous () =
  let n = (chunk / 4) + 4099 in
  check_transfers ~msg:"contiguous"
    (fun x -> Nx.add_s x 1l)
    (Nx.arange Nx.int32 0 n 1)

let test_chunked_offset () =
  let n = (chunk / 4) + 4099 in
  let x = Nx.slice [ Nx.R (3, n + 3) ] (Nx.arange Nx.int32 0 (n + 5) 1) in
  is_true ~msg:"contiguous at an offset"
    (Nx.is_c_contiguous x && Nx.offset x = 3);
  check_transfers ~msg:"offset" (fun x -> Nx.add_s x 1l) x

let test_chunked_strided () =
  let rows = 4100 and cols = 4099 in
  let x =
    Nx.matrix_transpose
      (Nx.reshape [| rows; cols |] (Nx.arange Nx.int32 0 (rows * cols) 1))
  in
  is_true ~msg:"strided" (not (Nx.is_c_contiguous x));
  check_transfers ~msg:"strided" (fun x -> Nx.add_s x 1l) x

(* A strided value whose rows are themselves larger than a chunk. *)
let test_chunked_strided_rows () =
  let cols = (chunk / 4) + 4099 in
  let x =
    Nx.flip ~axes:[ 1 ]
      (Nx.reshape [| 2; cols |] (Nx.arange Nx.int32 0 (2 * cols) 1))
  in
  is_true ~msg:"strided" (not (Nx.is_c_contiguous x));
  check_transfers ~msg:"strided rows" (fun x -> Nx.add_s x 1l) x

let test_chunked_capture () =
  let n = (chunk / 4) + 4099 in
  let w =
    Nx.matrix_transpose (Nx.reshape [| 1; n |] (Nx.arange Nx.int32 0 n 1))
  in
  let f s = Nx.add (Nx.reshape [| n |] w) s in
  let s = Nx.create Nx.int32 [| 1 |] [| 5l |] in
  let worst = Nx.max (Nx.abs (Nx.sub (Rune.jit' ~device:"CPU:1" f s) (f s))) in
  equal ~msg:"capture: difference from eager" int32 0l (Nx.item [] worst)

let test_feedback_chain_moves_no_bytes () =
  let f x = Nx.add_s (Nx.mul_s x 2.0) 1.0 in
  let g = Rune.jit' ~device:"CPU:1" f in
  let x = vec32 [| 1.0; 2.0; 3.0 |] in
  let h1 = g x in
  let h2, up2, down2 = delta (fun () -> g h1) in
  let h3, up3, down3 = delta (fun () -> g h2) in
  equal ~msg:"feeding h1 back uploads nothing" int 0 up2;
  equal ~msg:"producing h2 downloads nothing" int 0 down2;
  equal ~msg:"feeding h2 back uploads nothing" int 0 up3;
  equal ~msg:"producing h3 downloads nothing" int 0 down3;
  check_arr ~msg:"h3 matches the eager composition" (to_arr (f (f (f x)))) h3;
  (* Handles from earlier calls keep their own storage (R3). *)
  check_arr ~msg:"h1 still readable" (to_arr (f x)) h1;
  check_arr ~msg:"h2 still readable" (to_arr (f (f x))) h2

let test_forced_handle_feeds_current_bytes () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let h = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"a read leaves the value placed" [| 2.0; 4.0; 6.0 |] h;
  (* A value an eager operation derives from it lives with it, and feeds a call
     with no upload. *)
  let h = Nx.set [ I 0 ] (Nx.scalar f32 10.0) h in
  let h2, up, _ = delta (fun () -> g h) in
  equal ~msg:"a derived value feeds with no upload" int 0 up;
  check_arr ~msg:"the new value is observed" [| 20.0; 8.0; 12.0 |] h2

let test_same_handle_as_two_leaves () =
  let g =
    Rune.jit2 ~device:"CPU:1"
      (module Pair)
      (module Pair)
      (fun p -> { u = Nx.add p.u p.v; v = Nx.mul p.u p.v })
  in
  let h =
    Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 3.0) (vec32 [| 1.0; 2.0 |])
  in
  let r, up, _ = delta (fun () -> g { u = h; v = h }) in
  equal ~msg:"resident duplicate leaves upload nothing" int 0 up;
  check_arr ~msg:"u" [| 6.0; 12.0 |] r.u;
  check_arr ~msg:"v" [| 9.0; 36.0 |] r.v

let test_duplicate_outputs_share_one_handle () =
  let g =
    Rune.jit2 ~device:"CPU:1"
      (module Pair)
      (module Pair)
      (fun p ->
        let y = Nx.add p.u p.v in
        { u = y; v = y })
  in
  let r = g { u = vec32 [| 1.0 |]; v = vec32 [| 2.0 |] } in
  is_true ~msg:"both leaves are one handle" (r.u == r.v);
  check_arr ~msg:"readable" [| 3.0 |] r.u;
  check_arr ~msg:"readable through the other leaf" [| 3.0 |] r.v

let test_cross_jit_feedback () =
  let g1 = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let g2 = Rune.jit' ~device:"CPU:1" (fun x -> Nx.add_s x 1.0) in
  let h = g1 (vec32 [| 1.0; 2.0 |]) in
  let r, up, _ = delta (fun () -> g2 h) in
  equal ~msg:"a distinct jitted closure seeds the handle too" int 0 up;
  check_arr ~msg:"value" [| 3.0; 5.0 |] r

let test_cross_signature_feedback () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.sum ~axes:[ 0 ] x) in
  let h1 = g (Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]) in
  (* h1 has a new shape: feeding it back compiles a second signature, still
     without forcing the handle. *)
  let h2, up, down = delta (fun () -> g h1) in
  equal ~msg:"the retrace uploads nothing" int 0 up;
  equal ~msg:"the retrace downloads nothing" int 0 down;
  check_arr ~msg:"value" [| 21.0 |] h2;
  check_arr ~msg:"h1 still readable" [| 5.0; 7.0; 9.0 |] h1

let test_pass_through_output_survives () =
  let g =
    Rune.jit2 ~device:"CPU:1"
      (module Pair)
      (module Pair)
      (fun p -> { u = p.u; v = Nx.mul_s p.v 2.0 })
  in
  let r1 = g { u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 3.0; 4.0 |] } in
  let r2 = g { u = vec32 [| 5.0; 6.0 |]; v = vec32 [| 7.0; 8.0 |] } in
  check_arr ~msg:"pass-through survives a later call" [| 1.0; 2.0 |] r1.u;
  check_arr ~msg:"first call's computed output" [| 6.0; 8.0 |] r1.v;
  check_arr ~msg:"second call's pass-through" [| 5.0; 6.0 |] r2.u;
  check_arr ~msg:"second call's computed output" [| 14.0; 16.0 |] r2.v

let test_grad_over_jit_with_deferred_arg () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x x) in
  let h = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  (* Under grad the jitted function runs eagerly; the handle forces on its first
     operation. *)
  let dx = Rune.grad' (fun x -> Nx.sum (g x)) h in
  check_arr ~msg:"gradient at the deferred point" [| 2.0; 8.0; 18.0 |] dx

let test_vmap_over_jit_with_deferred_arg () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let h = g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]) in
  let y = Rune.vmap' g h in
  check_arr ~msg:"vmap over jit at a deferred point" [| 4.0; 8.0; 12.0; 16.0 |]
    y

let test_dispatch_on_handle_reads_no_bytes () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let h = g (vec32 [| 1.0; 2.0 |]) in
  (* Signature dispatch uses only metadata: replaying on a handle must not force
     it. *)
  let _h2, _, down = delta (fun () -> g h) in
  equal ~msg:"dispatching on a handle downloads nothing" int 0 down

let test_capture_uploaded_once_across_signatures () =
  let n = 256 in
  let c = vec32 (Array.init n float_of_int) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.add x c) in
  let _, up1, _ = delta (fun () -> g (vec32 (Array.make n 0.0))) in
  (* A new input shape compiles a second signature; the capture's device copy is
     shared, so only the input is uploaded. *)
  let _, up2, _ =
    delta (fun () -> g (Nx.create f32 [| 1; n |] (Array.make n 1.0)))
  in
  equal ~msg:"first compile uploads input and capture" int (2 * n * 4) up1;
  equal ~msg:"second signature re-uploads only the input" int (n * 4) up2

let test_dropped_handles_are_reclaimed () =
  let n = 1024 in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let x = vec32 (Array.make n 1.0) in
  let base = (Rune.jit_stats ()).resident_bytes in
  let _, _, down =
    delta (fun () ->
        for _ = 1 to 50 do
          ignore (g x)
        done)
  in
  equal ~msg:"unread outputs download nothing" int 0 down;
  (* Collect the dropped handles; the next calls drain their buffers. *)
  full_major ();
  ignore (g x);
  full_major ();
  ignore (g x);
  let s = Rune.jit_stats () in
  is_true ~msg:"resident bytes are bounded after gc"
    (s.resident_bytes - base <= 3 * n * 4)

(* A call returns while its kernels may still run. A read waits for them: right
   after one call, and after a chain of unread calls. *)
let test_read_after_call_waits () =
  let n = 256 in
  let f x = Nx.add_s (Nx.matmul (Nx.tanh x) (Nx.transpose x)) 1.0 in
  let g = Rune.jit' ~device:"CPU:1" f in
  let x =
    Nx.create f32 [| n; n |]
      (Array.init (n * n) (fun i -> float_of_int (i mod 13) /. 13.0))
  in
  check_arr ~eps:1e-3 ~msg:"right after one call" (to_arr (f x)) (g x);
  let step =
    Rune.jit' ~device:"CPU:1" (fun x -> Nx.add_s (Nx.mul_s x 0.5) 1.0)
  in
  let h = ref (place (vec32 (Array.make 4096 0.0))) in
  for _ = 1 to 50 do
    h := step !h
  done;
  let expected = 2.0 -. (2.0 *. (0.5 ** 50.0)) in
  check_arr ~eps:1e-6 ~msg:"after fifty unread calls" (Array.make 4096 expected)
    !h

(* Arenas. A program's planned intermediates are slices of its device's shared
   arena, so programs run in turn hold one arena between them. *)

let device_bytes name =
  Option.value ~default:0
    (Hashtbl.find_opt Tolk.Helpers.Global_counters.mem_used_per_device name)

(* Not inlined: once it returns, only [g]'s binding refers to the window's range
   of storage, through a buffer view the call made. *)
let[@inline never] read_a_window g n =
  let x = place (Nx.create f32 [| n + 1 |] (Array.make (n + 1) 1.0)) in
  check_arr ~msg:"the window" (Array.make n 2.0)
    (g (Nx.slice [ Nx.R (1, n + 1) ] x))

(* A call releases the buffer views of its ranges once it has run, so a dropped
   value's storage returns to the device while the program that read a window of
   it lives. *)
let test_a_window's_view_is_released () =
  let n = 1 lsl 18 in
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  read_a_window g n;
  let before = device_bytes "CPU:1" in
  full_major ();
  is_true ~msg:"the window's storage is freed"
    (device_bytes "CPU:1" <= before - (n * 4));
  check_arr ~msg:"the program still runs" [| 2.0 |]
    (g (place (vec32 [| 1.0 |])))

(* An [n x n] intermediate from an [n x 8] input: its arena outweighs the input
   and output storage the program also allocates. *)
let arena_program ~n act =
  let f x =
    let a = act (Nx.matmul x (Nx.transpose x)) in
    Nx.sum ~axes:[ 1 ] (Nx.matmul a a)
  in
  let x k =
    Nx.create f32 [| n; 8 |]
      (Array.init (n * 8) (fun i -> sin (float_of_int ((k * i) + 1)) /. 4.0))
  in
  (f, Rune.jit' ~device:"CPU:1" f, x)

let test_programs_share_an_arena () =
  let n = 512 in
  let f, f', x = arena_program ~n Nx.tanh in
  let g, g', _ = arena_program ~n Nx.sin in
  check_arr ~eps:1e-2 ~msg:"first program" (to_arr (f (x 1))) (f' (x 1));
  let before = device_bytes "CPU:1" in
  check_arr ~eps:1e-2 ~msg:"second program" (to_arr (g (x 2))) (g' (x 2));
  is_true ~msg:"the second program allocates no arena of its own"
    (device_bytes "CPU:1" - before < n * n * 4);
  for k = 3 to 5 do
    check_arr ~eps:1e-2 ~msg:"the first after the second"
      (to_arr (f (x k)))
      (f' (x k));
    check_arr ~eps:1e-2 ~msg:"the second after the first"
      (to_arr (g (x (k + 3))))
      (g' (x (k + 3)))
  done

(* A larger arena grows the shared one; the programs bound to the old buffer
   move to the new one at their next call, and the old buffer is freed: the
   device grows by less than the new arena. *)
let test_a_grown_arena_frees_the_old () =
  let f, f', x = arena_program ~n:256 Nx.tanh in
  check_arr ~eps:1e-2 ~msg:"small" (to_arr (f (x 1))) (f' (x 1));
  let n = 1024 in
  let g, g', y = arena_program ~n Nx.sin in
  (* Collect the views earlier tests' programs left of the old buffer. *)
  full_major ();
  let before = device_bytes "CPU:1" in
  check_arr ~eps:1e-2 ~msg:"large" (to_arr (g (y 1))) (g' (y 1));
  is_true ~msg:"the outgrown buffer is freed"
    (device_bytes "CPU:1" - before < (n * n * 4) - (128 * 1024));
  check_arr ~eps:1e-2 ~msg:"small, on the grown arena"
    (to_arr (f (x 2)))
    (f' (x 2));
  check_arr ~eps:1e-2 ~msg:"large again" (to_arr (g (y 2))) (g' (y 2))

(* Not inlined: once it returns, only the queued kernels use the placed
   input. *)
let[@inline never] run_on_a_dropped_input g data n =
  g (place (Nx.create f32 [| n; n |] data))

(* A placed input dropped while the kernels that read it are queued returns to
   the system only once they have run: the first result is intact after a value
   of the same size is placed, which may take the dropped input's memory. *)
let test_buffer_freed_under_a_running_kernel () =
  let n = 384 in
  let data = Array.init (n * n) (fun i -> float_of_int (i mod 7) /. 7.0) in
  let f x = Nx.matmul (Nx.tanh (Nx.matmul x x)) x in
  let expected = to_arr (f (Nx.create f32 [| n; n |] data)) in
  let g = Rune.jit' ~device:"CPU:1" f in
  ignore (to_arr (g (Nx.create f32 [| n; n |] data)));
  let noise = Nx.create f32 [| n; n |] (Array.make (n * n) 1e9) in
  for _ = 1 to 4 do
    let y = run_on_a_dropped_input g data n in
    full_major ();
    let z = place noise in
    check_arr ~eps:1e-2 ~msg:"the first result" expected y;
    ignore (to_arr z)
  done

(* Traced values carry no storage. OCaml counts a bigarray's bytes towards the
   major collector's pace even when its pages are never touched, so a
   placeholder with a buffer costs a slice of major collection per traced
   operation: the first call of a 20b-parameter decoder step spent 35 s of its
   47 s there. *)
let test_traced_values_have_no_storage () =
  let is_traced (type a b) (x : (a, b) Nx.t) =
    match x with Nx_effect.Traced _ -> true | Host _ | Placed _ -> false
  in
  let seen = ref [] in
  let f x =
    let y = Nx.add_s (Nx.reshape [| 2; 2 |] x) 1.0 in
    let q, r = Nx.qr y in
    seen := List.map is_traced [ x; y; q; r ];
    Nx.matmul q r
  in
  let x = vec32 [| 1.0; 2.0; 3.0; 5.0 |] in
  let compiled = Rune.jit' f x in
  let traced = !seen in
  check_arr ~eps:1e-4 ~msg:"value" (to_arr (f x)) compiled;
  equal ~msg:"input, result, and both results of a two-result operation"
    (list bool) [ true; true; true; true ] traced

(* A traced value exists only inside its trace: leaked out of it, it neither
   runs eagerly nor enters another trace. *)
let test_leaked_traced_value_raises () =
  let leaked = ref None in
  let f x =
    let y = Nx.mul_s x 2.0 in
    leaked := Some y;
    y
  in
  ignore (Rune.jit' f (vec32 [| 1.0; 2.0 |]));
  let y = Option.get !leaked in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Nx.add y y);
  raises_jit_error (fun () ->
      Rune.jit' (fun x -> Nx.add x y) (vec32 [| 1.0; 2.0 |]))

(* Donation. [jit_step] consumes the resident leaves of its state: their device
   buffers return to the allocator once the call completes, so a state-to-state
   loop holds ~2 generations of device memory instead of one per call, without
   any GC. A donated handle raises on read; the first argument's leaves, host
   tensors, already-read handles, and written-back leaves are unaffected. *)

let raises_donated f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg ->
          msg
          = "this value was donated to a compiled call and no longer exists; \
             read or copy it before the call"
      | _ -> false)
    (fun () -> ignore (f ()))

(* One tensor as a tree. *)
let leaf (type a b) () : (module Nx.Ptree.S with type t = (a, b) Nx.t) =
  (module struct
    type t = (a, b) Nx.t

    let map (f : 'p 'q. ('p, 'q) Nx.t -> ('p, 'q) Nx.t) x = f x

    let map2 (f : 'p 'q. ('p, 'q) Nx.t -> ('p, 'q) Nx.t -> ('p, 'q) Nx.t) a b =
      f a b

    let iter (f : 'p 'q. ('p, 'q) Nx.t -> unit) x = f x
  end)

(* [f] compiled as a step that reads nothing and consumes its state. *)
let consume state f =
  Rune.jit_step ~device:"CPU:1"
    (module Nx.Ptree)
    state
    (fun _ x -> f x)
    (Nx.Ptree.list [])

let consume' f = consume (leaf ()) f

let test_donate_bounds_resident_memory () =
  let n = 4096 in
  let step d =
    let f x = Nx.add_s x 1.0 in
    if d then consume' f else Rune.jit' ~device:"CPU:1" f
  in
  let x = vec32 (Array.make n 0.0) in
  (* Every handle created here stays reachable; retiring the handles earlier
     tests dropped unread keeps their release out of the measured window. *)
  let hold = Array.make 10 x in
  let run g =
    full_major ();
    let base = (Rune.jit_stats ()).resident_bytes in
    let h = ref (g x) in
    for i = 0 to 9 do
      hold.(i) <- !h;
      h := g !h
    done;
    let r = (Rune.jit_stats ()).resident_bytes - base in
    (!h, r)
  in
  let h, grew = run (step true) in
  is_true ~msg:"donate holds at most two generations" (grew <= 2 * n * 4);
  check_arr ~msg:"donated chain computes the right value" (Array.make n 11.0) h;
  let h', grew' = run (step false) in
  is_true ~msg:"without donate every generation stays resident"
    (grew' >= 10 * n * 4);
  check_arr ~msg:"undonated chain still correct" (Array.make n 11.0) h'

(* Elision: a donated input whose output reads it at the same index hands its
   storage to the output, so a carry loop holds one generation, not two. *)
let test_donate_reuses_storage () =
  let n = 4096 in
  let step = consume' (fun x -> Nx.add_s x 1.0) in
  let x = vec32 (Array.make n 0.0) in
  full_major ();
  let base = (Rune.jit_stats ()).resident_bytes in
  let h = ref (step x) in
  let hold = Array.make 10 !h in
  for i = 0 to 9 do
    hold.(i) <- !h;
    h := step !h
  done;
  let grew = (Rune.jit_stats ()).resident_bytes - base in
  is_true ~msg:"one generation stays resident" (grew <= n * 4);
  check_arr ~msg:"the chain computes the right value" (Array.make n 11.0) !h

(* A path through a movement op reads the input at another index, so the output
   must not take its storage; a chain stays correct. *)
let test_donate_refuses_movement_path () =
  let f x = Nx.add x (Nx.transpose x) in
  let step = consume' f in
  let x = Nx.create f32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let e = ref x and h = ref x in
  for _ = 1 to 3 do
    e := f !e;
    h := step !h
  done;
  check_arr ~msg:"transposed chain" (to_arr (f !e)) (step !h)

(* An input read by a kernel that runs after the output's store keeps its own
   storage: here the reduction over [u] must see the old value. *)
let test_donate_refuses_later_reader () =
  let f (p : Pair.t) =
    {
      Pair.u = Nx.add_s p.u 1.0;
      v = Nx.add p.v (Nx.broadcast_to (Nx.shape p.v) (Nx.sum p.u));
    }
  in
  let step = consume (module Pair) f in
  let p =
    { Pair.u = vec32 [| 1.0; 2.0; 3.0 |]; v = vec32 [| 0.0; 0.0; 0.0 |] }
  in
  let e = ref p and h = ref p in
  for _ = 1 to 3 do
    e := f !e;
    h := step !h
  done;
  let e = f !e and h = step !h in
  check_arr ~msg:"u" (to_arr e.Pair.u) h.Pair.u;
  check_arr ~msg:"v sums the pre-update values" (to_arr e.Pair.v) h.Pair.v

(* A donated input returned unchanged moves its storage to the output. *)
let test_donate_moves_pass_through () =
  let step =
    consume
      (module Pair)
      (fun (p : Pair.t) -> { Pair.u = p.u; v = Nx.add_s p.v 1.0 })
  in
  let p =
    { Pair.u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 3.0; 4.0 |] } |> step |> step
  in
  let base = (Rune.jit_stats ()).resident_bytes in
  let r, up, _ = delta (fun () -> step p) in
  equal ~msg:"resident leaves upload nothing" int 0 up;
  is_true ~msg:"no fresh buffer for the pass-through"
    ((Rune.jit_stats ()).resident_bytes - base <= 0);
  check_arr ~msg:"pass-through value" [| 1.0; 2.0 |] r.Pair.u;
  check_arr ~msg:"updated value" [| 6.0; 7.0 |] r.Pair.v;
  raises_donated (fun () -> to_arr p.Pair.u)

(* An input both updated and returned unchanged keeps its storage for the
   pass-through: the update must not write over the value the pass-through
   copies out after the kernels ran. *)
let test_donate_keeps_pass_through_readable () =
  let step =
    consume
      (module Pair)
      (fun (p : Pair.t) -> { Pair.u = Nx.add_s p.u 1.0; v = p.u })
  in
  let p = { Pair.u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 0.0; 0.0 |] } in
  let r = step (step p) in
  check_arr ~msg:"updated" [| 3.0; 4.0 |] r.Pair.u;
  check_arr ~msg:"pass-through holds the value before the update" [| 2.0; 3.0 |]
    r.Pair.v

(* Every leaf an output derives from is reused, whatever its position among the
   inputs. *)
let test_donate_reuses_every_leaf () =
  let step =
    consume
      (module Pair)
      (fun (p : Pair.t) -> { Pair.u = Nx.add_s p.u 1.0; v = Nx.add_s p.v 2.0 })
  in
  let p = { Pair.u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 3.0; 4.0 |] } in
  let r1 = step p in
  let before = (Rune.jit_stats ()).reused_bytes in
  let r2 = step r1 in
  equal ~msg:"both leaves reused" int 16
    ((Rune.jit_stats ()).reused_bytes - before);
  check_arr ~msg:"u" [| 3.0; 4.0 |] r2.Pair.u;
  check_arr ~msg:"v" [| 7.0; 8.0 |] r2.Pair.v

(* A staged loop refuses reuse only for the leaves it touches. *)
let test_donate_reuses_beside_a_scan () =
  let step =
    consume
      (module Pair)
      (fun (p : Pair.t) -> { Pair.u = Nx.add_s p.u 1.0; v = snd (cumsum p.v) })
  in
  let p = { Pair.u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 1.0; 2.0 |] } in
  let r1 = step p in
  let before = (Rune.jit_stats ()).reused_bytes in
  let r2 = step r1 in
  is_true ~msg:"the leaf beside the loop is reused"
    ((Rune.jit_stats ()).reused_bytes - before >= 8);
  check_arr ~msg:"u" [| 3.0; 4.0 |] r2.Pair.u;
  check_arr ~msg:"v" [| 1.0; 4.0 |] r2.Pair.v

(* A fresh output never takes an input's buffer node: a resident input fed to a
   later call keeps its bytes whatever the outputs are. *)
let test_outputs_never_write_into_inputs () =
  let step =
    Rune.jit2 ~device:"CPU:1"
      (module Pair)
      (module Pair)
      (fun (p : Pair.t) -> { Pair.u = Nx.add_s p.u 1.0; v = p.u })
  in
  let p = { Pair.u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 5.0; 6.0 |] } in
  let r1 = step p in
  let r2 = step r1 in
  check_arr ~msg:"second call's output" [| 3.0; 4.0 |] r2.Pair.u;
  check_arr ~msg:"the first call's pass-through is intact" [| 1.0; 2.0 |]
    r1.Pair.v;
  check_arr ~msg:"the first call's output is intact" [| 2.0; 3.0 |] r1.Pair.u

(* The decode shape: a window write at a run-time position reuses the cache's
   storage across steps. *)
let test_donate_reuses_window_write () =
  let v = vec32 [| 9.0; 8.0 |] in
  let f { x; pos } =
    { x = Nx.set [ Nx.D (pos, 2) ] v x; pos = Nx.add_s pos 2l }
  in
  let step = consume (module Windowed) f in
  let s0 = { x = vec32 (Array.make 8 0.0); pos = pos_at 0 } in
  let s = ref (step s0) in
  full_major ();
  let base = (Rune.jit_stats ()).resident_bytes in
  for _ = 1 to 3 do
    s := step !s
  done;
  let grew = (Rune.jit_stats ()).resident_bytes - base in
  is_true ~msg:"the cache holds one generation" (grew <= 0);
  check_arr ~msg:"every window written"
    [| 9.0; 8.0; 9.0; 8.0; 9.0; 8.0; 9.0; 8.0 |]
    !s.x

(* The slot-pool write of a key-value cache: every slot takes a new row or keeps
   its old one, and the same program reads the written pool back through an
   index. The read follows the store, so the pool still reuses its storage. *)
type pool = { slots : Nx.float32_t; writer : Nx.int32_t; read : Nx.float32_t }

module Pool = struct
  type t = pool

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { slots; writer; read } =
    { slots = f slots; writer = f writer; read = f read }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      slots = f p.slots q.slots;
      writer = f p.writer q.writer;
      read = f p.read q.read;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { slots; writer; read } =
    f slots;
    f writer;
    f read
end

let test_donate_reuses_pool_read_after_write () =
  let n = 1024 in
  let rows = vec32 [| 10.0; 20.0; 30.0 |] in
  let window = Nx.create Nx.int32 [| 4 |] [| 5l; 2l; 7l; 0l |] in
  let f { slots; writer; read = _ } =
    let fresh = Nx.take ~axis:0 ~indices:(Nx.maximum_s writer 0l) rows in
    let slots = Nx.where (Nx.greater_equal_s writer 0l) fresh slots in
    { slots; writer; read = Nx.take ~axis:0 ~indices:window slots }
  in
  let step = consume (module Pool) f in
  let writer =
    Nx.create Nx.int32 [| n |]
      (Array.init n (fun i ->
           match i with 2 -> 0l | 5 -> 1l | 7 -> 2l | _ -> -1l))
  in
  let s =
    ref
      (step { slots = vec32 (Array.make n 1.0); writer; read = vec32 [| 0. |] })
  in
  let before = (Rune.jit_stats ()).reused_bytes in
  s := step !s;
  is_true ~msg:"the pool is written over its donated input"
    ((Rune.jit_stats ()).reused_bytes - before >= n * 4);
  check_arr ~msg:"the read sees the written pool"
    [| 20.0; 10.0; 30.0; 1.0 |]
    !s.read

(* Two compiled programs take turns on one donated state. Each keeps its own
   planned intermediates, so a program that parked an intermediate in storage
   the other still owns would corrupt the state here. *)
let test_donate_alternates_two_programs () =
  let n = 8 in
  let mix (p : Pair.t) =
    let a = Nx.tanh (Nx.matmul p.u p.v) in
    let scale = Nx.add_s (Nx.sum ~axes:[ 1 ] ~keepdims:true (Nx.abs a)) 1.0 in
    {
      Pair.u = Nx.add p.u (Nx.div a scale);
      v = Nx.sub p.v (Nx.mul_s (Nx.transpose a) 0.1);
    }
  in
  let fold (p : Pair.t) =
    let m = Nx.mean ~axes:[ 0 ] ~keepdims:true (Nx.matmul p.v p.u) in
    let u = Nx.mul_s (Nx.sin (Nx.add p.u m)) 0.5 in
    { Pair.u; v = Nx.add (Nx.mul_s p.v 0.9) (Nx.matmul u u) }
  in
  let mix' = consume (module Pair) mix in
  let fold' = consume (module Pair) fold in
  let init k =
    Nx.create f32 [| n; n |]
      (Array.init (n * n) (fun i -> sin (float_of_int ((k * i) + 1))))
  in
  let p = { Pair.u = init 3; v = init 7 } in
  let e = ref p and h = ref p in
  for i = 1 to 8 do
    let f, f' = if i mod 2 = 0 then (fold, fold') else (mix, mix') in
    e := f !e;
    h := f' !h
  done;
  check_arr ~eps:1e-4 ~msg:"u" (to_arr !e.Pair.u) !h.Pair.u;
  check_arr ~eps:1e-4 ~msg:"v" (to_arr !e.Pair.v) !h.Pair.v

(* The same pool written through the token-to-slot map: a scatter over the
   tokens. The output is a copy of the pool plus a store at loaded indices, and
   it takes the donated pool's storage, so the copy has nothing to move. *)
let scatter_pool ~donate =
  let n = 1024 in
  let rows = Nx.create f32 [| 4; 1 |] [| 10.0; 20.0; 30.0; 40.0 |] in
  let window = Nx.create Nx.int32 [| 4 |] [| 5l; 2l; 7l; 0l |] in
  let f { slots; writer; read = _ } =
    let slots =
      Nx.scatter ~axis:0
        ~indices:(Nx.reshape [| 4; 1 |] writer)
        ~values:rows
        (Nx.reshape [| n; 1 |] slots)
    in
    let slots = Nx.reshape [| n |] slots in
    { slots; writer; read = Nx.take ~axis:0 ~indices:window slots }
  in
  let step =
    if donate then consume (module Pool) f
    else Rune.jit2 ~device:"CPU:1" (module Pool) (module Pool) f
  in
  (* Tokens 1 and 3 aim at slot 5: the later one wins. Token 2 has no slot. *)
  let writer = Nx.create Nx.int32 [| 4 |] [| 2l; 5l; -1l; 5l |] in
  let first =
    { slots = vec32 (Array.make n 1.0); writer; read = vec32 [| 0. |] }
  in
  (n, step, first)

let test_donate_reuses_pool_scatter () =
  let n, step, first = scatter_pool ~donate:true in
  let s = ref (step first) in
  let before = (Rune.jit_stats ()).reused_bytes in
  s := step !s;
  is_true ~msg:"the pool is written over its donated input"
    ((Rune.jit_stats ()).reused_bytes - before >= n * 4);
  check_arr ~msg:"the read sees the written pool" [| 40.0; 10.0; 1.0; 1.0 |]
    !s.read;
  let slots = to_arr !s.slots in
  equal ~msg:"an unwritten slot keeps its row" float_exact 1.0 slots.(9)

let test_scatter_without_donation_keeps_the_input () =
  let _, step, first = scatter_pool ~donate:false in
  let s1 = step first in
  let s2 = step s1 in
  check_arr ~msg:"second call" [| 40.0; 10.0; 1.0; 1.0 |] s2.read;
  equal ~msg:"the first call's pool is intact" float_exact 40.0
    (to_arr s1.slots).(5)

(* Bytes of donated storage the second of two donated steps reuses. *)
let reused_by_second_step step x =
  let y = step x in
  let before = (Rune.jit_stats ()).reused_bytes in
  let z = step y in
  (z, (Rune.jit_stats ()).reused_bytes - before)

(* The values written are read from the donated pool by a kernel that runs
   before the write: a reader of the old value in time, so the output still
   takes the pool's storage. *)
let test_scatter_of_values_read_from_the_pool () =
  let n = 8 in
  let indices = Nx.create Nx.int32 [| 2 |] [| 0l; 1l |] in
  let f x =
    let values = Nx.mul_s (Nx.slice [ Nx.R (6, 8) ] (Nx.flip x)) 10.0 in
    Nx.scatter ~axis:0 ~indices ~values x
  in
  let step = consume' f in
  let x = vec32 (Array.init n float_of_int) in
  let expected = to_arr (f (f x)) in
  let z, reused = reused_by_second_step step x in
  check_arr ~msg:"two donated steps" expected z;
  equal ~msg:"the pool's storage is reused" int (n * 4) reused

(* The donated pool is itself the values: the kernel would read through the
   storage it writes, so the output must not take it. *)
let test_scatter_of_the_pool_into_itself () =
  let indices = Nx.create Nx.int32 [| 4 |] [| 3l; 2l; 1l; 0l |] in
  let f x = Nx.scatter ~axis:0 ~indices ~values:x x in
  let step = consume' f in
  let x = vec32 [| 1.0; 2.0; 3.0; 4.0 |] in
  check_arr ~msg:"reversed" [| 4.0; 3.0; 2.0; 1.0 |] (step (step (step x)));
  let _, reused = reused_by_second_step step x in
  equal ~msg:"the pool's storage is not reused" int 0 reused

(* A kernel reads the old pool and the written one together, so it runs after
   the write: the output must not take the pool's storage, and the reader still
   sees the old value. *)
let test_scatter_refuses_a_later_reader_of_the_pool () =
  let indices = Nx.create Nx.int32 [| 2 |] [| 1l; 3l |] in
  let values = vec32 [| 50.0; 70.0 |] in
  let f (p : Pair.t) =
    let u = Nx.scatter ~axis:0 ~indices ~values p.u in
    { Pair.u; v = Nx.add (Nx.flip p.u) u }
  in
  let step = consume (module Pair) f in
  let p =
    { Pair.u = vec32 [| 1.0; 2.0; 3.0; 4.0 |]; v = vec32 (Array.make 4 0.) }
  in
  let e = f (f p) in
  let r1 = step p in
  let before = (Rune.jit_stats ()).reused_bytes in
  let r2 = step r1 in
  equal ~msg:"only the leaf the step never reads lends its storage" int 16
    ((Rune.jit_stats ()).reused_bytes - before);
  check_arr ~msg:"written" (to_arr e.Pair.u) r2.Pair.u;
  check_arr ~msg:"old value read" (to_arr e.Pair.v) r2.Pair.v

(* A reader of the old pool scheduled with the write: the output keeps the new
   value and the reader the old one, whether or not storage was reused. *)
let test_scatter_beside_a_reader_of_the_old_value () =
  let indices = Nx.create Nx.int32 [| 2 |] [| 1l; 3l |] in
  let values = vec32 [| 50.0; 70.0 |] in
  let f (p : Pair.t) =
    {
      Pair.u = Nx.scatter ~axis:0 ~indices ~values p.u;
      v = Nx.add (Nx.flip p.u) p.v;
    }
  in
  let step = consume (module Pair) f in
  let p () =
    { Pair.u = vec32 [| 1.0; 2.0; 3.0; 4.0 |]; v = vec32 (Array.make 4 0.) }
  in
  let e = f (f (p ())) in
  let r = step (step (p ())) in
  check_arr ~msg:"written" (to_arr e.Pair.u) r.Pair.u;
  check_arr ~msg:"old value read" (to_arr e.Pair.v) r.Pair.v

let test_place_then_donate_consumes () =
  let g = consume' (fun x -> Nx.mul_s x 2.0) in
  ignore (g (vec32 [| 0.0; 0.0 |]));
  let p = place (vec32 [| 1.0; 2.0 |]) in
  check_arr ~msg:"result" [| 2.0; 4.0 |] (g p);
  raises_donated (fun () -> to_arr p)

(* File-backed sources. An upload reads a tensor over a mapped file from the
   file itself. *)

(* An int32 tensor of [n] elements over a fresh mapping of a file whose bytes
   are [byte i] at offset [i], with the file's path. *)
let mapped_int32 ~byte n =
  let path = Filename.temp_file "rune_mapped_" ".bin" in
  let oc = open_out_bin path in
  let piece = 1 lsl 20 in
  let bytes = Bytes.create piece in
  let written = ref 0 in
  while !written < 4 * n do
    let len = Int.min piece ((4 * n) - !written) in
    for i = 0 to len - 1 do
      Bytes.unsafe_set bytes i (byte (!written + i))
    done;
    Stdlib.output oc bytes 0 len;
    written := !written + len
  done;
  close_out oc;
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let stat = Unix.fstat fd in
  let mapping =
    Nx_buffer.of_bigarray1
      (Bigarray.array1_of_genarray
         (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
            [| -1 |]))
  in
  Unix.close fd;
  Nx_buffer.register_file
    { path; size = 4 * n; mtime = stat.st_mtime; inode = stat.st_ino }
    mapping;
  ( Nx.of_buffer (Nx_buffer.reinterpret Nx_buffer.Int32 mapping) ~shape:[| n |],
    path )

let first_byte i = Char.chr (i * 7 land 0xff)
let other_byte i = Char.chr (i * 13 land 0xff)

let remove_mapped path =
  full_major ();
  try Sys.remove path with Sys_error _ when Sys.win32 -> ()

(* [x] placed from its file equals [x] placed from memory. *)
let check_placed_from_file ~msg x =
  is_true
    ~msg:(msg ^ ": over a mapped file")
    (Nx_buffer.file_range (Nx.data x) <> None);
  let from_memory = place (Nx.copy x) in
  let from_file, up, _ = delta (fun () -> place x) in
  equal ~msg:(msg ^ ": bytes uploaded") int (Nx.nbytes x) up;
  let differing =
    Nx.item [] (Nx.sum (Nx.cast Nx.int32 (Nx.not_equal from_file from_memory)))
  in
  equal ~msg:(msg ^ ": elements differing") int32 0l differing

let test_file_backed_upload () =
  let n = (chunk / 4) + 4099 in
  let x, path = mapped_int32 ~byte:first_byte n in
  Fun.protect
    ~finally:(fun () -> remove_mapped path)
    (fun () ->
      check_placed_from_file ~msg:"contiguous" x;
      check_placed_from_file ~msg:"offset" (Nx.slice [ Nx.R (3, n - 5) ] x);
      let rows = 4100 in
      let cols = n / rows in
      let m =
        Nx.reshape [| rows; cols |] (Nx.slice [ Nx.R (0, rows * cols) ] x)
      in
      check_placed_from_file ~msg:"transposed" (Nx.matrix_transpose m))

(* The path names another file by now: the upload must not read it. *)
let test_file_backed_upload_after_replace () =
  let n = 1 lsl 16 in
  let x, path = mapped_int32 ~byte:first_byte n in
  Fun.protect
    ~finally:(fun () -> remove_mapped path)
    (fun () ->
      let expected = Nx.copy x in
      let _, replacement = mapped_int32 ~byte:other_byte n in
      Unix.rename replacement path;
      let differing placed =
        Nx.item [] (Nx.sum (Nx.cast Nx.int32 (Nx.not_equal placed expected)))
      in
      equal ~msg:"contiguous: the mapped bytes" int32 0l (differing (place x));
      let m = Nx.matrix_transpose (Nx.reshape [| 256; 256 |] x) in
      equal ~msg:"transposed: the mapped bytes" int32 0l
        (Nx.item []
           (Nx.sum
              (Nx.cast Nx.int32
                 (Nx.not_equal (place m)
                    (Nx.matrix_transpose (Nx.reshape [| 256; 256 |] expected)))))))

(* Bound captures. A compiled function that captures a resident value on its own
   device reads that value's buffer as its constant. *)

let test_bound_capture_moves_no_bytes () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  let x = place (vec32 [| 2.0; 2.0; 2.0 |]) in
  let y, up, down = delta (fun () -> g x) in
  equal ~msg:"compiling and calling uploads nothing" int 0 up;
  equal ~msg:"and reads nothing back" int 0 down;
  check_arr ~msg:"result" [| 2.0; 4.0; 6.0 |] y;
  (* A second signature of the same closure binds the same buffer. *)
  let m = place (Nx.create f32 [| 2; 3 |] (Array.make 6 3.0)) in
  let y2, up, _ = delta (fun () -> g m) in
  equal ~msg:"a second signature uploads nothing" int 0 up;
  check_arr ~msg:"second signature" [| 3.0; 6.0; 9.0; 3.0; 6.0; 9.0 |] y2

let test_bound_capture_is_shared () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g1 = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  let g2 = Rune.jit' ~device:"CPU:1" (fun x -> Nx.add x w) in
  let x = vec32 [| 2.0; 2.0; 2.0 |] in
  let (), up, _ =
    delta (fun () ->
        check_arr ~msg:"first function" [| 2.0; 4.0; 6.0 |] (g1 x);
        check_arr ~msg:"second function" [| 3.0; 4.0; 5.0 |] (g2 x))
  in
  equal ~msg:"only the inputs are uploaded" int 24 up;
  equal ~msg:"one storage, bound by both functions" int 2 (cell_of w).bound;
  ignore (Sys.opaque_identity (g1, g2, w))

let test_bound_value_survives_a_read () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  let x = vec32 [| 2.0; 2.0; 2.0 |] in
  check_arr ~msg:"before the read" [| 2.0; 4.0; 6.0 |] (g x);
  let (), _, down =
    delta (fun () -> check_arr ~msg:"value" [| 1.0; 2.0; 3.0 |] w)
  in
  equal ~msg:"the read copies the value out" int 12 down;
  is_true ~msg:"and leaves the storage bound" (bound_by 1 w);
  let (), _, down =
    delta (fun () -> check_arr ~msg:"value again" [| 1.0; 2.0; 3.0 |] w)
  in
  equal ~msg:"a second read copies again: nothing is memoised" int 12 down;
  check_arr ~msg:"after the read" [| 2.0; 4.0; 6.0 |] (g x);
  (* As an input leaf it still seeds from its buffer. *)
  let double = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  ignore (double x);
  let y, up, _ = delta (fun () -> double w) in
  equal ~msg:"a read bound value feeds an input with no transfer" int 0 up;
  check_arr ~msg:"as an input" [| 2.0; 4.0; 6.0 |] y

let test_unbound_value_stays_after_a_read () =
  full_major ();
  let base = resident () in
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"value" [| 1.0; 2.0; 3.0 |] w;
  equal ~msg:"the read left the buffer" int 12 (resident () - base);
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  let x = place (vec32 [| 2.0; 2.0; 2.0 |]) in
  let y, up, _ = delta (fun () -> g x) in
  equal ~msg:"captured afterwards it is bound" int 0 up;
  check_arr ~msg:"result" [| 2.0; 4.0; 6.0 |] y

let test_bound_input_is_not_donated () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  ignore (g (vec32 [| 0.0; 0.0; 0.0 |]));
  let step = consume' (fun x -> Nx.mul_s x 2.0) in
  let y, up, _ = delta (fun () -> step w) in
  equal ~msg:"the bound input seeds with no transfer" int 0 up;
  check_arr ~msg:"result" [| 2.0; 4.0; 6.0 |] y;
  check_arr ~msg:"the bound value is still readable" [| 1.0; 2.0; 3.0 |] w;
  check_arr ~msg:"and still the constant" [| 2.0; 4.0; 6.0 |]
    (g (vec32 [| 2.0; 2.0; 2.0 |]));
  (* Returned unchanged, it comes back as a copy on the device. *)
  let pass = consume' (fun x -> x) in
  let z, up, _ = delta (fun () -> pass w) in
  equal ~msg:"the pass-through uploads nothing" int 0 up;
  is_true ~msg:"the pass-through is another value" (z != w);
  is_true ~msg:"with storage of its own" (cell_of z != cell_of w);
  check_arr ~msg:"pass-through value" [| 1.0; 2.0; 3.0 |] z;
  check_arr ~msg:"the constant after the pass-through" [| 2.0; 4.0; 6.0 |]
    (g (vec32 [| 2.0; 2.0; 2.0 |]))

let test_bound_capture_returned_is_a_copy () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g = consume' (fun (_ : Nx.float32_t) -> w) in
  let y = g (vec32 [| 0.0 |]) in
  is_true ~msg:"another value" (y != w);
  check_arr ~msg:"the copy" [| 1.0; 2.0; 3.0 |] y;
  check_arr ~msg:"a second call" [| 1.0; 2.0; 3.0 |] (g (vec32 [| 0.0 |]))

(* Not inlined: once it returns, nothing but the collector's own bookkeeping
   refers to the placed value or to the function that bound it. It returns a
   weak pointer to the value's cell, whose finaliser releases the storage. *)
let[@inline never] bind_and_drop () =
  let w = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul x w) in
  check_arr ~msg:"result" [| 2.0; 4.0; 6.0 |] (g (vec32 [| 2.0; 2.0; 2.0 |]));
  let cell = Weak.create 1 in
  Weak.set cell 0 (Some (cell_of w));
  cell

let test_bound_buffer_is_released_with_its_owners () =
  let cell = bind_and_drop () in
  let before = resident () in
  full_major ();
  is_true ~msg:"the cell is collected once its owners are"
    (Option.is_none (Weak.get cell 0));
  is_true ~msg:"and its storage released" (resident () <= before - 12)

let with_budget bytes f =
  Unix.putenv "RUNE_JIT_RESIDENT_BUDGET" (string_of_int bytes);
  Fun.protect ~finally:(fun () -> Unix.putenv "RUNE_JIT_RESIDENT_BUDGET" "") f

let majors () = (Gc.quick_stat ()).major_collections

(* The collection budget counts every device allocation since the last major
   collection: eager results count as outputs do, and weights placed before a
   collection weigh on none after it. *)
let test_budget_counts_every_allocation () =
  let w = place (Nx.create f32 [| 1024 |] (Array.make 1024 1.0)) in
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  ignore (g (vec32 [| 1.0 |]));
  with_budget 1024 (fun () ->
      let before = majors () in
      for _ = 1 to 20 do
        ignore (Nx.mul_s w 2.0)
      done;
      is_true ~msg:"each 4 KiB eager result past a 1 KiB budget collects"
        (majors () - before >= 20);
      let before = majors () in
      for _ = 1 to 100 do
        ignore (to_arr (g (vec32 [| 1.0 |])))
      done;
      is_true ~msg:"4-byte outputs collect about every 256 calls"
        (majors () - before < 50));
  ignore (Sys.opaque_identity w)

let test_out_of_memory () =
  let huge = Nx.broadcast_to [| 1 lsl 48 |] (Nx.scalar f32 1.0) in
  raises_match
    (function Nx.Device.Out_of_memory (_, n) -> n = 4 lsl 48 | _ -> false)
    (fun () -> place huge)

let test_donated_handle_raises_on_read () =
  let g = consume' (fun x -> Nx.mul_s x 2.0) in
  let h1 = g (vec32 [| 1.0; 2.0 |]) in
  let h2 = g h1 in
  (* h1 was donated to the second call: its storage is gone. *)
  raises_donated (fun () -> to_arr h1);
  check_arr ~msg:"the consuming call's output is fine" [| 4.0; 8.0 |] h2

let test_donated_handle_refeed_raises () =
  let g = consume' (fun x -> Nx.mul_s x 2.0) in
  let h1 = g (vec32 [| 1.0; 2.0 |]) in
  ignore (g h1);
  (* Seeding a donated handle forces it, which raises the same error. *)
  raises_donated (fun () -> g h1)

let test_donate_duplicate_leaves_once () =
  let g =
    consume (module Pair) (fun p -> { u = Nx.add p.u p.v; v = Nx.mul p.u p.v })
  in
  let h =
    Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 3.0) (vec32 [| 1.0; 2.0 |])
  in
  let base = (Rune.jit_stats ()).resident_bytes in
  let r = g { u = h; v = h } in
  (* One handle behind two leaves donates once; only the two fresh outputs
     remain resident. *)
  is_true ~msg:"the duplicate handle was released once"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * 2 * 4);
  check_arr ~msg:"u" [| 6.0; 12.0 |] r.u;
  check_arr ~msg:"v" [| 9.0; 36.0 |] r.v;
  raises_donated (fun () -> to_arr h)

let test_read_value_is_still_donated () =
  let g = consume' (fun x -> Nx.mul_s x 2.0) in
  let h = g (vec32 [| 1.0; 2.0 |]) in
  check_arr ~msg:"a read before the call" [| 2.0; 4.0 |] h;
  let y = g h in
  (* A read moved nothing, so the value is still resident and donated. *)
  raises_donated (fun () -> to_arr h);
  check_arr ~msg:"the result" [| 4.0; 8.0 |] y

let test_host_input_unaffected_by_donate () =
  let g = consume' (fun x -> Nx.mul_s x 2.0) in
  let x = vec32 [| 1.0; 2.0 |] in
  ignore (g x);
  check_arr ~msg:"a host tensor is never consumed" [| 1.0; 2.0 |] x

let test_jit_leaves_handle_readable () =
  let g = Rune.jit' ~device:"CPU:1" (fun x -> Nx.mul_s x 2.0) in
  let h1 = g (vec32 [| 1.0; 2.0 |]) in
  ignore (g h1);
  check_arr ~msg:"jit keeps the input handle alive" [| 2.0; 4.0 |] h1

(* A step reads its first argument: a resident leaf there is used in place, call
   after call, and stays readable. *)
let test_step_reads_its_first_argument () =
  let step =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun w x -> Nx.add (Nx.mul w x) w)
  in
  let w = place (vec32 [| 1.0; 2.0 |]) in
  let x = place (vec32 [| 3.0; 4.0 |]) in
  let y = step w x in
  let z, up, _ = delta (fun () -> step w y) in
  equal ~msg:"resident leaves upload nothing" int 0 up;
  check_arr ~msg:"two steps" [| 5.0; 22.0 |] z;
  raises_donated (fun () -> to_arr x);
  raises_donated (fun () -> to_arr y);
  check_arr ~msg:"the read leaf is readable" [| 1.0; 2.0 |] w

(* The next state takes the storage of the state leaf at its own position; a
   read leaf lends none, even to an output that derives from it alone. *)
let test_step_reuses_only_the_state () =
  let w = place (vec32 [| 1.0; 2.0 |]) in
  let add =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun w x -> Nx.add x w)
  in
  let x, reused =
    reused_by_second_step (add w) (place (vec32 [| 0.0; 0.0 |]))
  in
  equal ~msg:"the state's storage is reused" int 8 reused;
  check_arr ~msg:"value" [| 2.0; 4.0 |] x;
  let double =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun w _ -> Nx.mul_s w 2.0)
  in
  let y, reused =
    reused_by_second_step (double w) (place (vec32 [| 0.0; 0.0 |]))
  in
  equal ~msg:"the state lends its storage, the read leaf none" int 8 reused;
  check_arr ~msg:"value" [| 2.0; 4.0 |] y;
  check_arr ~msg:"the read leaf is intact" [| 1.0; 2.0 |] w

(* A handle passed as both arguments is read: the call consumes nothing, lends
   nothing, and the handle stays usable. *)
let test_step_reads_a_handle_in_both_arguments () =
  let step =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun w x -> Nx.add w x)
  in
  let h = place (vec32 [| 1.0; 2.0 |]) in
  let before = (Rune.jit_stats ()).reused_bytes in
  let y = step h h in
  let y' = step h h in
  equal ~msg:"it lends nothing" int 0 ((Rune.jit_stats ()).reused_bytes - before);
  check_arr ~msg:"first" [| 2.0; 4.0 |] y;
  check_arr ~msg:"second" [| 2.0; 4.0 |] y';
  check_arr ~msg:"the handle is readable" [| 1.0; 2.0 |] h

(* Only a value whose view covers its storage can be donated: a state leaf that
   is a view of part of one raises before the call, and the value stays. *)
let test_step_refuses_a_partial_view () =
  let step = consume' (fun x -> Nx.mul_s x 2.0) in
  let w = place (vec32 [| 1.0; 2.0; 3.0; 4.0 |]) in
  let part = Nx.slice [ Nx.R (0, 2) ] w in
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with ~prefix:"Rune.jit_step: state leaf 0" msg
      | _ -> false)
    (fun () -> step part);
  check_arr ~msg:"the value stays" [| 1.0; 2.0 |] part;
  check_arr ~msg:"and so does its storage" [| 1.0; 2.0; 3.0; 4.0 |] w

(* A storage both arguments reach is read, even when the read argument reaches
   it through a view of part of it. *)
let test_step_reads_a_storage_both_arguments_reach () =
  let step =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun r s -> Nx.add s (Nx.sum r))
  in
  let w = place (vec32 [| 1.0; 2.0; 3.0; 4.0 |]) in
  let view = Nx.slice [ Nx.R (0, 2) ] w in
  let y = step view w in
  check_arr ~msg:"result" [| 4.0; 5.0; 6.0; 7.0 |] y;
  check_arr ~msg:"the read view is readable" [| 1.0; 2.0 |] view;
  check_arr ~msg:"and so is the state" [| 1.0; 2.0; 3.0; 4.0 |] w

(* An indexed write into a read leaf lands in fresh storage: the leaf keeps its
   value. *)
let test_step_write_into_a_read_leaf () =
  let indices = Nx.create Nx.int32 [| 2 |] [| 0l; 2l |] in
  let step =
    Rune.jit_step ~device:"CPU:1"
      (module Csingle)
      (module Csingle)
      (fun pool x -> Nx.scatter ~axis:0 ~indices ~values:x pool)
  in
  let pool = place (vec32 [| 1.0; 2.0; 3.0 |]) in
  let a = step pool (vec32 [| 10.0; 30.0 |]) in
  let b = step pool (vec32 [| 40.0; 60.0 |]) in
  check_arr ~msg:"first write" [| 10.0; 2.0; 30.0 |] a;
  check_arr ~msg:"second write" [| 40.0; 2.0; 60.0 |] b;
  check_arr ~msg:"the pool keeps its value" [| 1.0; 2.0; 3.0 |] pool

(* One tensor behind both leaves on the tracing call: two inputs that happen to
   be equal, each bound to its own position, so a later call may pass distinct
   tensors to them. *)
let test_aliased_input_leaves () =
  let f (p : pair) = Nx.sub p.u (Nx.mul_s p.v 2.0) in
  let g = Rune.jit (module Pair) f in
  let x = vec32 [| 1.0; 2.0; 3.0 |] in
  check_arr ~msg:"aliased call" [| -1.0; -2.0; -3.0 |] (g { u = x; v = x });
  check_arr ~msg:"distinct call" [| -7.0; -8.0; -9.0 |]
    (g { u = x; v = vec32 [| 4.0; 5.0; 6.0 |] });
  (* Under grad inside jit the two leaves are separate parameters, as
     eagerly. *)
  let dg =
    Rune.jit2
      (module Pair)
      (module Pair)
      (fun p -> Rune.grad (module Pair) (fun p -> Nx.sum (f p)) p)
      { u = x; v = x }
  in
  check_arr ~msg:"d/du" [| 1.0; 1.0; 1.0 |] dg.u;
  check_arr ~msg:"d/dv" [| -2.0; -2.0; -2.0 |] dg.v

(* Failure modes *)

let test_data_dependent_read_raises () =
  let g = Rune.jit' (fun x -> if Nx.item [ 0 ] x > 0.0 then x else Nx.neg x) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0 |]))

let test_unsupported_op_raises () =
  let g = Rune.jit' (fun x -> Nx.eigvals x) in
  raises_jit_error (fun () ->
      g (Nx.create f32 [| 2; 2 |] [| 4.0; 2.0; 2.0; 3.0 |]))

let tests =
  [
    group "jit basics"
      [
        test "element-wise chain matches eager" test_elementwise_matches_eager;
        test "replay reads fresh input data" test_replay_reads_fresh_inputs;
        test "a new shape retraces" test_retrace_on_new_shape;
        test "zero-size outputs are empty tensors" test_zero_size_outputs;
        test "closure-captured weights (matmul)" test_closure_matmul;
        test "jit2 returns structured outputs" test_jit2_structured_output;
        test "aliased input leaves are separate inputs"
          test_aliased_input_leaves;
      ];
    group "composition"
      [
        test "grad inside jit matches eager grad" test_grad_inside_jit;
        test "jit under grad runs eagerly" test_jit_under_grad_is_transparent;
        test "jit under vmap runs eagerly" test_jit_under_vmap_is_transparent;
        test "scan matches eager" test_scan_matches_eager;
        test "grad through a scan matches eager"
          test_grad_through_scan_matches_eager;
        test "grad through a scan, stacked outputs only"
          test_grad_through_scan_ys_only;
        test "grad through a scan, final carry only"
          test_grad_through_scan_carry_only;
        test "grad through a scan with a multi-leaf carry"
          test_grad_through_scan_multi_leaf;
        test "grad through a scan with an asymmetric pair carry"
          test_grad_through_scan_asymmetric_pair;
        test "shape-unstable carry unrolls instead of staging"
          test_scan_shape_unstable_carry_unrolls;
        test "grad through nested scans" test_grad_through_scan_nested;
        test "grad through a scan with a captured weight"
          test_grad_through_scan_captured_weight;
        test "grad through a scan with a vector carry"
          test_grad_through_scan_vector_carry;
        test "grad through a scan with an external input"
          test_grad_through_scan_external_input;
        test "grad through a scan with external matrices"
          test_grad_through_scan_external_matrices;
        test "grad through a scan with a matrix carry"
          test_grad_through_scan_matrix_carry;
        test "scan rows short of 16 bytes" test_scan_rows_short_of_16_bytes;
        test "a scan over structured rows" test_scan_over_structured_rows;
        test "a scan rejects ragged or scalar rows"
          test_scan_rejects_ragged_rows;
        test "a scan carry is written in place" test_scan_carry_written_in_place;
        test "a scan reads its rows in place" test_scan_reads_rows_in_place;
      ];
    group "sliding windows"
      [
        test "unfold matches eager" test_unfold_matches_eager;
        test "fold of unfold matches eager" test_fold_matches_eager;
        test "sliding window matches eager" test_sliding_window_matches_eager;
        test "correlate matches eager" test_correlate_matches_eager;
      ];
    group "cumulative reductions"
      [
        slow "small integer scans keep their dtype"
          test_small_int_scans_keep_dtype;
        test "long scans match eager" test_long_scans_match_eager;
      ];
    group "indexed access"
      [
        test "scatter matches eager" test_scatter_matches_eager;
        test "scatter orders duplicate updates" test_scatter_duplicates;
        test "scatter along a middle axis" test_scatter_middle_axis;
        test "scatter with unique indices" test_scatter_unique_indices;
        test "scatter with unique indices broken at one row"
          test_scatter_unique_indices_broken_at_one_row;
        test "scatter drops an update outside the axis"
          test_scatter_out_of_range_writes_nothing;
        test "scatter carries int and bfloat16 payloads"
          test_scatter_payload_dtypes;
        test "scatter under vmap" test_scatter_under_vmap;
        test "gradient of take with repeated tokens"
          test_grad_of_take_with_repeated_tokens;
        test "take over a large table matches eager"
          test_take_large_table_matches_eager;
        test "top_k matches eager" test_top_k_matches_eager;
        test "gradient of top_k" test_grad_of_top_k;
        test "diag matches eager" test_diag_matches_eager;
      ];
    group "training"
      [
        test "jitted training follows the eager trajectory"
          test_jitted_training_matches_eager;
      ];
    group "state"
      [
        test "non-contiguous inputs fall back to copies"
          test_non_contiguous_input_matches_eager;
        test "offset views read the right span"
          test_offset_view_input_matches_eager;
        test "outputs have their own storage"
          test_outputs_have_their_own_storage;
      ];
    group "placement"
      [
        test "a placed value equals its argument" test_place_equals_its_argument;
        test "strided and offset values" test_place_strided_and_offset;
        test "a placed value feeds an input with no transfer"
          test_place_feeds_inputs_without_transfer;
        test "a resident value is returned as it is"
          test_place_resident_value_is_returned;
        test "on the host device" test_place_on_the_host_device;
        test "placement under grad, jvp, vmap and jit"
          test_place_is_the_identity_under_transformations;
        test "an unbound placed value is consumed by donation"
          test_place_then_donate_consumes;
        test "item reads one element" test_item_reads_one_element;
        test "a move to the host keeps its source"
          test_move_to_host_keeps_its_source;
        test "mixed placements raise" test_mixed_placements_raise;
        test "placing elsewhere inside jit raises"
          test_placing_elsewhere_inside_jit_raises;
        test "one device per name" test_one_device_per_name;
        test "a compiled function runs where its inputs live"
          test_runs_where_its_inputs_live;
        test "an input on another device raises" test_leaves_elsewhere_raise;
        test "a capture decides the device" test_capture_decides_the_device;
        test "a capture's device is remembered"
          test_capture_device_is_remembered;
        test "placed views bind without a copy" test_views_bind_without_a_copy;
        test "placed views share programs" test_views_share_programs;
        test "windows bind from aligned offsets"
          test_windows_bind_from_aligned_offsets;
        test "strides key programs" test_strides_key_programs;
        test "views of inputs as outputs" test_views_of_inputs_as_outputs;
        test "a window's view is released" test_a_window's_view_is_released;
        test "overlapping views are copied" test_overlapping_views_are_copied;
        test "captured views bind" test_captured_views_bind;
        test "a program on the host is on the host"
          test_host_program_is_on_the_host;
      ];
    group "file-backed sources"
      [
        slow "a mapped leaf larger than a chunk uploads from its file"
          test_file_backed_upload;
        test "a replaced file is not read" test_file_backed_upload_after_replace;
      ];
    group "bound captures"
      [
        test "binding a placed capture moves no bytes"
          test_bound_capture_moves_no_bytes;
        test "two compiled functions share one buffer"
          test_bound_capture_is_shared;
        test "a bound value keeps its buffer across a read"
          test_bound_value_survives_a_read;
        test "a read leaves an unbound value placed"
          test_unbound_value_stays_after_a_read;
        test "a bound input is not consumed by donation"
          test_bound_input_is_not_donated;
        test "a bound capture returned unchanged is a copy"
          test_bound_capture_returned_is_a_copy;
        test "a bound buffer is released with its owners"
          test_bound_buffer_is_released_with_its_owners;
        test "the collection budget counts every allocation"
          test_budget_counts_every_allocation;
        test "a device that cannot allocate raises Out_of_memory"
          test_out_of_memory;
      ];
    group "chunked transfers"
      [
        slow "a contiguous value larger than a chunk" test_chunked_contiguous;
        slow "a contiguous value at an offset" test_chunked_offset;
        slow "a strided value" test_chunked_strided;
        slow "a strided value with rows larger than a chunk"
          test_chunked_strided_rows;
        slow "a capture larger than a chunk" test_chunked_capture;
      ];
    group "residency"
      [
        test "feedback chain moves no bytes" test_feedback_chain_moves_no_bytes;
        test "forced handles feed current bytes"
          test_forced_handle_feeds_current_bytes;
        test "the same handle can seed two leaves"
          test_same_handle_as_two_leaves;
        test "duplicate output leaves share one handle"
          test_duplicate_outputs_share_one_handle;
        test "handles feed other jitted closures" test_cross_jit_feedback;
        test "handles feed new signatures without forcing"
          test_cross_signature_feedback;
        test "pass-through outputs survive later calls"
          test_pass_through_output_survives;
        test "grad over jit forces deferred arguments"
          test_grad_over_jit_with_deferred_arg;
        test "vmap over jit forces deferred arguments"
          test_vmap_over_jit_with_deferred_arg;
        test "signature dispatch never forces"
          test_dispatch_on_handle_reads_no_bytes;
        test "captures upload once across signatures"
          test_capture_uploaded_once_across_signatures;
        test "dropped handles are reclaimed" test_dropped_handles_are_reclaimed;
        test "a read after a call waits for it" test_read_after_call_waits;
        test "programs run in turn share an arena" test_programs_share_an_arena;
        test "a grown arena frees the old one" test_a_grown_arena_frees_the_old;
        test "a buffer freed under a running kernel is not reused"
          test_buffer_freed_under_a_running_kernel;
      ];
    group "donation"
      [
        test "donate bounds resident memory at two generations"
          test_donate_bounds_resident_memory;
        test "a donated input hands its storage to the output"
          test_donate_reuses_storage;
        test "a movement path refuses reuse and stays correct"
          test_donate_refuses_movement_path;
        test "a later reader refuses reuse and stays correct"
          test_donate_refuses_later_reader;
        test "a donated pass-through moves its storage"
          test_donate_moves_pass_through;
        test "a run-time window write reuses the cache"
          test_donate_reuses_window_write;
        test "a pool read after its write still reuses storage"
          test_donate_reuses_pool_read_after_write;
        test "two programs alternate on one donated state"
          test_donate_alternates_two_programs;
        test "donation reuses a pool written by scatter"
          test_donate_reuses_pool_scatter;
        test "a partial view of a storage is not donated"
          test_step_refuses_a_partial_view;
        test "a storage both arguments reach is read"
          test_step_reads_a_storage_both_arguments_reach;
        test "scatter without donation keeps its input"
          test_scatter_without_donation_keeps_the_input;
        test "scatter of values read from the donated pool"
          test_scatter_of_values_read_from_the_pool;
        test "scatter of the donated pool into itself"
          test_scatter_of_the_pool_into_itself;
        test "scatter refuses a later reader of the donated pool"
          test_scatter_refuses_a_later_reader_of_the_pool;
        test "scatter beside a reader of the old value"
          test_scatter_beside_a_reader_of_the_old_value;
        test "an updated input returned unchanged stays readable"
          test_donate_keeps_pass_through_readable;
        test "outputs never write into an input's buffer"
          test_outputs_never_write_into_inputs;
        test "every derived leaf is reused" test_donate_reuses_every_leaf;
        test "a staged loop refuses only the leaves it touches"
          test_donate_reuses_beside_a_scan;
        test "a donated handle raises on read"
          test_donated_handle_raises_on_read;
        test "re-feeding a donated handle raises"
          test_donated_handle_refeed_raises;
        test "duplicate leaves donate once" test_donate_duplicate_leaves_once;
        test "a value read before the call is still donated"
          test_read_value_is_still_donated;
        test "host inputs are unaffected" test_host_input_unaffected_by_donate;
        test "jit never consumes its inputs" test_jit_leaves_handle_readable;
        test "a step reads its first argument"
          test_step_reads_its_first_argument;
        test "a step reuses only its state's storage"
          test_step_reuses_only_the_state;
        test "a handle in both arguments is read"
          test_step_reads_a_handle_in_both_arguments;
        test "an indexed write into a read leaf keeps it"
          test_step_write_into_a_read_leaf;
      ];
    group "values"
      [
        test "set with a traced window start replays the position"
          test_set_traced_window_replays_position;
        test "set at a traced corner over two axes"
          test_set_traced_window_over_two_axes;
        test "set with static specs matches eager"
          test_set_static_window_matches_eager;
        test "slice with a traced window start replays the position"
          test_slice_traced_window_replays_position;
      ];
    group "linear algebra"
      [
        test "reduced QR matches eager" test_qr_reduced_matches_eager;
        test "a zero-tail column takes no reflector"
          test_qr_zero_tail_matches_eager;
        test "cholesky matches eager in both triangles"
          test_cholesky_matches_eager;
        test "the gradient of a Cholesky-using loss compiles"
          test_cholesky_gradient_compiles;
        test "triangular solve matches eager for every flag combination"
          test_solve_triangular_flags_match_eager;
        test "triangular solve takes a vector right-hand side"
          test_solve_triangular_vector_rhs;
        test "triangular solve is batched" test_solve_triangular_batched;
        test "solve and inv match eager" test_solve_matches_eager;
        test "the gradient of a QR-using loss compiles"
          test_qr_gradient_compiles;
      ];
    group "errors"
      [
        test "reading a traced value raises" test_data_dependent_read_raises;
        test "traced values have no storage" test_traced_values_have_no_storage;
        test "a leaked traced value raises" test_leaked_traced_value_raises;
        test "unsupported operations raise" test_unsupported_op_raises;
      ];
  ]

let () = run "rune jit" tests
