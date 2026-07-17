(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scaling regression tests for the schedule pipeline.

   [Rangeify.get_kernel_graph] must scale (at most) linearly with graph size.
   Two historical bugs made it superlinear:
   - unmemoised shape derivation re-walked shared DAGs, exploding exponentially
     on deep elementwise folds (the Lorenz workload);
   - the kernel splitter rescanned the whole graph history per kernel, giving
     quadratic behaviour on any graph that produces many kernels.

   Each case doubles the graph size and asserts the compile time grows by less
   than [threshold]x. Linear growth is ~2x; quadratic is ~4x; exponential blows
   past any bound (and would trip the per-test timeout). The threshold is
   deliberately generous, and timings use a warmup ladder plus a min over
   repetitions, so ordinary CI noise does not flake. *)

open Windtrap
module T = Tolk_frontend.Tensor
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype

(* Force [Op] to be linked so its initialiser installs the broadcasting hook
   the element-wise operations rely on. *)
let _op_linked = Sys.opaque_identity Op.broadcasted

let param slot dims =
  let shape = if dims = [] then None else Some (T.shape_uop dims) in
  T.of_uop (U.param ~slot ~dtype:D.float32 ?shape ~device:(U.Single "CPU") ())

(* Graph builders, one per graph shape. All are pure graph construction; no
   device or realization is involved. *)

(* Pure element-wise chain — the trivially linear baseline. *)
let chain n =
  let x = param 0 [ 256; 256 ] in
  let rec loop i x = if i = 0 then x else loop (i - 1) (El.add x (T.f 1.0)) in
  loop n x

(* Lorenz-like fold: [n] steps of ~9 element-wise ops on a 3-element state,
   reusing intermediates to create fan-out, ending in a sum. This is the
   workload whose shape derivation used to blow up exponentially. *)
let lorenz n =
  let s0 = param 0 [ 3 ] in
  let step s =
    let a = El.add s (T.f 0.1) in
    let b = El.mul a (T.f 0.9) in
    let c = El.sub b (T.f 0.2) in
    let d = El.mul c s in
    let e = El.add d a in
    let f = El.mul e (T.f 1.1) in
    let g = El.sub f b in
    let h = El.add g c in
    El.add h s
  in
  let rec fold i s = if i = 0 then s else fold (i - 1) (step s) in
  Rd.sum (fold n s0)

(* Reduce chain: each step adds a keepdim row-sum back to the input. *)
let wide_reduce n =
  let x = param 0 [ 256; 256 ] in
  let rec loop i x =
    if i = 0 then x
    else loop (i - 1) (El.add x (Rd.sum ~axis:[ -1 ] ~keepdim:true x))
  in
  loop n x

(* Matmul chain — [n] separate kernels, each of bounded size. *)
let matmul n =
  let xs = List.init (n + 1) (fun i -> param i [ 32; 32 ]) in
  match xs with
  | [] -> assert false
  | x0 :: rest -> List.fold_left Op.matmul x0 rest

(* Softmax chain — multi-consumer structure (max, exp, sum) per step. *)
let softmax n =
  let x = param 0 [ 64; 256 ] in
  let rec loop i x = if i = 0 then x else loop (i - 1) (Op.softmax x) in
  loop n x

(* Timing harness *)

let threshold = 3.0
let reps = 5

let compile_ms build n =
  let sink = U.sink [ U.contiguous ~src:(T.uop (build n)) () ] in
  let t0 = Unix.gettimeofday () in
  ignore (Sys.opaque_identity (Tolk.Rangeify.get_kernel_graph sink));
  (Unix.gettimeofday () -. t0) *. 1000.

let min_ms build n =
  let rec loop i best =
    if i = 0 then best else loop (i - 1) (Float.min best (compile_ms build n))
  in
  loop reps infinity

let assert_linear name build n =
  (* Warmup ladder: prime the shape caches and allocator at both sizes. *)
  ignore (compile_ms build n);
  ignore (compile_ms build (2 * n));
  let t_n = min_ms build n in
  let t_2n = min_ms build (2 * n) in
  let ratio = t_2n /. t_n in
  is_true
    ~msg:
      (Printf.sprintf
         "%s: get_kernel_graph superlinear — t(%d)=%.2fms t(%d)=%.2fms \
          ratio=%.2f (threshold %.1f)"
         name n t_n (2 * n) t_2n ratio threshold)
    (ratio < threshold)

(* A re-introduced exponential blows past the timeout rather than merely
   failing the ratio, so keep it well above a healthy run (~1s) yet finite. *)
let scaling name build n =
  test ~timeout:30. ~retries:3 name (fun () -> assert_linear name build n)

let scaling_tests =
  group "get_kernel_graph scaling"
    [
      scaling "chain" chain 400;
      scaling "lorenz" lorenz 40;
      scaling "wide_reduce" wide_reduce 40;
      scaling "matmul" matmul 40;
      scaling "softmax" softmax 25;
    ]

let () = run "Schedule.Scaling" [ scaling_tests ]
