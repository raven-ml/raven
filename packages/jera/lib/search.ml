(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectors over lanes *)

let lanes v = Array.sub (Nx.shape v) 0 (Nx.ndim v - 1)

let rows v =
  let width = (Nx.shape v).(Nx.ndim v - 1) in
  Nx.reshape [| Array.fold_left ( * ) 1 (lanes v); width |] v

let dot u v =
  let uv = Nx.mul u v in
  Nx.reshape (lanes uv) (Num.sum_rows (rows uv))

let norm v = Nx.sqrt (dot v v)

let hold m next last =
  let ones = Array.make (Nx.ndim next - Nx.ndim m) 1 in
  let m = Nx.reshape (Array.append (Nx.shape m) ones) m in
  Nx.where (Nx.broadcast_to (Nx.shape next) m) next last

let accepted tol ~e ~y =
  let r = Num.rms_rows (rows (Tol.ratio tol ~e ~y)) in
  Nx.less_equal_s (Nx.reshape (lanes e) r) 1.

(* Contraction *)

let contraction delta ~q =
  let shrinking = Nx.logical_and (Nx.greater_equal_s q 0.) (Nx.less_s q 1.) in
  let factor =
    Nx.where shrinking
      (Nx.div q (Nx.rsub_s 1. q))
      (Nx.full_like q Float.infinity)
  in
  (* A zero component counts zero, whatever the factor. *)
  let e =
    Nx.mul (Nx.abs delta)
      (Nx.reshape (Array.append (Nx.shape factor) [| 1 |]) factor)
  in
  Nx.where (Nx.equal_s delta 0.) (Nx.zeros_like delta) e

(* Backtracking

   The carry holds the trial length, the payload, whether each lane still
   searches, the trials each lane took and the count of trips. *)

let backtrack p dtype ~trials ~running ~accept ~shrink trial last =
  let step (alpha, (payload, (searching, (n, j)))) =
    let next, phi = trial alpha in
    let n = Nx.add n (Nx.cast Nx.int32 searching) in
    let ok = Nx.logical_and searching (accept alpha phi) in
    let payload = Nx.Ptree.map2 p (fun _ a b -> hold ok a b) next payload in
    let searching = Nx.logical_and searching (Nx.logical_not ok) in
    let alpha = Nx.where searching (shrink alpha phi) alpha in
    (alpha, (payload, (searching, (n, Nx.add_s j 1l))))
  in
  let _, (payload, (searching, (n, _))) =
    Rune.iterate
      Nx.Ptree.(pair tensor (pair p (pair tensor (pair tensor tensor))))
      ~max:trials
      ~until:(fun (_, (_, (searching, (_, j)))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any searching))
          (Nx.greater_equal_s j (Int32.of_int trials)))
      ~f:step
      ( Nx.ones dtype (Nx.shape running),
        ( last,
          ( running,
            (Nx.zeros Nx.int32 (Nx.shape running), Nx.scalar Nx.int32 0l) ) ) )
  in
  (payload, Nx.logical_and running (Nx.logical_not searching), n)

(* The sufficient-decrease constant: a step must take [c] of the decrease the
   slope predicts. *)
let c = 1e-4

let armijo ~phi0 ~slope0 =
  let accept alpha phi =
    Nx.less_equal phi (Nx.add phi0 (Nx.mul_s (Nx.mul alpha slope0) c))
  in
  let shrink alpha phi =
    let excess = Nx.sub (Nx.sub phi phi0) (Nx.mul slope0 alpha) in
    let quadratic =
      Nx.div (Nx.neg (Nx.mul slope0 (Nx.square alpha))) (Nx.mul_s excess 2.)
    in
    let lo = Nx.mul_s alpha 0.1 and hi = Nx.mul_s alpha 0.5 in
    Nx.where (Nx.isfinite quadratic)
      (Nx.minimum hi (Nx.maximum lo quadratic))
      hi
  in
  (accept, shrink)
