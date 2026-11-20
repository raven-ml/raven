(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

(* [fdiv a b] is [⌊a / b⌋] for [b > 0]. *)
let fdiv a b = if a >= 0 then a / b else -((b - 1 - a) / b)

(* [spans ~edge k n n'] is each new pixel's taps, in increasing [i], with their
   arguments. New pixel [j]'s [τ = (c - i - ½) / s] is [d / e] for the integers
   [d = (2j + 1) n - (2i + 1) n'] and [e], [2n] when shrinking and [2n']
   otherwise, so one division rounds it. [h e] is an integer, [2h] being whole
   and [e] even, so the support test [-h e <= d < h e] is exact. Under
   [`Renormalise] the taps outside are dropped here. *)
let spans ~edge kernel n n' =
  let h = Sample.half_width kernel in
  let e = if n > n' then 2 * n else 2 * n' in
  let he = Float.to_int (h *. Float.of_int e) in
  Array.init n' (fun j ->
      let num = ((2 * j) + 1) * n in
      let lo = fdiv (num - he - n') (2 * n') + 1 in
      let hi = fdiv (num + he - n') (2 * n') in
      let lo, hi =
        match edge with
        | `Renormalise -> (Int.max lo 0, Int.min hi (n - 1))
        | #extension -> (lo, hi)
      in
      Array.init
        (Int.max 0 (hi - lo + 1))
        (fun k ->
          let i = lo + k in
          let d = num - (((2 * i) + 1) * n') in
          (i, Float.of_int d /. Float.of_int e)))

(* [sinc x] at float64 on the host, for weights: [sin (πr) / (πx)] signed by the
   parity of [x]'s nearest integer [k], [r = x - k] exact, so it is [0] at every
   other integer. A weight needs no derivative, so the quotient serves near [0]
   too. *)
let sinc x =
  if x = 0. then 1.
  else
    let k = Float.round x in
    let q = Float.sin (Float.pi *. (x -. k)) /. (Float.pi *. x) in
    if Float.rem k 2. = 0. then q else -.q

(* [value k t] is [k] at [t] in its support, at float64 on the host, by the
   formulas [eval] stages. *)
let value kernel t =
  match kernel with
  | `Box -> 1.
  | `Triangle -> 1. -. Float.abs t
  | `Cubic a ->
      let t = Float.abs t in
      let t2 = t *. t in
      let t3 = t2 *. t in
      if t <= 1. then (t3 *. (a +. 2.)) -. (t2 *. (a +. 3.)) +. 1.
      else (t3 *. a) -. (t2 *. (5. *. a)) +. (t *. (8. *. a)) -. (4. *. a)
  | `Lanczos n -> sinc t *. sinc (t /. Float.of_int n)

(* [table fn ~edge k n n'] is each new pixel's taps that read a pixel with a
   nonzero weight, unfolded, and their weights: the kernel's float64 values
   divided by the row's float64 sum, added in order. A tap outside reads zero
   under [`Zero], and [`Renormalise] dropped it before the sum. A row whose sum
   is [0] has no tap. OCaml floats and integers alone, so that a trace stages
   none of it. *)
let table fn ~edge kernel n n' =
  Sample.check_kernel fn kernel;
  if n < 1 || n' < 1 then invalid_argf "%s: %d pixels to %d" fn n n';
  let reads i =
    match edge with
    | `Zero -> i >= 0 && i < n
    | `Renormalise | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic
      ->
        true
  in
  Array.map
    (fun span ->
      let values = Array.map (fun (_, t) -> value kernel t) span in
      let sum = Array.fold_left ( +. ) 0. values in
      if sum = 0. then [||]
      else
        let taps = Array.mapi (fun k (i, _) -> (i, values.(k) /. sum)) span in
        Array.of_list
          (List.filter (fun (i, w) -> reads i && w <> 0.) (Array.to_list taps)))
    (spans ~edge kernel n n')

(* [indices ~edge n shape is] is the pixels the unfolded positions [is] read,
   folded by {!Nx.extension_index}, and [-1] for [None], the padding. *)
let indices ~edge n shape is =
  let raw f = Nx.create Nx.int64 shape (Array.map f is) in
  match edge with
  | `Zero | `Renormalise ->
      raw (function Some i -> Int64.of_int i | None -> -1L)
  | #extension as e ->
      let pad = Nx.create Nx.bool shape (Array.map Option.is_none is) in
      let i = raw (function Some i -> Int64.of_int i | None -> 0L) in
      Nx.where pad (Nx.full_like i (-1L)) (Nx.extension_index e n i)

(* [slot rows k] is slot [k] of each row, [None] past its taps. *)
let slot rows k =
  Array.map (fun r -> if k < Array.length r then Some r.(k) else None) rows

let resize_weights ~edge kernel n n' =
  let edge = (edge :> Sample.edge) in
  let rows = table "Mynd.Sample.resize_weights" ~edge kernel n n' in
  let t = Array.fold_left (fun t r -> Int.max t (Array.length r)) 0 rows in
  let taps = Array.concat (List.init t (slot rows)) in
  (* [taps] is slot-major: transposed into rows. *)
  let rowwise a = Nx.copy (Nx.transpose (Nx.reshape [| t; n' |] a)) in
  ( rowwise (indices ~edge n [| t * n' |] (Array.map (Option.map fst) taps)),
    rowwise
      (Nx.create Nx.float64
         [| t * n' |]
         (Array.map (function Some (_, w) -> w | None -> 0.) taps)) )

(* [along fn ~edge k axis n' x] resizes axis [axis] of [x] to [n'] pixels: for
   each slot of the table, one gather of its pixels times its weights, rounded
   once into [x]'s dtype, the slots added in order. A padding slot reads zero
   with weight zero. A table of no slot has one of padding. *)
let along fn ~edge kernel axis n' x =
  let n = Nx.dim axis x in
  if n = n' then x
  else
    let rows = table fn ~edge kernel n n' in
    let t = Array.fold_left (fun t r -> Int.max t (Array.length r)) 1 rows in
    let at = Nx.beside x in
    let shape = Array.init (-axis) (fun k -> if k = 0 then n' else 1) in
    let term k =
      let taps = slot rows k in
      let i = indices ~edge n [| n' |] (Array.map (Option.map fst) taps) in
      (* Stored straight into [x]'s dtype, which rounds each float64 weight
         once, as a cast would, and stages no operation. *)
      let w =
        Array.map (function Some (_, w) -> w | None -> 0.) taps
        |> Nx.create (Nx.dtype x) shape
      in
      Nx.mul (Nx.take ~axis (Nx.place at i) x) (Nx.place at w)
    in
    let sum = ref (term 0) in
    for k = 1 to t - 1 do
      sum := Nx.add !sum (term k)
    done;
    !sum

let resize ~edge kernel (h', w') x =
  let fn = "Mynd.Sample.resize" in
  image fn x;
  if h' < 1 || w' < 1 then invalid_argf "%s: %d by %d pixels" fn h' w';
  if Nx.dim (-3) x < 1 || Nx.dim (-2) x < 1 then
    invalid_argf "%s: an image of %d by %d pixels" fn (Nx.dim (-3) x)
      (Nx.dim (-2) x);
  Sample.check_kernel fn kernel;
  let edge = (edge :> Sample.edge) in
  let go x =
    x |> along fn ~edge kernel (-2) w' |> along fn ~edge kernel (-3) h'
  in
  (* The formats narrower than 32 bits compute at float32 and round once. *)
  let dt = Nx.dtype x in
  if Nx.Dtype.bits dt < 32 then Nx.cast dt (go (Nx.cast Nx.float32 x)) else go x
