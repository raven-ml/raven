(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

(* Kernels and edges *)

type kernel = [ `Box | `Triangle | `Cubic of float | `Lanczos of int ]
type edge = [ extension | `Renormalise ]

let half_width = function
  | `Box -> 0.5
  | `Triangle -> 1.
  | `Cubic _ -> 2.
  | `Lanczos n -> Float.of_int n

let check_kernel fn = function
  | `Cubic a when not (Float.is_finite a) ->
      invalid_argf "%s: Keys' parameter %g is not finite" fn a
  | `Lanczos n when n < 1 -> invalid_argf "%s: Lanczos of %d lobes" fn n
  | `Box | `Triangle | `Cubic _ | `Lanczos _ -> ()

(* [eval k t] is [k] at the arguments [t], each in its support [[-h, h)], in
   [t]'s dtype: the formulas of mynd.mli, Keys' in powers of [|t|]. *)
let eval kernel t =
  match kernel with
  | `Box -> Nx.ones_like t
  | `Triangle -> Nx.rsub_s 1. (Nx.abs t)
  | `Cubic a ->
      let t = Nx.abs t in
      let t2 = Nx.square t in
      let t3 = Nx.mul t2 t in
      let near =
        Nx.add_s (Nx.sub (Nx.mul_s t3 (a +. 2.)) (Nx.mul_s t2 (a +. 3.))) 1.
      in
      let far =
        Nx.sub_s
          (Nx.add
             (Nx.sub (Nx.mul_s t3 a) (Nx.mul_s t2 (5. *. a)))
             (Nx.mul_s t (8. *. a)))
          (4. *. a)
      in
      Nx.where (Nx.less_equal_s t 1.) near far
  | `Lanczos n -> Nx.mul (Nx.sinc t) (Nx.sinc (Nx.div_s t (Float.of_int n)))

(* Sampling *)

(* The index of the pixel a tap at [t] reads, [-1] where it reads zero: under
   [`Renormalise] a tap outside reads zero, and weighs nothing. *)
let fold edge n t =
  match edge with
  | `Renormalise -> Nx.extension_index `Zero n t
  | #extension as e -> Nx.extension_index e n t

(* Positions past ±2⁶² read as if there: Zero and Replicate exactly, the folding
   rules from a far position. It keeps [⌊u⌋] and its taps in int64. *)
let far = 0x1p62

(* [below dt h] is [h] in [dt] and the largest value of [dt] below it, host
   floats so that they are constants under any trace. *)
let below dt h =
  let store = Nx.Dtype.of_float dt in
  let h = store h in
  let eps = (Nx.Dtype.float_format dt).epsilon in
  let rec refine lo =
    let mid = store ((lo +. h) /. 2.) in
    if mid > lo && mid < h then refine mid else lo
  in
  (h, refine (store (h -. (Float.abs h *. eps))))

(* [taps k u] is, along an axis, the [2h] taps reading coordinates [u]: each a
   pixel [t] in [(u - ½ - h, u - ½ + h]], not yet folded, and the kernel's
   argument [τ = u - ½ - t] there. They start at [⌊u⌋ + 1 - ⌊h + ½⌋], one less
   where [u]'s fraction lies below [h + ½]'s, [0] or [½].

   [τ] is the exact difference rounded once: [f - (t - ⌊u⌋ + ½)] with [f = u -
   ⌊u⌋], exact but for [u] in [(-1, 0)], where [u - (t + ½)] is, [t + ½] being a
   small half-integer. Rounding can take a [τ] just below [h] to [h], outside
   the support of the pixel it belongs to: it is held at the value below. It
   cannot take one to below [-h], which [u]'s dtype holds. *)
let taps kernel u =
  let h = half_width kernel in
  let m = Float.floor (h +. 0.5) in
  let fl = Nx.floor u in
  let f = Nx.sub u fl in
  let small = Nx.logical_and (Nx.less_s u 0.) (Nx.greater_s u (-1.)) in
  let base = Nx.cast Nx.int64 (Nx.clamp ~min:(-.far) ~max:far fl) in
  let r = h +. 0.5 -. m in
  let shift = if r = 0. then None else Some (Nx.less_s f r) in
  let h_dt, under = below (Nx.dtype u) h in
  List.init
    (Float.to_int (2. *. h))
    (fun d ->
      let delta = 1. -. m +. Float.of_int d in
      let t = Nx.add_s base (Int64.of_float delta) in
      let off = Nx.full_like f delta in
      let t, off =
        match shift with
        | None -> (t, off)
        | Some s ->
            (Nx.sub t (Nx.cast Nx.int64 s), Nx.sub off (Nx.cast (Nx.dtype f) s))
      in
      let half = Nx.add_s off 0.5 in
      let tau = Nx.where small (Nx.sub u (Nx.add fl half)) (Nx.sub f half) in
      let tau =
        Nx.where (Nx.greater_equal_s tau h_dt) (Nx.full_like tau under) tau
      in
      (t, tau))

(* [weights ~edge k n u] is the taps of an axis of [n] pixels at the coordinates
   [u]: each the index it reads and its weight, the kernel's values divided by
   their sum, added in order, and [0] where that sum is [0]. There the division
   is by [1], so that no [0 / 0] reaches the derivative. *)
let weights ~edge kernel n u =
  let taps =
    List.map
      (fun (t, tau) ->
        let i = fold edge n t in
        let k = eval kernel tau in
        match edge with
        | `Renormalise ->
            (i, Nx.where (Nx.greater_equal_s i 0L) k (Nx.zeros_like k))
        | #extension -> (i, k))
      (taps kernel u)
  in
  let sum =
    List.fold_left
      (fun s (_, k) -> Nx.add s k)
      (snd (List.hd taps))
      (List.tl taps)
  in
  let zero = Nx.equal_s sum 0. in
  let safe = Nx.where zero (Nx.ones_like sum) sum in
  List.map
    (fun (i, k) -> (i, Nx.where zero (Nx.zeros_like k) (Nx.div k safe)))
    taps

(* [flat ~w i j] is the flat index [i w + j] of a tap, [-1] where either reads
   zero, since [-1] in a flat index names the previous row's last pixel. *)
let flat ~w i j =
  let out = Nx.logical_or (Nx.less_s i 0L) (Nx.less_s j 0L) in
  Nx.where out (Nx.full_like i (-1L)) (Nx.add (Nx.mul_s i (Int64.of_int w)) j)

(* [gather x idx] reads the images [x], [[b…; h w; c]], at the flat indices
   [idx], [[b…; q]]: [[b…; q; c]]. *)
let gather x idx =
  let batch = Nx.ndim x - 2 in
  Nx.take_along_axis ~axis:batch (Nx.unsqueeze ~axes:[ -1 ] idx) x

let check_edge fn ~edge x =
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  match edge with
  | `Zero | `Renormalise -> ()
  | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ->
      if h = 0 || w = 0 then
        invalid_argf
          "%s: an image of %d by %d pixels reads nothing past its edge" fn h w

(* [prepare fn x p] is the batch shape, the points' shape, [x] with its pixels
   flat and the coordinates [u] and [v], [[b…; q]]. *)
let prepare fn ~edge x p =
  image fn x;
  check_edge fn ~edge x;
  let batch = Nx.ndim x - 3 in
  let pb, q, p = points fn ~batch p in
  let s = Nx.shape x in
  let xb = Array.sub s 0 batch in
  let b = broadcast fn xb pb in
  let h = s.(batch) and w = s.(batch + 1) and c = s.(batch + 2) in
  let xf = Nx.reshape (Array.concat [ xb; [| h * w; c |] ]) x in
  (b, q, (h, w, c), xf, coordinate p 0, coordinate p 1)

let finish ~b ~q ~c y = Nx.reshape (Array.concat [ b; q; [| c |] ]) y

let at ~edge kernel x p =
  let fn = "Mynd.Sample.at" in
  check_kernel fn kernel;
  let edge = (edge :> edge) in
  let b, q, (h, w, c), xf, u, v = prepare fn ~edge x p in
  let rows = weights ~edge kernel h v and cols = weights ~edge kernel w u in
  let sum = ref None in
  List.iter
    (fun (i, wi) ->
      List.iter
        (fun (j, wj) ->
          let weight = Nx.unsqueeze ~axes:[ -1 ] (Nx.mul wj wi) in
          let term = Nx.mul weight (gather xf (flat ~w i j)) in
          sum := Some (match !sum with None -> term | Some s -> Nx.add s term))
        cols)
    rows;
  let y = Option.get !sum in
  let finite =
    Nx.unsqueeze ~axes:[ -1 ] (Nx.logical_and (Nx.isfinite u) (Nx.isfinite v))
  in
  let y = Nx.where finite y (Nx.full_like y Float.nan) in
  finish ~b ~q ~c y

let nearest ~edge x p =
  let fn = "Mynd.Sample.nearest" in
  let edge = (edge :> extension) in
  let b, q, (h, w, c), xf, u, v = prepare fn ~edge:(edge :> edge) x p in
  let index n u =
    let i = Nx.cast Nx.int64 (Nx.clamp ~min:(-.far) ~max:far (Nx.floor u)) in
    let i = Nx.extension_index edge n i in
    Nx.where (Nx.isfinite u) i (Nx.full_like i (-1L))
  in
  finish ~b ~q ~c (gather xf (flat ~w (index h v) (index w u)))

let centres dt (h, w) =
  if h < 0 || w < 0 then invalid_argf "Mynd.Sample.centres: %d by %d pixels" h w;
  let half n = Nx.add_s (Nx.arange_f Nx.float64 0. (Float.of_int n) 1.) 0.5 in
  let x = Nx.broadcast_to [| h; w |] (Nx.reshape [| 1; w |] (half w)) in
  let y = Nx.broadcast_to [| h; w |] (Nx.reshape [| h; 1 |] (half h)) in
  Nx.cast dt (Nx.stack ~axis:(-1) [ x; y ])
