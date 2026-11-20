(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

let check_window fn (kh, kw) =
  if kh < 1 || kw < 1 then
    invalid_argf "%s: a window of %d by %d pixels" fn kh kw

(* [extend edge (kh, kw) x] is the images [x] with [kh - 1] rows and [kw - 1]
   columns more, [kh / 2] and [kw / 2] of them before: element [(i + p, j + q)]
   is tap [(p, q)] of pixel [(i, j)]. *)
let extend edge (kh, kw) x =
  x
  |> Nx.extend ~axis:(-3) edge (kh / 2, (kh - 1) / 2)
  |> Nx.extend ~axis:(-2) edge (kw / 2, (kw - 1) / 2)

(* The windows of an extended image: [[b…; h; w; c; kh; kw]]. Each
   [sliding_window] appends its axis, the rows' first. *)
let slide (kh, kw) e =
  e
  |> Nx.sliding_window ~axis:(-3) ~window:kh
  |> Nx.sliding_window ~axis:(-3) ~window:kw

let windows ~edge (kh, kw) x =
  let fn = "Mynd.Filter.windows" in
  image fn x;
  check_window fn (kh, kw);
  let edge = (edge :> Nx.extension) in
  check_extension fn ~edge x;
  slide (kh, kw) (extend edge (kh, kw) x)

(* [batch_of fn xb kb] is the broadcast of two batch shapes aligned at their
   ends. *)
let batch_of fn xb kb =
  let r = Int.max (Array.length xb) (Array.length kb) in
  let pad a = Array.append (Array.make (r - Array.length a) 1) a in
  broadcast fn (pad xb) (pad kb)

let elements a = Array.fold_left ( * ) 1 a

(* Every batch shape takes nx's depthwise path, one input and one output channel
   a group, which sums each output's taps as [Nx.sum] does: weights shared by
   every image as [c] groups over the batch, paired weights with the batch moved
   into the channels, [B c] groups over one image. A bank of weights over one
   image broadcasts the image to the bank, since the one-copy channel multiplier
   takes nx's grouped path, whose sums round otherwise, and an image's result
   must be the same in any batch. [`Zero] pads in the convolution itself. *)
let correlate ~edge k x =
  let fn = "Mynd.Filter.correlate" in
  image fn x;
  let edge = (edge :> Nx.extension) in
  if Nx.ndim k < 2 then invalid_argf "%s: weights have axes [b'...; kh; kw]" fn;
  let kh = Nx.dim (-2) k and kw = Nx.dim (-1) k in
  check_window fn (kh, kw);
  check_extension fn ~edge x;
  let s = Nx.shape x in
  let r = Array.length s in
  let xb = Array.sub s 0 (r - 3) and h = s.(r - 3) and w = s.(r - 2) in
  let c = s.(r - 1) in
  let kb = Array.sub (Nx.shape k) 0 (Nx.ndim k - 2) in
  let b = batch_of fn xb kb in
  let n = elements b in
  let out = Array.append b [| h; w; c |] in
  if n = 0 || c = 0 then Nx.copy (Nx.broadcast_to out x)
  else
    let padding, widen =
      match edge with
      | `Zero ->
          ([| (kh / 2, (kh - 1) / 2); (kw / 2, (kw - 1) / 2) |], fun x -> x)
      | e -> ([| (0, 0); (0, 0) |], extend e (kh, kw))
    in
    let depthwise ~groups x k =
      Nx.conv ~padding ~groups (widen x)
        (Nx.copy (Nx.broadcast_to [| groups; kh; kw; 1 |] k))
    in
    if elements kb = 1 then
      let x = Nx.reshape [| elements xb; h; w; c |] x in
      let k = Nx.reshape [| 1; kh; kw; 1 |] k in
      Nx.reshape out (depthwise ~groups:c x k)
    else
      let x =
        Nx.broadcast_to out x
        |> Nx.reshape [| n; h; w; c |]
        |> Nx.transpose ~axes:[ 1; 2; 0; 3 ]
        |> Nx.reshape [| 1; h; w; n * c |]
      in
      let k =
        Nx.broadcast_to (Array.append b [| kh; kw |]) k
        |> Nx.reshape [| n; 1; kh; kw |]
        |> Nx.broadcast_to [| n; c; kh; kw |]
        |> Nx.reshape [| n * c; kh; kw; 1 |]
      in
      depthwise ~groups:(n * c) x k
      |> Nx.reshape [| h; w; n; c |]
      |> Nx.transpose ~axes:[ 2; 0; 1; 3 ]
      |> Nx.reshape out

(* The sampled Gaussian's [2r + 1] weights along a last axis, [[b'…; 2r + 1]],
   in [sigma]'s dtype. Where the first sample off the centre, [e^(-1 / 2σ²)],
   underflows to [0], every other does, and the weights are the impulse: there
   [2σ²] is made [1], so that no [0 · ∞] reaches the derivative, which is below
   the dtype's least subnormal anyway. A NaN [σ] reads NaN. *)
let gaussian_weights ~radius sigma =
  let at = Nx.beside sigma in
  let taps = (2 * radius) + 1 in
  let host f =
    Nx.place at (Nx.create (Nx.dtype sigma) [| taps |] (Array.init taps f))
  in
  let squares = host (fun i -> Float.of_int ((i - radius) * (i - radius))) in
  let centre = host (fun i -> if i = radius then 1. else 0.) in
  let s2 = Nx.mul_s (Nx.square (Nx.unsqueeze ~axes:[ -1 ] sigma)) 2. in
  let impulse = Nx.equal_s (Nx.exp (Nx.neg (Nx.recip s2))) 0. in
  let s2 = Nx.where impulse (Nx.ones_like s2) s2 in
  let e = Nx.exp (Nx.neg (Nx.div squares s2)) in
  let g = Nx.div e (Nx.sum ~axes:[ -1 ] ~keepdims:true e) in
  Nx.where impulse (Nx.broadcast_to (Nx.shape g) centre) g

let gaussian ~edge ~radius sigma x =
  let fn = "Mynd.Filter.gaussian" in
  image fn x;
  if radius < 0 then invalid_argf "%s: a radius of %d" fn radius;
  check_extension fn ~edge:(edge :> Nx.extension) x;
  let go sigma x =
    let g = gaussian_weights ~radius sigma in
    x
    |> correlate ~edge (Nx.unsqueeze ~axes:[ -2 ] g)
    |> correlate ~edge (Nx.unsqueeze ~axes:[ -1 ] g)
  in
  (* Narrow formats widen before the extension, whose [`Antisymmetric_whole]
     rounds each tap outside, so that it rounds at float32. *)
  let dt = Nx.dtype x in
  if Nx.Dtype.bits dt < 32 then
    Nx.cast dt (go (Nx.cast Nx.float32 sigma) (Nx.cast Nx.float32 x))
  else go sigma x

(* [spans n size] cuts [[0, n)] into runs of [size], the last shorter. *)
let spans n size =
  if n = 0 then [ (0, 0) ]
  else
    List.init
      ((n + size - 1) / size)
      (fun b -> (b * size, Int.min n ((b + 1) * size)))

let join axis = function [ x ] -> x | xs -> Nx.concatenate ~axis xs

(* The lower median by a stable sort of each window, a band of about [h w / n]
   pixels an image at a time: [h / n] rows, or where [h < n] one row cut into
   runs of [h w / n] columns. A band stores its [n] windows' elements twice, the
   reshaped windows and the sorted values, and an [int64] position each, about
   twice [x]'s elements and as many positions; each band's medians are copied
   out of its sorted values, which then die. *)
let median ~edge (kh, kw) x =
  let fn = "Mynd.Filter.median" in
  image fn x;
  check_window fn (kh, kw);
  let edge = (edge :> Nx.extension) in
  check_extension fn ~edge x;
  let n = kh * kw in
  let e = extend edge (kh, kw) x in
  let lead = List.init (Nx.ndim x - 3) (fun _ -> Nx.A) in
  let band (i0, i1) (j0, j1) =
    let cut = lead @ [ Nx.R (i0, i1 + kh - 1); Nx.R (j0, j1 + kw - 1) ] in
    let w = slide (kh, kw) (Nx.slice cut e) in
    let s = Nx.shape w in
    let s = Array.append (Array.sub s 0 (Array.length s - 2)) [| n |] in
    let sorted, _ = Nx.sort ~axis:(-1) (Nx.reshape s w) in
    Nx.copy (Nx.slice (lead @ [ Nx.A; Nx.A; Nx.A; Nx.I ((n - 1) / 2) ]) sorted)
  in
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  let rows = Int.max 1 (h / n) in
  let cols = if h >= n then w else Int.max 1 (h * w / n) in
  spans h rows
  |> List.map (fun r -> join (-2) (List.map (band r) (spans w cols)))
  |> join (-3)
