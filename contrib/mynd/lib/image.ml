(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Shapes *)

(* [broadcast fn a b] is the broadcast of two batch shapes of one rank. *)
let broadcast fn a b =
  Array.map2
    (fun x y ->
      if x = y || y = 1 then x
      else if x = 1 then y
      else
        invalid_argf "%s: batch axes [%s] and [%s] do not broadcast" fn
          (String.concat "; " (Array.to_list (Array.map string_of_int a)))
          (String.concat "; " (Array.to_list (Array.map string_of_int b))))
    a b

let image fn x =
  if Nx.ndim x < 3 then
    invalid_argf "%s: an image has axes [...; h; w; c], not %d" fn (Nx.ndim x)

(* [points fn ~batch p] splits positions [[b…; q…; 2]] into their batch shape,
   their points' shape and the flat [[b…; q; 2]]. *)
let points fn ~batch p =
  let r = Nx.ndim p in
  if r < batch + 1 || Nx.dim (-1) p <> 2 then
    invalid_argf "%s: positions have axes [b...; q...; 2] with %d batch axes" fn
      batch;
  let s = Nx.shape p in
  let b = Array.sub s 0 batch and q = Array.sub s batch (r - batch - 1) in
  let n = Array.fold_left ( * ) 1 q in
  (b, q, Nx.reshape (Array.concat [ b; [| n; 2 |] ]) p)

(* [coordinate p k] is part [k] of positions or parameters along their last
   axis. *)
let coordinate p k =
  Nx.slice (List.init (Nx.ndim p - 1) (fun _ -> Nx.A) @ [ Nx.I k ]) p

(* [cut ~axis lo n x] is [x]'s [n] elements from [lo] along the axis [axis],
   counted from the end. *)
let cut ~axis lo n x =
  Nx.slice
    (List.init (Nx.ndim x + axis) (fun _ -> Nx.A) @ [ Nx.R (lo, lo + n) ])
    x

let dims x = String.concat "; " (List.map string_of_int (Array.to_list x))

(* Masks are boolean images of one channel. *)
let mask_shape fn m =
  if Nx.ndim m < 3 || Nx.dim (-1) m <> 1 then
    invalid_argf "%s: a mask has axes [...; h; w; 1]" fn

(* Reading outside the image *)

type extension =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]

(* Every rule but [`Zero] reads pixels past the edge, which an image of no row
   or no column lacks. *)
let check_extension fn ~(edge : Nx.extension) x =
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  if edge <> `Zero && (h = 0 || w = 0) then
    invalid_argf "%s: an image of %d by %d pixels reads nothing past its edge"
      fn h w

(* Dtypes *)

let narrow x = Nx.Dtype.bits (Nx.dtype x) < 32
