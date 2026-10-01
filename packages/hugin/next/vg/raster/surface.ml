(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

(* Pixels are premultiplied RGBA, 8 bits per component, row by row. *)
type pixels = (int, int8_unsigned_elt, c_layout) Array1.t
type t = { w : int; h : int; px : pixels }

let create w h =
  let px = Array1.create int8_unsigned c_layout (w * h * 4) in
  Array1.fill px 0;
  { w; h; px }

(* A clip is a window of pixels, [x1] and [y1] exclusive, and the fraction of
   each of its pixels that it lets through, row by row over the window, or
   [None] for all of each. *)
type clip = {
  x0 : int;
  y0 : int;
  x1 : int;
  y1 : int;
  mask : float array option;
}

let whole s = { x0 = 0; y0 = 0; x1 = s.w; y1 = s.h; mask = None }
let is_shut c = c.x1 <= c.x0 || c.y1 <= c.y0
let shut = { x0 = 0; y0 = 0; x1 = 0; y1 = 0; mask = None }

let[@inline] mask_at c x y =
  match c.mask with
  | None -> 1.
  | Some m -> Array.unsafe_get m (((y - c.y0) * (c.x1 - c.x0)) + (x - c.x0))

(* [blend px i sr sg sb sa k] composites the premultiplied colour [(sr, sg, sb,
   sa)], components in \[0;255\], with coverage [k] over the pixel at byte [i],
   rounding to 8 bits. *)
let[@inline] blend (px : pixels) i sr sg sb sa k =
  let inv = 1. -. (sa *. k /. 255.) in
  (* Written out rather than through a local function, which would keep the
     compiler from inlining [blend] and unboxing its arguments. *)
  Array1.unsafe_set px i
    (truncate ((sr *. k) +. (float (Array1.unsafe_get px i) *. inv) +. 0.5));
  Array1.unsafe_set px (i + 1)
    (truncate
       ((sg *. k) +. (float (Array1.unsafe_get px (i + 1)) *. inv) +. 0.5));
  Array1.unsafe_set px (i + 2)
    (truncate
       ((sb *. k) +. (float (Array1.unsafe_get px (i + 2)) *. inv) +. 0.5));
  Array1.unsafe_set px (i + 3)
    (truncate
       ((sa *. k) +. (float (Array1.unsafe_get px (i + 3)) *. inv) +. 0.5))

(* [composite dst clip src ox oy alpha] composites [src], with its top left
   pixel at [(ox, oy)] of [dst], faded by [alpha], through [clip]. *)
let composite dst clip src ox oy alpha =
  let x0 = Int.max clip.x0 ox and y0 = Int.max clip.y0 oy in
  let x1 = Int.min clip.x1 (ox + src.w) and y1 = Int.min clip.y1 (oy + src.h) in
  for y = y0 to y1 - 1 do
    for x = x0 to x1 - 1 do
      let si = (((y - oy) * src.w) + (x - ox)) * 4 in
      let sa = Array1.unsafe_get src.px (si + 3) in
      if sa > 0 then begin
        let k = alpha *. mask_at clip x y in
        if k > 0. then
          blend dst.px
            (((y * dst.w) + x) * 4)
            (float (Array1.unsafe_get src.px si))
            (float (Array1.unsafe_get src.px (si + 1)))
            (float (Array1.unsafe_get src.px (si + 2)))
            (float sa) k
      end
    done
  done

(* [to_straight s] is the pixels of [s] with straight alpha, as a [h; w; 4]
   tensor. *)
let to_straight s =
  let px = s.px in
  for i = 0 to (s.w * s.h) - 1 do
    let a = Array1.unsafe_get px ((4 * i) + 3) in
    if a > 0 && a < 255 then
      for c = 0 to 2 do
        let v = Array1.unsafe_get px ((4 * i) + c) in
        Array1.unsafe_set px
          ((4 * i) + c)
          (Int.min 255 (((v * 255) + (a / 2)) / a))
      done
  done;
  Nx.of_bigarray (reshape (genarray_of_array1 px) [| s.h; s.w; 4 |])
