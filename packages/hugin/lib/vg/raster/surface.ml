(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

(* Pixels are premultiplied RGBA in levels, \[0;255\] per component, row by row.
   They are single-precision floats so that compositing rounds no pixel to a
   level until [to_straight]: rounding each blend to 8 bits would keep a channel
   from rising once a blend moves it by less than half a level, which leaves
   dense translucent drawing up to [0.5 /. alpha] levels short of its colour. *)
type pixels = (float, float32_elt, c_layout) Array1.t
type t = { w : int; h : int; px : pixels }

let create w h =
  let px = Array1.create float32 c_layout (w * h * 4) in
  Array1.fill px 0.;
  { w; h; px }

let clear s = Array1.fill s.px 0.

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
   sa)], components in \[0;255\], with coverage [k] over the pixel at index
   [i]. *)
let[@inline] blend (px : pixels) i sr sg sb sa k =
  let inv = 1. -. (sa *. k /. 255.) in
  (* Written out rather than through a local function, which would keep the
     compiler from inlining [blend] and unboxing its arguments. *)
  Array1.unsafe_set px i ((sr *. k) +. (Array1.unsafe_get px i *. inv));
  Array1.unsafe_set px (i + 1)
    ((sg *. k) +. (Array1.unsafe_get px (i + 1) *. inv));
  Array1.unsafe_set px (i + 2)
    ((sb *. k) +. (Array1.unsafe_get px (i + 2) *. inv));
  Array1.unsafe_set px (i + 3)
    ((sa *. k) +. (Array1.unsafe_get px (i + 3) *. inv))

(* [composite dst clip src ox oy alpha] composites [src], with its top left
   pixel at [(ox, oy)] of [dst], faded by [alpha], through [clip]. *)
let composite dst clip src ox oy alpha =
  let x0 = Int.max clip.x0 ox and y0 = Int.max clip.y0 oy in
  let x1 = Int.min clip.x1 (ox + src.w) and y1 = Int.min clip.y1 (oy + src.h) in
  for y = y0 to y1 - 1 do
    for x = x0 to x1 - 1 do
      let si = (((y - oy) * src.w) + (x - ox)) * 4 in
      let sa = Array1.unsafe_get src.px (si + 3) in
      if sa > 0. then begin
        let k = alpha *. mask_at clip x y in
        if k > 0. then
          blend dst.px
            (((y * dst.w) + x) * 4)
            (Array1.unsafe_get src.px si)
            (Array1.unsafe_get src.px (si + 1))
            (Array1.unsafe_get src.px (si + 2))
            sa k
      end
    done
  done

(* [level v] is the nearest level to [v], within \[0;255\]. Compositing in
   floats can leave a component a hair outside its range. *)
let[@inline] level v =
  if v >= 254.5 then 255 else if v > 0. then truncate (v +. 0.5) else 0

(* [to_straight s] is the pixels of [s] as a [h; w; 4] tensor of levels with
   straight alpha: each component the level nearest its exact value. A pixel
   whose alpha rounds to [0] is [(0, 0, 0, 0)]. *)
let to_straight s =
  let n = s.w * s.h in
  let src = s.px and dst = Array1.create int8_unsigned c_layout (n * 4) in
  for i = 0 to n - 1 do
    let j = 4 * i in
    let a = Array1.unsafe_get src (j + 3) in
    let la = level a in
    if la = 0 then begin
      Array1.unsafe_set dst j 0;
      Array1.unsafe_set dst (j + 1) 0;
      Array1.unsafe_set dst (j + 2) 0;
      Array1.unsafe_set dst (j + 3) 0
    end
    else begin
      let f = 255. /. a in
      Array1.unsafe_set dst j (level (Array1.unsafe_get src j *. f));
      Array1.unsafe_set dst (j + 1) (level (Array1.unsafe_get src (j + 1) *. f));
      Array1.unsafe_set dst (j + 2) (level (Array1.unsafe_get src (j + 2) *. f));
      Array1.unsafe_set dst (j + 3) la
    end
  done;
  Nx.of_bigarray (reshape (genarray_of_array1 dst) [| s.h; s.w; 4 |])
