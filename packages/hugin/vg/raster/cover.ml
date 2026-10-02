(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exact area coverage. Each edge of a polygon deposits the signed area it
   sweeps into an accumulator, so that the running sum of a row is the winding
   number of each pixel, fractional at the pixels an edge crosses. Edges are cut
   to a window, the clip's pixel rectangle: what lies left of it winds from the
   window's first column, and what lies right of it from the column past its
   last, so that every row of a closed polygon sums back to zero.

   One accumulator serves every surface of a render, which draws one primitive
   at a time: [start] sets the window, edges are deposited, and [paint] or
   [take] reads and clears what they touched. *)

type t = {
  mutable acc : float array;  (** Zero outside a primitive in flight. *)
  mutable stride : int;
  mutable x0 : int;
  mutable y0 : int;
  mutable x1 : int;
  mutable y1 : int;
  mutable dx0 : int;  (** Columns and rows touched, empty if [dx0 > dx1]. *)
  mutable dy0 : int;
  mutable dx1 : int;
  mutable dy1 : int;
  edge : float array;
      (** [[|dir; x0; y0; x1; y1|]], the edge being added. Edges pass through it
          rather than as arguments, which a call that is not inlined boxes. *)
}

let create () =
  {
    acc = [||];
    stride = 0;
    x0 = 0;
    y0 = 0;
    x1 = 0;
    y1 = 0;
    dx0 = max_int;
    dy0 = max_int;
    dx1 = min_int;
    dy1 = min_int;
    edge = Array.create_float 5;
  }

(* [start c ~w ~h clip] prepares [c] for edges on a surface of [w] by [h] pixels
   in the window of [clip]. Two columns past the surface take what the edges of
   its last column spill. *)
let start c ~w ~h (clip : Surface.clip) =
  let stride = w + 2 in
  if Array.length c.acc < stride * h then c.acc <- Array.make (stride * h) 0.;
  c.stride <- stride;
  c.x0 <- clip.x0;
  c.y0 <- clip.y0;
  c.x1 <- clip.x1;
  c.y1 <- clip.y1;
  c.dx0 <- max_int;
  c.dy0 <- max_int;
  c.dx1 <- min_int;
  c.dy1 <- min_int

let is_untouched c = c.dx0 > c.dx1

(* [Float.min] and [Float.max] order NaNs and signed zeros at the price of a
   call; the coordinates here are finite. *)
let[@inline] fmin (a : float) b = if a < b then a else b
let[@inline] fmax (a : float) b = if a > b then a else b

(* [ceil_int v] is the least integer at least [v], for [v >= 0.]. *)
let[@inline] ceil_int v =
  let i = truncate v in
  if float i < v then i + 1 else i

(* [deposit c] adds the edge of [c.edge] from [(x0, y0)] down to [(x1, y1)], [y0
   < y1], with winding [dir], both ends within the window. *)
let deposit c =
  let e = c.edge in
  let dir = Array.unsafe_get e 0 in
  let x0 = Array.unsafe_get e 1 and y0 = Array.unsafe_get e 2 in
  let x1 = Array.unsafe_get e 3 and y1 = Array.unsafe_get e 4 in
  let acc = c.acc and stride = c.stride in
  let wx0 = float c.x0 and wx1 = float c.x1 in
  let dxdy = (x1 -. x0) /. (y1 -. y0) in
  let ystart = truncate y0 and ystop = ceil_int y1 in
  if ystart < c.dy0 then c.dy0 <- ystart;
  if ystop - 1 > c.dy1 then c.dy1 <- ystop - 1;
  let x = ref x0 in
  for y = ystart to ystop - 1 do
    let row = y * stride in
    (* Positive, since the row meets the edge's span of [y]. *)
    let dy = fmin (float (y + 1)) y1 -. fmax (float y) y0 in
    (* Rounding may step a hair outside the window, which indexes rely on. *)
    let xnext = fmin wx1 (fmax wx0 (!x +. (dxdy *. dy))) in
    let d = dy *. dir in
    let xa = fmin !x xnext and xb = fmax !x xnext in
    let xa_i = truncate xa in
    let xa_floor = float xa_i in
    let xb_i = ceil_int xb in
    if xa_i < c.dx0 then c.dx0 <- xa_i;
    if xb_i + 1 > c.dx1 then c.dx1 <- xb_i + 1;
    if xb_i <= xa_i + 1 then begin
      let xmf = (0.5 *. (!x +. xnext)) -. xa_floor in
      let i = row + xa_i in
      Array.unsafe_set acc i (Array.unsafe_get acc i +. d -. (d *. xmf));
      Array.unsafe_set acc (i + 1) (Array.unsafe_get acc (i + 1) +. (d *. xmf))
    end
    else begin
      let s = 1. /. (xb -. xa) in
      let xaf = xa -. xa_floor in
      let a0 = 0.5 *. s *. (1. -. xaf) *. (1. -. xaf) in
      let xbf = xb -. float xb_i +. 1. in
      let am = 0.5 *. s *. xbf *. xbf in
      let i = row + xa_i in
      Array.unsafe_set acc i (Array.unsafe_get acc i +. (d *. a0));
      if xb_i = xa_i + 2 then
        Array.unsafe_set acc (i + 1)
          (Array.unsafe_get acc (i + 1) +. (d *. (1. -. a0 -. am)))
      else begin
        let a1 = s *. (1.5 -. xaf) in
        Array.unsafe_set acc (i + 1)
          (Array.unsafe_get acc (i + 1) +. (d *. (a1 -. a0)));
        for xi = xa_i + 2 to xb_i - 2 do
          Array.unsafe_set acc (row + xi)
            (Array.unsafe_get acc (row + xi) +. (d *. s))
        done;
        let a2 = a1 +. (float (xb_i - xa_i - 3) *. s) in
        let j = row + xb_i - 1 in
        Array.unsafe_set acc j
          (Array.unsafe_get acc j +. (d *. (1. -. a2 -. am)))
      end;
      let j = row + xb_i in
      Array.unsafe_set acc j (Array.unsafe_get acc j +. (d *. am))
    end;
    x := xnext
  done

let[@inline] put c dir x0 y0 x1 y1 =
  let e = c.edge in
  Array.unsafe_set e 0 dir;
  Array.unsafe_set e 1 x0;
  Array.unsafe_set e 2 y0;
  Array.unsafe_set e 3 x1;
  Array.unsafe_set e 4 y1;
  deposit c

(* [span c dir x y0 y1] deposits the vertical edge at [x] from [y0] to [y1], if
   [y0 < y1]. *)
let[@inline] span c dir x y0 y1 = if y0 < y1 then put c dir x y0 x y1

(* [limit v] is [v] within [±1e15], NaN taken as [1e15], so that a coordinate
   overflowed far beyond the window stays beyond it and is finite. *)
let[@inline] limit v =
  if v < -1e15 then -1e15 else if v <= 1e15 then v else 1e15

(* [cut c] adds the edge [(x0, y0)] to [(x1, y1)] of [c.edge], cut to the
   window. *)
let cut c =
  let e = c.edge in
  let x0 = limit (Array.unsafe_get e 1) and y0 = limit (Array.unsafe_get e 2) in
  let x1 = limit (Array.unsafe_get e 3) and y1 = limit (Array.unsafe_get e 4) in
  let wy0 = float c.y0 and wy1 = float c.y1 in
  let down = y0 < y1 in
  (* The edge from its top [(ax, ay)] to its bottom [(bx, by)]. *)
  let ax = if down then x0 else x1 and ay = if down then y0 else y1 in
  let bx = if down then x1 else x0 and by = if down then y1 else y0 in
  if ay < by && by > wy0 && ay < wy1 then begin
    (* Coverage takes the winding's magnitude, whatever its sign. *)
    let dir = if down then 1. else -1. in
    (* Cut to the window's rows at fractions of the edge's height, below one, so
       that the cut stays finite however flat the edge. *)
    let h = by -. ay and w = bx -. ax in
    let top = fmax ay wy0 and bot = fmin by wy1 in
    let xt = if ay < wy0 then ax +. (w *. ((wy0 -. ay) /. h)) else ax in
    let xb = if by > wy1 then bx -. (w *. ((by -. wy1) /. h)) else bx in
    let wx0 = float c.x0 and wx1 = float c.x1 in
    if xt <= wx0 && xb <= wx0 then span c dir wx0 top bot
    else if xt >= wx1 && xb >= wx1 then span c dir wx1 top bot
    else if xt >= wx0 && xt <= wx1 && xb >= wx0 && xb <= wx1 then
      put c dir xt top xb bot
    else begin
      (* [ya] and [yb] are where the edge crosses the window's left and right
         sides, clamped to it; [xb <> xt] here. *)
      let hh = bot -. top and ww = xb -. xt in
      let ya = top +. (hh *. fmin 1. (fmax 0. ((wx0 -. xt) /. ww))) in
      let yb = top +. (hh *. fmin 1. (fmax 0. ((wx1 -. xt) /. ww))) in
      (* Going down, an edge running right crosses the left side, then the right
         one; one running left, the right side first. *)
      if xt < xb then begin
        span c dir wx0 top ya;
        if ya < yb then put c dir (fmax xt wx0) ya (fmin xb wx1) yb;
        span c dir wx1 yb bot
      end
      else begin
        span c dir wx1 top yb;
        if yb < ya then put c dir (fmin xt wx1) yb (fmax xb wx0) ya;
        span c dir wx0 ya bot
      end
    end
  end

(* [line c x0 y0 x1 y1] adds the edge from [(x0, y0)] to [(x1, y1)]. *)
let[@inline] line c x0 y0 x1 y1 =
  let e = c.edge in
  Array.unsafe_set e 1 x0;
  Array.unsafe_set e 2 y0;
  Array.unsafe_set e 3 x1;
  Array.unsafe_set e 4 y1;
  cut c

(* [coverage rule sum] is the fraction of a pixel of winding [sum] that the area
   of [rule] covers. *)
let[@inline] coverage rule sum =
  let c = Float.abs sum in
  match rule with
  | `Nonzero -> if c > 1. then 1. else c
  | `Even_odd ->
      let c = Float.rem c 2. in
      if c > 1. then 2. -. c else c

(* [paint c s clip rule (sr, sg, sb, sa)] composites the premultiplied colour
   over the pixels of [s] that the deposited edges cover, through [clip], and
   clears the accumulator. A coverage below a two-thousandth is none. *)
let paint c (s : Surface.t) (clip : Surface.clip) rule sr sg sb sa =
  if not (is_untouched c) then begin
    let acc = c.acc and stride = c.stride and px = s.px in
    for y = c.dy0 to c.dy1 do
      let row = y * stride in
      let sum = ref 0. in
      for x = c.dx0 to c.dx1 do
        let i = row + x in
        (* Coverage takes the winding's magnitude, whatever its sign. *)
        sum := !sum +. Array.unsafe_get acc i;
        Array.unsafe_set acc i 0.;
        (* The winding past the window's last column is zero. *)
        if x < c.x1 then begin
          let k = coverage rule !sum in
          if k > 0.0005 then
            let k = k *. Surface.mask_at clip x y in
            if k > 0. then Surface.blend px (((y * s.w) + x) * 4) sr sg sb sa k
        end
      done
    done;
    c.dx0 <- max_int;
    c.dx1 <- min_int
  end

(* [take c clip rule dst ~ox ~oy ~dw] writes into [dst], a window of rows of
   [dw] pixels from [(ox, oy)], the coverage of each pixel the deposited edges
   touch times what [clip] lets through, and clears the accumulator. *)
let take c (clip : Surface.clip) rule dst ~ox ~oy ~dw =
  if not (is_untouched c) then begin
    let acc = c.acc and stride = c.stride in
    for y = c.dy0 to c.dy1 do
      let row = y * stride in
      let sum = ref 0. in
      for x = c.dx0 to c.dx1 do
        let i = row + x in
        sum := !sum +. Array.unsafe_get acc i;
        Array.unsafe_set acc i 0.;
        if x < c.x1 then
          dst.(((y - oy) * dw) + (x - ox)) <-
            coverage rule !sum *. Surface.mask_at clip x y
      done
    done;
    c.dx0 <- max_int;
    c.dx1 <- min_int
  end

(* [clear c] clears what the deposited edges touched, unread. It is called when
   they touch only the columns past the window, where their deposits cancel, so
   that clearing guards against rounding alone. *)
let clear c =
  if not (is_untouched c) then begin
    for y = c.dy0 to c.dy1 do
      Array.fill c.acc ((y * c.stride) + c.dx0) (c.dx1 - c.dx0 + 1) 0.
    done;
    c.dx0 <- max_int;
    c.dx1 <- min_int
  end

(* [touched c] is the window's pixel rectangle that the deposited edges touch,
   [x1] and [y1] exclusive, empty when nothing was deposited. *)
let touched c =
  if is_untouched c then (0, 0, 0, 0)
  else (c.dx0, c.dy0, Int.min c.x1 (c.dx1 + 1), c.dy1 + 1)
