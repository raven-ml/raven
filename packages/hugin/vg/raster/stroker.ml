(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_vg

(* Strokes to fill polygons. Each segment body, join and cap becomes its own
   closed polygon with positive orientation, so the nonzero union of the pieces
   is the stroked outline and shared edges cancel exactly. *)

open Flatten

(* [emit xs ys acc] adds the polygon, reversed if needed so that its signed area
   is positive. *)
let emit xs ys acc =
  let n = Array.length xs in
  let area = ref 0. in
  for i = 0 to n - 1 do
    let j = if i = n - 1 then 0 else i + 1 in
    area := !area +. (xs.(i) *. ys.(j)) -. (xs.(j) *. ys.(i))
  done;
  if !area = 0. then acc
  else if !area > 0. then { xs; ys; closed = true } :: acc
  else begin
    let rev a = Array.init n (fun i -> a.(n - 1 - i)) in
    { xs = rev xs; ys = rev ys; closed = true } :: acc
  end

(* A regular polygon whose vertices straddle the circle so that its area is the
   disc's; the chord count keeps it within the flattening tolerance. *)
let circle cx cy r acc =
  let n =
    if r <= Flatten.tolerance then 8
    else
      Int.max 8
        (Int.min 128
           (int_of_float
              (Float.ceil
                 (Float.pi /. Float.acos (1. -. (Flatten.tolerance /. r))))))
  in
  let step = 2. *. Float.pi /. float n in
  let r = r *. Float.sqrt (step /. Float.sin step) in
  let xs = Array.make n 0. and ys = Array.make n 0. in
  for i = 0 to n - 1 do
    let a = step *. float i in
    xs.(i) <- cx +. (r *. Float.cos a);
    ys.(i) <- cy +. (r *. Float.sin a)
  done;
  emit xs ys acc

(* Dashing *)

(* [dash pattern poly] splits [poly] into the polylines its drawn dashes cover,
   restarting the pattern at each subpath. *)
let dash pattern (poly : polyline) =
  let pattern =
    if Array.length pattern mod 2 = 1 then Array.append pattern pattern
    else pattern
  in
  let np = Array.length pattern in
  let n = Array.length poly.xs in
  let px i = poly.xs.(i mod n) and py i = poly.ys.(i mod n) in
  let last = if poly.closed then n else n - 1 in
  let out = ref [] in
  let cur = buf_create () in
  let idx = ref 0 and remaining = ref pattern.(0) in
  let on () = !idx mod 2 = 0 in
  for i = 0 to last - 1 do
    let x0 = px i and y0 = py i and x1 = px (i + 1) and y1 = py (i + 1) in
    let len = Float.hypot (x1 -. x0) (y1 -. y0) in
    let pos = ref 0. in
    if on () && cur.n = 0 then push cur x0 y0;
    while !pos < len do
      let step = Float.min !remaining (len -. !pos) in
      pos := !pos +. step;
      remaining := !remaining -. step;
      let t = !pos /. len in
      let x = x0 +. (t *. (x1 -. x0)) and y = y0 +. (t *. (y1 -. y0)) in
      if on () then push cur x y;
      if !remaining <= 0. then begin
        if on () then out := flush cur ~closed:false !out;
        idx := (!idx + 1) mod np;
        remaining := pattern.(!idx);
        if on () then push cur x y
      end
    done
  done;
  List.rev (flush cur ~closed:false !out)

(* Outline *)

let outline (s : Stroke.t) ~scale polys =
  let hw = s.width *. scale /. 2. in
  let acc = ref [] in
  let stroke_one (poly : polyline) =
    (* Drop points within a twentieth of a pixel of the previous one, so that
       every segment has a direction and dense data does not pay for a join per
       sub-pixel step. *)
    let b = buf_create () in
    let n = Array.length poly.xs in
    for i = 0 to n - 1 do
      let x = poly.xs.(i) and y = poly.ys.(i) in
      if
        b.n = 0
        || Float.abs (x -. last_x b) > 0.05
        || Float.abs (y -. last_y b) > 0.05
      then push b x y
    done;
    if poly.closed && b.n > 1 && b.bx.(0) = last_x b && b.by.(0) = last_y b then
      b.n <- b.n - 1;
    let n = b.n in
    let xs = b.bx and ys = b.by in
    if n = 1 then
      (* A dot: only caps give it extent. *)
      begin match s.cap with
      | `Round -> acc := circle xs.(0) ys.(0) hw !acc
      | `Square ->
          let x = xs.(0) and y = ys.(0) in
          acc :=
            emit
              [| x -. hw; x +. hw; x +. hw; x -. hw |]
              [| y -. hw; y -. hw; y +. hw; y +. hw |]
              !acc
      | `Butt -> ()
      end
    else if n > 1 then begin
      let closed = poly.closed && n > 2 in
      let segs = if closed then n else n - 1 in
      let px i = xs.(i mod n) and py i = ys.(i mod n) in
      (* Unit direction of segment [i]. *)
      let dir i =
        let dx = px (i + 1) -. px i and dy = py (i + 1) -. py i in
        let l = Float.hypot dx dy in
        (dx /. l, dy /. l)
      in
      for i = 0 to segs - 1 do
        let dx, dy = dir i in
        let nx = -.dy *. hw and ny = dx *. hw in
        let x0 = px i and y0 = py i and x1 = px (i + 1) and y1 = py (i + 1) in
        acc :=
          emit
            [| x0 +. nx; x1 +. nx; x1 -. nx; x0 -. nx |]
            [| y0 +. ny; y1 +. ny; y1 -. ny; y0 -. ny |]
            !acc
      done;
      let join i =
        (* Join at vertex [i] between segments [i - 1] and [i]. *)
        let x = px i and y = py i in
        let d1x, d1y = dir ((i - 1 + segs) mod segs) and d2x, d2y = dir i in
        let cross = (d1x *. d2y) -. (d1y *. d2x) in
        let dot = (d1x *. d2x) +. (d1y *. d2y) in
        (* A join between nearly collinear segments leaves a notch narrower than
           the flattening tolerance, so it is not worth a polygon. *)
        if dot > 0. && Float.abs cross *. hw < Flatten.tolerance then ()
        else
          match s.join with
          | `Round -> acc := circle x y hw !acc
          | (`Bevel | `Miter) as j ->
              if cross <> 0. then begin
                (* The outer side is opposite to the turn. *)
                let side = if cross > 0. then -.hw else hw in
                let n1x = -.d1y *. side and n1y = d1x *. side in
                let n2x = -.d2y *. side and n2y = d2x *. side in
                let cos_t = dot in
                let miter_ratio = 1. /. Float.sqrt ((1. +. cos_t) /. 2.) in
                if j = `Miter && cos_t > -1. && miter_ratio <= s.miter_limit
                then begin
                  let k = 1. /. (1. +. cos_t) in
                  let mx = (n1x +. n2x) *. k and my = (n1y +. n2y) *. k in
                  acc :=
                    emit
                      [| x; x +. n1x; x +. mx; x +. n2x |]
                      [| y; y +. n1y; y +. my; y +. n2y |]
                      !acc
                end
                else
                  acc :=
                    emit
                      [| x; x +. n1x; x +. n2x |]
                      [| y; y +. n1y; y +. n2y |]
                      !acc
              end
      in
      for i = 1 to segs - 1 do
        join i
      done;
      if closed then join 0
      else begin
        let cap i (dx, dy) =
          (* Cap at vertex [i] with [(dx, dy)] pointing away from the line. *)
          let x = px i and y = py i in
          match s.cap with
          | `Round -> acc := circle x y hw !acc
          | `Square ->
              let nx = -.dy *. hw and ny = dx *. hw in
              let ex = dx *. hw and ey = dy *. hw in
              acc :=
                emit
                  [| x +. nx; x +. nx +. ex; x -. nx +. ex; x -. nx |]
                  [| y +. ny; y +. ny +. ey; y -. ny +. ey; y -. ny |]
                  !acc
          | `Butt -> ()
        in
        let dx, dy = dir 0 in
        cap 0 (-.dx, -.dy);
        cap (n - 1) (dir (n - 2))
      end
    end
  in
  List.iter
    (fun poly ->
      if Array.length s.dash = 0 then stroke_one poly
      else
        List.iter stroke_one
          (dash (Array.map (fun d -> d *. scale) s.dash) poly))
    polys;
  !acc
