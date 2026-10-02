(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

type t =
  | Circle
  | Square
  | Diamond
  | Triangle
  | Cross
  | Star
  | Wye
  | Plus
  | Times
  | Asterisk

let circle = Circle
let square = Square
let diamond = Diamond
let triangle = Triangle
let cross = Cross
let star = Star
let wye = Wye
let plus = Plus
let times = Times
let asterisk = Asterisk
let filled = [ Circle; Cross; Diamond; Square; Star; Triangle; Wye ]
let stroked = [ Circle; Plus; Times; Triangle; Asterisk; Square; Diamond ]

(* Outlines of size 1 *)

let sqrt3 = Float.sqrt 3.

(* The length of the outline of the circle of area 1. *)
let ink = 2. *. Float.sqrt Float.pi

(* A closed outline enclosing area 1, clockwise on screen, and the factor that
   scales it to an outline of length [ink]. *)
type closed = { xs : float array; ys : float array; stroke : float }

let closed pts =
  let xs = Array.map fst pts and ys = Array.map snd pts in
  let n = Array.length xs in
  let length = ref 0. in
  for i = 0 to n - 1 do
    let j = (i + 1) mod n in
    length := !length +. Float.hypot (xs.(j) -. xs.(i)) (ys.(j) -. ys.(i))
  done;
  { xs; ys; stroke = ink /. !length }

let square_outline =
  closed [| (-0.5, -0.5); (0.5, -0.5); (0.5, 0.5); (-0.5, 0.5) |]

(* Two equilateral triangles of height [y] base to base. *)
let diamond_outline =
  let y = Float.sqrt (sqrt3 /. 2.) in
  let x = y /. sqrt3 in
  closed [| (0., -.y); (x, 0.); (0., y); (-.x, 0.) |]

(* The centroid of an equilateral triangle is a third of its height above its
   base. *)
let triangle_outline =
  let h = Float.sqrt (1. /. (3. *. sqrt3)) in
  closed [| (0., -2. *. h); (sqrt3 *. h, h); (-.sqrt3 *. h, h) |]

(* Five squares of side [2 r]. *)
let cross_outline =
  let r = Float.sqrt (1. /. 5.) /. 2. in
  closed
    [|
      (-3. *. r, -.r);
      (-.r, -.r);
      (-.r, -3. *. r);
      (r, -3. *. r);
      (r, -.r);
      (3. *. r, -.r);
      (3. *. r, r);
      (r, r);
      (r, 3. *. r);
      (-.r, 3. *. r);
      (-.r, r);
      (-3. *. r, r);
    |]

(* Points at radius [r] and inner corners at radius [k r], on the lines that
   join the points. The ten triangles from the centre enclose [5 k r² sin (π /
   5)]. *)
let star_outline =
  let k = Float.sin (Float.pi /. 10.) /. Float.sin (7. *. Float.pi /. 10.) in
  let r = 1. /. Float.sqrt (5. *. k *. Float.sin (Float.pi /. 5.)) in
  closed
    (Array.init 10 (fun i ->
         let a = float i *. Float.pi /. 5. in
         let r = if i mod 2 = 0 then r else k *. r in
         (r *. Float.sin a, -.r *. Float.cos a)))

(* A triangle of side [r] with an [r × r] square on each side, one arm down: [3
   r² + √3 r² / 4] enclosed. *)
let wye_outline =
  let k = 1. /. Float.sqrt 12. in
  let r = 1. /. Float.sqrt (3. *. ((k /. 2.) +. 1.)) in
  let arm =
    [|
      (r /. 2., r *. k); (r /. 2., (r *. k) +. r); (-.r /. 2., (r *. k) +. r);
    |]
  in
  let c = -0.5 and s = sqrt3 /. 2. in
  let turn (x, y) = ((c *. x) -. (s *. y), (s *. x) +. (c *. y)) in
  let back (x, y) = ((c *. x) +. (s *. y), (c *. y) -. (s *. x)) in
  closed (Array.concat [ arm; Array.map turn arm; Array.map back arm ])

(* Strokes of total length [ink], from one end to the other. *)
let plus_strokes =
  let r = ink /. 4. in
  [ (-.r, 0., r, 0.); (0., r, 0., -.r) ]

let times_strokes =
  let r = ink /. (4. *. Float.sqrt 2.) in
  [ (-.r, -.r, r, r); (-.r, r, r, -.r) ]

let asterisk_strokes =
  let r = ink /. 6. in
  let t = r /. 2. in
  let u = t *. sqrt3 in
  [ (0., r, 0., -.r); (-.u, -.t, u, t); (-.u, t, u, -.t) ]

(* Drawing *)

type paint = [ `Fill | `Stroke ]

let polygon paint k o =
  let k = match paint with `Fill -> k | `Stroke -> k *. o.stroke in
  Path.polygon (Array.map (( *. ) k) o.xs) (Array.map (( *. ) k) o.ys)

let strokes k ss =
  List.fold_left
    (fun p (x0, y0, x1, y1) ->
      p
      |> Path.move_to (P2.v (k *. x0) (k *. y0))
      |> Path.line_to (P2.v (k *. x1) (k *. y1)))
    Path.empty ss

let path paint a s =
  if not (a >= 0. && Float.is_finite a) then
    invalid_arg (Printf.sprintf "Symbol.path: invalid size %g" a);
  let k = Float.sqrt a in
  match s with
  | Circle -> Path.circle (P2.v 0. 0.) (Float.sqrt (a /. Float.pi))
  | Square -> polygon paint k square_outline
  | Diamond -> polygon paint k diamond_outline
  | Triangle -> polygon paint k triangle_outline
  | Cross -> polygon paint k cross_outline
  | Star -> polygon paint k star_outline
  | Wye -> polygon paint k wye_outline
  | Plus -> strokes k plus_strokes
  | Times -> strokes k times_strokes
  | Asterisk -> strokes k asterisk_strokes

(* Comparing and formatting *)

let equal (s : t) s' = s = s'

let pp ppf s =
  Format.pp_print_string ppf
    (match s with
    | Circle -> "circle"
    | Square -> "square"
    | Diamond -> "diamond"
    | Triangle -> "triangle"
    | Cross -> "cross"
    | Star -> "star"
    | Wye -> "wye"
    | Plus -> "plus"
    | Times -> "times"
    | Asterisk -> "asterisk")
