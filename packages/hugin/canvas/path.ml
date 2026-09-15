(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type op =
  | Move of float * float
  | Line of float * float
  | Curve of float * float * float * float * float * float
  | Close
  | Poly of { xs : float array; ys : float array; closed : bool }

(* Operations in reverse drawing order, so that building is a cons. *)
type t = op list

let empty = []
let is_empty p = p = []

(* A segment needs a current point; without one it opens a subpath. *)
let has_current_point = function
  | [] | Close :: _ -> false
  | (Move _ | Line _ | Curve _ | Poly _) :: _ -> true

let move_to x y p = Move (x, y) :: p

let line_to x y p =
  if has_current_point p then Line (x, y) :: p else Move (x, y) :: p

let curve_to c1x c1y c2x c2y x y p =
  if has_current_point p then Curve (c1x, c1y, c2x, c2y, x, y) :: p
  else Move (x, y) :: p

let close p =
  match p with
  | [] | Close :: _ | Move _ :: _ -> p
  | Poly ({ closed = false; _ } as poly) :: rest ->
      Poly { poly with closed = true } :: rest
  | Poly { closed = true; _ } :: _ -> p
  | (Line _ | Curve _) :: _ -> Close :: p

let append p q = q @ p

let transform m p =
  let tx x y = (m.Affine.xx *. x) +. (m.xy *. y) +. m.x0 in
  let ty x y = (m.Affine.yx *. x) +. (m.yy *. y) +. m.y0 in
  List.map
    (function
      | Move (x, y) -> Move (tx x y, ty x y)
      | Line (x, y) -> Line (tx x y, ty x y)
      | Curve (c1x, c1y, c2x, c2y, x, y) ->
          Curve (tx c1x c1y, ty c1x c1y, tx c2x c2y, ty c2x c2y, tx x y, ty x y)
      | Close -> Close
      | Poly { xs; ys; closed } ->
          let n = Array.length xs in
          let xs' = Array.make n 0. and ys' = Array.make n 0. in
          for i = 0 to n - 1 do
            let x = Array.unsafe_get xs i and y = Array.unsafe_get ys i in
            Array.unsafe_set xs' i (tx x y);
            Array.unsafe_set ys' i (ty x y)
          done;
          Poly { xs = xs'; ys = ys'; closed })
    p

let rect x y w h =
  [
    Close; Line (x, y +. h); Line (x +. w, y +. h); Line (x +. w, y); Move (x, y);
  ]

(* Four cubic arcs approximate a circle to within 0.03% of the radius. *)
let kappa = 0.5522847498307936

let circle cx cy r =
  let k = kappa *. r in
  [
    Close;
    Curve (cx +. k, cy -. r, cx +. r, cy -. k, cx +. r, cy);
    Curve (cx -. r, cy -. k, cx -. k, cy -. r, cx, cy -. r);
    Curve (cx -. k, cy +. r, cx -. r, cy +. k, cx -. r, cy);
    Curve (cx +. r, cy +. k, cx +. k, cy +. r, cx, cy +. r);
    Move (cx +. r, cy);
  ]

let poly ~closed xs ys =
  if Array.length xs <> Array.length ys then
    invalid_arg "Path.polyline: xs and ys differ in length";
  if Array.length xs < 2 then [] else [ Poly { xs; ys; closed } ]

let polyline xs ys = poly ~closed:false xs ys
let polygon xs ys = poly ~closed:true xs ys

let fold ~move ~line ~curve ~close acc p =
  List.fold_left
    (fun acc op ->
      match op with
      | Move (x, y) -> move acc x y
      | Line (x, y) -> line acc x y
      | Curve (c1x, c1y, c2x, c2y, x, y) -> curve acc c1x c1y c2x c2y x y
      | Close -> close acc
      | Poly { xs; ys; closed } ->
          let acc = ref (move acc xs.(0) ys.(0)) in
          for i = 1 to Array.length xs - 1 do
            acc := line !acc (Array.unsafe_get xs i) (Array.unsafe_get ys i)
          done;
          if closed then close !acc else !acc)
    acc (List.rev p)
