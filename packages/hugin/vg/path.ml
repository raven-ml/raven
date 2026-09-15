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

(* Flattening *)

let default_tolerance = 0.1

let flatten ?(tolerance = default_tolerance) m ~move ~line ~close acc p =
  let tx x y = (m.Affine.xx *. x) +. (m.xy *. y) +. m.x0 in
  let ty x y = (m.Affine.yx *. x) +. (m.yy *. y) +. m.y0 in
  let finite x y = Float.is_finite x && Float.is_finite y in
  (* The current point in device space, or none after a break. *)
  let cur = ref None in
  let acc = ref acc in
  let start x y =
    cur := Some (x, y);
    acc := move !acc x y
  in
  let extend x y =
    cur := Some (x, y);
    acc := line !acc x y
  in
  let break () = cur := None in
  fold
    ~move:(fun () x y ->
      if finite x y then start (tx x y) (ty x y) else break ())
    ~line:(fun () x y ->
      if not (finite x y) then break ()
      else
        let x' = tx x y and y' = ty x y in
        match !cur with None -> start x' y' | Some _ -> extend x' y')
    ~curve:(fun () c1x c1y c2x c2y x y ->
      if not (finite c1x c1y && finite c2x c2y && finite x y) then break ()
      else
        let x3 = tx x y and y3 = ty x y in
        match !cur with
        | None -> start x3 y3
        | Some (x0, y0) ->
            let x1 = tx c1x c1y and y1 = ty c1x c1y in
            let x2 = tx c2x c2y and y2 = ty c2x c2y in
            (* Wang's bound: this many chords keep the curve within the
               tolerance of its second differences. *)
            let dd ax ay bx by cx cy =
              Float.hypot (ax -. (2. *. bx) +. cx) (ay -. (2. *. by) +. cy)
            in
            let d = Float.max (dd x0 y0 x1 y1 x2 y2) (dd x1 y1 x2 y2 x3 y3) in
            let n =
              Int.max 1
                (Int.min 256
                   (int_of_float
                      (Float.ceil (Float.sqrt (0.75 *. d /. tolerance)))))
            in
            for i = 1 to n do
              let t = float i /. float n in
              let u = 1. -. t in
              let a = u *. u *. u and b = 3. *. u *. u *. t in
              let c = 3. *. u *. t *. t and d = t *. t *. t in
              extend
                ((a *. x0) +. (b *. x1) +. (c *. x2) +. (d *. x3))
                ((a *. y0) +. (b *. y1) +. (c *. y2) +. (d *. y3))
            done)
    ~close:(fun () ->
      if !cur <> None then begin
        acc := close !acc;
        break ()
      end)
    () p;
  !acc

let bounds p =
  let grow acc x y =
    match acc with
    | None -> Some (Box.v x y x y)
    | Some b -> Some (Box.union b (Box.v x y x y))
  in
  flatten Affine.id ~move:grow ~line:grow ~close:Fun.id None p

let pp fmt p =
  let first = ref true in
  let sep () =
    if !first then first := false else Format.pp_print_space fmt ()
  in
  Format.pp_open_box fmt 0;
  fold
    ~move:(fun () x y ->
      sep ();
      Format.fprintf fmt "M %g %g" x y)
    ~line:(fun () x y ->
      sep ();
      Format.fprintf fmt "L %g %g" x y)
    ~curve:(fun () c1x c1y c2x c2y x y ->
      sep ();
      Format.fprintf fmt "C %g %g %g %g %g %g" c1x c1y c2x c2y x y)
    ~close:(fun () ->
      sep ();
      Format.pp_print_string fmt "Z")
    () p;
  Format.pp_close_box fmt ()
