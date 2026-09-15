(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Paths to device-space polylines. *)

type polyline = { xs : float array; ys : float array; closed : bool }

(* Growable point buffer *)

type buf = {
  mutable bx : float array;
  mutable by : float array;
  mutable n : int;
}

let buf_create () = { bx = Array.make 16 0.; by = Array.make 16 0.; n = 0 }

let push b x y =
  if b.n = Array.length b.bx then begin
    let grow a = Array.append a (Array.make (Array.length a) 0.) in
    b.bx <- grow b.bx;
    b.by <- grow b.by
  end;
  Array.unsafe_set b.bx b.n x;
  Array.unsafe_set b.by b.n y;
  b.n <- b.n + 1

let last_x b = b.bx.(b.n - 1)
let last_y b = b.by.(b.n - 1)

let flush b ~closed acc =
  let out =
    if b.n = 0 then acc
    else { xs = Array.sub b.bx 0 b.n; ys = Array.sub b.by 0 b.n; closed } :: acc
  in
  b.n <- 0;
  out

(* Flattening tolerance, in device pixels. *)
let tolerance = 0.1

(* [path m p] is [p] under [m] as polylines in drawing order. Curves are split
   into enough lines to stay within [tolerance] of the true curve, and a segment
   with a non-finite endpoint ends its subpath. *)
let path m p =
  let tx x y = (m.Affine.xx *. x) +. (m.xy *. y) +. m.x0 in
  let ty x y = (m.Affine.yx *. x) +. (m.yy *. y) +. m.y0 in
  let b = buf_create () in
  let acc = ref [] in
  let move x y =
    acc := flush b ~closed:false !acc;
    if Float.is_finite x && Float.is_finite y then push b (tx x y) (ty x y)
  in
  let line x y =
    if Float.is_finite x && Float.is_finite y then push b (tx x y) (ty x y)
    else acc := flush b ~closed:false !acc
  in
  let curve c1x c1y c2x c2y x y =
    if
      not
        (Float.is_finite c1x && Float.is_finite c1y && Float.is_finite c2x
       && Float.is_finite c2y && Float.is_finite x && Float.is_finite y)
    then acc := flush b ~closed:false !acc
    else if b.n = 0 then push b (tx x y) (ty x y)
    else begin
      let x0 = last_x b and y0 = last_y b in
      let x1 = tx c1x c1y and y1 = ty c1x c1y in
      let x2 = tx c2x c2y and y2 = ty c2x c2y in
      let x3 = tx x y and y3 = ty x y in
      (* Wang's bound: this many chords keep the curve within [tolerance] of its
         second differences. *)
      let dd ax ay bx by cx cy =
        Float.hypot (ax -. (2. *. bx) +. cx) (ay -. (2. *. by) +. cy)
      in
      let d = Float.max (dd x0 y0 x1 y1 x2 y2) (dd x1 y1 x2 y2 x3 y3) in
      let n =
        Int.max 1
          (Int.min 256
             (int_of_float (Float.ceil (Float.sqrt (0.75 *. d /. tolerance)))))
      in
      for i = 1 to n do
        let t = float i /. float n in
        let u = 1. -. t in
        let a = u *. u *. u and bb = 3. *. u *. u *. t in
        let c = 3. *. u *. t *. t and dd = t *. t *. t in
        push b
          ((a *. x0) +. (bb *. x1) +. (c *. x2) +. (dd *. x3))
          ((a *. y0) +. (bb *. y1) +. (c *. y2) +. (dd *. y3))
      done
    end
  in
  let close () = acc := flush b ~closed:true !acc in
  Path.fold
    ~move:(fun () x y -> move x y)
    ~line:(fun () x y -> line x y)
    ~curve:(fun () a b' c d e f -> curve a b' c d e f)
    ~close:(fun () -> close ())
    () p;
  acc := flush b ~closed:false !acc;
  List.rev !acc

(* [bounds polys] is the bounding box of [polys], or [None] when they have no
   points. *)
let bounds polys =
  let x0 = ref infinity and y0 = ref infinity in
  let x1 = ref neg_infinity and y1 = ref neg_infinity in
  List.iter
    (fun { xs; ys; _ } ->
      for i = 0 to Array.length xs - 1 do
        let x = Array.unsafe_get xs i and y = Array.unsafe_get ys i in
        if x < !x0 then x0 := x;
        if x > !x1 then x1 := x;
        if y < !y0 then y0 := y;
        if y > !y1 then y1 := y
      done)
    polys;
  if !x0 > !x1 then None else Some (!x0, !y0, !x1, !y1)
