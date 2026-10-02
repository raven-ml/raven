(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { minx : float; miny : float; maxx : float; maxy : float }

let finite4 a b c d =
  Float.is_finite a && Float.is_finite b && Float.is_finite c
  && Float.is_finite d

let v x y w h =
  let maxx = x +. w and maxy = y +. h in
  if not (finite4 x y maxx maxy) then
    invalid_arg
      (Printf.sprintf "Box2.v: corner not finite in %g %g %g %g" x y w h);
  if w < 0. || h < 0. then
    invalid_arg (Printf.sprintf "Box2.v: negative size %g x %g" w h);
  { minx = x; miny = y; maxx; maxy }

let of_pts p q =
  let px = P2.x p and py = P2.y p and qx = P2.x q and qy = P2.y q in
  if not (finite4 px py qx qy) then
    invalid_arg
      (Printf.sprintf "Box2.of_pts: corner not finite in (%g, %g) (%g, %g)" px
         py qx qy);
  {
    minx = Float.min px qx;
    miny = Float.min py qy;
    maxx = Float.max px qx;
    maxy = Float.max py qy;
  }

let minx b = b.minx
let miny b = b.miny
let maxx b = b.maxx
let maxy b = b.maxy
let w b = b.maxx -. b.minx
let h b = b.maxy -. b.miny

(* Halving each corner first keeps the sum finite. *)
let mid b =
  P2.v ((0.5 *. b.minx) +. (0.5 *. b.maxx)) ((0.5 *. b.miny) +. (0.5 *. b.maxy))

let union a b =
  {
    minx = Float.min a.minx b.minx;
    miny = Float.min a.miny b.miny;
    maxx = Float.max a.maxx b.maxx;
    maxy = Float.max a.maxy b.maxy;
  }

let inter a b =
  let minx = Float.max a.minx b.minx and miny = Float.max a.miny b.miny in
  let maxx = Float.min a.maxx b.maxx and maxy = Float.min a.maxy b.maxy in
  if minx <= maxx && miny <= maxy then Some { minx; miny; maxx; maxy } else None

let grow d b =
  let minx = b.minx -. d and miny = b.miny -. d in
  let maxx = b.maxx +. d and maxy = b.maxy +. d in
  if not (finite4 minx miny maxx maxy) then
    invalid_arg (Printf.sprintf "Box2.grow: corner not finite growing by %g" d);
  if minx > maxx || miny > maxy then
    invalid_arg (Printf.sprintf "Box2.grow: %g shrinks past a side" d);
  { minx; miny; maxx; maxy }

let transform m b =
  let p = P2.transform m (P2.v b.minx b.miny) in
  let q = P2.transform m (P2.v b.maxx b.miny) in
  let r = P2.transform m (P2.v b.maxx b.maxy) in
  let s = P2.transform m (P2.v b.minx b.maxy) in
  let minx =
    Float.min (Float.min (P2.x p) (P2.x q)) (Float.min (P2.x r) (P2.x s))
  and miny =
    Float.min (Float.min (P2.y p) (P2.y q)) (Float.min (P2.y r) (P2.y s))
  and maxx =
    Float.max (Float.max (P2.x p) (P2.x q)) (Float.max (P2.x r) (P2.x s))
  and maxy =
    Float.max (Float.max (P2.y p) (P2.y q)) (Float.max (P2.y r) (P2.y s))
  in
  if not (finite4 minx miny maxx maxy) then
    invalid_arg "Box2.transform: image not finite";
  { minx; miny; maxx; maxy }

let equal a b =
  Float.equal a.minx b.minx && Float.equal a.miny b.miny
  && Float.equal a.maxx b.maxx && Float.equal a.maxy b.maxy

let compare a b =
  let c = Float.compare a.minx b.minx in
  if c <> 0 then c
  else
    let c = Float.compare a.miny b.miny in
    if c <> 0 then c
    else
      let c = Float.compare a.maxx b.maxx in
      if c <> 0 then c else Float.compare a.maxy b.maxy

let pp ppf b =
  Format.fprintf ppf "@[<1>[(%g, %g)@ (%g, %g)]@]" b.minx b.miny b.maxx b.maxy
