(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  xx : float;
  yx : float;
  xy : float;
  yy : float;
  x0 : float;
  y0 : float;
}

let id = { xx = 1.; yx = 0.; xy = 0.; yy = 1.; x0 = 0.; y0 = 0. }
let translate dx dy = { id with x0 = dx; y0 = dy }
let scale sx sy = { id with xx = sx; yy = sy }

let rotate a =
  let c = Float.cos a and s = Float.sin a in
  { xx = c; yx = s; xy = -.s; yy = c; x0 = 0.; y0 = 0. }

let ( * ) m n =
  {
    xx = (m.xx *. n.xx) +. (m.xy *. n.yx);
    yx = (m.yx *. n.xx) +. (m.yy *. n.yx);
    xy = (m.xx *. n.xy) +. (m.xy *. n.yy);
    yy = (m.yx *. n.xy) +. (m.yy *. n.yy);
    x0 = (m.xx *. n.x0) +. (m.xy *. n.y0) +. m.x0;
    y0 = (m.yx *. n.x0) +. (m.yy *. n.y0) +. m.y0;
  }

(* The linear part is divided by its largest magnitude [s] so that the
   determinant neither overflows nor underflows. One of [a], [b], [c], [d] is
   then [1.] or [-1.], so a zero determinant makes a coefficient non-finite. *)
let invert m =
  let s =
    Float.max
      (Float.max (Float.abs m.xx) (Float.abs m.yx))
      (Float.max (Float.abs m.xy) (Float.abs m.yy))
  in
  let a = m.xx /. s and b = m.xy /. s and c = m.yx /. s and d = m.yy /. s in
  let det = (a *. d) -. (b *. c) in
  let xx = d /. det /. s and xy = -.b /. det /. s in
  let yx = -.c /. det /. s and yy = a /. det /. s in
  let x0 = -.((xx *. m.x0) +. (xy *. m.y0)) in
  let y0 = -.((yx *. m.x0) +. (yy *. m.y0)) in
  let finite = Float.is_finite in
  if finite xx && finite yx && finite xy && finite yy && finite x0 && finite y0
  then Some { xx; yx; xy; yy; x0; y0 }
  else None

let equal m n =
  Float.equal m.xx n.xx && Float.equal m.yx n.yx && Float.equal m.xy n.xy
  && Float.equal m.yy n.yy && Float.equal m.x0 n.x0 && Float.equal m.y0 n.y0

let compare m n =
  let c = Float.compare m.xx n.xx in
  if c <> 0 then c
  else
    let c = Float.compare m.yx n.yx in
    if c <> 0 then c
    else
      let c = Float.compare m.xy n.xy in
      if c <> 0 then c
      else
        let c = Float.compare m.yy n.yy in
        if c <> 0 then c
        else
          let c = Float.compare m.x0 n.x0 in
          if c <> 0 then c else Float.compare m.y0 n.y0

let pp ppf m =
  Format.fprintf ppf "@[<1>(%g %g %g %g %g %g)@]" m.xx m.yx m.xy m.yy m.x0 m.y0
