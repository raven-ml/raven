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
let translate x y = { id with x0 = x; y0 = y }
let scale sx sy = { id with xx = sx; yy = sy }

let rotate a =
  let c = Float.cos a and s = Float.sin a in
  { xx = c; yx = s; xy = -.s; yy = c; x0 = 0.; y0 = 0. }

let ( * ) a b =
  {
    xx = (a.xx *. b.xx) +. (a.xy *. b.yx);
    yx = (a.yx *. b.xx) +. (a.yy *. b.yx);
    xy = (a.xx *. b.xy) +. (a.xy *. b.yy);
    yy = (a.yx *. b.xy) +. (a.yy *. b.yy);
    x0 = (a.xx *. b.x0) +. (a.xy *. b.y0) +. a.x0;
    y0 = (a.yx *. b.x0) +. (a.yy *. b.y0) +. a.y0;
  }

let apply m x y =
  ((m.xx *. x) +. (m.xy *. y) +. m.x0, (m.yx *. x) +. (m.yy *. y) +. m.y0)

let invert m =
  let det = (m.xx *. m.yy) -. (m.xy *. m.yx) in
  if det = 0. then invalid_arg "Affine.invert: singular transform";
  let xx = m.yy /. det and xy = -.m.xy /. det in
  let yx = -.m.yx /. det and yy = m.xx /. det in
  {
    xx;
    yx;
    xy;
    yy;
    x0 = -.((xx *. m.x0) +. (xy *. m.y0));
    y0 = -.((yx *. m.x0) +. (yy *. m.y0));
  }

let is_translation m = m.xx = 1. && m.yx = 0. && m.xy = 0. && m.yy = 1.
