(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { x0 : float; y0 : float; x1 : float; y1 : float }

let v x0 y0 x1 y1 =
  {
    x0 = Float.min x0 x1;
    y0 = Float.min y0 y1;
    x1 = Float.max x0 x1;
    y1 = Float.max y0 y1;
  }

let width b = b.x1 -. b.x0
let height b = b.y1 -. b.y0

let union a b =
  {
    x0 = Float.min a.x0 b.x0;
    y0 = Float.min a.y0 b.y0;
    x1 = Float.max a.x1 b.x1;
    y1 = Float.max a.y1 b.y1;
  }

let transform m b =
  let corner x y = Affine.apply m x y in
  let ax, ay = corner b.x0 b.y0 and bx, by = corner b.x1 b.y0 in
  let cx, cy = corner b.x1 b.y1 and dx, dy = corner b.x0 b.y1 in
  {
    x0 = Float.min (Float.min ax bx) (Float.min cx dx);
    y0 = Float.min (Float.min ay by) (Float.min cy dy);
    x1 = Float.max (Float.max ax bx) (Float.max cx dx);
    y1 = Float.max (Float.max ay by) (Float.max cy dy);
  }
