(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { x : float; y : float }

let v x y = { x; y }
let x p = p.x
let y p = p.y

let transform (m : Affine.t) p =
  {
    x = (m.xx *. p.x) +. (m.xy *. p.y) +. m.x0;
    y = (m.yx *. p.x) +. (m.yy *. p.y) +. m.y0;
  }

let equal p q = Float.equal p.x q.x && Float.equal p.y q.y

let compare p q =
  let c = Float.compare p.x q.x in
  if c <> 0 then c else Float.compare p.y q.y

let pp ppf p = Format.fprintf ppf "(%g, %g)" p.x p.y
