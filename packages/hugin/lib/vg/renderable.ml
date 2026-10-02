(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { w : float; h : float; picture : Picture.t }

let v w h picture =
  let valid x = x > 0. && Float.is_finite x in
  if not (valid w && valid h) then
    invalid_arg (Printf.sprintf "Renderable.v: invalid page size %g x %g" w h);
  { w; h; picture }

let w r = r.w
let h r = r.h
let picture r = r.picture

let equal r r' =
  Float.equal r.w r'.w && Float.equal r.h r'.h
  && Picture.equal r.picture r'.picture

let pp ppf r =
  Format.fprintf ppf "@[<1>(renderable %g %g@ %a)@]" r.w r.h Picture.pp
    r.picture
