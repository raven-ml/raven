(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type cap = [ `Butt | `Round | `Square ]
type join = [ `Miter | `Round | `Bevel ]

type t = {
  width : float;
  cap : cap;
  join : join;
  dash : float array;
  miter_limit : float;
}

let v ?(cap = `Round) ?(join = `Round) ?(dash = [||]) ?(miter_limit = 4.) width
    =
  if not (width >= 0.) then invalid_arg "Stroke.v: negative width";
  if not (miter_limit >= 1.) then invalid_arg "Stroke.v: miter limit below 1";
  if Array.length dash > 0 then begin
    if not (Array.for_all (fun d -> d >= 0.) dash) then
      invalid_arg "Stroke.v: negative dash length";
    if not (Array.fold_left ( +. ) 0. dash > 0.) then
      invalid_arg "Stroke.v: dash pattern sums to zero"
  end;
  { width; cap; join; dash; miter_limit }
