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
  miter_limit : float;
  dash : float list;
  dash_offset : float;
}

let err fmt = Printf.ksprintf invalid_arg ("Stroke.v: " ^^ fmt)

(* The length of the pattern made even, or [0.] for a solid line. *)
let period dash =
  let sum = List.fold_left ( +. ) 0. dash in
  if List.length dash mod 2 = 1 then 2. *. sum else sum

let v ?(cap = `Round) ?(join = `Round) ?(miter_limit = 4.) ?(dash = [])
    ?(dash_offset = 0.) width =
  if not (width >= 0. && Float.is_finite width) then
    err "invalid width %g" width;
  if not (miter_limit >= 1. && Float.is_finite miter_limit) then
    err "invalid miter limit %g" miter_limit;
  if not (Float.is_finite dash_offset) then
    err "dash offset %g not finite" dash_offset;
  List.iter (fun l -> if not (l >= 0.) then err "invalid dash length %g" l) dash;
  let dash_offset =
    match dash with
    | [] -> 0.
    | _ :: _ ->
        let l = period dash in
        if not (l > 0. && Float.is_finite l) then
          err "dash pattern sums to %g" l;
        (* [o +. l] is [l] for a zero [o], and for a tiny negative one by
           rounding: both are [0.] modulo [l]. *)
        let o = Float.rem dash_offset l in
        let o = if o <= 0. then o +. l else o in
        if o >= l then 0. else o
  in
  { width; cap; join; miter_limit; dash; dash_offset }

let width s = s.width
let cap s = s.cap
let join s = s.join
let miter_limit s = s.miter_limit
let dash s = s.dash
let dash_offset s = s.dash_offset

let reach s =
  let k =
    match (s.join, s.cap) with
    | `Miter, `Square -> Float.max s.miter_limit (Float.sqrt 2.)
    | `Miter, _ -> s.miter_limit
    | _, `Square -> Float.sqrt 2.
    | _ -> 1.
  in
  0.5 *. s.width *. k

let cap_rank = function `Butt -> 0 | `Round -> 1 | `Square -> 2
let join_rank = function `Miter -> 0 | `Round -> 1 | `Bevel -> 2

let equal s s' =
  Float.equal s.width s'.width
  && cap_rank s.cap = cap_rank s'.cap
  && join_rank s.join = join_rank s'.join
  && Float.equal s.miter_limit s'.miter_limit
  && List.equal Float.equal s.dash s'.dash
  && Float.equal s.dash_offset s'.dash_offset

let compare s s' =
  let c = Float.compare s.width s'.width in
  if c <> 0 then c
  else
    let c = Int.compare (cap_rank s.cap) (cap_rank s'.cap) in
    if c <> 0 then c
    else
      let c = Int.compare (join_rank s.join) (join_rank s'.join) in
      if c <> 0 then c
      else
        let c = Float.compare s.miter_limit s'.miter_limit in
        if c <> 0 then c
        else
          let c = List.compare Float.compare s.dash s'.dash in
          if c <> 0 then c else Float.compare s.dash_offset s'.dash_offset

let cap_name = function
  | `Butt -> "butt"
  | `Round -> "round"
  | `Square -> "square"

let join_name = function
  | `Miter -> "miter"
  | `Round -> "round"
  | `Bevel -> "bevel"

let pp ppf s =
  Format.fprintf ppf "@[<1>(stroke %g@ cap %s@ join %s@ miter %g" s.width
    (cap_name s.cap) (join_name s.join) s.miter_limit;
  (match s.dash with
  | [] -> ()
  | dash ->
      Format.fprintf ppf "@ @[<1>dash [%a]@]@ offset %g"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf l ->
             Format.fprintf ppf "%g" l))
        dash s.dash_offset);
  Format.fprintf ppf ")@]"
