(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg

type t = Ring2.t list

let v rs = rs
let rings p = p
let area p = List.fold_left (fun a r -> a +. Ring2.area r) 0. p

let mem pt p =
  let px = P2.x pt and py = P2.y pt in
  Float.is_finite px && Float.is_finite py
  && List.fold_left
       (fun w r ->
         w + Winding.number (Ring2.length r) (Ring2.x r) (Ring2.y r) px py)
       0 p
     <> 0

let bounds p =
  List.fold_left
    (fun acc r ->
      match (acc, Ring2.bounds r) with
      | None, b | b, None -> b
      | Some a, Some b -> Some (Box2.union a b))
    None p

let to_path p =
  List.fold_left (fun path r -> Path.append (Ring2.to_path r) path) Path.empty p

let equal p p' = List.equal Ring2.equal p p'

let pp ppf p =
  Format.pp_open_box ppf 0;
  Format.pp_print_list ~pp_sep:Format.pp_print_space Ring2.pp ppf p;
  Format.pp_close_box ppf ()
