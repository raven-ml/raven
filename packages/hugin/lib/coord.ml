(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P2 = Hugin_gg.P2
module Box2 = Hugin_gg.Box2
module Affine = Hugin_gg.Affine
open Common

type t = Cartesian of { aspect : float option }

let cartesian ?aspect () =
  (match aspect with
  | Some a when not (is_pos a) ->
      err "Coord.cartesian" "aspect %g is not finite and positive" a
  | _ -> ());
  Cartesian { aspect }

let equal (Cartesian c) (Cartesian c') =
  Option.equal Float.equal c.aspect c'.aspect

let pp ppf (Cartesian { aspect }) =
  match aspect with
  | None -> Format.pp_print_string ppf "cartesian"
  | Some a -> Format.fprintf ppf "cartesian ~aspect:%g" a

(* A cartesian projection maps the unit square onto its box, y up. *)
type projection = Box of Box2.t

let project (Cartesian _) box = Box box

let point (Box b) x y =
  P2.v (Box2.minx b +. (x *. Box2.w b)) (Box2.maxy b -. (y *. Box2.h b))

let invert (Box b) pt =
  let w = Box2.w b and h = Box2.h b in
  if w = 0. || h = 0. then None
  else Some ((P2.x pt -. Box2.minx b) /. w, (Box2.maxy b -. P2.y pt) /. h)

let affine (Box b) =
  {
    Affine.xx = Box2.w b;
    yx = 0.;
    xy = 0.;
    yy = -.Box2.h b;
    x0 = Box2.minx b;
    y0 = Box2.maxy b;
  }
