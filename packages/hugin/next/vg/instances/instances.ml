(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg
open Hugin_next_vg

let everywhere = Box2.v (-1e15) (-1e15) 2e15 2e15
let bounds p = try Picture.bounds p with Invalid_argument _ -> Some everywhere

let rec exists f (p : Picture.t) =
  f p
  ||
  match p with
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> false
  | Group ps -> List.exists (exists f) ps
  | Clip { picture; _ }
  | Transform { picture; _ }
  | Opacity { picture; _ }
  | Tag { picture; _ }
  | Stamp { picture; _ } ->
      exists f picture

let rec fold_strokes f m (p : Picture.t) acc =
  match p with
  | Stroke { stroke; _ } -> f m stroke acc
  | Empty | Fill _ | Glyphs _ | Image _ -> acc
  | Group ps -> List.fold_left (fun acc p -> fold_strokes f m p acc) acc ps
  | Transform { m = m'; picture } -> fold_strokes f Affine.(m * m') picture acc
  | Clip { picture; _ }
  | Opacity { picture; _ }
  | Tag { picture; _ }
  | Stamp { picture; _ } ->
      fold_strokes f m picture acc

let reach m k p =
  fold_strokes
    (fun m s r -> Float.max r (Stroke.reach s *. (k *. Affine.stretch m)))
    (Affine.linear m) p 0.

(* Boxes that touch meet on a line, where nothing paints, so the comparisons may
   be strict or not. *)
let shows cut ~reach at s b =
  let x = P2.x at and y = P2.y at in
  x +. (s *. Box2.minx b) -. reach <= Box2.maxx cut
  && x +. (s *. Box2.maxx b) +. reach >= Box2.minx cut
  && y +. (s *. Box2.miny b) -. reach <= Box2.maxy cut
  && y +. (s *. Box2.maxy b) +. reach >= Box2.miny cut

let positions cut ~reach b =
  let x0 = Box2.minx cut -. reach -. Box2.maxx b
  and y0 = Box2.miny cut -. reach -. Box2.maxy b in
  let x1 = Box2.maxx cut +. reach -. Box2.minx b
  and y1 = Box2.maxy cut +. reach -. Box2.miny b in
  if
    Float.is_finite x0 && Float.is_finite y0 && Float.is_finite x1
    && Float.is_finite y1
  then Some (Box2.of_pts (P2.v x0 y0) (P2.v x1 y1))
  else None

let color cs i ~own inherited =
  match cs with None -> inherited | Some cs -> own cs.(i)
