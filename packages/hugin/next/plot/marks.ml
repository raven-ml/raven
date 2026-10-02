(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Picture = Hugin_next_vg.Picture
module Scale = Hugin_next_kit.Scale
module Curve = Hugin_next_kit.Curve
open Common
open Channel
open Figure

let on = Mark.bind
let opt role = Option.map (on role)

(* Rows are inhabited once figures are drawn: until then no draw function is
   called. *)
let not_drawn (r : rows) : Picture.t = match r with _ -> .

(* [length ch] implies [zero] on the scale of [ch] when it holds quantities: a
   position without its other end is a length. *)
let length : type d r. (d, r) Role.t -> (d, r) Channel.t -> binding =
 fun role ch ->
  match data ch with
  | Some { lift; _ } -> (
      match lift_kind lift with
      | Scale.Quantitative -> on ~imply:(Scale.linear ~zero:true ()) role ch
      | Scale.Temporal | Scale.Categorical -> on role ch)
  | None -> on role ch

let position role ~alone = function
  | None -> None
  | Some ch -> Some (if alone then length role ch else on role ch)

let facets fx fy = [ opt Role.fx fx; opt Role.fy fy ]

let make fn ?reduce ?coord ?base l =
  make_mark fn ~name:fn ?reduce ?coord ?base (List.filter_map Fun.id l)
    not_drawn

let dot ?fill ?stroke ?opacity ?size ?symbol ?fx ?fy ~x ~y () =
  Mark
    (make "dot" ~reduce:Raster
       ([
          Some (on Role.x x);
          Some (on Role.y y);
          opt Role.fill fill;
          opt Role.stroke stroke;
          opt Role.opacity opacity;
          opt Role.size size;
          opt Role.symbol symbol;
        ]
       @ facets fx fy))

let line ?x ?stroke ?fill ?width ?opacity ?(curve = Curve.linear) ?fx ?fy ~y ()
    =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  Mark
    (make "line" ~reduce:M4
       ([
          Some x;
          Some (on Role.y y);
          opt Role.stroke stroke;
          opt Role.fill fill;
          opt Role.width width;
          opt Role.opacity opacity;
          Some (on Role.curve (const curve));
        ]
       @ facets fx fy))

let rect ?x ?x2 ?y ?y2 ?fill ?stroke ?opacity ?fx ?fy () =
  Mark
    (make "rect" ~reduce:Cells
       ([
          position Role.x ~alone:(Option.is_none x2) x;
          opt Role.x2 x2;
          position Role.y ~alone:(Option.is_none y2) y;
          opt Role.y2 y2;
          opt Role.fill fill;
          opt Role.stroke stroke;
          opt Role.opacity opacity;
        ]
       @ facets fx fy))

let rule ?x ?x2 ?y ?y2 ?stroke ?width ?opacity ?fx ?fy () =
  let has = Option.is_some in
  let positions =
    if has x && not (has x2) then
      [
        position Role.x ~alone:false x;
        position Role.y ~alone:(not (has y2)) y;
        opt Role.y2 y2;
      ]
    else if has y && not (has y2) then
      [
        position Role.y ~alone:false y;
        position Role.x ~alone:(not (has x2)) x;
        opt Role.x2 x2;
      ]
    else if has x && has x2 && has y && has y2 then
      [ opt Role.x x; opt Role.x2 x2; opt Role.y y; opt Role.y2 y2 ]
    else
      err "rule"
        "the channels match no case: give x without x2, y without y2, or x, \
         x2, y and y2"
  in
  Mark
    (make "rule"
       (positions
       @ [
           opt Role.stroke stroke;
           opt Role.width width;
           opt Role.opacity opacity;
         ]
       @ facets fx fy))

let text ?fill ?opacity ?(dx = 0.) ?(dy = 0.) ?fx ?fy ~x ~y ~text () =
  Mark
    (make "text"
       ([
          Some (on Role.x x);
          Some (on Role.y y);
          Some (on Role.text text);
          opt Role.fill fill;
          opt Role.opacity opacity;
          Some (on Role.dx (const dx));
          Some (on Role.dy (const dy));
        ]
       @ facets fx fy))

let is_pixel : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.UInt8 | Nx.Float16 | Nx.Float32 | Nx.Float64 | Nx.BFloat16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      true
  | _ -> false

let image ?fx ?fy px =
  let shape = Nx.shape px in
  let rank = Array.length shape in
  let dtype = Nx.dtype px in
  if not (is_pixel dtype) then
    err "image" "the dtype %s is neither uint8 nor floating point"
      (Nx_dtype.to_string dtype);
  if rank < 2 then
    err "image" "the shape %a has fewer than two axes" pp_shape shape;
  let lead, h, w =
    if rank = 2 then ([||], shape.(0), shape.(1))
    else
      match shape.(rank - 1) with
      | 1 | 3 | 4 ->
          (Array.sub shape 0 (rank - 3), shape.(rank - 3), shape.(rank - 2))
      | c -> err "image" "the last axis has %d channels, not 1, 3 or 4" c
  in
  let fixed = Scale.linear ~nice:false () in
  let x = Data { lift = Scalar 0.; scale = None; title = None } in
  let x2 = Data { lift = Scalar (float w); scale = None; title = None } in
  let y = Data { lift = Scalar 0.; scale = None; title = None } in
  let y2 = Data { lift = Scalar (float h); scale = None; title = None } in
  Mark
    (make "image"
       ~coord:(Coord.cartesian ~aspect:1. ())
       ~base:lead
       ([
          Some (on ~imply:fixed ~guide:false Role.x x);
          Some (on Role.x2 x2);
          Some
            (on
               ~imply:(Scale.linear ~nice:false ~reverse:true ())
               ~guide:false Role.y y);
          Some (on Role.y2 y2);
          Some (on Role.pixels (const (Nx.P px)));
        ]
       @ facets fx fy))

(* [varies shape b a] is [true] iff the channel of [b] can vary along axis [a]
   of [shape]. *)
let varies shape (B b) a =
  let rank = Array.length shape in
  let along s =
    let off = rank - Array.length s in
    a >= off && s.(a - off) > 1
  in
  match data b.ch with
  | None -> false
  | Some d -> (
      match d.lift with
      | Num { x; _ } -> along (Nx.shape x)
      | Cat { codes; _ } -> along (Nx.shape codes)
      | Strings s -> along [| Array.length s |]
      | Index k | Dim { axis = k; _ } -> axis_of shape k = Some a
      | Scalar _ -> false)

let contour ?x ?y ?opacity ?fx ?fy ~fill () =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  let y =
    match y with Some y -> on Role.y y | None -> on Role.y (index (-2))
  in
  if Option.is_none (data fill) then err "contour" "fill is a constant";
  let m =
    make "contour"
      ([ Some x; Some y; Some (on Role.fill fill); opt Role.opacity opacity ]
      @ facets fx fy)
  in
  let shape = m.shape in
  let rank = Array.length shape in
  if rank < 2 then
    err "contour" "the shape %a has fewer than two axes" pp_shape shape;
  if varies shape x (rank - 2) then
    err "contour" "x can vary along the rows of the grid";
  if varies shape y (rank - 1) then
    err "contour" "y can vary along the columns of the grid";
  Mark m
