(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Symbol = Hugin_next_kit.Symbol
module Dash = Hugin_next_kit.Dash
module Scale = Hugin_next_kit.Scale
open Common

type 'r param = { id : 'r Type.Id.t; equal : 'r -> 'r -> bool }

type _ range =
  | Floats : float range
  | Colors : Color.t range
  | Symbols : Symbol.t range
  | Dashes : Dash.t range
  | Texts : Text.t range
  | Panels : string range
  | Param : 'r param -> 'r range

let equal_range : type r s. r range -> s range -> (r, s) Type.eq option =
 fun r s ->
  match (r, s) with
  | Floats, Floats -> Some Type.Equal
  | Colors, Colors -> Some Type.Equal
  | Symbols, Symbols -> Some Type.Equal
  | Dashes, Dashes -> Some Type.Equal
  | Texts, Texts -> Some Type.Equal
  | Panels, Panels -> Some Type.Equal
  | Param p, Param q -> Type.Id.provably_equal p.id q.id
  | _ -> None

let equal_in : type r. r range -> r -> r -> bool =
 fun r v v' ->
  match r with
  | Floats -> Float.equal v v'
  | Colors -> Color.equal v v'
  | Symbols -> Symbol.equal v v'
  | Dashes -> Dash.equal v v'
  | Texts -> Text.equal v v'
  | Panels -> String.equal v v'
  | Param p -> p.equal v v'

type axis = X | Y
type map = Color | Opacity | Area | Width | Shape | Pattern

type use =
  | Position of { axis : axis; far : bool }
  | Facet of axis
  | Encoding of { scale : string; map : map }
  | Value

type ('d, 'r) t = { name : string; range : 'r range; use : use }

(* [make name range use] is a role whose use maps into [range]. *)
let make : type r. string -> r range -> use -> ('d, r) t =
 fun name range use ->
  let fits =
    match (use, range) with
    | Value, _ -> true
    | Position _, Floats | Facet _, Panels -> true
    | Encoding { map = Color; _ }, Colors -> true
    | Encoding { map = Opacity | Area | Width; _ }, Floats -> true
    | Encoding { map = Shape; _ }, Symbols -> true
    | Encoding { map = Pattern; _ }, Dashes -> true
    | _ -> false
  in
  if not fits then err "Role" "the use of %s maps outside its range" name;
  { name; range; use }

let position axis ~far = Position { axis; far }
let encoding scale map = Encoding { scale; map }
let x = make "x" Floats (position X ~far:false)
let x2 = make "x2" Floats (position X ~far:true)
let y = make "y" Floats (position Y ~far:false)
let y2 = make "y2" Floats (position Y ~far:true)
let fill = make "fill" Colors (encoding "color" Color)
let stroke = make "stroke" Colors (encoding "color" Color)
let opacity = make "opacity" Floats (encoding "opacity" Opacity)
let size = make "size" Floats (encoding "size" Area)
let width = make "width" Floats (encoding "width" Width)
let symbol = make "symbol" Symbols (encoding "symbol" Shape)
let dash = make "dash" Dashes (encoding "dash" Pattern)
let text = make "text" Texts Value
let fx = make "fx" Panels (Facet X)
let fy = make "fy" Panels (Facet Y)

let names =
  [
    x.name;
    x2.name;
    y.name;
    y2.name;
    fill.name;
    stroke.name;
    opacity.name;
    size.name;
    width.name;
    symbol.name;
    dash.name;
    text.name;
    fx.name;
    fy.name;
  ]

let check_name fn name =
  if name = "" then err fn "the name is empty";
  if List.mem name names then err fn "%S names a built-in role" name

let value ~name =
  check_name "Role.value" name;
  make name Floats Value

let param ~name ~equal =
  check_name "Role.param" name;
  make name (Param { id = Type.Id.make (); equal }) Value

(* Meaning *)

let scale = function
  | Position { axis = X; _ } -> Some "x"
  | Position { axis = Y; _ } -> Some "y"
  | Facet X -> Some "fx"
  | Facet Y -> Some "fy"
  | Encoding e -> Some e.scale
  | Value -> None

type shown = [ `Axis of axis | `Header of axis | `Legend ]

let shown_on : use -> shown option = function
  | Position p -> Some (`Axis p.axis)
  | Facet a -> Some (`Header a)
  | Encoding _ -> Some `Legend
  | Value -> None

let implied : type d. use -> d Scale.kind -> d Scale.t option =
 fun u k ->
  match (u, k) with
  | Encoding { map = Area; _ }, Scale.Quantitative ->
      Some (Scale.linear ~zero:true ())
  | Position { axis = Y; _ }, Scale.Categorical ->
      Some (Scale.band ~reverse:true ())
  | _ -> None

let reads n u = scale u = Some n
let by_cell n = List.exists (reads n) [ x.use; y.use; fx.use; fy.use ]

let by_kind n =
  List.exists (reads n)
    [ fill.use; opacity.use; size.use; width.use; symbol.use; dash.use ]
