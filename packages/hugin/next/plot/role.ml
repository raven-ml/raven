(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Symbol = Hugin_next_kit.Symbol
module Curve = Hugin_next_kit.Curve
open Common

type _ range =
  | Floats : float range
  | Colors : Color.t range
  | Symbols : Symbol.t range
  | Texts : Text.t range
  | Panels : string range
  | Curves : Curve.t range
  | Pixels : Nx.packed range

let equal_range : type r s. r range -> s range -> (r, s) Type.eq option =
 fun r s ->
  match (r, s) with
  | Floats, Floats -> Some Type.Equal
  | Colors, Colors -> Some Type.Equal
  | Symbols, Symbols -> Some Type.Equal
  | Texts, Texts -> Some Type.Equal
  | Panels, Panels -> Some Type.Equal
  | Curves, Curves -> Some Type.Equal
  | Pixels, Pixels -> Some Type.Equal
  | _ -> None

let equal_in : type r. r range -> r -> r -> bool =
 fun r v v' ->
  match r with
  | Floats -> Float.equal v v'
  | Colors -> Color.equal v v'
  | Symbols -> Symbol.equal v v'
  | Texts -> Text.equal v v'
  | Panels -> String.equal v v'
  | Curves -> Curve.equal v v'
  | Pixels ->
      let (Nx.P px) = v in
      let (Nx.P px') = v' in
      equal_tensor px px'

type ('d, 'r) t = { name : string; range : 'r range; scale : string option }

let x = { name = "x"; range = Floats; scale = Some "x" }
let x2 = { name = "x2"; range = Floats; scale = Some "x" }
let y = { name = "y"; range = Floats; scale = Some "y" }
let y2 = { name = "y2"; range = Floats; scale = Some "y" }
let fill = { name = "fill"; range = Colors; scale = Some "color" }
let stroke = { name = "stroke"; range = Colors; scale = Some "color" }
let opacity = { name = "opacity"; range = Floats; scale = Some "opacity" }
let size = { name = "size"; range = Floats; scale = Some "size" }
let width = { name = "width"; range = Floats; scale = Some "width" }
let symbol = { name = "symbol"; range = Symbols; scale = Some "symbol" }
let text = { name = "text"; range = Texts; scale = None }
let fx = { name = "fx"; range = Panels; scale = Some "fx" }
let fy = { name = "fy"; range = Panels; scale = Some "fy" }

let names =
  [
    "x";
    "x2";
    "y";
    "y2";
    "fill";
    "stroke";
    "opacity";
    "size";
    "width";
    "symbol";
    "text";
    "fx";
    "fy";
  ]

let value ~name =
  if name = "" then err "Role.value" "the name is empty";
  if List.mem name names then err "Role.value" "%S names a built-in role" name;
  { name; range = Floats; scale = None }

(* The parameters of built-in marks. *)
let curve = { name = "curve"; range = Curves; scale = None }
let dx = { name = "dx"; range = Floats; scale = None }
let dy = { name = "dy"; range = Floats; scale = None }
let pixels = { name = "pixels"; range = Pixels; scale = None }
