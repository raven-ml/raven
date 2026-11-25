(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Api

(* Figures *)

type id = Common.id
type warning = Common.warning

let pp_warning = Common.pp_warning
let equal = Figure.equal

(* Marks *)

let dot = Marks.dot
let line = Marks.line
let area = Marks.area
let rect = Marks.rect
let frame = Marks.frame
let rule = Marks.rule
let abline = Marks.abline
let text = Marks.text
let image = Marks.image
let contour = Marks.contour

(* Views *)

module View = View

(* Composing *)

type sharing = Figure.sharing
type side = Figure.side
type corner = Figure.corner
type inside = Figure.inside

let layer = Figure.layer
let grid = Figure.grid
let span = Figure.span
let share = Figure.share
let title = Figure.title
let title' = Figure.title'
let coord = Figure.coord
let name = Figure.name
let bind = Figure.bind
let axis = Figure.axis
let axis' = Figure.axis'
let legend = Figure.legend

(* Sizes *)

module Size = Size

(* Stages *)

module Resolved = Resolved
module Layout = Layout
module Drawing = Draw

let resolve = Resolved.resolve
let layout = Layout.layout
let draw = Draw.draw

let render ?view ?theme ~density size f =
  draw ~density (layout ?theme size (resolve ?view f))

(* Output *)

let default_size = Size.figure 360. 240.
let default_density = 2.

let save ?(warn = Format.eprintf "%a@." pp_warning) ?view ?theme
    ?(size = default_size) ?(density = default_density) file f =
  let write =
    match String.lowercase_ascii (Filename.extension file) with
    | ".png" -> Hugin_vg_raster.png ~density
    | ".svg" -> fun r -> Hugin_vg_svg.render r
    | ".pdf" -> Hugin_vg_pdf.render
    | ext ->
        Common.err "save" "%S: the extension %S is not .png, .svg or .pdf" file
          ext
  in
  let d = render ?view ?theme ~density size f in
  List.iter warn (Drawing.warnings d);
  let data = write (Drawing.renderable d) in
  Out_channel.with_open_bin file (fun oc -> Out_channel.output_string oc data)

(* The header of an SVG display tag in Quill's display protocol: the line
   [quill.display], the MIME type's line and the display id's line. The document
   follows. *)
let svg_display id = "quill.display\nimage/svg+xml\n" ^ id ^ "\n"

(* The width of a prose column of about 66 characters of running text in IBM
   Plex Sans at 16 px: 496 px, 31 rem, which is 372 points at 3/4 point a CSS
   pixel. *)
let prose_width = 496. *. 0.75

(* At the prose width, the aspect of [default_size]. *)
let page_size = Size.figure prose_width (prose_width *. 2. /. 3.)

let pp_with ?(id = "") ?(size = page_size) () =
  if String.contains id '\n' then
    Common.err "pp_with" "the display id %S holds a newline" id;
  fun ppf f ->
    let r = resolve f in
    let drawn theme = draw ~density:default_density (layout ~theme size r) in
    let light = drawn Theme.default and dark = drawn Theme.dark in
    let lr = Drawing.renderable light in
    (* A mark that reads its theme may draw the schemes apart; the light one
       then displays alone. *)
    let svg =
      match Hugin_vg_svg.render ~dark:(Drawing.renderable dark) lr with
      | svg -> svg
      | exception Invalid_argument _ -> Hugin_vg_svg.render lr
    in
    let summary =
      match List.length (Drawing.warnings light) with
      | 0 -> "hugin figure"
      | 1 -> "hugin figure (1 warning)"
      | n -> Printf.sprintf "hugin figure (%d warnings)" n
    in
    Format.pp_open_stag ppf (Format.String_tag (svg_display id ^ svg));
    Format.pp_print_string ppf summary;
    Format.pp_close_stag ppf ()

let pp ppf f = pp_with () ppf f
