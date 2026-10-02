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

let layer = Figure.layer
let grid = Figure.grid
let span = Figure.span
let share = Figure.share
let title = Figure.title
let coord = Figure.coord
let name = Figure.name
let bind = Figure.bind
let axis = Figure.axis
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

let render ?view ?theme ?(density = 2.) size f =
  draw ~density (layout ?theme size (resolve ?view f))

(* Output *)

let default_size = Size.figure 360. 240.
let default_density = 2.

let save ?(warn = Format.eprintf "%a@." pp_warning) ?view ?theme
    ?(size = default_size) ?(density = default_density) file f =
  let write =
    match String.lowercase_ascii (Filename.extension file) with
    | ".png" -> Hugin_vg_raster.png ~density
    | ".svg" -> Hugin_vg_svg.render
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
   [quill.display], the MIME type's line and an empty display id line. The
   document follows. *)
let svg_display = "quill.display\nimage/svg+xml\n\n"

let pp ppf f =
  let d = render default_size f in
  let tag = svg_display ^ Hugin_vg_svg.render (Drawing.renderable d) in
  let summary =
    match List.length (Drawing.warnings d) with
    | 0 -> "hugin figure"
    | 1 -> "hugin figure (1 warning)"
    | n -> Printf.sprintf "hugin figure (%d warnings)" n
  in
  Format.pp_open_stag ppf (Format.String_tag tag);
  Format.pp_print_string ppf summary;
  Format.pp_close_stag ppf ()
