(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Color = Color
module Cmap = Cmap
module Theme = Theme

type t = Spec.t
type marker = Spec.marker = Circle | Square | Triangle | Plus | Star

type legend_loc = Spec.legend_loc =
  | Upper_right
  | Upper_left
  | Lower_right
  | Lower_left
  | Center
  | Right
  | Upper_center
  | Lower_center

type line_style = Spec.line_style
type scale = Spec.scale
type stretch = Spec.stretch

(* Mark constructors *)

let line = Spec.line
let point = Spec.point
let bar = Spec.bar
let hist = Spec.hist
let image = Spec.image
let text = Spec.text
let hline = Spec.hline
let vline = Spec.vline
let abline = Spec.abline
let fill_between = Spec.fill_between
let hspan = Spec.hspan
let vspan = Spec.vspan
let errorbar = Spec.errorbar
let heatmap = Spec.heatmap
let imshow = Spec.imshow
let contour = Spec.contour

(* Composition *)

let layers = Spec.layers

(* Decorations *)

let title = Spec.title
let xlabel = Spec.xlabel
let ylabel = Spec.ylabel
let xlim = Spec.xlim
let ylim = Spec.ylim
let xscale = Spec.xscale
let yscale = Spec.yscale
let xinvert = Spec.xinvert
let yinvert = Spec.yinvert
let grid_lines = Spec.grid_lines
let legend = Spec.legend
let xticks = Spec.xticks
let yticks = Spec.yticks
let with_theme = Spec.with_theme
let xtick_format = Spec.xtick_format
let ytick_format = Spec.ytick_format
let frame = Spec.frame
let no_axes = Spec.no_axes

(* Layout *)

let grid = Spec.grid_layout
let hstack ?gap specs = Spec.grid_layout ?gap [ specs ]
let vstack ?gap specs = Spec.grid_layout ?gap (List.map (fun s -> [ s ]) specs)

(* Rendering *)

let default_width = 1600.
let default_height = 1200.

(* Use Cairo text measurement for all backends for consistent layout *)
let resolve_with_cairo ~theme ~width ~height spec =
  let surface = Ucairo.Image.create ~w:1 ~h:1 in
  let cr = Ucairo.create surface in
  let tm = Cairo_backend.text_measurer cr in
  let scene = Resolve.resolve ~text_measurer:tm ~theme ~width ~height spec in
  Ucairo.Surface.finish surface;
  scene

let show ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) spec =
  let prepared = Prepared.compile ~theme spec in
  Cairo_backend.show_interactive ~theme ~width ~height prepared

let render_png ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) filename spec =
  let scene = resolve_with_cairo ~theme ~width ~height spec in
  Cairo_backend.render_to_png filename ~width ~height scene

let render_pdf ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) filename spec =
  let scene = resolve_with_cairo ~theme ~width ~height spec in
  Cairo_backend.render_to_pdf filename ~width ~height scene

let render_svg ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) filename spec =
  let scene = resolve_with_cairo ~theme ~width ~height spec in
  Svg_backend.render_to_file filename scene

let render_svg_to_string ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) spec =
  let scene = resolve_with_cairo ~theme ~width ~height spec in
  Svg_backend.render scene

let render_to_buffer ?(theme = Theme.default) ?(width = default_width)
    ?(height = default_height) spec =
  let scene = resolve_with_cairo ~theme ~width ~height spec in
  Cairo_backend.render_to_buffer ~width ~height scene

let infer_dimensions spec =
  let rec grid_shape = function
    | Spec.Grid { rows; _ } ->
        let nrows = List.length rows in
        let ncols =
          List.fold_left (fun acc row -> max acc (List.length row)) 0 rows
        in
        Some (nrows, ncols)
    | Spec.Decorated { inner; _ } -> grid_shape inner
    | _ -> None
  in
  match grid_shape spec with
  | Some (nrows, ncols) when ncols > 0 ->
      let cell_w = default_width /. float ncols in
      (default_width, cell_w *. float nrows)
  | _ -> (default_width, default_height)

let pp fmt spec =
  let width, height = infer_dimensions spec in
  let buf = render_to_buffer ~width ~height spec in
  let b64 = Image_util.base64_encode buf in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" b64
