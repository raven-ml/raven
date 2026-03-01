(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Declarative plotting and visualization.

    Hugin turns immutable plot specifications into rendered output. A plot is a
    value of type {!t} built from mark constructors ({!line}, {!point}, {!bar},
    {!hist}), composed with {!layers}, decorated with {!title}, {!xlabel}, etc.
    via the [|>] pipeline, and rendered with {!show}, {!render_png}, or
    {!render_svg}.

    {[
      let x = Nx.linspace Float32 0. 6.28 100 in
      let y = Nx.map (fun v -> Float.sin v) x in
      Hugin.line ~x ~y () |> Hugin.title "Sine wave"
      |> Hugin.render_png "sine.png"
    ]} *)

(** {1:sub Sub-modules} *)

module Color = Color
(** Perceptually uniform OKLCH colors. *)

module Cmap = Cmap
(** Colormaps. *)

module Theme = Theme
(** Visual themes. *)

(** {1:types Types} *)

type t
(** The type for plot specifications. Immutable and composable. *)

type marker =
  | Circle
  | Square
  | Triangle
  | Plus
  | Star  (** The type for point marker shapes. *)

type legend_loc =
  | Upper_right
  | Upper_left
  | Lower_right
  | Lower_left
  | Center
  | Right
  | Upper_center
  | Lower_center  (** The type for legend placement. *)

type line_style = [ `Solid | `Dashed | `Dotted | `Dash_dot ]
(** The type for line dash patterns. *)

type scale = [ `Linear | `Log | `Sqrt | `Asinh | `Symlog of float ]
(** The type for axis scales. [`Sqrt] and [`Asinh] handle zero gracefully.
    [`Symlog linthresh] is linear within \[[-linthresh];[linthresh]\] and
    logarithmic outside. *)

type stretch = [ `Linear | `Log | `Sqrt | `Asinh | `Power of float ]
(** The type for image stretch functions. [`Power a] raises normalized values to
    the power [a]. *)

(** {1:marks Mark constructors}

    Each constructor builds a single-layer specification from data arrays and
    optional visual properties. A mark is already a valid {!t} that can be
    rendered directly. *)

val line :
  x:Nx.float32_t ->
  y:Nx.float32_t ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?step:[ `Pre | `Post | `Mid ] ->
  ?marker:marker ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [line ~x ~y ()] is a line plot connecting the points [(x.(i), y.(i))].

    [color] defaults to the next color in the theme palette. [line_width]
    defaults to the theme line width. [line_style] defaults to [`Solid]. [step]
    draws a staircase line: [`Post] holds each value until the next x-point,
    [`Pre] steps to the new value at the current x-point, [`Mid] steps at the
    midpoint between consecutive x-points. *)

val point :
  x:Nx.float32_t ->
  y:Nx.float32_t ->
  ?color:Color.t ->
  ?color_by:Nx.float32_t ->
  ?size:float ->
  ?size_by:Nx.float32_t ->
  ?marker:marker ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [point ~x ~y ()] is a scatter plot of discrete markers at [(x.(i), y.(i))].

    [color_by] maps per-point values through the theme's sequential colormap.
    [size_by] scales marker area per point. [marker] defaults to {!Circle}. *)

val bar :
  x:Nx.float32_t ->
  height:Nx.float32_t ->
  ?width:float ->
  ?bottom:float ->
  ?color:Color.t ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [bar ~x ~height ()] is a bar chart with bars centered on [x] values,
    extending from [bottom] (default [0.0]) to [bottom + height]. [width]
    defaults to [0.8]. *)

val hist :
  x:Nx.float32_t ->
  ?bins:[ `Num of int | `Edges of float array ] ->
  ?density:bool ->
  ?color:Color.t ->
  ?label:string ->
  unit ->
  t
(** [hist ~x ()] is a histogram of the values in [x].

    [bins] defaults to [`Num 10]. When [density] is [true], the histogram is
    normalized so the total area equals [1.0]. *)

val image : ?extent:float * float * float * float -> Nx.uint8_t -> t
(** [image ?extent data] displays [data] as an image. [data] has shape
    [[|h; w; 3|]] (RGB) or [[|h; w; 4|]] (RGBA).

    When [extent] is [(xmin, xmax, ymin, ymax)], the image is placed in data
    coordinates. Without [extent], the image is centered in the plot area
    preserving aspect ratio. *)

val text :
  x:float ->
  y:float ->
  string ->
  ?color:Color.t ->
  ?font_size:float ->
  unit ->
  t
(** [text ~x ~y s ()] places the string [s] at data coordinates [(x, y)]. *)

val hline :
  y:float ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [hline ~y ()] draws a horizontal reference line at [y] spanning the full
    plot width. *)

val vline :
  x:float ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [vline ~x ()] draws a vertical reference line at [x] spanning the full plot
    height. *)

val abline :
  slope:float ->
  intercept:float ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [abline ~slope ~intercept ()] draws a diagonal line
    [y = slope * x + intercept] spanning the full plot area. Useful for
    regression lines and [y = x] references. *)

val fill_between :
  x:Nx.float32_t ->
  y1:Nx.float32_t ->
  y2:Nx.float32_t ->
  ?where:Nx.float32_t ->
  ?color:Color.t ->
  ?alpha:float ->
  ?label:string ->
  unit ->
  t
(** [fill_between ~x ~y1 ~y2 ()] fills the area between curves [y1] and [y2]
    over the shared [x] axis. [alpha] defaults to [0.3].

    [where] is an optional mask array of the same length as [x]: the fill is
    only drawn where [where.(i) > 0.], producing separate filled regions. *)

val hspan :
  y0:float ->
  y1:float ->
  ?color:Color.t ->
  ?alpha:float ->
  ?label:string ->
  unit ->
  t
(** [hspan ~y0 ~y1 ()] is a horizontal shaded band between [y0] and [y1],
    spanning the full plot width. [alpha] defaults to [0.2]. *)

val vspan :
  x0:float ->
  x1:float ->
  ?color:Color.t ->
  ?alpha:float ->
  ?label:string ->
  unit ->
  t
(** [vspan ~x0 ~x1 ()] is a vertical shaded band between [x0] and [x1], spanning
    the full plot height. [alpha] defaults to [0.2]. *)

val errorbar :
  x:Nx.float32_t ->
  y:Nx.float32_t ->
  yerr:
    [ `Symmetric of Nx.float32_t | `Asymmetric of Nx.float32_t * Nx.float32_t ] ->
  ?xerr:
    [ `Symmetric of Nx.float32_t | `Asymmetric of Nx.float32_t * Nx.float32_t ] ->
  ?color:Color.t ->
  ?line_width:float ->
  ?cap_size:float ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [errorbar ~x ~y ~yerr ()] draws error bars at [(x.(i), y.(i))].

    [yerr] specifies vertical error: [`Symmetric e] draws [y +/- e],
    [`Asymmetric (lo, hi)] draws [[y - lo, y + hi]]. [xerr] adds horizontal
    error bars. [cap_size] defaults to half the theme marker size. *)

val heatmap :
  data:Nx.float32_t ->
  ?annotate:bool ->
  ?cmap:Cmap.t ->
  ?vmin:float ->
  ?vmax:float ->
  ?fmt:(float -> string) ->
  unit ->
  t
(** [heatmap ~data ()] displays a 2D array as a grid of colored cells. [data]
    has shape [[|rows; cols|]]. Row 0 appears at the top.

    [cmap] defaults to the theme's sequential colormap. [vmin] and [vmax]
    override the automatic value range. When [annotate] is [true], each cell
    shows its value formatted by [fmt] (default [Printf.sprintf "%.2g"]). *)

val imshow :
  data:Nx.float32_t ->
  ?stretch:stretch ->
  ?cmap:Cmap.t ->
  ?vmin:float ->
  ?vmax:float ->
  unit ->
  t
(** [imshow ~data ()] displays a 2D float array as a colormapped image. [data]
    has shape [[|rows; cols|]].

    [stretch] controls the transfer function applied before colormap lookup:
    [`Linear] (default), [`Log], [`Sqrt], [`Asinh], or [`Power a]. [cmap]
    defaults to the theme's sequential colormap. [vmin] and [vmax] override the
    automatic value range. *)

val contour :
  data:Nx.float32_t ->
  x0:float ->
  x1:float ->
  y0:float ->
  y1:float ->
  ?levels:[ `Num of int | `Values of float array ] ->
  ?filled:bool ->
  ?cmap:Cmap.t ->
  ?color:Color.t ->
  ?line_width:float ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [contour ~data ~x0 ~x1 ~y0 ~y1 ()] draws iso-level contour lines through the
    2D grid [data] of shape [[|rows; cols|]], mapped to the data-space rectangle
    \[[x0];[x1]\] x \[[y0];[y1]\].

    [levels] defaults to [`Num 8]. When [filled] is [true], regions between
    adjacent levels are filled. [color] sets a single stroke color for unfilled
    contours; [cmap] assigns per-level colors from the theme's sequential
    colormap. *)

(** {1:composition Composition} *)

val layers : t list -> t
(** [layers marks] overlays [marks] on shared axes. A single mark is already a
    valid {!t}; [layers] is only needed to combine multiple marks into one plot.
*)

(** {1:decorations Decorations}

    Decoration functions add metadata to a specification. They are designed for
    the [|>] pipeline:
    {[
      line ~x ~y () |> title "My Plot" |> xlabel "Time"
    ]} *)

val title : string -> t -> t
(** [title s t] is [t] with plot title [s]. *)

val xlabel : string -> t -> t
(** [xlabel s t] is [t] with x-axis label [s]. *)

val ylabel : string -> t -> t
(** [ylabel s t] is [t] with y-axis label [s]. *)

val xlim : float -> float -> t -> t
(** [xlim lo hi t] is [t] with x-axis range fixed to \[[lo];[hi]\]. *)

val ylim : float -> float -> t -> t
(** [ylim lo hi t] is [t] with y-axis range fixed to \[[lo];[hi]\]. *)

val xscale : scale -> t -> t
(** [xscale s t] is [t] with x-axis scale [s]. Defaults to [`Linear].

    [`Sqrt] and [`Asinh] handle zero gracefully. [`Symlog linthresh] is linear
    within \[[-linthresh];[linthresh]\] and logarithmic outside. *)

val yscale : scale -> t -> t
(** [yscale s t] is [t] with y-axis scale [s]. Defaults to [`Linear]. *)

val xinvert : t -> t
(** [xinvert t] is [t] with the x-axis inverted (values increase right-to-left).
    Useful for right ascension in sky charts. *)

val yinvert : t -> t
(** [yinvert t] is [t] with the y-axis inverted (values increase top-to-bottom).
    Useful for magnitude axes in HR diagrams. *)

val grid_lines : bool -> t -> t
(** [grid_lines visible t] is [t] with grid lines shown or hidden. *)

val legend : ?loc:legend_loc -> ?ncol:int -> t -> t
(** [legend ?loc ?ncol t] is [t] with the legend shown at [loc]. [loc] defaults
    to {!Upper_right}. [ncol] defaults to [1]; set higher for multi-column
    layouts with many series. The legend is automatically visible when any mark
    has a [~label]. *)

val xticks : (float * string) list -> t -> t
(** [xticks ticks t] is [t] with explicit x-axis tick positions and labels.
    Overrides auto-generated ticks. *)

val yticks : (float * string) list -> t -> t
(** [yticks ticks t] is [t] with explicit y-axis tick positions and labels.
    Overrides auto-generated ticks. *)

val with_theme : Theme.t -> t -> t
(** [with_theme th t] is [t] rendered with theme [th] instead of the default. *)

val xtick_format : (float -> string) -> t -> t
(** [xtick_format fmt t] is [t] with x-axis tick labels formatted by [fmt].
    Overrides auto-generated labels while preserving tick positions. *)

val ytick_format : (float -> string) -> t -> t
(** [ytick_format fmt t] is [t] with y-axis tick labels formatted by [fmt].
    Overrides auto-generated labels while preserving tick positions. *)

(** {1:layout Layout} *)

val grid : ?gap:float -> t list list -> t
(** [grid rows] arranges specifications in a grid. Each inner list is a row of
    panels. [gap] defaults to [0.05] (fraction of total size). *)

val hstack : ?gap:float -> t list -> t
(** [hstack specs] arranges [specs] in a single row. *)

val vstack : ?gap:float -> t list -> t
(** [vstack specs] arranges [specs] in a single column. *)

(** {1:rendering Rendering} *)

val show : ?theme:Theme.t -> ?width:float -> ?height:float -> t -> unit
(** [show t] displays [t] in an interactive SDL window.

    [width] defaults to [1600.0]. [height] defaults to [1200.0]. The window
    supports resize (re-resolves at new dimensions) and closes on Escape or Q.
*)

val render_png :
  ?theme:Theme.t -> ?width:float -> ?height:float -> string -> t -> unit
(** [render_png filename t] writes [t] as a PNG image to [filename].

    [width] defaults to [1600.0]. [height] defaults to [1200.0]. *)

val render_pdf :
  ?theme:Theme.t -> ?width:float -> ?height:float -> string -> t -> unit
(** [render_pdf filename t] writes [t] as a PDF document to [filename].

    [width] defaults to [1600.0]. [height] defaults to [1200.0]. *)

val render_svg :
  ?theme:Theme.t -> ?width:float -> ?height:float -> string -> t -> unit
(** [render_svg filename t] writes [t] as an SVG document to [filename].

    [width] defaults to [1600.0]. [height] defaults to [1200.0]. *)

val render_svg_to_string :
  ?theme:Theme.t -> ?width:float -> ?height:float -> t -> string
(** [render_svg_to_string t] is [t] rendered as an SVG document string.

    [width] defaults to [1600.0]. [height] defaults to [1200.0]. *)

val render_to_buffer :
  ?theme:Theme.t -> ?width:float -> ?height:float -> t -> string
(** [render_to_buffer t] is [t] rendered as a PNG image, returned as a string of
    bytes. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] renders the specification as a PNG data URI. Intended for use with
    [#install_printer] in the toplevel and Quill.

    Output format: [![figure](data:image/png;base64,...)] *)
