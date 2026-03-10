(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Immutable plot specifications.

    {b Internal module.} The specification tree is the user-facing
    representation of a plot. {!Prepared.compile} resolves data-dependent work;
    {!Resolve} turns the result into a {!Scene.t}. *)

(** {1:types Types} *)

type line_style = [ `Solid | `Dashed | `Dotted | `Dash_dot ]
(** The type for line dash patterns. *)

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

type scale = [ `Linear | `Log | `Sqrt | `Asinh | `Symlog of float ]
(** The type for axis scales. [`Sqrt] and [`Asinh] handle zero gracefully.
    [`Symlog linthresh] is linear within \[[-linthresh];[linthresh]\] and
    logarithmic outside. *)

type stretch = [ `Linear | `Log | `Sqrt | `Asinh | `Power of float ]
(** The type for image stretch functions. [`Power a] raises normalized values to
    the power [a]. *)

type mark =
  | Line of {
      x : Nx.float32_t;
      y : Nx.float32_t;
      color : Color.t option;
      line_width : float option;
      line_style : line_style option;
      step : [ `Pre | `Post | `Mid ] option;
      marker : marker option;
      label : string option;
      alpha : float option;
    }
  | Point of {
      x : Nx.float32_t;
      y : Nx.float32_t;
      color : Color.t option;
      color_by : Nx.float32_t option;
      size : float option;
      size_by : Nx.float32_t option;
      marker : marker option;
      label : string option;
      alpha : float option;
    }
  | Bar of {
      x : Nx.float32_t;
      height : Nx.float32_t;
      width : float option;
      bottom : float;
      color : Color.t option;
      label : string option;
      alpha : float option;
    }
  | Hist of {
      x : Nx.float32_t;
      bins : [ `Num of int | `Edges of float array ];
      density : bool;
      color : Color.t option;
      label : string option;
    }
  | Image of {
      data : Nx.uint8_t;
      extent : (float * float * float * float) option;
    }
  | Text_mark of {
      x : float;
      y : float;
      content : string;
      color : Color.t option;
      font_size : float option;
    }
  | Hline of {
      y : float;
      color : Color.t option;
      line_width : float option;
      line_style : line_style option;
      label : string option;
      alpha : float option;
    }
  | Vline of {
      x : float;
      color : Color.t option;
      line_width : float option;
      line_style : line_style option;
      label : string option;
      alpha : float option;
    }
  | Abline of {
      slope : float;
      intercept : float;
      color : Color.t option;
      line_width : float option;
      line_style : line_style option;
      label : string option;
      alpha : float option;
    }
  | Fill_between of {
      x : Nx.float32_t;
      y1 : Nx.float32_t;
      y2 : Nx.float32_t;
      where : Nx.float32_t option;
      color : Color.t option;
      alpha : float option;
      label : string option;
    }
  | Hspan of {
      y0 : float;
      y1 : float;
      color : Color.t option;
      alpha : float option;
      label : string option;
    }
  | Vspan of {
      x0 : float;
      x1 : float;
      color : Color.t option;
      alpha : float option;
      label : string option;
    }
  | Errorbar of {
      x : Nx.float32_t;
      y : Nx.float32_t;
      yerr :
        [ `Symmetric of Nx.float32_t
        | `Asymmetric of Nx.float32_t * Nx.float32_t ];
      xerr :
        [ `Symmetric of Nx.float32_t
        | `Asymmetric of Nx.float32_t * Nx.float32_t ]
        option;
      color : Color.t option;
      line_width : float option;
      cap_size : float option;
      label : string option;
      alpha : float option;
    }
  | Heatmap of {
      data : Nx.float32_t;
      cmap : Cmap.t option;
      annotate : bool;
      vmin : float option;
      vmax : float option;
      fmt : (float -> string) option;
    }
  | Imshow of {
      data : Nx.float32_t;
      stretch : stretch;
      cmap : Cmap.t option;
      vmin : float option;
      vmax : float option;
    }
  | Contour of {
      data : Nx.float32_t;
      x0 : float;
      x1 : float;
      y0 : float;
      y1 : float;
      levels : [ `Num of int | `Values of float array ];
      filled : bool;
      cmap : Cmap.t option;
      color : Color.t option;
      line_width : float option;
      label : string option;
      alpha : float option;
    }
      (** The type for visual marks. Each constructor carries the data arrays
          and visual properties for one layer. *)

type decoration =
  | Title of string
  | Xlabel of string
  | Ylabel of string
  | Xlim of float * float
  | Ylim of float * float
  | Xscale of scale
  | Yscale of scale
  | Xinvert
  | Yinvert
  | Grid_visible of bool
  | Legend of legend_loc * int
  | Xticks of (float * string) list
  | Yticks of (float * string) list
  | With_theme of Theme.t
  | Xtick_format of (float -> string)
  | Ytick_format of (float -> string)
  | Frame of bool
      (** The type for plot decorations. Applied via {!Decorated} nodes. *)

type t =
  | Mark of mark
  | Layers of t list
  | Decorated of { inner : t; decorations : decoration list }
  | Grid of { rows : t list list; gap : float }
      (** The type for plot specifications. An immutable tree composed via mark
          constructors, {!layers}, decoration functions, and {!grid_layout}. *)

(** {1:marks Mark constructors} *)

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
(** [line ~x ~y ()] is a line mark. *)

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
(** [point ~x ~y ()] is a scatter mark. *)

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
(** [bar ~x ~height ()] is a bar mark. [bottom] defaults to [0.]. *)

val hist :
  x:Nx.float32_t ->
  ?bins:[ `Num of int | `Edges of float array ] ->
  ?density:bool ->
  ?color:Color.t ->
  ?label:string ->
  unit ->
  t
(** [hist ~x ()] is a histogram mark. [bins] defaults to [`Num 10]. *)

val image : ?extent:float * float * float * float -> Nx.uint8_t -> t
(** [image ?extent data] is an image mark. When [extent] is
    [(xmin, xmax, ymin, ymax)], the image is placed in data coordinates. *)

val text :
  x:float ->
  y:float ->
  string ->
  ?color:Color.t ->
  ?font_size:float ->
  unit ->
  t
(** [text ~x ~y s ()] is a text mark at [(x, y)]. *)

val hline :
  y:float ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [hline ~y ()] is a horizontal reference line. *)

val vline :
  x:float ->
  ?color:Color.t ->
  ?line_width:float ->
  ?line_style:line_style ->
  ?label:string ->
  ?alpha:float ->
  unit ->
  t
(** [vline ~x ()] is a vertical reference line. *)

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
(** [abline ~slope ~intercept ()] is a diagonal line [y = slope * x + intercept]
    spanning the full plot area. *)

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
(** [fill_between ~x ~y1 ~y2 ()] is a filled area between two curves. [where] is
    a mask array: only fill where [where.(i) > 0.]. *)

val hspan :
  y0:float ->
  y1:float ->
  ?color:Color.t ->
  ?alpha:float ->
  ?label:string ->
  unit ->
  t
(** [hspan ~y0 ~y1 ()] is a horizontal shaded band. *)

val vspan :
  x0:float ->
  x1:float ->
  ?color:Color.t ->
  ?alpha:float ->
  ?label:string ->
  unit ->
  t
(** [vspan ~x0 ~x1 ()] is a vertical shaded band. *)

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
(** [errorbar ~x ~y ~yerr ()] is an error bar mark. *)

val heatmap :
  data:Nx.float32_t ->
  ?annotate:bool ->
  ?cmap:Cmap.t ->
  ?vmin:float ->
  ?vmax:float ->
  ?fmt:(float -> string) ->
  unit ->
  t
(** [heatmap ~data ()] is a heatmap mark. [data] has shape [[|rows; cols|]]. *)

val imshow :
  data:Nx.float32_t ->
  ?stretch:stretch ->
  ?cmap:Cmap.t ->
  ?vmin:float ->
  ?vmax:float ->
  unit ->
  t
(** [imshow ~data ()] is a colormapped image mark. [stretch] defaults to
    [`Linear]. *)

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
(** [contour ~data ~x0 ~x1 ~y0 ~y1 ()] is a contour mark. [levels] defaults to
    [`Num 8]. [filled] defaults to [false]. *)

(** {1:composition Composition} *)

val layers : t list -> t
(** [layers marks] overlays [marks] on shared axes. *)

(** {1:decorations Decorations} *)

val title : string -> t -> t
(** [title s t] adds plot title [s]. *)

val xlabel : string -> t -> t
(** [xlabel s t] adds x-axis label [s]. *)

val ylabel : string -> t -> t
(** [ylabel s t] adds y-axis label [s]. *)

val xlim : float -> float -> t -> t
(** [xlim lo hi t] fixes the x-axis range. *)

val ylim : float -> float -> t -> t
(** [ylim lo hi t] fixes the y-axis range. *)

val xscale : scale -> t -> t
(** [xscale s t] sets the x-axis scale. *)

val yscale : scale -> t -> t
(** [yscale s t] sets the y-axis scale. *)

val xinvert : t -> t
(** [xinvert t] inverts the x-axis direction (values increase right-to-left). *)

val yinvert : t -> t
(** [yinvert t] inverts the y-axis direction (values increase top-to-bottom). *)

val grid_lines : bool -> t -> t
(** [grid_lines visible t] shows or hides grid lines. *)

val legend : ?loc:legend_loc -> ?ncol:int -> t -> t
(** [legend t] shows the legend. [loc] defaults to {!Upper_right}. [ncol]
    defaults to [1]; set higher for multi-column layouts. *)

val xticks : (float * string) list -> t -> t
(** [xticks ticks t] sets explicit x-axis tick positions and labels. *)

val yticks : (float * string) list -> t -> t
(** [yticks ticks t] sets explicit y-axis tick positions and labels. *)

val with_theme : Theme.t -> t -> t
(** [with_theme th t] overrides the rendering theme. *)

val xtick_format : (float -> string) -> t -> t
(** [xtick_format fmt t] formats x-axis tick labels with [fmt]. *)

val ytick_format : (float -> string) -> t -> t
(** [ytick_format fmt t] formats y-axis tick labels with [fmt]. *)

val frame : bool -> t -> t
(** [frame visible t] shows or hides the axis border rectangle. *)

val no_axes : t -> t
(** [no_axes t] hides the axis frame, ticks, and tick labels. The full panel
    area is used for marks. Title is preserved. Useful for image grids. *)

(** {1:layout Layout} *)

val grid_layout : ?gap:float -> t list list -> t
(** [grid_layout rows] arranges specs in a grid. [gap] defaults to [0.05]. *)
