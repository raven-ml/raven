(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type line_style = [ `Solid | `Dashed | `Dotted | `Dash_dot ]
type marker = Circle | Square | Triangle | Plus | Star

type legend_loc =
  | Upper_right
  | Upper_left
  | Lower_right
  | Lower_left
  | Center
  | Right
  | Upper_center
  | Lower_center

type scale = [ `Linear | `Log | `Sqrt | `Asinh | `Symlog of float ]
type stretch = [ `Linear | `Log | `Sqrt | `Asinh | `Power of float ]

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

type t =
  | Mark of mark
  | Layers of t list
  | Decorated of { inner : t; decorations : decoration list }
  | Grid of { rows : t list list; gap : float }

(* Mark constructors *)

let line ~x ~y ?color ?line_width ?line_style ?step ?marker ?label ?alpha () =
  Mark
    (Line { x; y; color; line_width; line_style; step; marker; label; alpha })

let point ~x ~y ?color ?color_by ?size ?size_by ?marker ?label ?alpha () =
  Mark (Point { x; y; color; color_by; size; size_by; marker; label; alpha })

let bar ~x ~height ?width ?(bottom = 0.) ?color ?label ?alpha () =
  Mark (Bar { x; height; width; bottom; color; label; alpha })

let hist ~x ?(bins = `Num 10) ?(density = false) ?color ?label () =
  Mark (Hist { x; bins; density; color; label })

let image ?extent data = Mark (Image { data; extent })

let text ~x ~y s ?color ?font_size () =
  Mark (Text_mark { x; y; content = s; color; font_size })

let hline ~y ?color ?line_width ?line_style ?label ?alpha () =
  Mark (Hline { y; color; line_width; line_style; label; alpha })

let vline ~x ?color ?line_width ?line_style ?label ?alpha () =
  Mark (Vline { x; color; line_width; line_style; label; alpha })

let abline ~slope ~intercept ?color ?line_width ?line_style ?label ?alpha () =
  Mark
    (Abline { slope; intercept; color; line_width; line_style; label; alpha })

let fill_between ~x ~y1 ~y2 ?where ?color ?alpha ?label () =
  Mark (Fill_between { x; y1; y2; where; color; alpha; label })

let hspan ~y0 ~y1 ?color ?alpha ?label () =
  Mark (Hspan { y0; y1; color; alpha; label })

let vspan ~x0 ~x1 ?color ?alpha ?label () =
  Mark (Vspan { x0; x1; color; alpha; label })

let errorbar ~x ~y ~yerr ?xerr ?color ?line_width ?cap_size ?label ?alpha () =
  Mark
    (Errorbar { x; y; yerr; xerr; color; line_width; cap_size; label; alpha })

let heatmap ~data ?(annotate = false) ?cmap ?vmin ?vmax ?fmt () =
  Mark (Heatmap { data; cmap; annotate; vmin; vmax; fmt })

let imshow ~data ?(stretch = `Linear) ?cmap ?vmin ?vmax () =
  Mark (Imshow { data; stretch; cmap; vmin; vmax })

let contour ~data ~x0 ~x1 ~y0 ~y1 ?(levels = `Num 8) ?(filled = false) ?cmap
    ?color ?line_width ?label ?alpha () =
  Mark
    (Contour
       {
         data;
         x0;
         x1;
         y0;
         y1;
         levels;
         filled;
         cmap;
         color;
         line_width;
         label;
         alpha;
       })

(* Composition *)

let layers ts = Layers ts

(* Decorations *)

let decorate d = function
  | Decorated r -> Decorated { r with decorations = d :: r.decorations }
  | t -> Decorated { inner = t; decorations = [ d ] }

let title s t = decorate (Title s) t
let xlabel s t = decorate (Xlabel s) t
let ylabel s t = decorate (Ylabel s) t
let xlim lo hi t = decorate (Xlim (lo, hi)) t
let ylim lo hi t = decorate (Ylim (lo, hi)) t
let xscale s t = decorate (Xscale s) t
let yscale s t = decorate (Yscale s) t
let xinvert t = decorate Xinvert t
let yinvert t = decorate Yinvert t
let grid_lines visible t = decorate (Grid_visible visible) t
let legend ?(loc = Upper_right) ?(ncol = 1) t = decorate (Legend (loc, ncol)) t
let xticks ticks t = decorate (Xticks ticks) t
let yticks ticks t = decorate (Yticks ticks) t
let with_theme th t = decorate (With_theme th) t
let xtick_format fmt t = decorate (Xtick_format fmt) t
let ytick_format fmt t = decorate (Ytick_format fmt) t
let frame v t = decorate (Frame v) t

let no_axes t =
  t |> decorate (Frame false) |> decorate (Xticks []) |> decorate (Yticks [])
  |> decorate (Grid_visible false)

(* Layout *)

let grid_layout ?(gap = 0.05) rows = Grid { rows; gap }
