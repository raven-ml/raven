(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type color = { r : float; g : float; b : float; a : float }
type cmap = Viridis | Plasma | Inferno | Magma | Cividis | Coolwarm | Gray

module Color = struct
  let rgb r g b = { r = r /. 255.; g = g /. 255.; b = b /. 255.; a = 1. }
  let blue = rgb 31. 119. 180.
  let orange = rgb 255. 127. 14.
  let green = rgb 44. 160. 44.
  let red = rgb 214. 39. 40.
  let magenta = rgb 148. 103. 189. (* purple *)
  let brown = rgb 140. 86. 75.
  let pink = rgb 227. 119. 194.
  let gray = rgb 127. 127. 127.
  let olive = rgb 188. 189. 34.
  let cyan = rgb 23. 190. 207.
  let black = { r = 0.; g = 0.; b = 0.; a = 1. }
  let white = { r = 1.; g = 1.; b = 1.; a = 1. }
  let lightgray = rgb 211. 211. 211.
  let darkgray = rgb 169. 169. 169.
  let yellow = olive
end

module Colormap = struct
  let viridis = Viridis
  let plasma = Plasma
  let inferno = Inferno
  let magma = Magma
  let cividis = Cividis
  let coolwarm = Coolwarm
  let gray = Gray

  let viridis_lut =
    [|
      (* 0.0 *)
      (0.267004, 0.004874, 0.329415);
      (* 0.25 *)
      (0.190631, 0.407061, 0.556089);
      (* 0.5 *)
      (0.128729, 0.563265, 0.550194);
      (* 0.75 *)
      (0.49259, 0.773646, 0.389166);
      (* 1.0 *)
      (0.993248, 0.906157, 0.143936);
    |]

  let plasma_lut =
    [|
      (* 0.0 *)
      (0.050383, 0.029803, 0.527975);
      (* 0.25 *)
      (0.457721, 0.049901, 0.71019);
      (* 0.5 *)
      (0.78396, 0.20376, 0.559144);
      (* 0.75 *)
      (0.976039, 0.479697, 0.215519);
      (* 1.0 *)
      (0.940015, 0.975198, 0.131326);
    |]

  let inferno_lut =
    [|
      (* 0.0 *)
      (0.001462, 0.000466, 0.013866);
      (* 0.25 *)
      (0.239404, 0.052262, 0.318459);
      (* 0.5 *)
      (0.647059, 0.15751, 0.138346);
      (* 0.75 *)
      (0.974998, 0.54451, 0.07199);
      (* 1.0 *)
      (0.988362, 0.998364, 0.644924);
    |]

  let magma_lut =
    [|
      (* 0.0 *)
      (0.001462, 0.000466, 0.013866);
      (* 0.25 *)
      (0.263399, 0.090783, 0.35913);
      (* 0.5 *)
      (0.67503, 0.180444, 0.30488);
      (* 0.75 *)
      (0.97875, 0.563173, 0.30923);
      (* 1.0 *)
      (0.988621, 0.991277, 0.749974);
    |]

  let cividis_lut =
    [|
      (* 0.0 *)
      (0.0, 0.128113, 0.294719);
      (* 0.25 *)
      (0.212092, 0.31128, 0.39814);
      (* 0.5 *)
      (0.458655, 0.492975, 0.40661);
      (* 0.75 *)
      (0.724916, 0.691162, 0.313601);
      (* 1.0 *)
      (0.993578, 0.98313, 0.20794);
    |]

  let coolwarm_lut =
    [|
      (* 0.0 *)
      (0.2298057, 0.29871797, 0.75368315);
      (* 0.25 *)
      (0.41799106, 0.51892369, 0.89460784);
      (* 0.5 *)
      (0.88024788, 0.88024788, 0.88024788);
      (* 0.75 *)
      (0.91587714, 0.56176471, 0.38039216);
      (* 1.0 *)
      (0.70567316, 0.01555616, 0.15023281);
    |]

  let lerp v0 v1 t = v0 +. (t *. (v1 -. v0))

  let apply_lut (lut : (float * float * float) array) (v : float) :
      float * float * float =
    let n = Array.length lut - 1 in
    if n <= 0 then lut.(0)
    else
      let v_clamped = max 0.0 (min 1.0 v) in
      let f_idx = v_clamped *. float_of_int n in
      let idx0 = int_of_float f_idx in
      let idx1 = min n (idx0 + 1) in

      if idx0 = n then lut.(n)
      else
        let t = f_idx -. float_of_int idx0 in
        let r0, g0, b0 = lut.(idx0) in
        let r1, g1, b1 = lut.(idx1) in
        let r = lerp r0 r1 t in
        let g = lerp g0 g1 t in
        let b = lerp b0 b1 t in
        (r, g, b)

  let apply_colormap (t : cmap) (v : float) : int * int * int =
    let r_f, g_f, b_f =
      match t with
      | Gray ->
          let v_clamped = max 0.0 (min 1.0 v) in
          (v_clamped, v_clamped, v_clamped)
      | Viridis -> apply_lut viridis_lut v
      | Plasma -> apply_lut plasma_lut v
      | Inferno -> apply_lut inferno_lut v
      | Magma -> apply_lut magma_lut v
      | Cividis -> apply_lut cividis_lut v
      | Coolwarm -> apply_lut coolwarm_lut v
    in
    let clamp_8bit f = max 0 (min 255 (int_of_float ((f *. 255.) +. 0.5))) in
    (clamp_8bit r_f, clamp_8bit g_f, clamp_8bit b_f)
end

type line_style = Solid | Dashed | DashDot | Dotted | None

type marker_style =
  | Circle
  | Point
  | Pixel
  | Square
  | Triangle
  | Plus
  | Star
  | None

type plot_style = {
  fmt_color : color option;
  fmt_linewidth : float option;
  fmt_linestyle : line_style option;
  fmt_marker : marker_style option;
}

let plot_style ?color ?linewidth ?linestyle ?marker () : plot_style =
  {
    fmt_color = color;
    fmt_linewidth = linewidth;
    fmt_linestyle = linestyle;
    fmt_marker = marker;
  }

type line2d = {
  xdata : Nx.float32_t;
  ydata : Nx.float32_t;
  color : color;
  linewidth : float;
  linestyle : line_style;
  marker : marker_style;
  label : string option;
}

type line3d = {
  xdata : Nx.float32_t;
  ydata : Nx.float32_t;
  zdata : Nx.float32_t;
  color : color;
  linewidth : float;
  linestyle : line_style;
  marker : marker_style;
  label : string option;
}

type scatter_size = Uniform of float | Varying of Nx.float32_t

type scatter = {
  xdata : Nx.float32_t;
  ydata : Nx.float32_t;
  s : scatter_size;
  c : color;
  marker : marker_style;
  label : string option;
}

type bar = {
  x : Nx.float32_t;
  height : Nx.float32_t;
  width : float;
  bottom : float;
  color : color;
  label : string option;
}

type text = {
  x : float;
  y : float;
  content : string;
  color : color;
  fontsize : float;
}

type image_data = Uint8_image of Nx.uint8_t | Float32_image of Nx.float32_t

type image = {
  data : image_data;
  shape : int array;
  extent : (float * float * float * float) option;
  cmap : cmap option;
  aspect : string option;
}

type errorbar_style = {
  yerr : Nx.float32_t option;
  xerr : Nx.float32_t option;
  color : color;
  linewidth : float;
  capsize : float;
}

type step_where = Pre | Post | Mid

type step = {
  xdata : Nx.float32_t;
  ydata : Nx.float32_t;
  color : color;
  linewidth : float;
  linestyle : line_style;
  where : step_where;
  label : string option;
}

type fill_between = {
  xdata : Nx.float32_t;
  y1data : Nx.float32_t;
  y2data : Nx.float32_t;
  color : color; (* Fill color, alpha is part of the color type *)
  where : Nx.float32_t option; (* Optional mask *)
  interpolate : bool; (* Handle NaNs by interpolation? *)
  label : string option;
}

type contour = {
  x : Nx.float32_t; (* 1D array of x coordinates *)
  y : Nx.float32_t; (* 1D array of y coordinates *)
  z : Nx.float32_t; (* 2D array of z values *)
  levels : float array; (* Contour levels *)
  colors : color array option; (* Colors for each level *)
  linewidths : float array option; (* Line widths for each level *)
  linestyles : line_style array option; (* Line styles for each level *)
}

type contour_filled = {
  x : Nx.float32_t; (* 1D array of x coordinates *)
  y : Nx.float32_t; (* 1D array of y coordinates *)
  z : Nx.float32_t; (* 2D array of z values *)
  levels : float array; (* Contour levels *)
  cmap : cmap option; (* Colormap *)
  alpha : float; (* Alpha transparency *)
}

type ref_line = {
  position : float;
  ref_color : color;
  ref_linewidth : float;
  ref_linestyle : line_style;
  ref_label : string option;
}

type diag_line = {
  slope : float;
  intercept : float;
  diag_color : color;
  diag_linewidth : float;
  diag_linestyle : line_style;
  diag_label : string option;
}

type text_labels = {
  tl_xdata : Nx.float32_t;
  tl_ydata : Nx.float32_t;
  labels : string array;
  tl_color : color;
  tl_fontsize : float;
}

(* The main Artist variant type *)
type t =
  | Line2D of line2d
  | Line3D of line3d
  | Scatter of scatter
  | Bar of bar
  | Text of text
  | Image of image
  | Errorbar of line2d * errorbar_style
  | Step of step
  | FillBetween of fill_between
  | Contour of contour
  | ContourFilled of contour_filled
  | HLine of ref_line
  | VLine of ref_line
  | ALine of diag_line
  | TextLabels of text_labels

let line2d ?(color = Color.blue) ?(linewidth = 1.5) ?(linestyle = Solid)
    ?(marker = None) ?label x y =
  if Nx.ndim x <> Nx.ndim y then
    invalid_arg "Artist.line2d: x and y dimensions must match";
  Line2D { xdata = x; ydata = y; color; linewidth; linestyle; marker; label }

let line3d ?(color = Color.blue) ?(linewidth = 1.5) ?(linestyle = Solid)
    ?(marker = None) ?label x y z =
  if Nx.ndim x <> Nx.ndim y || Nx.ndim x <> Nx.ndim z then
    invalid_arg "Artist.line3d: x, y, and z dimensions must match";
  Line3D
    {
      xdata = x;
      ydata = y;
      zdata = z;
      color;
      linewidth;
      linestyle;
      marker;
      label;
    }

let scatter ?(s = 20.0) ?s_data ?(c = Color.blue) ?(marker = Circle) ?label x y
    =
  if Nx.ndim x <> Nx.ndim y then
    invalid_arg "Artist.scatter: x and y dimensions must match";
  let s =
    match s_data with
    | Some arr ->
        if Nx.size arr <> Nx.size x then
          invalid_arg "Artist.scatter: s_data length must match x length";
        Varying arr
    | None -> Uniform s
  in
  Scatter { xdata = x; ydata = y; s; c; marker; label }

let bar ?(width = 0.8) ?(bottom = 0.0) ?(color = Color.blue) ?label ~height x =
  if Nx.size x <> Nx.size height then
    invalid_arg "Artist.bar: x and height lengths must match";
  Bar { x; height; width; bottom; color; label }

let text ?(color = Color.black) ?(fontsize = 10.0) ~x ~y content =
  Text { x; y; content; color; fontsize }

let image : type a b.
    ?extent:float * float * float * float ->
    ?cmap:cmap ->
    ?aspect:string ->
    (a, b) Nx.t ->
    t =
 fun ?extent ?cmap ?aspect data ->
  match Nx.dtype data with
  | Nx.UInt8 ->
      let shape = Nx.shape data in
      let data = Uint8_image data in
      Image { data; shape; extent; cmap; aspect }
  | Nx.Float32 ->
      let shape = Nx.shape data in
      let data = Float32_image data in
      Image { data; shape; extent; cmap; aspect }
  | _ ->
      invalid_arg
        "Artist.image: Unsupported data type. Only UInt8 and Float32 supported"

let image_uint8 ?extent ?cmap ?aspect data =
  let shape = Nx.shape data in
  Image { data = Uint8_image data; shape; extent; cmap; aspect }

let image_float32 ?extent ?cmap ?aspect data =
  let shape = Nx.shape data in
  Image { data = Float32_image data; shape; extent; cmap; aspect }

let step ?(color = Color.blue) ?(linewidth = 1.5) ?(linestyle = Solid)
    ?(where = Post) ?label x y =
  if Nx.ndim x <> 1 || Nx.ndim y <> 1 || Nx.size x <> Nx.size y then
    invalid_arg "Artist.step: x and y must be 1D arrays of the same size";
  Step { xdata = x; ydata = y; color; linewidth; linestyle; where; label }

let fill_between ?(color = Color.blue) ?where ?(interpolate = false) ?label x y1
    y2 =
  if Nx.size x <> Nx.size y1 || Nx.size x <> Nx.size y2 then
    invalid_arg "Artist.fill_between: x, y1, and y2 must have the same size";
  (match where with
  | Some w when Nx.size w <> Nx.size x ->
      invalid_arg "Artist.fill_between: where mask must have same size as x"
  | _ -> ());
  FillBetween
    { xdata = x; y1data = y1; y2data = y2; color; where; interpolate; label }

let contour ?colors ?linewidths ?linestyles ~levels x y z =
  if Nx.ndim x <> 1 || Nx.ndim y <> 1 then
    invalid_arg "Artist.contour: x and y must be 1D arrays";
  if Nx.ndim z <> 2 then invalid_arg "Artist.contour: z must be a 2D array";
  let z_shape = Nx.shape z in
  if z_shape.(0) <> Nx.size y || z_shape.(1) <> Nx.size x then
    invalid_arg "Artist.contour: z shape must match (len(y), len(x))";

  (* Validate optional arrays if provided *)
  (match colors with
  | Some c when Array.length c <> Array.length levels ->
      invalid_arg "Artist.contour: colors array must match levels length"
  | _ -> ());
  (match linewidths with
  | Some lw when Array.length lw <> Array.length levels ->
      invalid_arg "Artist.contour: linewidths array must match levels length"
  | _ -> ());
  (match linestyles with
  | Some ls when Array.length ls <> Array.length levels ->
      invalid_arg "Artist.contour: linestyles array must match levels length"
  | _ -> ());

  Contour { x; y; z; levels; colors; linewidths; linestyles }

let contourf ?(cmap = Viridis) ?(alpha = 1.0) ~levels x y z =
  if Nx.ndim x <> 1 || Nx.ndim y <> 1 then
    invalid_arg "Artist.contourf: x and y must be 1D arrays";
  if Nx.ndim z <> 2 then invalid_arg "Artist.contourf: z must be a 2D array";
  let z_shape = Nx.shape z in
  if z_shape.(0) <> Nx.size y || z_shape.(1) <> Nx.size x then
    invalid_arg "Artist.contourf: z shape must match (len(y), len(x))";
  if alpha < 0.0 || alpha > 1.0 then
    invalid_arg "Artist.contourf: alpha must be between 0.0 and 1.0";

  ContourFilled { x; y; z; levels; cmap = Some cmap; alpha }

let hline ?(color = Color.gray) ?(linewidth = 1.0) ?(linestyle = Dashed) ?label
    y =
  HLine
    {
      position = y;
      ref_color = color;
      ref_linewidth = linewidth;
      ref_linestyle = linestyle;
      ref_label = label;
    }

let vline ?(color = Color.gray) ?(linewidth = 1.0) ?(linestyle = Dashed) ?label
    x =
  VLine
    {
      position = x;
      ref_color = color;
      ref_linewidth = linewidth;
      ref_linestyle = linestyle;
      ref_label = label;
    }

let abline ?(color = Color.gray) ?(linewidth = 1.0) ?(linestyle = Dashed) ?label
    ~slope ~intercept () =
  ALine
    {
      slope;
      intercept;
      diag_color = color;
      diag_linewidth = linewidth;
      diag_linestyle = linestyle;
      diag_label = label;
    }

let text_labels ?(color = Color.black) ?(fontsize = 10.0) x y labels =
  if Nx.size x <> Nx.size y then
    invalid_arg "Artist.text_labels: x and y must have the same size";
  if Nx.size x <> Array.length labels then
    invalid_arg "Artist.text_labels: labels length must match x length";
  TextLabels
    {
      tl_xdata = x;
      tl_ydata = y;
      labels;
      tl_color = color;
      tl_fontsize = fontsize;
    }
