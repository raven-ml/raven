(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type font = { family : string; size : float; weight : [ `Normal | `Bold ] }
type line = { color : Color.t; width : float; dash : float list }

type t = {
  background : Color.t;
  palette : Color.t array;
  sequential : Cmap.t;
  diverging : Cmap.t;
  font_title : font;
  font_label : font;
  font_tick : font;
  axis : line;
  grid : line option;
  tick_length : float;
  padding : float;
  title_gap : float;
  label_gap : float;
  scale_factor : float;
  line_width : float;
  marker_size : float;
}

let axis_color = Color.oklch ~l:0.3 ~c:0. ~h:0. ()
let grid_color = Color.with_alpha 0.15 axis_color

let default =
  {
    background = Color.oklch ~l:0.985 ~c:0. ~h:0. ();
    palette =
      [|
        Color.orange;
        Color.sky_blue;
        Color.green;
        Color.darken 0.1 Color.yellow;
        Color.blue;
        Color.vermillion;
        Color.purple;
        Color.black;
      |];
    sequential = Cmap.viridis;
    diverging = Cmap.coolwarm;
    font_title = { family = "sans-serif"; size = 28.; weight = `Bold };
    font_label = { family = "sans-serif"; size = 22.; weight = `Normal };
    font_tick = { family = "sans-serif"; size = 18.; weight = `Normal };
    axis = { color = axis_color; width = 2.; dash = [] };
    grid = Some { color = grid_color; width = 1.; dash = [] };
    tick_length = 10.;
    padding = 24.;
    title_gap = 16.;
    label_gap = 12.;
    scale_factor = 1.;
    line_width = 3.;
    marker_size = 10.;
  }

let dark_bg = Color.oklch ~l:0.15 ~c:0. ~h:0. ()
let dark_fg = Color.oklch ~l:0.8 ~c:0. ~h:0. ()
let dark_grid = Color.with_alpha 0.2 dark_fg

let dark =
  {
    default with
    background = dark_bg;
    palette =
      [|
        Color.orange;
        Color.sky_blue;
        Color.green;
        Color.yellow;
        Color.blue;
        Color.vermillion;
        Color.purple;
        Color.white;
      |];
    axis = { color = dark_fg; width = 2.; dash = [] };
    grid = Some { color = dark_grid; width = 1.; dash = [] };
  }

let minimal =
  { default with axis = { default.axis with width = 1. }; grid = None }

let paper t = { t with scale_factor = 1.0 }
let notebook t = { t with scale_factor = 1.3 }
let talk t = { t with scale_factor = 1.6 }
let poster t = { t with scale_factor = 2.0 }
