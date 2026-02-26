(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type padding = { left : float; right : float; top : float; bottom : float }

module Defaults = struct
  let font_face = "Sans"
  let tick_font_size = 20.0
  let label_font_size = 22.0
  let title_font_size = 28.0
  let tick_length = 10.0
  let tick_label_gap = 10.0
  let axis_label_gap = 20.0
  let title_gap = 16.0
  let min_padding = 5.0
end

let calculate_axes_layout cr (ax : Axes.t) (xticks : float list)
    (yticks : float list) (axes_pixel_bounds : Transforms.pixel_bounds) =
  let padding =
    let measure_text font_size text =
      if text = "" then (0., 0.)
      else (
        Cairo.select_font_face cr Defaults.font_face;
        Cairo.set_font_size cr font_size;
        let te = Cairo.text_extents cr text in
        (te.Cairo.width, te.Cairo.height))
    in
    let measure_tick_labels font_size ticks formatter =
      List.fold_left
        (fun (max_w, max_h) tick ->
          let label = formatter tick in
          let w, h = measure_text font_size label in
          (Float.max max_w w, Float.max max_h h))
        (0.0, 0.0) ticks
    in

    let max_y_tick_width, _ =
      measure_tick_labels Defaults.tick_font_size yticks (Printf.sprintf "%.3g")
    in
    let _, max_x_tick_height =
      measure_tick_labels Defaults.tick_font_size xticks (Printf.sprintf "%.3g")
    in
    let _, x_label_height = measure_text Defaults.label_font_size ax.xlabel in
    let _y_label_width, y_label_height =
      measure_text Defaults.label_font_size ax.ylabel
    in
    let y_label_eff_width = y_label_height in
    let _, title_height = measure_text Defaults.title_font_size ax.title in

    let left_pad =
      Defaults.min_padding
      +. (if y_label_eff_width > 0. then
            y_label_eff_width +. Defaults.axis_label_gap
          else 0.)
      +. (if max_y_tick_width > 0. then
            max_y_tick_width +. Defaults.tick_label_gap
          else 0.)
      +. Defaults.tick_length
    in
    let bottom_pad =
      Defaults.min_padding
      +. (if x_label_height > 0. then x_label_height +. Defaults.axis_label_gap
          else 0.)
      +. (if max_x_tick_height > 0. then
            max_x_tick_height +. Defaults.tick_label_gap
          else 0.)
      +. Defaults.tick_length
    in
    let top_pad =
      Defaults.min_padding
      +. if title_height > 0. then title_height +. Defaults.title_gap else 0.
    in
    let right_pad = Defaults.min_padding in
    { left = left_pad; bottom = bottom_pad; top = top_pad; right = right_pad }
  in

  let plot_left = axes_pixel_bounds.left +. padding.left in
  let plot_top = axes_pixel_bounds.top +. padding.top in
  let plot_width =
    Float.max 0. (axes_pixel_bounds.width -. padding.left -. padding.right)
  in
  let plot_height =
    Float.max 0. (axes_pixel_bounds.height -. padding.top -. padding.bottom)
  in

  let plot_pixel_bounds =
    {
      Transforms.left = plot_left;
      top = plot_top;
      width = plot_width;
      height = plot_height;
    }
  in

  (plot_pixel_bounds, padding)
