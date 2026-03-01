(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Visual themes.

    A theme controls every non-data visual element: background color,
    typography, axes, grid, spacing, and default data palettes.

    Themes separate two orthogonal concerns: {e style} (aesthetic appearance)
    and {e context} (scaling for the output medium). The {!paper}, {!notebook},
    {!talk}, and {!poster} functions adjust {!field-scale_factor} to uniformly
    scale all visual elements for the target medium. *)

(** {1:types Types} *)

type font = { family : string; size : float; weight : [ `Normal | `Bold ] }
(** The type for font specifications. [size] is in points before
    {!field-scale_factor} is applied. *)

type line = { color : Color.t; width : float; dash : float list }
(** The type for line styles. [dash] is a list of on/off lengths; empty means
    solid. *)

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
(** The type for themes. All dimensional values (font sizes, line widths, gaps)
    are multiplied by {!field-scale_factor} at render time. *)

(** {1:predefined Predefined themes} *)

val default : t
(** [default] is a light theme with subtle grid, Okabe-Ito categorical palette,
    and Tufte-informed defaults. *)

val dark : t
(** [dark] is a dark-background theme. *)

val minimal : t
(** [minimal] is a theme with no grid and thin axes. *)

(** {1:context Context scaling} *)

val paper : t -> t
(** [paper t] is [t] with [scale_factor = 1.0]. *)

val notebook : t -> t
(** [notebook t] is [t] with [scale_factor = 1.3]. *)

val talk : t -> t
(** [talk t] is [t] with [scale_factor = 1.6]. *)

val poster : t -> t
(** [poster t] is [t] with [scale_factor = 2.0]. *)
