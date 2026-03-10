(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Data-only compilation stage.

    {b Internal module.} Compiles a {!Spec.t} tree into a {!t} tree with all
    data-dependent work done: decoration collection, histogram binning,
    auto-coloring, data-bound computation, imshow rasterization, contour
    tracing, and guide-range detection.

    The result is independent of output dimensions and can be resolved
    repeatedly at different sizes by {!Resolve.resolve_prepared}. *)

(** {1:bounds Data bounds} *)

val nx_finite_range : Nx.float32_t -> float * float
(** [nx_finite_range arr] is [(lo, hi)] of the finite values in [arr]. *)

(** {1:marks Mark introspection} *)

val mark_color : Spec.mark -> Color.t option
(** [mark_color m] is the color of [m], if set. *)

(** {1:contour Contour tracing} *)

type contour_paths = { level : float; paths : (float * float) array list }
(** The type for traced contour paths at a single iso-level. Coordinates are in
    data space. *)

val prepare_contour :
  x0:float ->
  x1:float ->
  y0:float ->
  y1:float ->
  data:Nx.float32_t ->
  levels:[ `Num of int | `Values of float array ] ->
  contour_paths list
(** [prepare_contour ~x0 ~x1 ~y0 ~y1 ~data ~levels] traces contour paths through
    [data] and maps grid coordinates to the data-space rectangle \[[x0];[x1]\] x
    \[[y0];[y1]\]. *)

(** {1:panel Prepared panel} *)

type panel = {
  marks : Spec.mark list;
  x : Axis.t;
  y : Axis.t;
  title : string option;
  legend_loc : Spec.legend_loc option;
  legend_ncol : int;
  grid_visible : bool option;
  frame_visible : bool option;
  theme_override : Theme.t option;
  colorbar_range : (float * float) option;
  size_by_range : (float * float) option;
}
(** The type for prepared panels. All data-only work is done: marks are
    auto-colored and histograms normalized to bars, data bounds are computed,
    and guide ranges are detected. *)

(** {1:grid Grid decorations} *)

type grid_decorations = {
  gd_title : string option;
  gd_xlabel : string option;
  gd_ylabel : string option;
  gd_legend_loc : Spec.legend_loc option;
  gd_legend_ncol : int;
  gd_theme_override : Theme.t option;
}
(** The type for grid-level decorations extracted from a decorated grid spec. *)

(** {1:tree Prepared tree} *)

type t =
  | Panel of panel
  | Grid of { rows : t list list; gap : float }
  | Decorated_grid of {
      decorations : grid_decorations;
      inner : t;
      all_marks : Spec.mark list;
    }
      (** The type for prepared spec trees. Mirrors {!Spec.t} structure with all
          data-only work pre-computed. *)

(** {1:compile Compilation} *)

val compile : theme:Theme.t -> Spec.t -> t
(** [compile ~theme spec] is the prepared tree for [spec]. Collects decorations,
    normalizes histograms, auto-colors marks, computes data bounds, and detects
    colorbar/size-guide ranges. *)
