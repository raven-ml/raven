(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Per-axis configuration and resolution.

    {b Internal module.} Consolidates per-axis state into two stages: {!config}
    holds user-set options from decorations (all optional), and {!t} holds the
    resolved axis with defaults applied and data bounds merged. Used by
    {!Prepared} and {!Resolve}. *)

(** {1:config Configuration} *)

type config = {
  label : string option;
  lim : (float * float) option;
  scale : Spec.scale option;
  invert : bool;
  ticks : (float * string) list option;
  tick_format : (float -> string) option;
}
(** The type for per-axis user options collected from decorations. *)

val empty_config : config
(** [empty_config] is the default configuration: no label, no limits, [invert]
    is [false], scale/ticks/format are [None]. *)

(** {1:resolved Resolved axis} *)

type t = {
  scale : Spec.scale;
  invert : bool;
  lo : float;
  hi : float;
  label : string option;
  ticks : (float * string) list option;
  tick_format : (float -> string) option;
}
(** The type for resolved axes. [scale] defaults to [`Linear], [lo] and [hi]
    come from data bounds unless overridden by {!config.lim}. *)

val resolve : data_lo:float -> data_hi:float -> config -> t
(** [resolve ~data_lo ~data_hi c] is a resolved axis from [c]. Uses [data_lo]
    and [data_hi] when [c.lim] is [None], and [`Linear] when [c.scale] is
    [None]. *)

val make_scale_and_ticks : t -> Scale.t * (float * string) list
(** [make_scale_and_ticks a] is [(scale, ticks)] for [a]. Generates ticks via
    {!Ticks.generate} when [a.ticks] is [None], then applies [a.tick_format] if
    set. *)
