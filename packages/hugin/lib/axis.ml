(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Per-axis configuration and resolution.

   config holds the user-set options (all optional, from decorations). t holds
   the resolved axis with defaults applied and data bounds merged. *)

type config = {
  label : string option;
  lim : (float * float) option;
  scale : Spec.scale option;
  invert : bool;
  ticks : (float * string) list option;
  tick_format : (float -> string) option;
}

let empty_config =
  {
    label = None;
    lim = None;
    scale = None;
    invert = false;
    ticks = None;
    tick_format = None;
  }

type t = {
  scale : Spec.scale;
  invert : bool;
  lo : float;
  hi : float;
  label : string option;
  ticks : (float * string) list option;
  tick_format : (float -> string) option;
}

let resolve ~data_lo ~data_hi (c : config) =
  let scale = Option.value ~default:`Linear c.scale in
  let lo, hi = Option.value ~default:(data_lo, data_hi) c.lim in
  {
    scale;
    invert = c.invert;
    lo;
    hi;
    label = c.label;
    ticks = c.ticks;
    tick_format = c.tick_format;
  }

let make_scale_and_ticks (a : t) =
  let s = Scale.make ~invert:a.invert a.scale ~lo:a.lo ~hi:a.hi () in
  let ticks =
    match a.ticks with
    | Some t -> t
    | None -> Ticks.generate a.scale ~lo:a.lo ~hi:a.hi ()
  in
  let ticks =
    match a.tick_format with
    | None -> ticks
    | Some f -> List.map (fun (v, _) -> (v, f v)) ticks
  in
  (s, ticks)
