(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stroke styles. *)

type cap = [ `Butt | `Round | `Square ]
(** The type for line caps. *)

type join = [ `Miter | `Round | `Bevel ]
(** The type for line joins. *)

type t = private {
  width : float;
  cap : cap;
  join : join;
  dash : float array;
  miter_limit : float;
}
(** The type for stroke styles. Invariants: [width >= 0.], [miter_limit >= 1.],
    and [dash] is empty or has non-negative entries and a positive sum. *)

val v :
  ?cap:cap ->
  ?join:join ->
  ?dash:float array ->
  ?miter_limit:float ->
  float ->
  t
(** [v ~cap ~join ~dash ~miter_limit width] is the stroke style with line width
    [width]. [cap] and [join] default to [`Round], [dash] to no dashes and
    [miter_limit] to [4.]. Dash entries alternate drawn and skipped lengths in
    the same units as [width]; an odd number of entries repeats.

    Raises [Invalid_argument] if an invariant is violated. *)
