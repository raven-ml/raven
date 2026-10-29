(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functions at the points where they have no derivative.

    rune differentiates [sqrt] at 0 and [atan2] at (0, 0) to infinities, and a
    [where] that selects another value after the operation still multiplies
    its zero cotangent by them, which gives NaN. Each function here replaces
    the operands at [singular] by harmless ones before the operation and
    selects [value] after it, so the result there is [value] with derivative
    0. *)

val sqrt :
  'd Nx.bool_t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t
(** [sqrt singular value x] is [sqrt x], and [value] where [singular]
    holds. *)

val atan2 :
  'd Nx.bool_t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t
(** [atan2 singular value y x] is [atan2 y x], and [value] where [singular]
    holds. *)
