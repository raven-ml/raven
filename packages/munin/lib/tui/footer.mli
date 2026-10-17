(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's bottom bar of key hints. *)

val view :
  mode:[< `Dashboard | `Detail of Theme.smooth | `Info ] -> 'a Mosaic.t
(** [view ~mode] is the bar of key hints for the screen [mode]. Its
    [`Detail s] form also shows the smoothing level [s] unless it is
    [Off]. *)
