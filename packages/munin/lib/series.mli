(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The representation of {!Metric.series}: the first [length] samples of
    columns that a history shares with later ones, which append past [length]
    and never write below it. *)

type t = {
  steps : int array;
  times : Float.Array.t;
  values : Float.Array.t;
  length : int;
}
