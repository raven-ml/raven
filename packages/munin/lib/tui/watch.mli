(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the dashboard has seen of a run since it began watching it: the arrivals
   of its metric samples, timed on the dashboard's clock, from which the state
   word comes (the States section of [Munin_tui]). *)

type t

val start : now:float -> samples:int -> t
(** [start ~now ~samples] is a watch that began at [now] on a run whose log held
    [samples] metric samples. Those are no arrival. *)

val observe : t -> now:float -> samples:int -> t
(** [observe w ~now ~samples] is [w] after a poll at [now] read [samples] metric
    samples: an arrival when they are more than before. *)

val since : t -> float
val samples : t -> int
val last_arrival : t -> float option

val usual_gap : t -> float option
(** [usual_gap w] is the median gap between the last 32 arrivals, once two are
    known. *)

val stalled : t -> now:float -> bool
(** [stalled w ~now] is [true] iff no arrival came for longer than three usual
    gaps, and at least 10 s. *)

val word : Munin.Run.status -> t option -> now:float -> string
(** [word status w ~now] is the state word of a run of [status] watched by [w].
*)
