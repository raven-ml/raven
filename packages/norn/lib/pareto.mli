(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pareto smoothing of one set of log importance ratios, on the host at
    float64, as {!Norn.Diag.psis} states it. *)

val tail_length : r_eff:float -> int -> int
(** [tail_length ~r_eff s] is [floor (min (0.2 s) (3 sqrt (s / r_eff)))], the
    ratios that [s] draws of relative efficiency [r_eff] smooth. *)

val smooth : tail:int -> float array -> float array * float
(** [smooth ~tail lr] is the smoothed log weights of the log ratios [lr], their
    [tail] largest replaced, and the tail's shape [k̂]. [tail] is at most a
    fifth of the ratios. *)

val threshold : int -> float
(** [threshold s] is [min (1 - 1 / log10 s) 0.7], the largest [k̂] for which
    [s] draws give a reliable estimate. *)

val log_sum_exp : float array -> float
(** [log_sum_exp xs] is [log (Σ exp xs)], [-inf] for no [xs] or all [-inf]. *)
