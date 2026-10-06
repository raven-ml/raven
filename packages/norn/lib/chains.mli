(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The statistics of one element's chains, on the host at float64.

    An element's draws are [float array array], a row per chain. *)

val leaf : ('a, 'b) Nx.t -> (int array * float array array) array
(** [leaf x] is the index and chains of each element of the draws [x], of shape
    [[c; n; ...]], in C order. *)

val pooled : float array array -> float array
(** [pooled m] is every draw of [m]. *)

val mean : float array -> float

val var1 : float array -> float
(** [var1 xs] is the variance of [xs] with one degree of freedom less. *)

val quantile : float array -> float -> float
(** [quantile xs q] interpolates linearly between the order statistics. *)

val undefined : float array array -> bool
(** [undefined m] is [true] if [m] has no diagnostic: chains of fewer than four
    draws, a draw that is not finite, or every draw equal. *)

val rank_rhat : (float array array -> float) -> float array array -> float
(** [rank_rhat stat m] is the larger of [stat] of the split chains' normal
    scores and of the normal scores of their distance to the median. *)

val rhat_of : float array array -> float
(** [rhat_of m] is the R-hat of the chains [m]. *)

val nested_rank_rhat : int -> float array array -> float
(** [nested_rank_rhat k m] is the rank-normalised nested R-hat of [m] in [k]
    consecutive superchains. *)

val ess_bulk_of : float array array -> float
val ess_tail_of : float array array -> float
val mcse_mean_of : float array array -> float
