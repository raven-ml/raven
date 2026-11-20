(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exactly rounded sums, for {!Mynd_coco.summary}. *)

val mean : float array -> int -> float
(** [mean a n] is the exactly rounded sum of [a]'s first [n] values divided by
    [n], and NaN when [n] is [0]. The values must be finite. *)
