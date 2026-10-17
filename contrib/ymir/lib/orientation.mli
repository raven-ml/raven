(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The fixed frames' orientations from ICRS, which [test/gen/frames.py]
    writes.

    Each value is a frame's orientation [R], with [v_frame = R · v_icrs], as
    its nine entries in row-major order. Each entry is its definition
    evaluated at 60 digits and rounded once to float64. {!Frame.entries}
    returns these arrays themselves, so callers must not change them. *)

val fk5_j2000 : float array
val galactic : float array
val ecliptic_j2000 : float array
val supergalactic : float array
