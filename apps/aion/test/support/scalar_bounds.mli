(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where a scalar codec's float32 transforms may land, by nx.mli's bounds
    ({!Bounds}): [log] within 1 float, [exp] 2, [asinh] and [sinh] 3, each
    applied to the input it is given. *)

val forward : Aion.Scalar.transform -> float -> float * float
(** [forward tr x] is where a float32 [tr] of [x] may land. *)

val inverse : Aion.Scalar.transform -> float -> float * float
(** [inverse tr e] is where a float32 inverse of [tr] at the edge [e] may land.
*)

val crosses :
  Aion.Scalar.transform -> float array -> float -> int -> int -> bool
(** [crosses tr edges x a b] is [true] iff [a] and [b] may both be the token of
    [x] over a channel's [edges]: they are equal, or [tr] rounds and the edges
    between them are of one value, which {!forward}[ tr x] holds. *)
