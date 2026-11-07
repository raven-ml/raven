(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Means of finite values that stay finite where the values' sum passes the
   largest float. A sum is kept scaled by 2^-[shift]: [shift] is [0] until a
   sum would overflow, and then the sum is scaled down by a power of two,
   exactly. *)

val add : float array -> int array -> int -> float -> unit
(** [add sums shifts k x] adds the finite [x] to the sum [sums.(k)] scaled by
    2^-[shifts.(k)]. *)

val plus : float -> int -> float -> float * int
(** [plus sum shift x] is the sum and shift of [sum] scaled by 2^-[shift] with
    the finite [x] added. *)

val get : float -> int -> int -> float
(** [get sum shift n] is the mean of [n] values whose sum is [sum] scaled by
    2^-[shift], within the finite floats. *)
