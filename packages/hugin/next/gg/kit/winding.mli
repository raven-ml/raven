(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Winding numbers, shared by {!Ring2} and {!Pgon2}. *)

val number : int -> (int -> float) -> (int -> float) -> float -> float -> int
(** [number n x y px py] is the winding number around [(px, py)] of the ring
    through the points [(x i, y i)] for [i] in \[[0];[n - 1]\], a point on the
    ring decided as the library's conventions state. *)
