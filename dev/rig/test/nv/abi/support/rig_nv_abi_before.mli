(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The words allocated before rig_nv_abi initialises. *)

val allocated : unit -> float
(** [allocated ()] is the words the program has allocated so far. *)

val start : float
(** [start] is {!allocated} when this module initialises. *)

val before : float
(** [before] is {!allocated} right after {!start}: [before -. start] is the cost
    of a reading. *)
