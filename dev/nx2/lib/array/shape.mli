(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Shapes: the facts about extents that movements and layouts share.

    A shape is an [int array] of extents. Each function that refuses one raises
    [Invalid_argument] whose message starts with the name of the public function
    the caller passes as [fn]. *)

val max_rank : int
(** {!Nx_array.Layout.max_rank}. *)

val check_rank : string -> int -> unit
(** [check_rank fn r] raises [Invalid_argument] if [r > max_rank]. *)

val max_numel : int
(** {!Nx_array.Layout.max_numel}. *)

val numel : string -> int array -> int
(** [numel fn s] is the product of [s]'s extents, [1] for no extent. Raises
    [Invalid_argument] if an extent is negative or the product of the extents
    other than [0] exceeds {!max_numel}. *)

val zeros : int -> int array
(** [zeros r] is a fresh array of [r] zeros. *)

val copy : int array -> int array
(** [copy s] is a fresh copy of [s]. *)

val pp : Format.formatter -> int array -> unit
(** [pp] formats a shape or an index for messages, as [[2; 3]]. *)
