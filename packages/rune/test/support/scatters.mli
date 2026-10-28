(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scatters past the work bound: 33 to 48 updates over 33 to 40 positions,
    along the middle axis of three, which a mask would compute in more than 16
    times eager's work. *)

(** The type for how the positions reach the scatter: as drawn, or broadcast
    or transposed inside the traced function from fewer elements. *)
type positions =
  | Full  (** Drawn at the updates' shape [[p; k; q]]. *)
  | Trailing  (** Drawn at [[p; k; 1]], broadcast along the last axis. *)
  | Leading  (** Drawn at [[1; k; q]], broadcast along the first axis. *)
  | Transposed
      (** Drawn at [[1; k; p]], broadcast to [[q; k; p]], then transposed:
          broadcast along the last axis through a permutation. *)

type t = {
  dtype : Nx.Dtype.any;
  combine : Nx.combine;
  into_shape : int array;  (** [[p; n; q]]. *)
  updates_shape : int array;  (** [[p; k; q]]. *)
  positions : positions;
  pos : int64 array;  (** At the shape [positions] draws them at. *)
  into : Operand.t;
  updates : Operand.t;
  donated : bool;  (** Whether the call donates [into]. *)
}
(** The type for cases. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws every dtype of {!examples}, each combine it takes, positions
    that repeat and fall outside the axis, each kind of {!positions}, and
    operands of {!Operand.draw}. An [Add] of floats draws halves of at most 8
    in magnitude and zeros of both signs, which every order sums exactly. *)

val examples : t list
(** [examples] holds a case of each combine of each kind of dtype, of each
    kind of {!positions}, with its target donated, with its updates an offset
    view, and of each narrow float's [Add] and each complex combine. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that a compiled call on [s]'s first device gives eager's
    values on the host, with no call evaluated: integers and booleans exactly,
    floats with every NaN read as one ({!Witness.codes}). It covers each
    combine, each kind of dtype and of {!positions}, positions broadcast along
    the trailing axes and positions not, a donated target, a position outside
    the axis, a narrow float's [Add], an offset view of the updates, and each
    complex combine. *)
