(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scatters by [Set] of rows along the middle axis of three, a position for
    each row of the last axis, as a model writes rows of its cache: of fewer
    updates than a GPU's scatter elects at ({!Rune.Lower.scatter_election}),
    of more, and of positions written in the program, which are distinct. *)

(** The type for how the positions reach the scatter. *)
type positions =
  | Trailing  (** Drawn at [[p; k; 1]], broadcast along the last axis. *)
  | Transposed
      (** Drawn at [[1; k; p]], broadcast to [[q; k; p]], then transposed:
          broadcast along the last axis through a permutation. *)
  | Written
      (** [k] distinct positions inside the axis, written in the program as
          {!Nx.set}'s [L]: the scatter's targets are promised distinct. *)

type t = {
  dtype : Nx.Dtype.any;
  into_shape : int array;  (** [[p; n; q]]. *)
  updates_shape : int array;  (** [[p; k; q]]. *)
  positions : positions;
  pos : int64 array;
      (** At the shape [positions] draws them at, or the [k] written. *)
  seed : int;  (** The seed of the operands' random bytes. *)
  donated : bool;  (** Whether the call donates [into]. *)
}
(** The type for cases. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws floats of 16, 32 and 64 bits, [complex64] and integers, [p]
    from 1 to 3, [q] from 1 to 4, each kind of {!positions}, drawn
    positions that repeat and fall outside the axis, operands of random
    bytes, NaN payloads included, and a donated target or not. *)

val examples : t list
(** [examples] holds a case of each regime of a GPU's scatter of rows: a
    walk outside the election's bounds, elections of 16 to 4096 rows, of
    repeating positions with one outside the axis and [p > 1], and written
    positions of a float and of a complex number. *)

val law : Tolk.Renderer.t -> (module Nx.Devices) -> t -> unit
(** [law ren s c] checks that a compiled call on [s]'s first device, whose
    kernels [ren] renders, with no call evaluated, gives the bits of eager's
    result on the host: the last update in C order of each target, every
    other element unchanged. Under written positions it also checks the
    tangent along the updates, which adds at the promised distinct targets,
    against eager's, with every NaN read as one. It covers each regime of
    {!examples}, positions broadcast along the trailing axes, a repeated
    position, a position outside the axis, a float, a complex number, an
    integer, a donated target, and a tangent of a float and of a complex
    number added at distinct targets. *)
