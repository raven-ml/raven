(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scatters whose target holds signalling NaNs, negative zeros and subnormals
    where no update lands, compiled on a set's first device. *)

type t = {
  dtype : Nx.Dtype.any;  (** [float16], [bfloat16] or [float32]. *)
  combine : Nx.combine;
  n : int;
      (** The target's positions along the scattered axis, and the updates
          along it. *)
  transposed : bool;
      (** Whether the target is a transposed view, as the traced function
          reads it. *)
}
(** The type for cases. The target is [[n; 4; 4]], scattered along its
    first axis; every update lands on its first row. *)

val pp : Format.formatter -> t -> unit

val cases : t list
(** [cases] holds each 16-bit float and [float32] under each combine, at
    [n = 4] and at [n = 40], each plain and transposed. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that a scatter compiled on [s]'s first device, with no
    call evaluated, keeps the bits of each element of the target no update
    lands on, as {!Nx.scatter} states: its signalling NaNs of both signs, quiet
    NaNs, zeros of both signs, subnormals of both signs and numbers. *)
