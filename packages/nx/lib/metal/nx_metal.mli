(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Metal GPUs as nx devices.

    A device of this library is the Metal GPU of a Mac, whose memory
    [Nx_metal_device] opens, named [METAL]. It computes eagerly with no backend:
    [Rune.jit] compiles for it, and an eager operation on its values raises
    [Invalid_argument] naming the remedies. Constants, views, reads and
    [Nx.place] work on it.

    It flushes [float32] subnormal numbers to zero: a subnormal operand reads
    as zero and a subnormal result is written as zero, so nx's accuracy bounds
    hold on it where the operands and the result are normal.

    The library builds on every system; off macOS no device opens. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is Metal GPU [i], opened now, or [Error msg] with the reason, such
    as that a Mac has one GPU or that Metal exists on macOS only. Every [get i]
    that succeeds gives an equal device until the device is lost
    ({!Nx_device.Lost}); a [get i] then opens the GPU anew, an unequal device.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)
