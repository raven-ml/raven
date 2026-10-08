(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packets: the 32-bit words a GPU's queue reads, around the caller's values.

    An encoder ({!Pm4}, {!Aql}, {!Sdma}, {!Thread_trace}) describes a packet
    whose values are of any type ['v]: integers for a driver that writes its
    ring now, or values a compiler computes later. The words and terms are
    {!Rig_packet}'s, which interprets them: {!Rig_packet.encode} with integers,
    {!Rig_packet.template} with some values left for later, {!Rig_packet.load}
    as a driver's C template, a compiler as its own nodes.

    Packets concatenate with [@]: the words of [p @ q] are [p]'s, then [q]'s. *)

(** {1:words Terms and words} *)

(** The type for computations on a value ({!Rig_packet.term}). *)
type 'v term = 'v Rig_packet.term =
  | Value of 'v
  | Add of 'v term * int64
  | Shift of 'v term * int
  | Or of 'v term * int64

(** The type for the words of a packet ({!Rig_packet.word}). *)
type 'v word = 'v Rig_packet.word =
  | Dword of int
  | W32 of 'v term
  | W64 of 'v term

type 'v t = 'v word list
(** The type for packets, and for sequences of packets, in queue order. *)

(** {1:operands Operands} *)

(** The type for the comparisons of a wait: the bits a wait reads are equal to,
    or at least, the reference, as unsigned integers. *)
type comparison = Equal | Greater_equal

(** The type for the scope of a cache operation: work on this GPU ([Agent]), for
    which the L2 keeps memory coherent, or anyone ([System]), the host and other
    devices included. *)
type scope = Agent | System
