(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The ring writer's templates, as rig_amd_stubs.c copies them into a device's
    state. *)

val flatten : int Rig_amd_abi.Packet.t -> string * string
(** [flatten p] is [(words, holes)]: [words] the bytes of
    {!Rig_amd_abi.Packet.template} of [p] with every value unknown, [holes] its
    holes, each ten little-endian 64-bit words: the index of its first word, its
    argument, its words (1 or 2), its operations' count, then three pairs of an
    operation (0 adds the constant, 1 shifts right by it, 2 ors it) and its
    constant, innermost first.

    Raises [Invalid_argument] if [p] exceeds 16 words or 4 holes, or a hole
    takes more than 3 operations, a shift outside \[[0];[63]\] or an argument
    outside \[[0];[2]\]. *)
