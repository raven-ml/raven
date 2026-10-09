(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The ring writer's templates, as rig_nv_stubs.c copies them into a device's
    state. *)

val flatten :
  (int -> int64 option) -> int Rig_nv_abi.Packet.t -> string * string
(** [flatten known p] is [(words, holes)]: [words] the bytes of
    {!Rig_nv_abi.Packet.template}[ known p], [holes] its holes, each ten
    little-endian 64-bit words: the index of its first word, its value's slot,
    its words (1 or 2), its operations' count, then three pairs of an operation
    (0 adds the constant, 1 shifts right by it) and its constant, innermost
    first.

    Raises [Invalid_argument] if [p] exceeds 16 words or 6 holes, or a hole
    takes more than 3 operations, a shift outside \[[0];[63]\] or a slot outside
    \[[0];[2]\]. *)
