(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The ring writer's templates, as rig_nv_stubs.c reads them into a device's
    state. *)

val flatten :
  (int -> int64 option) -> int Rig_nv_abi.Packet.t -> string * string
(** [flatten known p] is [(words, holes)]: [words] the bytes of
    {!Rig_nv_abi.Packet.template}[ known p], [holes] its holes, each a record of
    little-endian 64-bit words: the index of its first word, its value's slot,
    its words (1 or 2), its operations' count, then each operation (0 adds the
    constant, 1 shifts right by it) and its constant, innermost first.

    Raises [Invalid_argument] if the template exceeds the bounds of
    [rig_nv_stubs.h]: 16 words, 6 holes, a hole of 1 or 2 words that reads
    a value of 0 to 2 through at most 3 operations, each shift 0 to 63. *)

val structure_bytes : int
(** [structure_bytes] is the most bytes of a structure, 1024. *)

(** The type for a structure's values: known now, or the writer's value
    number. *)
type operand = Known of int | Slot of int

val structure : operand Rig_nv_abi.Structure.t -> string * string
(** [structure s] is [(bytes, fields)]: [bytes] those of [s] with every hole
    of a [Known] value filled and every other hole [0], [fields] those other
    holes, each a record of little-endian 64-bit words: its byte, its bits, its
    value's slot, its operations' count, then each operation and its constant,
    as {!flatten}'s.

    Raises [Invalid_argument] if [s] exceeds the bounds of [rig_nv_stubs.h]:
    1024 bytes, 16 such holes, each of 1 to 64 bits within the bytes that
    reads a launch's value through at most 3 operations, each shift 0 to
    63. *)
