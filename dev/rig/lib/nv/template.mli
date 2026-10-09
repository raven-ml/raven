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

    The C state checks them against the template's bounds as it reads them. *)
