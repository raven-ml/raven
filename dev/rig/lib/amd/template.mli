(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The ring writer's templates, as rig_amd_stubs.c reads them into a device's
    state. *)

val flatten :
  ('v -> int64 option) ->
  ('v -> int) ->
  'v Rig_amd_abi.Packet.t ->
  string * string
(** [flatten known arg p] is [(words, holes)]: [words] the bytes of
    {!Rig_amd_abi.Packet.template}[ known p], [holes] its holes, each a record
    of little-endian 64-bit words: the index of its first word, its argument,
    [arg] of its value, its words (1 or 2), its operations' count, then each
    operation (0 adds the constant, 1 shifts right by it, 2 ors it) and its
    constant, innermost first.

    The C state checks them against the template's bounds as it reads them. *)
