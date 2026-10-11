(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packet templates in the bytes the ring writer's C state reads
    (rig_amd_stubs.c). *)

(** The type for what a template is: one of the ring writer's packet
    templates, or a function's launch. *)
type form = Packet | Launch

val flatten :
  form ->
  ('v -> int64 option) ->
  ('v -> int) ->
  'v Rig_amd_abi.Packet.t ->
  string * string
(** [flatten form known arg p] is {!Rig_amd_abi.Packet.template}[ known p] as
    [(words, holes)]. [words] holds the template's words. [holes] holds one
    record per hole, of little-endian 64-bit words: the index of its first
    word, [arg] of its value, its width in words (1 or 2), the count of its
    operations, then each operation's code (0 adds, 1 shifts right, 2 ors)
    and constant, innermost first.

    Raises [Invalid_argument] if [p] exceeds the bounds [rig_amd_stubs.h]
    sets for [form]: 16 words, 4 holes and 3 arguments for a [Packet], 64
    words, 12 holes and 11 arguments for a [Launch], and for either a hole
    of 1 or 2 words through at most 3 operations, each shift 0 to 63. *)
