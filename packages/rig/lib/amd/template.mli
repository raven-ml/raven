(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packet templates in the bytes the ring writer's C state reads
    (rig_amd_stubs.c). *)

(** The type for what a template is: one of the ring writer's packet
    templates, a function's launch, or a program of packets around a
    measured launch. *)
type form = Packet | Launch | Program

val flatten :
  form ->
  ('v -> int64 option) ->
  ('v -> int * int64) ->
  'v Rig_amd_abi.Packet.t ->
  string * string
(** [flatten form known arg p] is {!Rig_amd_abi.Packet.template}[ known p] as
    [(words, holes)]. [words] holds the template's words. [holes] holds one
    record per hole, six little-endian 64-bit words: the index of its first
    word, its width in words (1 or 2), the argument [i] of [arg v = (i, o)]
    for its term's value [v], then the term's [add] plus [o], its [shift]
    and its [set].

    Raises [Invalid_argument] if [p] exceeds the bounds [rig_amd_stubs.h]
    sets for [form], which [flatten] reads from it: the most words and holes
    and the arguments a hole reads, [RIG_AMD_TEMPLATE_WORDS],
    [RIG_AMD_TEMPLATE_HOLES] and [RIG_AMD_TEMPLATE_ARGS] for a [Packet] and
    their [RIG_AMD_LAUNCH_] and [RIG_AMD_PROGRAM_] counterparts for a
    [Launch] and a [Program], and for any a hole whose shift is outside 0
    to 63. *)
