(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's packets as the nodes of a batch.

    {!Ops_nv} encodes its commands with [Nx_nv_packet], over nodes that a host
    program computes when it runs or that a batch's link writes. This module is
    that instance: the values, and how packets become a queue's words and
    regions. *)

module Value : Nx_nv_packet.VALUE with type t = Ops.t
(** Nodes as values: [add v n] is [Ops.add v (Ops.int ~dtype:Uint64 n)], and
    [shift_right v n] is [Ops.shr v (Ops.int n)]. *)

val words : Ops.t Nx_nv_packet.word list -> Ops.t list
(** [words ws] is the words [ws] as a queue takes them ({!Hcq2.Queue.q}): a
    [Dword n] is {!Hcq2.Queue.dword}[ n], a [W32 v] is [v] as a [Uint32]
    ({!Ops.ccast}), and a [W64 v] is [v] as a [Uint64]. *)

val region : string -> string -> (int * Ops.t) list -> Ops.t
(** [region name bytes patches] is the region named [name] of 256-byte alignment
    ({!Ops.arg}'s [Region]) holding [bytes], each of [patches], [(offset, w)],
    written over the bytes from [offset] for the width of [w]'s type. *)

val structure : string -> Ops.t Nx_nv_packet.structure -> Ops.t
(** [structure name s] is {!region}[ name] of [s]'s bytes, each hole of [n]
    bytes its value as the unsigned type of [n] bytes. *)
