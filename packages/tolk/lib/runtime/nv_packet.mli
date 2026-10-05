(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's packets as the nodes of a batch.

    {!Ops_nv} encodes its commands with [Nx_nv_packet], around nodes that a host
    program computes when it runs or that a batch's link writes. This module
    interprets the packets on nodes: their terms, and how they become a queue's
    words and regions. *)

val term : Ops.t Nx_nv_packet.term -> Ops.t
(** [term t] is [t] computed on nodes, in its order: an [Add (t, n)] is
    [Ops.add (term t) c], with [c] the [Uint64] constant [n] as an unsigned
    integer, and a [Shift (t, n)] is [Ops.shr (term t) (Ops.int n)]. *)

val words : Ops.t Nx_nv_packet.word list -> Ops.t list
(** [words ws] is the words [ws] as a queue takes them ({!Hcq2.Queue.q}): a
    [Dword n] is {!Hcq2.Queue.dword}[ n], a [W32 t] is {!term}[ t] as a [Uint32]
    ({!Ops.ccast}), and a [W64 t] is {!term}[ t] as a [Uint64]. *)

val region : string -> string -> (int * Ops.t) list -> Ops.t
(** [region name bytes patches] is the region named [name] of 256-byte alignment
    ({!Ops.arg}'s [Region]) holding [bytes], each of [patches], [(offset, w)],
    written over the bytes from [offset] for the width of [w]'s type. *)

val structure : string -> Ops.t Nx_nv_packet.structure -> Ops.t
(** [structure name s] is {!region}[ name] of [s]'s bytes, each hole of [n]
    bytes its {!term} as the unsigned type of [n] bytes. *)
