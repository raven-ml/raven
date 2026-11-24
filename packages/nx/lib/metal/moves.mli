(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Maps that move bits: plans of the copy kernels. *)

val map :
  Nx_kernel.Prog.t -> Operands.t -> (Plan.t option, Nx_array.answer) result
(** [map p xs] plans [p] into the destinations of the call of facts [xs]
    ({!Operands}), each C-contiguous, where each output is an operand's bits,
    the operand itself or a [Copy], a [Bitcast], or a [Cast] of it into its own
    dtype or between integers of one width, of a [Plain] load of a dtype of a
    byte or more: launches of the copy kernels for each output, of one thread
    for 16 bytes where the operand's elements lie at consecutive positions and
    both ends' first bytes on 16 bytes, and for an element otherwise, as many as
    keep each within {!Plan.allowed}. An output whose operand's buffer overlaps
    another output's destination copies first into the workspace, before any
    destination is written, so that every output reads the operands as the call
    found them: the run of positions the operand's layout reaches, where it
    holds fewer elements than the operand, and the operand's elements otherwise.
    A call of more launches than two an output takes a submission of its own. It
    is [Ok None] for a call of no element, and [Error Declined] for a
    destination that is not C-contiguous and for every other program.

    nx.metal has checked the call: [p] and [xs] fit. *)

(** {1:loops Copies in a loop}

    A loop's pass whose outputs each move an operand's bits into a C-contiguous
    destination copies them as {!map} does. *)

type site
(** The type for a copy's ends: elements of a dtype of a byte or more at a
    layout's positions in a plan's slot. *)

val site_at :
  slot:int ->
  base:int ->
  residue:int ->
  Nx_array.Dtype.any ->
  Nx_array.Layout.t ->
  site
(** [site_at ~slot ~base ~residue dt l] is the end of [dt] elements at [l]'s
    positions from byte [base] of [slot], whose first byte's address is
    [residue] modulo 32, as {!Operands.operand} holds it: 0 for the workspace.
*)

val moved : Nx_kernel.Prog.t -> int -> int option
(** [moved p k] is the operand whose bits [p]'s output [k] is, as {!map} takes
    it, if it is an operand's. *)

val launch : Plan.builder -> n:int -> dst:site -> site -> float
(** [launch b ~n ~dst src] adds to [b] the launches that copy [src]'s [n]
    elements, in C order, into consecutive ones of [dst] from its position 0, as
    {!map} copies an output, and is their {!Plan.seconds} together. *)

val bound_copy : wide:int -> unit
(** [bound_copy ~wide] is {!Nx_metal.bound_copy}. *)

val bigbytes : string -> Plan.bigbytes
(** [bigbytes s] is a constant's bytes of [s]. *)

(** {1:widenings Widenings} *)

val place : Nx_array.any -> int * int * int
(** [place x] is [x] as a kernel that reads or writes elements by their
    positions addresses it from its buffer's slot: the byte offset in it of its
    address; the position of its index 0 counted from its lowest element's,
    which every position it reaches is at or past; and the bit of the address's
    byte at which its lowest element starts. The address is its lowest
    element's, or for a dtype narrower than a byte the 4-byte word's that holds
    it: a kernel reads and writes such an element in whole words, which start at
    multiples of 4 from its allocation's first byte (rig.mli's Words). *)

val twin : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [twin dt] is the dtype of a byte that holds each value of [dt], where [dt]
    is narrower than a byte: [Int8] for [Int4], [Uint8] for [Uint4], [Bool] for
    [Bit] and [Float8_e4m3fn] for [Float4_e2m1fn]. It is [dt] otherwise. *)

val widen :
  Plan.builder ->
  dst:site ->
  slot:int ->
  Nx_array.any ->
  Nx_array.Layout.t ->
  unit
(** [widen b ~dst ~slot x l] adds to [b] the launches of the widen kernel that
    cast the elements of [x], of a dtype narrower than a byte, in [slot], at the
    positions [l] gives its indices, in C order of them, into consecutive bytes
    of {!twin}'s dtype of [dst] from its position 0, each within
    {!Plan.allowed}[ ()]. Each cast keeps its value. [l] reaches positions of
    [x]'s layout alone. *)
