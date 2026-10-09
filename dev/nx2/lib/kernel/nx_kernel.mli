(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels: what they compute and the contract they implement.

    A kernel library computes nx's operations on the memory of some devices. Its
    [.mli] states what it computes on and which cases it declines, then includes
    {!S}, so that its build checks it against the contract:
    {[
    (** Kernels for host memory. … *)

    include Nx_kernel.S
    ]}
    {!Prog} names what a kernel computes. In C, [nx_kinds.h] computes each kind
    and [nx_spec.h] holds the code {!not_computed}, with no OCaml header, so
    device sources include them. *)

module Prog = Prog
(** Scalar kinds. *)

(** {1:contract The contract} *)

(** The type for kernel libraries.

    Every kernel writes its result into [dst]: fresh, C-contiguous, reaching no
    element twice, of the result's shape and dtype, on the operands' device.
    Operands may be strided, broadcast or offset. The caller has checked their
    shapes, dtypes and axes against the operation's rule.

    A kernel claims [dst] and its operands for the extent of its call through a
    door: [nx_read] of [nx_array.h] for host kernels. [nx_read] waits, under its
    claims, for earlier device work on the operands. A kernel answers [0] once
    its work is done on the host or queued on the device's timeline; the door's
    code if the door refused an operand, before any write; {!not_computed} if it
    does not compute the case, before any write or queued work.
    {!Nx_array.refused} raises a door's code. Value-dependent failures are NaN.
    A kernel's result is a function of its operands' values alone: neither
    layouts nor threads change a bit. *)
module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t

  val name : string
  (** [name] names the kernels in messages, as ["nx.cpu"]. *)

  val computes_on : Rig.t -> bool
  (** [computes_on d] is [true] iff these kernels read and write [d]'s
      memory. *)

  (** {1:elementwise Elementwise} *)

  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> int
  (** [apply1 k ~dst x] stores [k] of each element of [x] into [dst]. *)
end

val not_computed : int
(** [not_computed] is the code a kernel answers for a case it does not compute,
    before any write or queued work. It is none of the door's codes. *)
