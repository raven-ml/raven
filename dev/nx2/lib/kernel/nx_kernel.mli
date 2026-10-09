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
    {!Prog} names what an elementwise kernel computes and {!Spec} describes
    what the others compute. In C, [nx_kinds.h] computes each kind and
    [nx_spec.h] holds the descriptors' structs, with no OCaml header, so device
    sources include them. *)

module Prog = Prog
(** Scalar kinds. *)

module Spec = Spec
(** Descriptors. *)

(** {1:contract The contract} *)

(** The type for kernel libraries.

    Every kernel writes its result into [dst]: C-contiguous, reaching no
    element twice, of the result's shape and dtype, on the operands' device.
    Operands may be strided, broadcast or offset. The caller has checked their
    shapes, dtypes and axes against the operation's rule.

    [dst] shares no byte with an operand, except one it is identical to
    ({!Nx_array.door}) that the kernel reads only at the result's own index:
    an operand of [apply1] to [apply3], a [Plain] load of [map], or
    [contract]'s [init]. A kernel reads such an operand at an index before it
    writes [dst] there, so the result is the one a fresh [dst] receives. The
    door refuses an operand that shares a byte with [dst] without being
    identical to it; an identical operand the kernel reads at other indices,
    as a gather's source or a contraction's [a] and [b], is the caller's
    error.

    A kernel claims [dst] and its operands for the extent of its call through a
    door: [nx_read] of [nx_array.h] for host kernels, {!Nx_array.door} for
    kernels that submit device work. [nx_read] waits, under its claims, for
    earlier device work on the operands; {!Nx_array.door} holds its claims until
    the work is submitted. A kernel answers [Done] once its work is done on the
    host or queued on the device's timeline; the door's refusal, before any
    write; [Declined] if it does not compute the case, before any write or
    queued work. Value-dependent failures are NaN. A kernel's result is a
    function of its operands' values alone: neither layouts nor threads change a
    bit. A kernel raises [Out_of_memory] if host memory runs out and
    {!Rig.Out_of_memory} if its device's does; [Declined] says only that it
    does not compute the case.

    An [apply] entry answers [Wrong_dtype] for dtypes its kind does not take
    ({!Prog.accepts0} to {!Prog.accepts3}), whatever its caller checked. A
    decline of [apply0] to [apply3] at a base dtype (float32, float64, the 8-
    to 64-bit integers and bool), or of [reduce] or [scan] of one [Sum],
    [Prod], [Max] or [Min] of a program's one operand into its own dtype, read
    plain, at a base dtype, is an error its caller raises; elsewhere, and for
    [map], its caller computes the case from other operations. *)
module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  (** [name] names the kernels in messages, as ["nx.cpu"]. *)

  val computes_on : Rig.t -> bool
  (** [computes_on d] is [true] iff these kernels compute on [d]'s memory. *)

  (** {1:elementwise Elementwise} *)

  val apply0 : Prog.op0 -> dst:('v, 's) a -> answer
  (** [apply0 k ~dst] stores [k] at each index of [dst]. *)

  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
  (** [apply1 k ~dst x] stores [k] of each element of [x] into [dst]. *)

  val apply2 : Prog.op2 -> dst:('v, 's) a -> ('a, 'b) a -> ('a, 'b) a -> answer
  (** [apply2 k ~dst x y] stores [k] of each element of [x] and [y] into
      [dst]. *)

  val apply3 :
    Prog.op3 -> dst:('v, 's) a -> ('c, 'e) a -> ('a, 'b) a -> ('a, 'b) a -> answer
  (** [apply3 k ~dst c x y] stores [k] of each element of [c], [x] and [y]
      into [dst]. *)

  val map : Spec.map Spec.t -> dsts:any array -> any array -> answer
  (** [map s ~dsts ops] stores [s]'s results into [dsts], one per output of
      its program, from [ops], one per load. *)

  (** {1:reductions Reductions and scans} *)

  val reduce : Spec.reduce Spec.t -> dsts:any array -> any array -> answer
  (** [reduce s ~dsts ops] stores [s]'s results into [dsts], in the order
      {!Spec.reduce} gives them, from [ops], one per load. *)

  val scan : Spec.scan Spec.t -> dsts:any array -> any array -> answer
  (** [scan s ~dsts ops] stores [s]'s results into [dsts] from [ops], one per
      load. *)

  (** {1:contraction Contraction} *)

  val contract : Spec.contract Spec.t -> dst:any -> any array -> answer
  (** [contract s ~dst ops] stores the contraction [s] of [ops] into [dst]: [a],
      [b], then [init] where [s] has one. *)
end
