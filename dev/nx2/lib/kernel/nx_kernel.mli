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
    an operand of [apply1] to [apply3], a [Plain] load of [map],
    [contract]'s [init], [scatter]'s [into], or [assemble]'s first piece
    where its region is the whole result. A kernel reads such an operand at
    an index before it writes [dst] there, so the result is the one a fresh
    [dst] receives. The door refuses an operand that shares a byte with
    [dst] without being identical to it; an identical operand the kernel
    reads at other indices, as a gather's source or a contraction's [a] and
    [b], is the caller's error.

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

    An [apply], [fft] or [linalg] entry answers [Wrong_dtype] for dtypes its
    kind ({!Prog.accepts0} to {!Prog.accepts3}), transform or routine
    ({!Spec.dtypes}) does not take or give, whatever its caller checked. A
    decline of [apply0] to [apply3] at a base dtype (float32, float64, the 8-
    to 64-bit integers and bool), of [reduce] or [scan] of one [Sum],
    [Prod], [Max] or [Min] of a program's one operand into its own dtype, read
    plain, at a base dtype, of [gather] at a dtype of eight bits or more, or
    of [scatter] with [unique] at such a dtype, is an error its caller
    raises; elsewhere, and for [map], [sort], [assemble], [fold], [fft] and
    [linalg], its caller computes the case from other operations. *)
module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t
  type answer := Nx_array.answer
  type any := Nx_array.any
  type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

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
    Prog.op3 -> dst:('v, 's) a -> ('c, 'e) a -> ('v, 's) a -> ('v, 's) a -> answer
  (** [apply3 k ~dst c x y] stores [k] of each element of [c], [x] and [y]
      into [dst], of [x]'s and [y]'s dtype, as [Where] and [Fma] give. *)

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

  (** {1:index Gathers and scatters} *)

  val gather : Spec.gather Spec.t -> dst:('v, 's) a -> index -> ('v, 's) a -> answer
  (** [gather s ~dst idx x] stores into [dst] the elements of [x] at the
      positions [idx] holds. *)

  val scatter :
    Spec.scatter Spec.t ->
    dst:('v, 's) a ->
    into:('v, 's) a ->
    index ->
    ('v, 's) a ->
    answer
  (** [scatter s ~dst ~into idx updates] stores into [dst] the elements of
      [into] with [updates] combined at the positions [idx] holds. It answers
      [Wrong_dtype] for an [Add] of a dtype {!Prog.Add} does not take. *)

  (** {1:sorts Sorts} *)

  val sort :
    Spec.sort Spec.t -> values:('v, 's) a -> positions:index -> ('v, 's) a -> answer
  (** [sort s ~values ~positions x] stores [x]'s ordered elements into
      [values] and their positions into [positions]. *)

  (** {1:assembly Assemblies and folds} *)

  val assemble : Spec.assemble Spec.t -> dst:('v, 's) a -> ('v, 's) a array -> answer
  (** [assemble s ~dst pieces] stores [s]'s assembly of [pieces] into [dst].
      It answers [Wrong_dtype] if [s]'s fill is not an element of [dst]'s
      dtype. *)

  val fold : Spec.fold Spec.t -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [fold s ~dst x] stores the fold [s] of [x] into [dst]. It answers
      [Wrong_dtype] for a dtype {!Prog.Add} does not take. *)

  (** {1:contraction Contraction} *)

  val contract : Spec.contract Spec.t -> dst:any -> any array -> answer
  (** [contract s ~dst ops] stores the contraction [s] of [ops] into [dst]: [a],
      [b], then [init] where [s] has one. *)

  (** {1:transforms Transforms and factorisations} *)

  val fft : Spec.fft Spec.t -> dst:any -> any -> answer
  (** [fft s ~dst x] stores the transform [s] of [x] into [dst]. *)

  val linalg : Spec.linalg Spec.t -> dsts:any array -> any array -> answer
  (** [linalg s ~dsts ops] stores [s]'s results into [dsts], in the order
      {!Spec.linalg} gives them, from [ops]: [a], then [b] for a triangular
      solve. *)
end
