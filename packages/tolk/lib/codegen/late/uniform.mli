(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values the same for every thread of a workgroup.

    A workgroup's threads run a kernel together. A barrier ({!Op.Barrier}) waits
    until each of them reaches it, so a thread that skips one, or reaches it a
    different number of times, hangs the workgroup or reads memory another
    thread has not written. A matrix multiply-accumulate ({!Op.Wmma}) computes
    on the operands of several threads at once, so each of them must run it.
    Such an operation may run in a loop only when the loop runs as many trips in
    every thread: when its bound is uniform, the same for every thread of the
    workgroup.

    {!varying} decides which values may differ between a workgroup's threads,
    from what makes them differ: a thread's index, data read where another
    thread's writes may race with the read, and a value carried out of a loop
    whose trips differ. It is sound and incomplete: a value it calls uniform is
    one; a value it calls varying may still be the same everywhere. A matrix
    multiply-accumulate needs only the threads that share it to agree, which are
    those of a warp, but which threads form a warp is decided later, so it is
    held to the workgroup's rule.

    {b Precondition.} No two parameters of a kernel overlap in memory where one
    of them is written: a load of a parameter the kernel never stores into reads
    memory no thread of the launch changes. *)

val varying : Ops.t -> Ops.t -> bool
(** [varying sink] is [v] such that [v u] is [false] only when [u], a value of
    the kernel [sink], is the same for every thread of a workgroup that computes
    it.

    Uniform are constants, a launched kernel's parameters and variables
    ([sink]'s {!Ops.kernel_info.kernel}), workgroup indices
    ({!Ops.Axis_type.Global} ranges and a {!Op.Special} named [gidx]), a serial
    loop's index ({!Ops.Axis_type.Weak}, {!Ops.Axis_type.Loop}) whose bound is
    uniform, arithmetic, selections, casts, indices and vectors of uniform
    values, an {!Op.After} of a uniform value, and a reduction of a value whose
    every varying part is a thread index or a vector lane it reduces over.

    Varying are thread indices and vector lanes ({!Ops.Axis_type.Local},
    {!Ops.Axis_type.Warp} and {!Ops.Axis_type.Upcast} ranges, any other
    {!Op.Special}), the parameters and variables of a body another kernel calls,
    which each thread's call passes, matrix multiply-accumulates, instructions,
    custom code and calls, a loop's index whose bound varies, an unbounded
    loop's whose condition varies, a reduction over such a loop, and a load:
    - at a varying index, or under a varying gate;
    - of a global buffer [sink] writes, or of a volatile parameter, which the
      host may change during the launch;
    - of workgroup memory, unless each write to that buffer is ordered before
      the load by a barrier;
    - of a register buffer that one of its writes leaves varying: a store of a
      varying value, at a varying index or under a varying gate, a store in a
      loop whose trips vary, or a call or custom code that writes it.

    Register buffers are marked varying until none is left to mark, from none
    marked: a buffer whose stores carry only uniform values stays uniform. *)

val check : Ops.t -> unit
(** [check sink] raises [Invalid_argument], naming the kernel (or a called body,
    which has no name), the operation and the loop, if a barrier, a matrix
    multiply-accumulate, or a call whose body holds one, runs in a serial loop
    whose trips may differ between the threads of a workgroup: a loop whose
    index {!varying} calls varying. *)
