(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Staging a tensor core's operands through shared memory.

    A tensor core ({!Op.Wmma}) reads each operand as fragments of a tile: the
    box of addresses its index spans over the workgroup's threads and the
    thread's lanes, at each trip of the loop of its sum. Read from global
    memory, each warp loads its own fragments in a few bytes at a time, and
    warps that share a tile load it again. Staged, every thread of the
    workgroup copies the tile into memory the workgroup shares, [16] bytes at a
    time, once a trip, and the cores read their fragments back from it:

    {v
      global  ──copy, 16 bytes a thread──▶  shared, swizzled  ──▶  fragments
    v}

    Staging moves values and never rounds them: a staged kernel computes the
    bits of the kernel of the same optimisations unstaged.

    {b Two stages.} Where the target copies asynchronously
    ({!Renderer.t.async_copy}), the copy of the next trip's tile overlaps this
    trip's cores, in a second slot:

    {v
      before the loop:  copy chunk 0 into slot 0;  barrier
      trip k:           copy chunk k + 1 into slot (k + 1) mod 2
                        read slot k mod 2, compute;  barrier
    v}

    One barrier a trip retires every read of slot [k mod 2] before trip [k + 1]
    overwrites it, and publishes chunk [k + 1] before it is read. The barriers
    are the ones code generation places between a store to shared memory and
    the reads after it. Without asynchronous copies a single stage copies each
    trip's tile at the start of the trip. *)

val depth : ?used:int -> Renderer.t -> int list -> int
(** [depth ~used ren tiles] is the number of stages a core whose staged
    operands' tiles hold [tiles] bytes takes on [ren], beside [used] bytes
    (default [0]) of shared memory other cores take: [2] where [ren] copies
    asynchronously and two stages fit {!Renderer.t.shared_max}, [1] where one
    does, and [0], staging nothing, where neither does, where [tiles] is empty
    or a tile holds less than one copy of [16] bytes, where [ren] has no local
    threads or no shared memory, or where {!Setting.stage} is off. *)

val operands : Renderer.t -> Ops.t -> Ops.t
(** [operands ren sink] is the kernel [sink], whose optimisations are applied,
    with each tensor core's operands staged ({!depth}) that read global memory
    through an index whose terms that read the workgroup's threads and the
    thread's lanes form a dense box of strided dimensions: each element read
    once, and none of those terms reading another range. A gate on such an
    operand, its index's validity or a selection of [0], copies [0] where it
    fails. An operand whose terms mix the box's ranges with others stays
    unstaged, as do the cores of a kernel that holds other shared memory or
    reduces across threads. *)

val admits : Renderer.t -> Ops.t -> bool
(** [admits ren sink] is [true] iff {!operands} stages every operand of each
    tensor core of [sink] on [ren], tiles permitting ({!depth}): the question
    the heuristic asks before it gives a core the staged tile
    ({!Heuristic.hand_coded_optimizations}). *)

(**/**)

val layout : itemsize:int -> cols:int -> Ops.t -> Ops.t -> Ops.t
(** [layout ~itemsize ~cols col row] is the offset in a staged tile of the
    element at [col] in [row], rows of [cols] elements of [itemsize] bytes:
    each row's [16]-byte chunks permuted by an XOR with bits of [row], so that
    the [16]-byte chunks of eight consecutive rows at one column, and eight
    consecutive chunks of the tile in row order, fall on eight different
    [16]-byte slots of a [128]-byte line of shared memory's banks. *)

(**/**)
