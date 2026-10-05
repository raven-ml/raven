(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hand-coded kernel optimisations.

    Rules of thumb that pick a kernel's optimisations ({!Opt.t}) from its shape
    and accesses, without measuring it. *)

val hand_coded_optimizations : Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_optimizations k] is a copy of [k] with optimisations applied;
    [k] itself is left as it is. The first of these that applies decides:

    - {b tensor cores}, when the setting {!Setting.use_tc} is positive and [k]
      reduces over one axis, or {!Setting.tc_opt} admits more: the first of the
      three candidate axes on which a tensor core applies ({!Opt.Tc}), then
      upcasts and a local split of its [N] and [M] axes, by the first of 5, 4,
      3 and 2 (4 and 2 for the local) that divides them.
      With {!Setting.tc_min_globals}, [M] is upcast only while enough global
      threads remain;
    - {b matrix-vector}, for a renderer with local indices and shared memory
      and a sum over the first reduce axis of products of a matrix and a
      vector: computations of accesses with no reduce, the vector's axes some of
      the matrix's, the matrix's more, and the first reduce axis indexing one of
      the vector's accesses alone or at a constant stride. The axis along which
      an access of the matrix has unit stride decides the layout:
      {ul
       {- the first reduce axis (rows along the reduce): it splits into the
          largest of 32, 16, 8, 4 and 2 local threads that divides it, the
          first global axis that 4 divides into 4 local threads, and what is
          left of the reduce axis is unrolled by its largest divisor up to 8;}
       {- a global axis (columns along an output), when 64 divides it and the
          upcastable axes hold at most 32768 elements: it splits into 32
          local threads, then 2 upcast lanes, and the first reduce axis into
          the least power of two of local threads that makes 32768 threads,
          at most 32, or the largest that divides it.}}
      {!Setting.mv} turns it on;
    - {b grouping}: when the upcastable axes hold at most 2048 elements, the
      first of the first three reduce axes that splits by 16 into local threads
      from the top; nothing more is applied then.

    Otherwise, in order:

    - masked axes, those a selection's condition runs in, of size at most 7, are
      upcast whole, while their product stays at most 49;
    - while the upcastable axes hold at least 1024 elements and fewer than 32
      lanes are upcast, one more axis is upcast by 3 or 4, chosen among axes
      some access does not index while indexing every upcast axis, least indexed
      and of least stride first;
    - the last unrollable axis is unrolled whole if of size at most 32 (and a
      second one if both are at most 3), or else by 4, when at most 4 lanes are
      upcast or none are unrolled, and fewer than 64;
    - if nothing is upcast, the last upcastable axis is upcast by 4 when 4
      divides it;
    - for a renderer with local indices, global and weak axes of a constant size
      become local: up to three axes, those some access does not index first,
      split by the first of 32 (axis [0] only), 16, 8, 4, 3 and 2 that divides
      them while the workgroup stays within 128 threads. *)
