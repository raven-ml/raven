(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hand-coded kernel optimisations.

    Rules of thumb that pick a kernel's optimisations ({!Opt.t}) from its shape
    and accesses, without measuring it. A reduce axis is one a reduction closes;
    the upcastable axes are the global, local and weak axes of a known size
    that no reduction closes, and the unrollable axes those that one does. An
    unroll is an upcast of a reduce axis. *)

val hand_coded_optimizations : Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_optimizations k] is a copy of [k] with optimisations applied;
    [k] itself is left as it is. The first of these that applies decides:

    - {b streaming}, below, when the reuse of the streamed operand is at most
      32;
    - {b tensor cores}, when the setting {!Setting.use_tc} is positive and [k]
      reduces over one axis, or {!Setting.tc_opt} admits more: the first of the
      three candidate axes on which a tensor core applies ({!Opt.Tc}), then
      upcasts of its [M] and [N] axes and a local split of [N], by the first
      of 5, 4, 3 and 2 (4 and 2 for the local) that divides them. Each split
      applies only when it leaves at least 100 workgroups, the fewest that
      keep every compute unit of a large GPU busy, or when the count is not
      known: a split gives each workgroup more of the outputs and the kernel
      fewer workgroups, which leaves compute units idle once they are fewer
      than the units.
      With {!Setting.tc_min_globals}, [M] is upcast and [N] split locally
      whatever the workgroups, and [N] is upcast only while that many
      workgroups remain;
    - {b streaming} at any reuse, when the streamed operand's main access reads
      elements of fewer than 4 bytes;
    - {b grouping}: when the upcastable axes hold at most 2048 elements, the
      first of the first three reduce axes that splits by 16 into local threads
      from the top; nothing more is applied then.

    {b Streaming} applies on a renderer with local indices and shared memory,
    when {!Setting.mv} is on, to [k]'s first reduce that sums a product of two
    computations of accesses, each of whose own reduces runs inside none of
    the summed axes. Each element of an operand
    feeds as many outputs as its reuse: the product of the sizes of the global
    axes it does not read, times [d] for each it reads only as its quotient by
    [d]. The operand of least reuse is streamed, or of two of equal reuse the
    one laid out along a contracted axis. Its main access is the one with an
    axis of unit stride, a term of its index, whose terms of an axis the
    reduce contracts span the most bytes. When that axis [u] is contracted:
    {ul
     {- with reuse, the axes it comes from are upcast, each by its share of the
        largest divisor of the reuse up to 16;}
     {- [u] splits into the largest of 32, 16, 8, 4 and 2 local threads that
        divides it, and when they take [u] whole and are fewer than 32, the
        contracted axis of the next stride in the main access into as many
        more as make 32 or divide it;}
     {- the first global axis that the streamed operand reads, the other does
        not, and 4 divides, or else one both read, splits by 4: into local
        threads without reuse, into upcast lanes with it;}
     {- what the threads leave of [u], or else the axis of the next stride, is
        unrolled by its largest divisor up to 8, or 4 with reuse.}}
    When that axis is a global axis, after the upcast of the reuse, it splits
    into 32 local threads and 2 upcast lanes, and the first reduce axis into
    the least power of two of local threads that makes 32768 threads, at most
    32, or the largest that divides it; with few outputs fewer threads take
    the global axis, down to 8, and more the reduce. It does not apply when
    more than 32768 outputs are left after the upcast.

    Otherwise, in order:

    - masked axes, those a selection's condition runs in, of size at most 7, are
      upcast whole, while their product stays at most 49;
    - while the upcastable axes hold at least 1024 elements and fewer than 32
      lanes are upcast, one more axis is upcast by 3 or 4, chosen among axes
      some access does not index while indexing every upcast axis, least indexed
      and of least stride first;
    - the last unrollable axis is unrolled whole if of size at most 32 (and a
      second one if both are at most 3), or else by 4, when at most 4 lanes are
      upcast or no reduce axis is, and fewer than 64;
    - if nothing is upcast, the last upcastable axis is upcast by 4 when 4
      divides it;
    - for a renderer with local indices, global and weak axes of a constant size
      become local: up to three axes, those some access does not index first,
      split by the first of 32 (axis [0] only), 16, 8, 4, 3 and 2 that divides
      them while the workgroup stays within 128 threads.

    A kernel of the host without a reduce, whose lanes the host computes as
    vectors, takes in place of the upcasts after its masked axes one upcast of
    its last output axis that 2 divides, by the largest power of two, at most
    16, that keeps its lanes within 64 bytes of the widest type it reads,
    writes or computes, and again while they have room. It takes none when its
    stored values hold 5 operations or more for each operation of their
    longest chain, or when it computes a type the target emulates. *)
