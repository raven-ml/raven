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

    - {b tensor cores}, when the setting {!Helpers.use_tc} is positive and [k]
      reduces over one axis, or {!Helpers.tc_opt} admits more: the first of the
      three candidate axes on which a tensor core applies ({!Opt.Tc}), then
      upcasts and a local split of its [N] and [M] axes, by the first of 5, 4, 3
      and 2 (4 and 2 for the local) that divides them. With
      {!Helpers.tc_min_globals}, [M] is upcast only while enough global threads
      remain;
    - {b matrix-vector}, for a renderer with local indices and shared memory and
      a sum of products of two accesses where the first reduce axis indexes the
      first and its axes are all the second's: the first reduce axis split into
      [MV_THREADS_PER_ROW] local threads (default 8), and the first global axis
      whose size [MV_BLOCKSIZE * MV_ROWS_PER_THREAD] divides split into
      [MV_BLOCKSIZE] local threads (default 4), then [MV_ROWS_PER_THREAD] upcast
      lanes (default 4). [MV=0] turns it off;
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
