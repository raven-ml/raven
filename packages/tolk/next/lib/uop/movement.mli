(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Cleaning up movements.

    Rewrites that shorten chains of movements and indexing without changing the
    values they denote. *)

val mop_cleanup : (unit, Ops.t) Ops.Pattern_matcher.t
(** [mop_cleanup] merges and removes movements and indexing:

    - a shrink of a shrink is one shrink of the inner one's source, starting at
      the sum of their starts, of the outer one's sizes;
    - a reshape of a reshape is one reshape of the inner one's source, and a
      reshape to its source's shape is its source;
    - a permutation of a permutation is their composition, and the identity
      permutation is its source;
    - a stack of the elements [0], [1], ... of a node [x], in order and of [x]'s
      shape, is [x];
    - indexing a stack by a constant [i] and further indices is its [i]th source
      indexed by those indices;
    - indexing an index, when every index of both is a scalar, is one index by
      the inner indices followed by the outer ones;
    - indexing the storage [b] indexed by [i] with as many indices as [i] has
      axes is indexing [b] by the element of [i] they select. *)
