(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Moving gates from indices to memory accesses.

    An index [where g i invalid] ({!Ops.invalid}) addresses [i] where [g] holds
    and nothing elsewhere. No target renders {!Ops.invalid}, so before rendering
    the condition moves onto the access: a load reads only where its gate holds,
    and gives an alternate value elsewhere; a store writes only where its gate
    holds. *)

val pm_move_gates_from_index : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_move_gates_from_index] moves gates onto loads and stores, with [g] a
    boolean node:

    - a load or a store through an {!Op.Index} or {!Op.Shrink} whose first
      index is [where g i invalid] becomes the same access with each of its
      sources [where g j invalid] replaced by [j], gated by [g];
    - a load gated this way reads [0] where its gate fails;
    - [where g l a], with [l] a load gated by [g], is [l] reading [a], cast to
      [l]'s type, where [g] fails. [where g a l], with [l] gated by [not g], is
      the same. *)
