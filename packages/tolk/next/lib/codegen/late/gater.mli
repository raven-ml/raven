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

    - a load or a store through the {!Op.Index} of a storage by two indices
      [where g y invalid] and [where g x invalid] becomes the same access
      through the index by [y] and [x], gated by [g];
    - otherwise, a load or a store through an {!Op.Index} or {!Op.Shrink} whose
      first index is [where g i invalid] becomes the same access with [i] in its
      place, gated by [g];
    - a load gated this way reads [0] where its gate fails;
    - [where g l a], with [l] a load gated by [g] or a cast of one, is [l]
      reading [a] where [g] fails, cast to the type of the [where]: [0] if [a]
      is {!Ops.invalid}, [a]'s value if [a] is a constant, the source of [a] if
      [a] is a cast of a node of [l]'s type, and [a] cast to [l]'s type
      otherwise. [where g a l], with [l] gated by [not g], is the same. *)
