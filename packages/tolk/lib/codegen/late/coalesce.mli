(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late index simplification and memory coalescing.

    After devectorization, an access's index is simplified under the condition
    that gates it, and accesses to consecutive elements of one buffer are merged
    into vector accesses the target can make. *)

val indexing_simplify : (unit, Ops.t) Ops.Pattern_matcher.t
(** [indexing_simplify] rewrites an index [where cond x invalid] into a buffer
    to [where cond x' invalid], with [x'] the simplification of [x] given that
    [cond] holds ({!Symbolic.uop_given_valid}), when the condition simplifies
    [x] further than {!Ops.simplify} does alone. A load in [x] runs whatever
    [cond], so [cond] simplifies the arithmetic around it, which sees only its
    bounds, and the load keeps its own index and gate. *)

val merges : Renderer.t -> Dtype.t -> bool
(** [merges r dt] is [true] iff {!memory_coalescing} merges accesses to
    consecutive elements of [dt] on [r]: [r] supports such accesses
    ([supports_float4]) and [dt] is {!Dtype.Float32}, {!Dtype.Float16},
    {!Dtype.Int32}, {!Dtype.Uint32} or an 8-bit float. *)

val memory_coalescing : Ops.t -> Renderer.t -> Ops.t
(** [memory_coalescing sink r] merges the loads, and the stores, of [sink] that
    access consecutive elements of one buffer, under one gate and with one
    argument, into loads and stores of [4] or [2] elements, if the buffer's
    elements merge on [r] ({!merges}); [8], [4] or [2] for {!Dtype.Float16} under
    {!Setting.allow_half8}. A group starts at an element whose count from the
    boundary behind the buffer's first element, a multiple of its alignment (its
    phase and alignment, {!Ops.param_arg}), the group's length divides, and is
    no wider than that alignment, so that every vector access is aligned to its
    width. A merged load is a {!Op.Shrink} of the buffer, whose
    elements each former load reads by index; a merged store stores the stack of
    the former values. Register memory and volatile parameters are left as they
    are, and so is everything under {!Setting.dmc}.

    Raises [Invalid_argument] if a load or store is gated, is not through an
    {!Op.Index} of one index, or two stores write one element. *)
