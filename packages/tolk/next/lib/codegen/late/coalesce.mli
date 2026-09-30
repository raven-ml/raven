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
    [cond] holds ({!Symbolic.uop_given_valid}), when that changes [x]. *)

val memory_coalescing : Ops.t -> Renderer.t -> Ops.t
(** [memory_coalescing sink r] merges the loads, and the stores, of [sink] that
    access consecutive elements of one buffer, under one gate and with one
    argument, into loads and stores of [4] or [2] elements, if the buffer's
    elements are {!Dtype.Float32}, {!Dtype.Float16}, {!Dtype.Int32},
    {!Dtype.Uint32} or 8-bit floats and [r] supports such accesses
    ([supports_float4]); [8], [4] or [2] for {!Dtype.Float16} when the variable
    [ALLOW_HALF8] is nonzero. A group starts at an offset that the group's
    length divides. A merged load is a {!Op.Shrink} of the buffer, whose
    elements each former load reads by index; a merged store stores the stack of
    the former values. Register memory and volatile parameters are left as they
    are, and so is everything when the variable [DMC] is nonzero.

    Raises [Invalid_argument] if a load or store is gated, is not through an
    {!Op.Index}, or two stores write one element. *)
