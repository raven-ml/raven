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
    [x] further than {!Shape.simplify} does alone. A load in [x] runs whatever
    [cond], so [cond] simplifies the arithmetic around it, which sees only its
    bounds, and the load keeps its own index and gate. *)

val memory_coalescing : Ops.t -> Renderer.t -> Ops.t
(** [memory_coalescing sink r] merges the loads, and the stores, of [sink] that
    access consecutive elements of one buffer, under one gate, with one
    argument and at one depth, into vector loads and stores, if [r] supports
    them ([supports_float4]):
    - where [r] knows its vector registers ({!Renderer.t.registers}), of a
      power of two elements that fills at most one register, if the buffer's
      elements are {!Dtype.Float64}, {!Dtype.Float32}, {!Dtype.Float16},
      {!Dtype.Int32}, {!Dtype.Uint32}, {!Dtype.Int8}, {!Dtype.Uint8} or 8-bit
      floats, but for a store of doubles or bytes whose values are not all
      loads, which merges as below;
    - otherwise of [4] or [2] elements, if they are {!Dtype.Float32},
      {!Dtype.Float16}, {!Dtype.Int32}, {!Dtype.Uint32} or 8-bit floats, or
      {!Dtype.Bfloat16} for a renderer other than the [CPU] device's; [8],
      [4] or [2] for those 16-bit floats where [r] writes vectors of eight
      ({!Renderer.t.half8}) and every access of the group (one operation,
      buffer, base index, gate, argument and depth) copies into local
      memory: a load of global memory whose only use is as the value of a
      store into local memory, or that store.

    The longest length that fits is taken first. A group starts at an element
    whose count from the boundary behind the buffer's first element, a multiple
    of its alignment (its phase and alignment, {!Ops.type-param_arg}), the
    group's length divides, and is no wider than that alignment, so that every
    vector access is aligned to its width. On the ["CPU"] device, whose vectors
    are aligned to their element, every buffer is taken to start on a boundary
    of one vector register, or of 16 bytes where [r] does not know its
    registers. A merged load is a {!Op.Shrink} of the buffer from its first
    access's index, gate included, whose elements each former load reads by
    index; a merged store stores the stack of the former values. An access
    merged with no other is left as it is, and so are register memory, volatile
    parameters, the bodies of calls, and everything under {!Setting.dmc}.

    The depth of an access is the greatest pair, in lexicographic order, of
    the counts of global and of local stores on a chain of sources it waits
    on; for an access of global memory, the count of global stores alone. Two
    accesses at different depths would make a merged store wait on itself.

    Raises [Invalid_argument] if a load or store is gated, is not through an
    {!Op.Index} of one index, or two stores write one element. *)
