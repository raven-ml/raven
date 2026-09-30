(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Operations as the operations a target has.

    Code generation lowers the operations a target lacks, and some it has, to
    cheaper ones: floor divisions to truncating ones, divisions by constants to
    multiplications and shifts, the {!Op.Threefry} hash to integer arithmetic,
    and comparisons, negations and fused multiply-adds to the forms renderers
    expect. A target states its operations as a set of {!Op.t}, the keys of its
    renderer's [code_for_op]. *)

(** {1:idiv Integer division} *)

val fast_idiv : Renderer.t -> Ops.t -> Z.t -> Ops.t option
(** [fast_idiv r x d] is [x] divided by the positive constant [d], rounding
    towards zero, as a multiplication and a right shift: [(x * m) >> s], for
    [x]'s values from [0] to its upper bound. It shifts the powers of two out of
    [d] first when the product would overflow [x]'s type, and otherwise computes
    in the next wider integer type if [r] supports it
    ({!Renderer.supported_dtypes}). It is [const_like x 0] if [x]'s upper bound
    is below [d], and [None] if [d] is not positive, [x] can be negative, or no
    such computation fits. *)

(** {1:threefry Threefry} *)

val threefry2x32 : Ops.t -> Ops.t -> Ops.t
(** [threefry2x32 x key] is the Threefry-2x32 hash of the {!Dtype.Uint64}
    counter [x] under [key], with 20 rounds, as 32-bit additions, rotations and
    exclusive ors: the two 32-bit words of the result, high word second, in a
    {!Dtype.Uint64}. *)

(** {1:patterns Patterns} *)

val simplifying_patterns : Op.Set.t -> (unit, Ops.t) Ops.Pattern_matcher.t
(** [simplifying_patterns ops] lowers, for a target with the operations [ops]:
    - a floor division ({!Op.Floordiv}) of an integer by a power of two to a
      right shift if [ops] has {!Op.Shr}, and any other to a truncating division
      ({!Op.Cdiv}), corrected where the operands' signs can differ;
    - a floor remainder ({!Op.Floormod}) of an integer by a power of two to a
      mask if [ops] has {!Op.And}, and any other to a truncating remainder
      ({!Op.Cmod}), corrected where the operands' signs can differ;
    - {!Op.Threefry} to {!threefry2x32} if [ops] lacks it. *)

val late_patterns :
  disable_fast_idiv:bool ->
  Op.Set.t ->
  (Renderer.t, Ops.t) Ops.Pattern_matcher.t
(** [late_patterns ~disable_fast_idiv ops] rewrites, for a target with the
    operations [ops] and rendered by the context:
    - {!Op.Max} to a comparison and a selection if [ops] lacks it and has
      {!Op.Cmplt};
    - the conjunction of two negated booleans to the negated disjunction if
      [ops] has {!Op.Or};
    - a multiplication of an integer by a power of two to a left shift if [ops]
      has {!Op.Shl};
    - if [ops] has {!Op.Shr}, a truncating division of an integer by a power of
      two to a right shift, rounding negative dividends up first; and, unless
      [disable_fast_idiv], a truncating division or remainder by any other
      constant with {!fast_idiv} where it applies;
    - multiplications by [-1] to {!Op.Neg}, and additions of a negation to
      {!Op.Sub}, if [ops] has them;
    - if [ops] has {!Op.Cmplt}, negated comparisons of signed integers with
      constants to comparisons, [x * -1 < y * c] to [y * -c < x], [x * -1 < c]
      to [-c < x], and [c1 < x && x < c2] to [x == c1 + 1] when it is the one
      integer between;
    - negated {!Op.Cmpne} to {!Op.Cmpeq} if [ops] has it;
    - [a * b + c], and [(x << n) + c], to {!Op.Mulacc} if [ops] has it;
    - if [ops] has {!Op.Fdiv}, reciprocals to a division of [1.0], and a float's
      multiplication by such a division to a division. *)
