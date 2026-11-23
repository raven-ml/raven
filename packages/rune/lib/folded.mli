(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Elementwise arithmetic on tolk nodes that folds constants as it builds.

    Each function builds the node {!Tolk.Ops}' function of the same name builds.
    Where every operand is a {e constant}, whose elements all hold one value,
    and the node is none of those left to code generation below, it is instead
    its value as a literal of the node's type, expanded to its shape. A constant
    is a literal (a {!Tolk.Ops.is_literal} node) or a reshape, expansion,
    permutation or shrink of a literal or of a stack of literals that leaves one
    value: a part of a complex constant is the shrink of its stacked parts to
    one lane. A lowering of an operation on constants then links its value's
    literal where it would link the operation's whole expansion: a [lgamma] of a
    constant is a few nodes, not a thousand.

    {b Values.} The value is the one tolk gives the node where it folds
    constants: the operation built over the operands' scalar literals and
    rewritten by {!Tolk.Shape.symbolic_simple}, the rules by which its code
    generation folds them. So folding here changes no value a compiled program
    computes. An integer division by [0] is [0] and its remainder the dividend,
    as tolk computes them.

    {b Operations left to code generation.} Every node folds but these, which
    are built as they are:
    - the transcendental operations ([Exp2], [Log2], [Sin], [Pow]), which tolk's
      rules compute correctly rounded where a kernel computes them by their
      decomposition, and [Threefry];
    - a [Mulacc] of floats narrower than [float32], which a target that emulates
      them may round twice;
    - a shift by a count outside the width of a committed integer type, which
      tolk's rules do not compute, and a conversion of a float to an integer
      type that does not hold its truncation, NaN and infinities included, which
      C leaves undefined and tolk's code generation decides;
    - an operation that builds a node of neither a literal nor arithmetic, such
      as a movement;
    - an operation with an [`Invalid] operand. *)

include Tolk.Ops.Elementwise with type t = Tolk.Ops.t

val bitcast : t -> Tolk.Dtype.t -> t
(** [bitcast x dt] is {!Tolk.Ops.bitcast}[ x dt], folded as above. *)
