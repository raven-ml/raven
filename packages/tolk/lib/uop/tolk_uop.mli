(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hash-consed DAG IR for the tolk compiler.

    This module aggregates the uop layer: a single hash-consed node type
    ({!Uop.t}) that flows through every stage of the pipeline — tensor
    graph, kernel AST, and linearized program — together with its
    operation tags, pattern DSL, stage specifications, and symbolic
    simplifier.

    Structurally equal nodes are physically identical. Stage membership is
    not encoded in the type system; it is enforced by {!Spec} validators
    at pass boundaries.

    {1:usage Usage}

    Open this module to bring every submodule into scope:

    {[
      open Tolk_uop

      let x = Uop.const_int 5
      let sum = Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:(Uop.const_int 7)

      let pat =
        Upat.op ~src:[Upat.var "x"; Upat.const_int 0] Ops.Add
    ]}

    {1:modules Modules}

    {2:core Core}

    - {!Uop} — the hash-consed node type, its smart constructors, view
      accessors, traversal, and rewrite engine.
    - {!Ops} — the flat enumeration of operation tags. Declaration order
      is part of the public contract (drives commutative canonicalisation
      and toposort stability).
    - {!Render} — tinygrad-shaped graph listings for debugging and golden
      tests.

    {2:primitives Primitives}

    - {!Dtype} — value and pointer data types, promotion lattice, and
      float/integer truncation.
    - {!Const} — typed compile-time constants.
    - {!Axis_type} — classification of kernel loop axes (global, local,
      reduce, unroll, …) used during scheduling.

    {2:rewrite Patterns and rewriting}

    - {!Upat} — pattern DSL and matcher for writing rewrite rules over
      {!Uop.t}.
    - {!Movement} — movement-op cleanup rewrites (merge and drop redundant
      reshapes, permutes, and stacks).
    - {!Validate} — deterministic memory-access validation used by
      {!Spec}.
    - {!Spec} — stage specifications used to validate that a DAG is
      well-formed at a given pipeline stage.
    - {!Symbolic} — algebraic simplification rules wired into
      {!Uop.simplify}. *)

module Ops = Ops
module Axis_type = Axis_type
module Dtype = Dtype
module Const = Const
module Uop = Uop
module Render = Render
module Upat = Upat
module Movement = Movement
module Validate = Validate
module Spec = Spec
module Symbolic = Symbolic
module Divandmod = Divandmod
