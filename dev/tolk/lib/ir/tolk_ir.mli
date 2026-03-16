(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Intermediate representations for tensor computation.

    [Tolk_ir] provides three IR stages that lower a tensor program from
    high-level operations down to render-ready linear code:
    - {!Tensor} — value-graph of high-level tensor operations.
    - {!Kernel} — codegen-oriented DAG of indexed buffer accesses and loops.
    - {!Program} — linear SSA instruction sequence for backend emission.

    Supporting modules define the shared type vocabulary:
    {!modules:
    Dtype
    Const
    Shape
    Op
    Axis_kind
    Special_dim
    Symbolic} *)

module Dtype = Dtype
(** Scalar, vector, and pointer data types. *)

module Const = Const
(** Typed compile-time constants. *)

module Shape = Shape
(** Tensor shapes with static and symbolic dimensions. *)

module Axis_kind = Axis_kind
(** Kernel axis kinds (thread, local, reduce, etc.). *)

module Special_dim = Special_dim
(** Backend-provided hardware execution indices. *)

module Op = Op
(** Arithmetic and logical operations grouped by arity. *)

module Kernel = Kernel
(** Codegen-oriented DAG IR (memory-level graph stage). *)

module Symbolic = Symbolic
(** Symbolic simplification rules for {!Kernel} IR. *)

module Tensor = Tensor
(** High-level tensor graph IR (value-graph stage). *)

module Program = Program
(** Render-ready linear SSA IR (backend emission stage). *)
