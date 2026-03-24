(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel-to-program linearization.

    Transforms a late {!Tolk_ir.Kernel.t} DAG into a flat
    {!Tolk_ir.Program.t} SSA instruction sequence by splitting
    multi-range ends, computing control-flow dependencies between
    ranges, choosing a priority-respecting topological emission order,
    and lowering scheduled kernel nodes into linear program
    instructions.

    The input kernel must be in late codegen form: exactly one
    {!Tolk_ir.Kernel.view.Sink}, no unlowered nodes ({!Tolk_ir.Kernel.view.Reduce},
    {!Tolk_ir.Kernel.view.Bufferize}, {!Tolk_ir.Kernel.view.Ptrcat},
    {!Tolk_ir.Kernel.view.Cat}, {!Tolk_ir.Kernel.view.Unroll},
    {!Tolk_ir.Kernel.view.Contract}), and gated loads must already
    carry an alternate value. *)

val linearize : Tolk_ir.Kernel.t -> Tolk_ir.Program.t
(** [linearize kernel] lowers [kernel] to a flat {!Tolk_ir.Program.t}.

    Raises [Failure] if [kernel] does not satisfy the late-kernel
    preconditions, does not contain exactly one sink, or has
    inconsistent control-flow dependencies. *)
