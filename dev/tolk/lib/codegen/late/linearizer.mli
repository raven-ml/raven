(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel-to-program linearization.

    Transforms late {!Tolk_ir.Kernel.t} into flat {!Tolk_ir.Program.t}.
    The pass keeps a single public boundary while internally:
    - splitting multi-range {!Tolk_ir.Kernel.view.End} nodes into
      nested single-range ends;
    - computing the control-flow dependencies between ranges;
    - choosing a priority-respecting topological emission order;
    - lowering the scheduled kernel into linear {!Tolk_ir.Program}
      instructions.

    This is the final pure DAG-to-SSA lowering stage before
    post-linearize cleanup and rendering. It performs split-ends,
    control-flow insertion, and linearization.

    {1:contract Contract}

    The input kernel must be in the late codegen form expected by the
    renderer: exactly one {!Tolk_ir.Kernel.view.Sink} must be present;
    scheduling and control-flow nodes such as
    {!Tolk_ir.Kernel.view.Range}, {!Tolk_ir.Kernel.view.End},
    {!Tolk_ir.Kernel.view.Special}, {!Tolk_ir.Kernel.view.Group}, and
    {!Tolk_ir.Kernel.view.After} may remain; higher-level nodes that
    should have been lowered earlier, such as
    {!Tolk_ir.Kernel.view.Reduce}, {!Tolk_ir.Kernel.view.Bufferize},
    {!Tolk_ir.Kernel.view.Ptrcat}, {!Tolk_ir.Kernel.view.Cat},
    {!Tolk_ir.Kernel.view.Unroll}, and
    {!Tolk_ir.Kernel.view.Contract}, must not remain.

    Gated {!Tolk_ir.Kernel.view.Load} instructions must already carry
    an alternate value.

    The result is a flat SSA program with explicit loop structure:
    [Range]/[End_range] are balanced, [After] nodes remain explicit
    scheduling aliases, gated stores remain ordinary stores, and the
    program is suitable for later cleanup, {!Tolk_ir.Program.validate},
    and rendering. *)

val linearize : Tolk_ir.Kernel.t -> Tolk_ir.Program.t
(** [linearize kernel] lowers [kernel] to a flat {!Tolk_ir.Program.t}.

    Raises [Failure] if [kernel] does not satisfy the late-kernel preconditions,
    if it does not contain exactly one sink, or if the control-flow dependencies
    implied by nested ranges are inconsistent. *)
