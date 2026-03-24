(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedule pipeline orchestrator.

    Transforms a tensor-level SINK into a graph with CALL nodes wrapping
    {!Tolk_ir.Kernel.t} ASTs.

    The pipeline:
    + multi_pm — multi-device rewriting
    + earliest_rewrites — canonicalization, call resolution, allreduce
    + run_rangeify — core range analysis
    + pm_apply_rangeify — apply analysis, create BUFFERIZE/INDEX/REDUCE
    + post-rangeify optimization — buffer folding, buffer removal
    + pm_add_buffers — BUFFERIZE → STORE + BUFFER
    + split_kernels — create CALL(Ast of Kernel.t)
    + WAR dependency fixup *)

val get_kernel_graph : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.t
(** [get_kernel_graph program] transforms a tensor-level graph into a graph
    with CALL nodes wrapping {!Tolk_ir.Kernel.t} ASTs. This is the main entry
    point for the schedule. *)
