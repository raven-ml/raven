(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen lowering — all passes after optimization, up to linearization.

    {!lower} runs expansion, devectorization, GPU dimension mapping, image
    lowering, index dtype concretization, decompositions, and renderer-specific
    rewrites.

    This module has no dependency on Search, Postrange, or Heuristic,
    so beam search can safely call {!lower_and_linearize} without cycles. *)

val lower : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [lower renderer sink] runs all non-optimization codegen passes on an
    optimized kernel AST. Returns a linearizer-ready {!Tolk_ir.Kernel.t}. *)
