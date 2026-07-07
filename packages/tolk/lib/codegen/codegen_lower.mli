(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen lowering after optimization.

    {!lower} runs expansion, devectorization, GPU dimension mapping, image
    lowering, index dtype concretization, decompositions, and renderer-specific
    rewrites.

    This module is the post-optimization half of tinygrad's
    [codegen/__init__.py] pipeline. It intentionally has no dependency on
    {!Search}, {!Postrange}, or {!Heuristic}, so beam search can lower and
    linearize candidate schedules without depending on {!Codegen}. *)

val lower : Renderer.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [lower renderer sink] is [sink] after the non-optimization codegen
    passes. The result is ready for {!Linearizer.linearize}. *)
