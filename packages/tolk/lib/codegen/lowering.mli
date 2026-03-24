(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Non-optimization codegen passes.

    {!lower} runs the lowering pipeline: postopt symbolic, expander,
    devectorize, gpudims, index dtype lowering, and final rewrite.

    {!compile} chains {!lower} with linearization, rendering, and device
    compilation.

    This module is safe for {!Search} to depend on — it has no dependency on
    [Postrange], [Heuristic], [Search], or [Pipeline]. *)

val lower : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [lower renderer sink] runs all non-optimization codegen passes. Returns a
    linearizer-ready {!Tolk_ir.Kernel.t}. *)

val lower_and_linearize :
  Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Program.t
(** [lower_and_linearize renderer sink] is {!lower} followed by linearization.
    Returns the flat SSA program before rendering and device compilation.
    Useful when you need to inspect the program (e.g. count uops) before
    committing to compilation. *)

val compile :
  Device.t -> Renderer.t -> Tolk_ir.Kernel.t -> Device.Program.t
(** [compile device renderer sink] is {!lower} followed by linearize, render,
    and device compilation. *)
