(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen entry point — optimization dispatch, lowering, and compilation.

    {!to_program} is the main entry point: it optimizes a kernel AST
    (load collapse, range splitting/simplification, beam search or
    hand-coded optimizations), lowers it (expansion, devectorization,
    GPU dims, decompositions), linearizes, renders, and compiles it into an
    on-graph {!Tolk_uop.Ops.Program} node carrying the rendered source and
    compiled binary.

    This boundary is intentionally sink-only: it accepts a kernel {!Ops.Sink}
    and produces the compiled program. The engine caches the compiled programs
    it dispatches, and {!Compiler.compile_cached} caches the render/compile
    results.

    Tolk keeps post-optimization lowering in {!Codegen_lower}. The split is
    smaller than tinygrad's single [codegen/__init__.py] file, but it avoids a
    cycle with beam search: candidate schedules need lowering and
    linearization without depending on this optimization entry point. *)

val full_rewrite_to_sink :
  ?optimize:bool ->
  ?beam_device:Device.t ->
  Renderer.t ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [full_rewrite_to_sink ?optimize ?beam_device ren sink] optimizes and
    lowers kernel [sink] to a linearize-ready form.

    When [optimize] is [true] (default) and [sink] is untagged, runs
    load collapse, range splitting, symbolic simplification, range
    tightening, and dispatches to beam search or hand-coded optimizations.
    This block requires [sink] to carry {!Tolk_uop.Uop.kernel_info}.
    Tagged sinks skip this optimization block. When [false], skips
    directly to lowering. Post-optimization lowering parity lives in
    {!Codegen_lower}.

    [beam_device] supplies the runtime for beam search when the kernel's
    beam setting is positive. {!Realize.compile_linear} resolves the [BEAM]
    context before calling codegen. [SPEC=1] output program validation is handled by
    {!Codegen_lower}; this module does not run an input spec check because
    Tolk has no exact tinygrad [spec_tensor] equivalent for this sink stage. *)

val to_program :
  ?optimize:bool ->
  ?beam_device:Device.t ->
  Renderer.t ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [to_program ?optimize ?beam_device ren input] completes a kernel into an
    on-graph {!Tolk_uop.Ops.Program} node [PROGRAM(SINK, LINEAR, SOURCE, BINARY)].

    A [SINK] input must carry {!Tolk_uop.Uop.kernel_info}. It is optimized and
    lowered through {!full_rewrite_to_sink}, and its argument and launch
    metadata are captured before linearization. [optimize] defaults to [true];
    tagged sinks skip optimization. [beam_device] supplies the runtime when
    an optimized sink requests beam search.

    A [PROGRAM] input is already prepared: supplied stages are retained without
    repeating optimization or lowering. Missing [LINEAR], [SOURCE], and [BINARY]
    stages are produced in order. Existing program metadata is preserved; absent
    metadata is derived from its sink and the renderer target. Missing estimates
    are computed when rendering a [LINEAR] stage. A complete program is returned
    unchanged and does not require a compiler.

    Raises [Invalid_argument] for malformed stages, when compilation is needed
    but the renderer has no compiler, or when an optimized sink requests beam
    search without [beam_device]. *)
