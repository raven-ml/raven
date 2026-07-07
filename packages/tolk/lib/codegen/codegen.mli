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
    Tagged sinks skip this optimization block. When [false], skips
    directly to lowering. Post-optimization lowering parity lives in
    {!Codegen_lower}.

    [beam_device] enables beam search when [BEAM >= 1] is set or the
    kernel's beam setting is positive. Environment settings are read at the
    point of use. [SPEC=1] output program validation is handled by
    {!Codegen_lower}; this module does not run an input spec check because
    Tolk has no exact tinygrad [spec_tensor] equivalent for this sink stage. *)

val to_program :
  ?optimize:bool ->
  ?beam_device:Device.t ->
  Device.t ->
  Renderer.t ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [to_program ?optimize ?beam_device dev ren sink] compiles kernel [sink]
    into an on-graph {!Tolk_uop.Ops.Program} node
    [PROGRAM(SINK, LINEAR, SOURCE, BINARY)].

    It runs {!full_rewrite_to_sink}, derives program metadata with
    {!Tolk_uop.Uop.program_info_from_sink}, linearizes, renders the kernel
    source, and compiles it to a binary. The rendered source and compiled
    binary are attached as {!Tolk_uop.Ops.Source} and {!Tolk_uop.Ops.Binary}
    children; launch dimensions, scalar variables, and buffer slots are
    carried on the node's {!Tolk_uop.Uop.program_info} arg.

    [sink] must carry {!Tolk_uop.Uop.kernel_info}. When [optimize] is [true]
    (default) and [sink] is untagged, the optimization block in
    {!full_rewrite_to_sink} runs; tagged sinks skip it. [beam_device] overrides
    the runtime device used for beam-search buffers; when omitted, [dev] is
    used.

    Raises [Invalid_argument] if the device renderer has no compiler. *)
