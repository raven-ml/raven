(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen pipeline orchestrator.

    Chains preprocessing, optimization strategy dispatch, and lowering.
    The optimization strategy (manual opts / beam search / heuristic) is
    dispatched via {!Postrange.apply_opts}. Lowering is delegated to
    {!Lowering.lower}. *)

val full_rewrite_to_sink :
  ?optimize:bool ->
  ?device:Device.t ->
  Renderer.t ->
  Tolk_ir.Kernel.t ->
  Tolk_ir.Kernel.t
(** [full_rewrite_to_sink ?optimize ?device renderer sink] runs all codegen
    passes from raw kernel AST to linearizer-ready form.

    When [optimize] is [true] (default), range simplification and
    optimization strategy dispatch are applied. When [false], only the
    lowering passes run (via {!Lowering.lower}).

    When [device] is provided and [BEAM >= 1], uses beam search for
    kernel optimization. Otherwise falls back to hand-coded heuristics
    (unless [NOOPT] is set). *)

val get_program :
  ?optimize:bool ->
  ?device:Device.t ->
  Device.t ->
  Renderer.t ->
  Tolk_ir.Kernel.t ->
  Device.Program.t
(** [get_program ?optimize ?device dev renderer sink] runs the full codegen
    pipeline from kernel AST to compiled program. Chains
    {!full_rewrite_to_sink}, linearization, rendering, and compilation.

    Debug output is controlled by the [DEBUG] environment variable:
    - [>= 3]: prints applied optimization options;
    - [>= 4]: prints rendered source code;
    - [>= 5]: prints the AST before codegen (in {!full_rewrite_to_sink});
    - [>= 6]: prints per-stage UOp dumps and the linearized program. *)
