(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen entry point — optimization dispatch and lowering.

    {!get_program} is the main entry point: it optimizes a kernel AST
    (load collapse, range splitting/simplification, beam search or
    hand-coded optimizations), lowers it (expansion, devectorization,
    GPU dims, decompositions), linearizes, renders, and compiles to a
    {!Program_spec.t}. *)

val full_rewrite_to_sink :
  ?optimize:bool ->
  ?device:Device.t ->
  Renderer.t ->
  Tolk_ir.Kernel.t ->
  Tolk_ir.Kernel.t
(** [full_rewrite_to_sink ?optimize ?device ren sink] optimizes and
    lowers kernel [sink] to a linearizer-ready form.

    When [optimize] is [true] (default), runs load collapse, range
    splitting, symbolic simplification, range tightening, and dispatches
    to beam search or hand-coded optimizations. When [false], skips
    directly to lowering.

    [device] enables beam search when [BEAM >= 1] is set. *)

val get_program :
  ?optimize:bool ->
  ?device:Device.t ->
  Device.t ->
  Renderer.t ->
  Tolk_ir.Kernel.t ->
  Program_spec.t
(** [get_program ?optimize ?device dev ren sink] compiles kernel [sink]
    to a {!Program_spec.t}. Calls {!full_rewrite_to_sink} then
    linearizes, renders, and compiles. *)
