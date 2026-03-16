(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late expansion of structured vector markers.

    This pass owns late vector-shape realization between post-range
    optimization and bufferization. It normalizes scheduler-introduced {!Tolk_ir.Kernel.view.Unroll}/
    {!Tolk_ir.Kernel.view.Contract} structure, rewrite grouped reductions
    through {!Tolk_ir.Kernel.view.Bufferize}, and consume [Upcast]/[Unroll]
    ranges on {!Tolk_ir.Kernel.view.Reduce},
    {!Tolk_ir.Kernel.view.Store}, and {!Tolk_ir.Kernel.view.End} before
    later lowering.

    The result is still late Kernel IR, not render-ready IR. Later stages
    may still see transient vectorized reduce sources or vectorized pointer
    indexing forms that are consumed by {!Devectorizer}. *)

val expand : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [expand kernel] simplifies and lowers reachable [Contract] and [Unroll]
    markers in [kernel].

    The result preserves kernel semantics while replacing structured expansion
    markers with ordinary late-kernel vector structure and keeping range fields
    scalar-only. *)
