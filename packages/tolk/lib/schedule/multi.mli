(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Multi-device sharding transformations.

    Rewrites operations on {!Tolk_ir.Tensor.Multi}-wrapped (sharded)
    tensors into per-shard operations. Each rule strips the MULTI
    wrapper, applies the operation to the inner per-shard tensor, and
    re-wraps the result.

    Covers ALU, movement, reduction, copy, allreduce, store, and
    passthrough ops. CALL bodies are resolved recursively. *)

val multi_pm :
  shapes:(Tolk_ir.Tensor.t -> int list option) ->
  devices:(Tolk_ir.Tensor.t -> Tolk_ir.Tensor.device option) ->
  Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.t option
(** [multi_pm ~shapes ~devices node] rewrites [node] if it involves
    multi-device sharding.

    [shapes] maps a tensor node to its concrete shape, if known.
    [devices] maps a tensor node to its device placement, if known.

    Returns [Some node'] when the node is rewritten, [None] when no
    rule applies. Intended as the rewrite function for
    {!Tolk_ir.Tensor.graph_rewrite}. *)
