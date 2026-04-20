(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Memory planning.

    Reduces peak memory by reusing buffers whose lifetimes don't overlap.
    Each schedule step lists the buffers it touches; the planner computes
    live ranges per base buffer, then either suballocates from a per-lane
    {!Tlsf} arena (when the device supports offset views) or recycles freed
    buffers from a pool keyed by (device, dtype, spec, nbytes).

    Copy and compute buffers are separated into distinct lanes so that
    freeing a copy buffer never forces a dependency between the copy and
    compute queues.

    Disabled when the [NO_MEMORY_PLANNER] environment variable is non-zero. *)

(** {1:planner Core planner} *)

val internal_memory_planner :
  ?copies:(Device.Buffer.t * Device.Buffer.t) list ->
  ?ignore_checks:bool ->
  ?debug_prefix:string ->
  Device.Buffer.t list list ->
  (int, Device.Buffer.t) Hashtbl.t
(** [internal_memory_planner ?copies ?ignore_checks ?debug_prefix buffers]
    is a buffer replacement table that minimises peak memory.

    [buffers] is a list of per-step buffer lists (one inner list per
    schedule item). [copies] is the [(dst, src)] pairs from copy
    operations; defaults to [\[\]]. Buffers involved in copies are
    placed in a separate lane and held longer to avoid cross-queue
    dependencies.

    The returned hashtable maps {!Device.Buffer.id} to a replacement
    {!Device.Buffer.t}. Buffers absent from the table keep their
    original allocation.

    When [ignore_checks] is [false] (the default), buffers that are
    already allocated or have a positive reference count are skipped.

    When [DEBUG >= 1], prints memory reduction statistics to stdout,
    prefixed by [debug_prefix] (defaults to [""]). *)

(** {1:schedule Schedule integration} *)

val memory_planner :
  Realize.Exec_item.t list -> Realize.Exec_item.t list
(** [memory_planner schedule] applies {!internal_memory_planner} to
    [schedule] and returns a new schedule with buffers replaced.

    Copy operations are detected by matching on
    {!Tolk_ir.Tensor.Copy} nodes and their first two buffer
    arguments. Each exec item is reconstructed with replaced buffers;
    the AST and variable bindings are preserved. *)
