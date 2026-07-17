(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compile-pipeline benchmark workloads.

    Each workload is a tensor-level {!Tolk_uop.Uop.t} [Sink] — the input to
    the rangeify stage. These builders are the single source of truth for the
    benchmarked graph shapes; every bench executable links this module so the
    lab gate and any later comparison time the identical graph. *)

type t
(** A named workload graph. *)

val name : t -> string
(** [name w] is [w]'s short identifier, used as the benchmark-case prefix. *)

val size : t -> string
(** [size w] is [w]'s shape descriptor (e.g. ["256x256"]), used to label a
    comparison row. *)

val sink : t -> Tolk_uop.Uop.t
(** [sink w] is the tensor-level [Sink] for [w], ready for
    {!Tolk.Rangeify.get_kernel_graph}. *)

val kernels : Tolk_uop.Uop.t -> Tolk_uop.Uop.t list
(** [kernels kg] is the inline kernel-body ASTs carried by the [Call] nodes of
    a kernel graph [kg] — the output of {!Tolk.Rangeify.get_kernel_graph}. Each
    body is a per-kernel input to the codegen stage. *)

val elementwise : t
(** [a + b * c] over 256x256 float32: one fused elementwise kernel. *)

val reduce : t
(** [sum] over the last axis of a 512x512 float32 tensor. *)

val matmul_small : t
(** [a @ b] for a 128x128 by 128x128 float32 matrix product, expressed as a
    broadcast-multiply reduced over the contraction axis. *)

val all : t list
(** [[ elementwise; reduce; matmul_small ]]. *)
