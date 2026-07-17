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

val lorenz : int -> t
(** [lorenz n_steps] is an Euler-integrated Lorenz attractor unrolled for
    [n_steps]. The state [(x, y, z)] is three small float32 vectors; each step
    applies [dx = σ(y-x)], [dy = x(ρ-z)-y], [dz = xy-βz] and the update
    [state += dt·d]. Every step's outputs feed the next through shared
    subgraphs, producing a deep straight-line DAG that fuses into one reduce
    kernel. The size descriptor is [nN]. *)

val rnn : int -> t
(** [rnn horizon] is a forward-only recurrence unrolled for [horizon] steps.
    Each step forms [h ← x@W + h@U] (affine; no autodiff) over modest matrices
    and accumulates a squared-magnitude scalar loss. Matmuls and reduces
    dominate the graph. The size descriptor is [hH]. *)

val lorenz_ladder : int list
(** Step counts sweept by {!lorenz}: [[ 10; 25; 50; 100; 200 ]]. *)

val rnn_ladder : int list
(** Horizons sweept by {!rnn}: [[ 2; 5; 10; 20 ]]. *)

val all : t list
(** [[ elementwise; reduce; matmul_small ]]: the fixed-size reference
    workloads. *)

val scaling : t list
(** {!lorenz} and {!rnn} at every ladder point, for the comparative sweep. *)
