(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's operations over OCaml arrays of unboxed elements.

    Place values with [Nx.Placement.device ~backend Nx.Device.host] to compute
    on them with these kernels. *)

val backend : Nx_effect.Backend.t
(** [backend] holds values in arrays of its own, on the host device only, and
    computes on them with {!Kernels}. [Nx.place] onto its placement copies the
    elements in, and a read copies them out. It refuses, with
    [Nx.Backend.Refused], the dtypes it has no arrays for (all but float64,
    float32, int8, int16, int32, int64 and bool), the Fourier transforms and
    the linear algebra beyond [matmul]. A host operand raises
    [Invalid_argument], as operands of two placements do: place it first. *)

module Kernels = Kernels
(** The kernels over their own tensors: {!Nx_core.Backend_intf.S} with their
    view, dtype and context. *)
