(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    Each kernel computes into [dst], a written operand of the result's shape and
    dtype, from operands of the same shape and any layouts, on memory the host
    addresses. It claims every operand through the door of [nx_array.h] for the
    extent of its call. It answers [0] once its work is done, or, before any
    write, a refusal's code: the door's, or [NX_SHAPE] if the operands' shapes
    differ. {!Nx_array.refused} raises any such code. The result is a function
    of the operands' values alone: neither layouts nor threads change a bit. *)

val copy : dst:('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
(** [copy ~dst a] stores [a]'s elements into [dst], bits for bits, NaN payloads
    included. *)

val cast : dst:('w, 'r) Nx_array.t -> ('v, 's) Nx_array.t -> int
(** [cast ~dst a] stores each of [a]'s elements into [dst]'s dtype. A float
    stores as {!Nx_array.Dtype.of_float} says. An integer stores as its exact
    value rounded once to a float format, modulo the width to an integer dtype,
    and as [x <> 0] to a boolean. A boolean stores as [0] or [1]. Into a complex
    dtype, these rules give the real part and the imaginary part is [0.]. A
    complex number stores part by part into a complex dtype, as [true] into a
    boolean if either part is non-zero, and by its real part into any other
    dtype. A float that keeps its format, as a float32 into a complex64's real
    part, keeps its bits. Into [a]'s own dtype it is {!copy}. *)
