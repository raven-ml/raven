(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What computes on arrays.

    A backend is kernels over the arrays of one device: one function per
    operation nx computes, each reading its operands and writing its result into
    [dst], an array nx allocated on the same device, C-contiguous from its first
    element. nx places the operands, allocates the results and calls the backend
    once per device, so a backend never sees nx's values, a placement or another
    device. The kinds of nx's operations are here too: operations of one kind
    share their operands' and results' types, and the kind names the
    mathematical function, after the [Nx] function it implements. *)

include module type of Nx_backend_intf
(** @inline *)

(** {1:backends Backends} *)

exception Refused of string
(** [Refused reason] is raised by a kernel that does not run its arguments,
    before it writes anything. [reason] names the backend, the operation and
    what it refuses, as in ["counting: matmul: no float64"]. *)

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }
(** [Linalg_error { op; kind }] is raised by a linear-algebra kernel whose
    computation fails on its values: [op] is the operation, as in ["cholesky"],
    and [kind] the failure: a matrix that is not positive-definite, a singular
    matrix, or an iteration that did not converge. A precondition on shapes or
    dtypes that slips past nx raises [Invalid_argument] or [Failure]. *)

type t
(** The type for backends. *)

val make : (module S) -> t
(** [make k] is the backend of kernels [k]. Each [make] is a backend of its own:
    a library makes its backend once and shares the value. *)

val kernels : t -> (module S)
(** [kernels b] is [b]'s kernels, for nx to call. *)

val name : t -> string
(** [name b] is [b]'s name. *)

val runs_on : t -> Nx_device.t -> bool
(** [runs_on b d] is [true] iff [b] computes on arrays in [d]'s memory. *)

val equal : t -> t -> bool
(** [equal b b'] is [true] iff [b] and [b'] are the same {!make}. *)
