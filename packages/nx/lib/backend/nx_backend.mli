(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Backends: what computes nx's operations on arrays.

    A backend ({!S}) is a name, the memories it computes on, and one kernel per
    operation nx computes, each reading its operands and writing its result into
    [dst], an array allocated in the same memory, C-contiguous from its first
    element. A kernel never sees nx's values, a placement or another device.
    Every backend, and jit's compiled programs, compute by this contract:
    nx.cpu, which computes eager operations on the host and test devices, and
    every backend a program pairs with a device ([Nx.Device.with_backend]). The
    kinds of nx's operations are here too: operations of one kind share their
    operands' and results' types, and the kind names the mathematical function,
    after the [Nx] function it implements. *)

include module type of Nx_backend_intf
(** @inline *)

(** {1:errors Errors} *)

exception Refused of string
(** [Refused reason] is raised by a kernel that does not run its arguments,
    before it writes anything. [reason] says what it refuses, as in
    ["no float64"]: nx raises [Invalid_argument] in its place, naming the
    backend, the device and the operation, and the operation reaches no other
    backend. *)

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
