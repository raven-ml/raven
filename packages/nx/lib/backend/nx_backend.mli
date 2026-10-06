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
    every backend a program pairs with a device ([Nx.Device.with_backend]). A
    kernel library makes its backend ({!type-t}) once, with {!v}, and exports
    it, as [Nx_cpu.backend]. The kinds of nx's operations are here too:
    operations of one kind share their operands' and results' types, and the
    kind names the mathematical function, after the [Nx] function it implements.
*)

include module type of Nx_backend_intf
(** @inline *)

(** {1:backends Backends} *)

type t
(** The type for backends: the kernels of one library, as a value devices carry.
    A backend is equal only to itself: two calls of {!v} make two backends, even
    over one module. *)

val v : (module S) -> t
(** [v (module K)] is a new backend computed by [K]'s kernels. A kernel library
    calls it once and exports the result, so that every device it computes on
    carries one backend. *)

val kernels : t -> (module S)
(** [kernels b] is the kernels [b] was made from. *)

val name : t -> string
(** [name b] is the name of [b]'s kernels ([K.name]). *)

val equal : t -> t -> bool
(** [equal b b'] is [true] iff one call of {!v} made [b] and [b']. *)

(** {1:errors Errors} *)

exception Refused of string
(** [Refused reason] is raised by a kernel that does not run its arguments,
    before it writes anything. [reason] says what it refuses, as in
    ["no float64"]: nx raises [Invalid_argument] in its place, naming the
    backend, the device and the operation, and the operation reaches no other
    backend. *)
