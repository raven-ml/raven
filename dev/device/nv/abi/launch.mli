(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels set up for launch on a GPU.

    A launch of a cubin's kernel ({!Cubin.kernel}) depends on the GPU as well:
    its compute engine's class sets the launch descriptor's version and the
    layout of the driver's parameters, and the GPU's limits bound the shared
    memory and threads a launch may take.

    Constant bank [0] is written anew for each launch: the driver's parameters
    ({!driver_parameters}), then the kernel's. *)

type t = private Repr.launch
(** The type for kernels set up for launch on a GPU. *)

val make : Gpu.t -> Cubin.kernel -> (t, string) result
(** [make g k] is [k] launched on [g]. It is [Error] if [k] takes more than a
    launch can:
    - more shared memory than 100 KiB, the 1 KiB the driver reserves included;
    - more than 255 registers a thread;
    - a constant bank of index outside \[[0];[7]\], the banks a launch
      descriptor names, or of more than 64 KiB.

    Raises [Invalid_argument] if [g.compute_class] is none of the classes
    {!Gpu.t} names. *)

val banks : t -> Cubin.bank list
(** [banks l] is the constant banks a launch addresses: the kernel's
    ({!Cubin.kernel}), with a bank [0] of 352 bytes at offset [0] first if it
    has none. *)

val driver_parameters : t -> string
(** [driver_parameters l] is the start of constant bank [0]: the GPU's shared
    and local memory windows ({!Gpu.t}) and the stack limit, at the places the
    class sets, and zeros elsewhere. Its length is the kernel's [params_offset],
    or the length of the class's parameters if that is longer. *)

val local_bytes : t -> int
(** [local_bytes l] is the local memory each thread of a launch needs: the
    kernel's stack and the 576 bytes the driver reserves. *)

val max_threads : t -> int
(** [max_threads l] is the most threads a block of [l] may have: [1024], or
    fewer for the registers its threads use. *)
