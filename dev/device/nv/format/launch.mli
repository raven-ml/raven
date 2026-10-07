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

type t
(** The type for kernels set up for launch on a GPU. *)

val make :
  Gpu.t ->
  shared_window:int ->
  local_window:int ->
  Cubin.kernel ->
  (t, string) result
(** [make g ~shared_window ~local_window k] is [k] launched on [g], whose
    kernels see their shared and local memory at the addresses [shared_window]
    and [local_window] ({!Method.shared_memory_window}). It is [Error] if [k]
    declares more shared memory than a launch can take: 100 KiB, the 1 KiB the
    driver reserves included. *)

val code : t -> int
(** [code l] is the offset of the kernel's first instruction in its cubin's
    image. *)

val banks : t -> Cubin.bank list
(** [banks l] is the constant banks a launch addresses: the kernel's
    ({!Cubin.kernel}), with a bank [0] of 352 bytes at offset [0] first if it
    has none. *)

val driver_parameters : t -> string
(** [driver_parameters l] is the start of constant bank [0]: the shared and
    local memory windows and the stack limit, at the places the class sets, and
    zeros elsewhere. Its length is the kernel's [params_offset], or the length
    of the class's parameters if that is longer. *)

val local_bytes : t -> int
(** [local_bytes l] is the local memory each thread of a launch needs: the
    kernel's stack and the 576 bytes the driver reserves. *)

val max_threads : t -> int
(** [max_threads l] is the most threads a block may have for the registers its
    threads use. *)
