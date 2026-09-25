(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Metal GPU device backend.

    [Tolk_metal] provides a {!Tolk.Device.t} that executes compiled kernels on
    Apple Metal GPUs. Construct a device with {!create} and interact with it
    through the {!Tolk.Device} interface.

    {!Tolk.Realize.compile_linear} encodes kernel batches into indirect
    command buffers and compiles their host submission. Linking allocates the
    commands and resolves native functions; replay fences the previous use
    before updating buffer addresses, scalar arguments and dispatch sizes.

    {1:compilation Kernel compilation}

    Kernels are compiled in two stages. The compiler first attempts offline
    compilation via Apple's private MTLCompiler framework (source to MTLB
    binary). When that framework is unavailable, it falls back to runtime source
    compilation through the Metal API.

    {1:env Environment variables}

    - [METAL_FAST_MATH] — when set to a non-zero integer, enables fast-math mode
      for runtime source compilation (the fallback path). Defaults to [0]
      (disabled).
    - [FIX_METAL_ICB] — whether queue submissions issue an empty dispatch per
      pipeline before executing the indirect command buffer, which pre-M3 GPUs
      need to not crash. Defaults to [1] on those GPUs and [0] elsewhere. *)

(** {1:device Device} *)

val create : string -> Tolk.Device.t
(** [create name] is a Metal device identified by [name].

    The device uses the system default Metal GPU, an LRU-cached shared-memory
    allocator with shared compiled buffer transfers, and a {!Tolk.Cstyle.metal}
    renderer built from the device's Metal GPU family. An {!Stdlib.at_exit}
    handler synchronizes in-flight work and releases the underlying Metal
    device and command queue.

    Raises [Failure] if no Metal GPU is available (e.g. running in a VM or on
    unsupported hardware). *)

(** {1:state Device state} *)

module State : sig
  type t
  (** The type for Metal device state. Holds the GPU device handle, command
      queue, completion timeline, in-flight command buffers, and pipelines
      shared by compiled schedules. *)

  val create : unit -> t
  (** [create ()] initializes the system default Metal device and command queue.

      Raises [Failure] if no Metal GPU is available. *)

  val synchronize : t -> unit
  (** [synchronize t] blocks until all in-flight command buffers complete. After
      return, the in-flight list is empty.

      Raises [Failure] if any command buffer completed with an error. Compiled
      queues retain the first failure and reject subsequent submissions. *)

  val shutdown : t -> unit
  (** [shutdown t] synchronizes and releases cached pipelines, the command queue,
      and the device. Subsequent calls are no-ops. Raises [Failure] if
      synchronization fails; resources remain retained in that case. *)

end
