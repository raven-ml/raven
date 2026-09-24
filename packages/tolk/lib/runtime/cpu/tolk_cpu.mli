(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU device backend.

    [Tolk_cpu] provides a CPU execution backend for tolk. It compiles kernels to
    native object code via an external C compiler, loads them into executable
    memory, and executes each kernel synchronously on the calling domain.

    {!val-create} returns a {!Tolk.Device.t} ready for use with the runtime. *)

(** {1:device Device creation} *)

val link_symbol : ?libs:string list -> string -> nativeint
(** [link_symbol ?libs name] resolves [name] in the current process or [libs].
    Loaded libraries stay resident while generated host calls can address them.
    Raises [Failure] if the symbol cannot be found. *)

val create : ?aligned:bool -> string -> Tolk.Device.t
(** [create name] is a CPU device named [name].

    The device uses clang (or the compiler specified by the [CC] environment
    variable) to compile kernel source to native object code. [DEV=CPU:CLANG]
    selects this renderer; a third field overrides the architecture, for example
    [DEV=CPU:CLANG:arm64,generic]. The default is the host architecture and CPU.
    Compiled objects
    are loaded into executable memory via an ELF loader and JIT stubs.

    Kernel calls complete before returning, including calls without timing.
    Synchronization is a no-op. Kernels retain vectorized arithmetic but do
    not spawn workers or reserve scalar parameter names.

    Memory allocation uses [calloc]/[free]. The allocator supports byte-offset
    views and is wrapped in an LRU cache to reuse recently freed buffers.

    [aligned] is passed to {!Tolk.Cstyle.clang}: [false] compiles kernels that
    accept buffers at any address, which a caller that binds memory the device
    did not allocate needs.

    [CC] defaults to ["clang"] when unset. *)
