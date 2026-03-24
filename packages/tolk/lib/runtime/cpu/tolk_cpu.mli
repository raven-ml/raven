(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU device backend.

    [Tolk_cpu] provides a CPU execution backend for tolk. It compiles kernels to
    native object code via an external C compiler, loads them into executable
    memory, and dispatches execution across multiple CPU cores using a domain
    pool.

    The single entry point is {!val-create}, which returns a {!Tolk.Device.t}
    ready for use with the tolk runtime. *)

(** {1:device Device creation} *)

val create : string -> Tolk.Device.t
(** [create name] is a CPU device named [name].

    The device uses clang (or the compiler specified by the [CC] environment
    variable) to compile kernel source to native object code. Compiled objects
    are loaded into executable memory via an ELF loader and JIT stubs.

    Kernel execution is dispatched through a background worker thread.
    Multi-threaded kernels fan out across a shared domain pool, providing true
    OS-level parallelism via OCaml 5 domains.

    Memory allocation uses [calloc]/[free]. An LRU cache wraps the raw allocator
    to reuse recently freed buffers.

    [CC] defaults to ["clang"] when unset. *)
