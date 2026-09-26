(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel source compiler.

    A compiler turns rendered source code into a compiled binary. Each
    backend provides its own compiler (e.g. clang for CPU, nvcc for
    CUDA). Renderers carry an optional compiler via
    {!Renderer.compiler}; the device selects the active renderer and
    uses its compiler at {!Device.compile_program} time.

    {!compile_cached} is the primary entry point: it checks the
    on-disk cache before invoking the underlying compiler. Disk
    caching is enabled when the compiler is created, using the [CCACHE]
    context (default [1], set to [0] to disable). The [CACHELEVEL] context
    controls disk reads and writes on each call. *)

(** {1:types Types} *)

type t
(** The type for kernel compilers. *)

exception Compile_error of string
(** Raised by {!compile} when compilation fails. The payload is a
    human-readable error message. *)

(** {1:constructors Constructors} *)

val make :
  name:string ->
  ?cachekey:string ->
  compile:(string -> bytes) ->
  unit ->
  t
(** [make ~name ?cachekey ~compile ()] is a compiler with the given
    name and compilation function.

    [cachekey] is the disk cache table name (e.g., ["compile_clang_jit"]).
    When [None] (default) or when the [CCACHE] context is [0] at creation,
    {!compile_cached} bypasses the disk cache for this compiler. *)

(** {1:accessors Accessors} *)

val name : t -> string
(** [name c] is [c]'s name. *)

val cachekey : t -> string option
(** [cachekey c] is [c]'s disk cache table name, or [None] when [c] was
    created without one or with caching disabled. The key identifies the exact
    compilation target (e.g. ["compile_cuda_sm_90"]), so it doubles as a compiler and
    architecture fingerprint for callers keying their own caches. *)

(** {1:compiling Compiling} *)

val compile : t -> string -> bytes
(** [compile c src] compiles [src] using [c], bypassing the disk
    cache. *)

val compile_cached : t -> string -> bytes
(** [compile_cached c src] compiles [src] using [c]. Checks the
    disk cache first when {!cachekey} is present and [CACHELEVEL] is positive;
    stores the result on cache miss when caching is enabled.

    Raises {!Compile_error} before invoking the compiler if no cached binary
    is available and the [ASSERT_COMPILE] environment variable is nonzero. *)
