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
    caching is controlled by the [CCACHE] environment variable
    (default [1], set to [0] to disable). *)

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
    When [None] (default) or when the [CCACHE] environment variable is
    [0], {!compile_cached} bypasses the disk cache. *)

(** {1:accessors Accessors} *)

val name : t -> string
(** [name c] is [c]'s name. *)

(** {1:compiling Compiling} *)

val compile : t -> string -> bytes
(** [compile c src] compiles [src] using [c], bypassing the disk
    cache. *)

val compile_cached : t -> string -> bytes
(** [compile_cached c src] compiles [src] using [c]. Checks the
    disk cache first when a [cachekey] was provided and [CCACHE] is
    enabled; stores the result on cache miss. Falls back to
    {!compile} when caching is disabled. *)
