(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU backend Clang compiler for the tolk JIT runtime.

    Compiles C source to relocatable ELF objects by invoking clang as a
    subprocess. The compiler uses the selected architecture in freestanding mode
    ([-ffreestanding], [-nostdlib], [-fPIC]), producing position-independent
    objects suitable for JIT loading via {!Compiler}.

    Source is fed on stdin and the object is read from stdout, so no temporary
    files are created.

    The compiler executable is controlled by the [CC] environment variable (via
    {!Helpers.Context_var.string}), defaulting to ["clang"]. *)

(** {1:compiling Compiling} *)

val cc : unit -> string
(** [cc ()] is the compiler executable, the current value of [CC]. Objects
    differ by compiler, so the compiler's cache key records it. *)

val host_arch : unit -> string
(** [host_arch ()] is the normalized host architecture followed by [,native],
    suitable as the default CPU target. *)

val compile_clang : ?arch:string -> string -> bytes
(** [compile_clang ?arch src] compiles C source [src] to a relocatable ELF object.

    The returned {!bytes} contains the raw object file contents.

    [arch] defaults to {!host_arch}. It has the form [arch,cpu,features], with
    zero or more comma-separated features; prefix a feature with [-] to disable
    it. For example, [x86_64,znver2,-avx512f] or [arm64,generic].
    Compilation uses [-O2] and architecture-specific tuning flags. On ARM64,
    [-ffixed-x18] reserves the platform register used by macOS and Windows.

    [-fno-math-errno] is always passed so that intrinsics like [sqrt] compile to
    single instructions rather than function calls.

    Host machine names are normalized like tinygrad's CPU device:
    [amd64] to [x86_64] and [aarch64] to [arm64].

    Raises {!Compiler.Compile_error} if clang cannot be started or exits with a
    non-zero status, or if [arch] is malformed or unsupported. The error
    message includes clang's stderr output when available. *)
