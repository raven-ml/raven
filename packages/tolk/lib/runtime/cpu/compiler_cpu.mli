(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU backend Clang compiler for the tolk JIT runtime.

    Compiles C source to relocatable ELF objects by invoking clang as a
    subprocess. The compiler targets the host architecture in freestanding mode
    ([-ffreestanding], [-nostdlib], [-fPIC]), producing position-independent
    objects suitable for JIT loading via {!Compiler}.

    Source is fed on stdin and the object is read from stdout, so no temporary
    files are created.

    The compiler executable is controlled by the [CC] environment variable (via
    {!Helpers.Context_var.string}), defaulting to ["clang"]. *)

(** {1:compiling Compiling} *)

val compile_clang : string -> bytes
(** [compile_clang src] compiles C source [src] to a relocatable ELF object.

    The returned {!bytes} contains the raw object file contents.

    Compilation uses [-O2] and the following architecture-specific flags:
    - x86_64: [-march=native]
    - ARM64/AArch64: [-ffixed-x18] and [-mcpu=native] (avoids the
      platform-reserved register on macOS and Windows)
    - RISC-V 64: [-march=rv64g]

    [-fno-math-errno] is always passed so that intrinsics like [sqrt] compile to
    single instructions rather than function calls.

    Host machine names are normalized like tinygrad's CPU device:
    [amd64] to [x86_64] and [aarch64] to [arm64].

    Raises {!Compiler.Compile_error} if clang cannot be started or exits with a
    non-zero status, or if the host architecture is unsupported. The error
    message includes clang's stderr output when available. *)
