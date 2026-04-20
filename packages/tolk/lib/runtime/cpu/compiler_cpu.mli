(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CPU backend compiler for the tolk JIT runtime.

    Compiles C or LLVM IR source to relocatable ELF objects by invoking clang as
    a subprocess. The compiler targets the host architecture in freestanding
    mode ([-ffreestanding], [-nostdlib], [-fPIC]), producing
    position-independent objects suitable for JIT loading via
    {!Compiler}.

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
    - ARM64: [-mcpu=native] and [-ffixed-x18] (avoids the platform-reserved
      register on macOS and Windows)
    - RISC-V 64: [-march=rv64g]

    [-fno-math-errno] is always passed so that intrinsics like [sqrt] compile to
    single instructions rather than function calls.

    {b Note.} On Windows the target is forced to [x86_64] regardless of the
    reported host architecture.

    Raises {!Compiler.Compile_error} if clang exits with a non-zero
    status. The error message includes clang's stderr output when available. *)

val compile_llvmir : string -> bytes
(** [compile_llvmir src] compiles LLVM IR source [src] to a relocatable ELF
    object.

    Behaves identically to {!compile_clang} except the input language is LLVM IR
    ([-x ir]) instead of C.

    {b Note.} This invokes clang as a subprocess rather than using the
    LLVM C API directly. This avoids a library dependency on LLVM at the
    cost of per-compilation subprocess overhead.

    Raises {!Compiler.Compile_error} if clang exits with a non-zero
    status. *)
