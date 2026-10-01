(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of C for processors.

    Clang compiles the source to a relocatable ELF object that is freestanding:
    it calls no library, so a loader places it in memory and runs it as it is.
    The Clang run is the program named by the environment variable [CC], [clang]
    by default, read once, at the first compile ({!Helpers.getenv_string}). *)

val clang : string -> Renderer.Compiler.t
(** [clang arch] is the compiler of C source to objects for [arch], written
    [MACHINE,CPU[,FEATURE]...], as in ["x86_64,znver2,avx,-avx512f"]:
    - [MACHINE] is [x86_64], [arm64] or [riscv64];
    - [CPU] is a processor model that Clang knows, [native] for the host's. On
      [riscv64], [native] is [rv64g];
    - each [FEATURE] is an instruction set extension to enable, or to disable if
      it starts with [-]. On [riscv64] features cannot be disabled: each is an
      extension that the processor model gains.

    Objects are optimized at [-O2] and position independent. Math functions do
    not set [errno], so that a square root is one instruction. On [arm64] the
    register [x18] is left alone, since macOS and Windows clobber it.

    With {!Helpers.ccache}, objects are cached in the table
    [compile_clang_obj_A], where [A] is [arch] with its commas replaced by
    underscores. {!Renderer.Compiler.compile} raises
    {!Renderer.Compiler.Compile_error} with Clang's diagnostics if Clang rejects
    the source or does not run, and {!Renderer.Compiler.disassemble} prints what
    [objdump -d] prints ({!Helpers.cpu_objdump}).

    Raises [Invalid_argument] naming [arch] if it has fewer than two fields, and
    naming its [MACHINE] if that is another. *)
