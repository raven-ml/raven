(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of C for processors.

    Clang compiles the source to a relocatable ELF object that is freestanding:
    it calls no library, so a loader places it in memory and runs it as it is.
    The Clang run is {!Setting.cc} when the compiler is made. *)

val clang : string -> Renderer.Compiler.t
(** [clang arch] is the compiler of C source to objects for [arch], written
    [MACHINE[-SYSTEM],CPU[,FEATURE]...], as in ["x86_64,znver2,avx,-avx512f"]:
    - [MACHINE] is [x86_64], [arm64] or [riscv64];
    - [SYSTEM] is the operating system the code runs on. [x86_64-windows]
      compiles for Windows' calling convention, that of the process's
      functions, which the object calls by name or which Clang calls itself,
      such as [fma] or [memcpy], and for its LLP64 types, where a [long] has
      32 bits ({!Cstyle.clang} renders for them). The function rig calls must
      be declared [__attribute__((sysv_abi))], since rig calls by System V.
      Any other system compiles as no system;
    - [CPU] is a processor model that Clang knows, [native] for the host's. On
      [riscv64], [native] is [rv64g]. On an arm64 Mac, [native] has the half
      arithmetic that macOS lists ([hw.optional.arm.FEAT_FP16]), which Clang
      misses on a virtual Mac;
    - each [FEATURE] is an instruction set extension to enable, or to disable if
      it starts with [-]. On [riscv64] features cannot be disabled: each is an
      extension that the processor model gains.

    Objects are ELF, optimized at [-O2] and position independent. Math functions
    do not set [errno], so that a square root is one instruction. On [arm64] the
    register [x18] is left alone, since macOS and Windows clobber it. Under a
    Clang of LLVM 18 or 19, the straight-line vectorizer, which packs
    independent scalar operations into vectors, is off, since it writes
    invalid code for some packed integer lanes; under LLVM 18 so are AVX10.1
    and AVX-512's half arithmetic ({!flags}). On x86_64 Windows the stack is not probed, since
    rig commits a thread's stack before it runs linked code, and nothing is
    kept below the stack pointer, which Windows may write. A source of more
    than 400 000 bytes compiles without the loop vectorizer, which packs a
    loop's iterations into vectors, so that a long source compiles in a
    fraction of the time.

    Clang's compiler proper runs as Clang states it runs for these options
    ([clang -###]), its [-cc1] command, with LLVM's IR verifier on, which a
    release Clang turns off: code that an optimization makes invalid raises
    {!Renderer.Compiler.Compile_error} with Clang's diagnostic, which names the
    function, where it would compile into wrong values or crash Clang. Several
    sources ({!Renderer.Compiler.compile_all}) compile in one run of it, each
    source from a file of its own, so that Clang starts once for them; each
    object is the one Clang makes of its source alone, from standard input.

    Objects do not depend on the working directory. With {!Setting.ccache}, they
    are cached in the table [compile_C_obj_A_D], where [C] is the Clang run, [A]
    is [arch] with its commas replaced by underscores, and [D] the digest of
    that Clang's version and installation ([clang --version]), of the command
    of its compiler proper that compiles a source of at most and of more than
    400 000 bytes, and of that length: the command names the processor and
    features it compiles for and every option, or is replaced by why Clang
    rejects them or does not run. An object is read back only for its source,
    compiled by the same Clang with the same options.
    Clang is asked once per process for its version and for each set of
    options, when the table or a compile first needs it.
    {!Renderer.Compiler.compile} raises {!Renderer.Compiler.Compile_error} with
    Clang's diagnostics if Clang rejects the source or does not run, and
    {!Renderer.Compiler.disassemble} prints what [objdump -d] prints
    ({!Helpers.cpu_objdump}).

    Raises [Invalid_argument] naming [arch] if it has fewer than two fields, and
    naming its [MACHINE] if that is another. *)

val flags : version:string -> string -> string list
(** [flags ~version arch] is the arguments that make a Clang whose [--version]
    states [version], such as ["Ubuntu clang version 18.1.3 (1ubuntu1)"],
    compile for [arch] as {!clang} reads it: its target, processor and
    features, in that order. Under LLVM 18 an x86_64 [arch]'s arguments end
    by disabling AVX10.1 and AVX-512's half arithmetic, [avx512fp16], which
    AVX10.1 implies, whatever [arch] or [native] enables: LLVM 18 crashes
    compiling some loops over halves with them. Without them halves convert
    to and from [float] by other instructions, to the same values. Any other
    [version], including one that names no
    LLVM release, such as Apple's, and any other machine take [arch]'s
    arguments alone.

    Raises [Invalid_argument] as {!clang} does. *)

val registers : string -> Renderer.registers option
(** [registers arch] is the vector register file of the processor
    {!clang}[ arch] compiles for, as the macros of the Clang run {!Setting.cc}
    state its features: on [x86_64], [16] registers, [32] with AVX-512, of [32]
    bytes with AVX and [16] without (a processor with AVX-512 is taken at [32]
    bytes); on [arm64], [32] of [16]; on [riscv64], [None]. On [x86_64] a Clang
    that does not run gives [16] of [16]. Each Clang and [arch] is asked once
    per process.

    Raises [Invalid_argument] as {!clang} does. *)

val win64 : string -> bool
(** [win64 arch] is [true] iff [arch], as {!clang} reads it, is x86_64's
    under Windows: its [MACHINE-SYSTEM] is [x86_64-windows]. *)
