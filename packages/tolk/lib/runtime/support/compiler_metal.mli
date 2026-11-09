(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Metal: the compiler of Metal Shading Language for Apple GPUs.

    MTLCompiler, the private framework behind Metal's own compiler, compiles the
    source. It reads its options into LLVM's, which are global to a process, at
    every compile, so compiles run in processes of their own, which tolk starts
    from a program it carries: compiles from several domains run at once, and a
    compile that crashes ends its own process only. Each process serves compiles
    in turn, from any domain, and waits idle between them, so that loading
    MTLCompiler is paid once per process; a compile that finds none idle starts
    another, so a process's first compile also pays a process start. This
    process keeps as many as it ran compiles at once, a search's parallelism or
    the domains that compile a program's kernels, each holding about 13 MB of
    its own memory, more after a large kernel, until it exits. A process forked
    from this one starts its own and never uses its parent's. The program is
    written to {!Helpers.cache_dir}, under a name holding its digest, or to the
    temporary directory if the cache cannot be written or the system refuses to
    run it from there. It runs only as a regular file of this user's that no
    other user can write, reached through no link and not quarantined, checked
    before each start and written again otherwise: where the system runs no
    program this process writes, every compile fails. The library is the file
    {!C.findlib} finds for [MTLCompiler] when the compiler is made: the file
    [MTLCOMPILER_PATH] names, then macOS's private frameworks. Each process
    loads it. Only macOS has it. Making a compiler, and a compile its table
    holds, start no process. *)

val compiler : unit -> Renderer.Compiler.t
(** [compiler ()] is the compiler of Metal source to Metal libraries, the
    binaries that Metal loads. The language is the latest version that the
    running macOS compiles kernels of: Metal 4.0 from macOS 26, 3.1 from 14, 3.0
    from 13, and macOS Metal 2.0 before. Fast math is off. The modules of
    Metal's standard library are cached in {!Helpers.cache_dir}, which makes
    later compiles faster.

    With {!Setting.ccache}, libraries are cached in the table
    [compile_metal_direct_D], where [D] is the digest of the build of macOS,
    which MTLCompiler is part of, MTLCompiler's file ({!C.identity}), the
    language and the options: a library is read back only for its source,
    compiled by the same MTLCompiler with the same options.

    Raises [Failure] naming [MTLCOMPILER_PATH] if MTLCompiler is not found.
    {!Renderer.Compiler.compile} raises {!Renderer.Compiler.Compile_error} with
    the compiler's message if it rejects the source, with the reason if
    MTLCompiler cannot be loaded or makes no service, and with how the process
    ended and what it printed if it ends without answering, which fails that
    compile only; it raises [Failure] if the compiler replies with no Metal
    library or a process cannot be started. {!Renderer.Compiler.disassemble}
    prints nothing. *)
