(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Metal: the compiler of Metal Shading Language for Apple GPUs.

    MTLCompiler, the private framework behind Metal's own compiler, compiles the
    source in the process. The framework is loaded at the first compile, from
    where {!C.findlib} finds [MTLCompiler]: the file [MTLCOMPILER_PATH] names,
    then macOS's private frameworks. Only macOS has it. *)

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
    {!Renderer.Compiler.compile} raises {!Renderer.Compiler.Compile_error} with
    the compiler's message if it rejects the source, and with the reason if
    MTLCompiler cannot be loaded, and [Failure] if the compiler replies with no
    Metal library. {!Renderer.Compiler.disassemble} prints nothing. *)
