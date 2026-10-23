(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of CUDA C++ for NVIDIA GPUs.

    NVRTC, NVIDIA's runtime compilation library, compiles the source. NVRTC's
    assembler crashes on some valid sources, so compiles run in processes of
    their own, which tolk starts from a program it carries: a compile that
    crashes ends its own process only. Each process serves compiles in turn,
    from any domain, and waits idle between them, so that NVRTC's setup is paid
    once per process; a compile that finds none idle starts another. This
    process keeps as many as it ran compiles at once, a search's parallelism,
    each holding about 18 MB of its own memory, until it exits. A process
    forked from this one starts its own and never uses its parent's. On Linux the program runs from memory;
    elsewhere it is written to {!Helpers.cache_dir}, under a name holding its
    digest. The library, [libnvrtc], is loaded by each process from where
    {!C.findlib} finds [nvrtc]: the file [NVRTC_PATH] names, then the system's
    library directories and those of the CUDA toolkit. NVRTC's own cache of
    binaries, which NVIDIA's libraries keep under [CUDA_CACHE_PATH], is
    disabled in those processes alone: its key misses some of the options,
    math and headers among them, so it could serve a binary compiled under
    others. *)

val nvrtc : ?ptx:bool -> ?cache_key:string -> string -> Renderer.Compiler.t
(** [nvrtc ~ptx ~cache_key arch] is the compiler of CUDA C++ source for the GPU
    architecture [arch], as in ["sm_89"]: to PTX if [ptx] is [true] (the
    default), which the driver translates for the GPU it loads it on, and else
    to a cubin, the binary of [arch] itself.

    Headers are searched in [include] of the directory {!Setting.cuda_path}
    names when the compiler is made, or if it is empty in
    [/usr/local/cuda/include], [/usr/include] and [/opt/cuda/include]. From
    version 12.4, NVRTC compiles faster without textures, surfaces and the
    device runtime API ([--minimal]), which rendered kernels do not use.

    With {!Setting.ccache}, binaries are cached in the table [compile_K_A_D],
    where [K] is [cache_key] (default ["cuda"]), [A] is [arch], and [D] the
    digest of NVRTC's library ({!C.identity}), the program that runs it,
    whether it makes PTX, and its options: a binary is read back only for its
    source, compiled by the same NVRTC with the same options.
    {!Renderer.Compiler.compile} raises {!Renderer.Compiler.Compile_error} with
    NVRTC's error and log if NVRTC rejects the source, with the reason if NVRTC
    cannot be loaded or reports no version, and with how the process ended, as
    the signal that killed it, and what it printed if it ends without
    answering, which fails that compile only; it raises [Failure] if a process
    cannot be started.
    {!Renderer.Compiler.disassemble} prints the GPU's instructions, which the
    CUDA toolkit's [ptxas] and [nvdisasm] find, or else why they could not. *)
