(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of HIP for AMD GPUs.

    comgr, ROCm's code object manager library, compiles the source. comgr
    compiles one source at a time in a process, so compiles run in processes
    of their own, which tolk starts from a program it carries: compiles from
    several domains run at once, and a compile that crashes ends its own
    process only. Each process serves compiles in turn, from any domain, and
    waits idle between them, so that loading comgr is paid once per process;
    a compile that finds none idle starts another. This process keeps as many
    as it ran compiles at once, a search's parallelism, each holding about
    15 MB of its own memory, until it exits. A process forked from this one
    starts its own and never uses its parent's. On Linux the program runs
    from memory; elsewhere it is written to {!Helpers.cache_dir}, under a name
    holding its digest. The library, [libamd_comgr], is the file {!C.findlib}
    finds when the compiler is made: the file [COMGR_PATH] names, then
    [lib/libamd_comgr.so] in the directory {!Setting.rocm_path} names, then
    the system's library directories. Each process loads it. Its major
    version picks its numbering of languages and actions, which version 3
    changed. comgr's own cache of the code it generates, which it keeps under
    [~/.cache/comgr] or [AMD_COMGR_CACHE_DIR], is disabled in those processes
    alone: code objects are cached in one place, {!Setting.ccache}'s table. *)

val hip : string -> Renderer.Compiler.t
(** [hip arch] is the compiler of HIP source to executable code objects for the
    GPU architecture [arch], as in ["gfx1100"]. A source whose first line is
    [.text] is assembly, which is assembled instead.

    HIP is compiled with the device libraries, at [-O3], in CU mode, as HIP 6.0
    without HIP's own headers, whose declarations rendered kernels carry. The
    kernels' symbols are internalized before code generation, so that only
    kernels stay visible. From comgr 2.8 (ROCm 6.2), LLVM's verifier checks
    the IR before it is optimized and before code generation: IR that an
    optimization makes invalid raises {!Renderer.Compiler.Compile_error}
    naming the verifier's finding instead of compiling. Earlier comgr
    compiles without it.

    With {!Setting.ccache}, code objects are cached in the table
    [compile_hip_A_D], where [A] is [arch] and [D] the digest of comgr's library
    ({!C.identity}), the program that runs it, and the options it compiles and
    links with: a code object is read back only for its source, compiled by the
    same comgr with the same options. Raises [Failure] naming [COMGR_PATH] if no comgr is found.
    {!Renderer.Compiler.compile} raises {!Renderer.Compiler.Compile_error} with
    comgr's status and log if comgr rejects the source, with the reason if
    comgr cannot be loaded, and with how
    the process ended and what it printed if it ends without answering, and
    [Failure] if the process cannot be started. {!Renderer.Compiler.disassemble}
    prints what [llvm-objdump -d] prints ({!Helpers.amdgpu_disassemble}). *)
