(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of HIP for AMD GPUs.

    comgr, ROCm's code object manager library, compiles the source. comgr
    compiles one source at a time in a process, so each compile runs in a
    process of its own, which tolk starts from a program it carries: compiles
    from several domains run at once, and a compile that crashes ends its own
    process only. On Linux the program runs from memory; elsewhere it is written
    to {!Helpers.cache_dir}, under a name holding its digest. The library,
    [libamd_comgr], is loaded by that process from where {!C.findlib} finds it:
    the file [COMGR_PATH] names, then [lib/libamd_comgr.so] in the directory
    [ROCM_PATH] names ([/opt/rocm] by default, read when the program starts),
    then the system's library directories. Its major version picks its numbering
    of languages and actions, which version 3 changed. *)

val hip : string -> Renderer.Compiler.t
(** [hip arch] is the compiler of HIP source to executable code objects for the
    GPU architecture [arch], as in ["gfx1100"]. A source whose first line is
    [.text] is assembly, which is assembled instead.

    HIP is compiled with the device libraries, at [-O3], in CU mode, as HIP 6.0
    without HIP's own headers, whose declarations rendered kernels carry. The
    kernels' symbols are internalized before code generation, so that only
    kernels stay visible.

    With {!Helpers.ccache}, code objects are cached in the table
    [compile_hip_A_D], where [A] is [arch] and [D] the digest of comgr's library
    ({!C.identity}) and the options it compiles and links with: a code object is
    read back only for its source, compiled by the same comgr with the same
    options. {!Renderer.Compiler.compile} raises
    {!Renderer.Compiler.Compile_error} with comgr's status and log if comgr
    rejects the source, with the reason if comgr cannot be loaded, and with how
    the process ended and what it printed if it ends without answering, and
    [Failure] if the process cannot be started. {!Renderer.Compiler.disassemble}
    prints what [llvm-objdump -d] prints ({!Helpers.amdgpu_disassemble}). *)
