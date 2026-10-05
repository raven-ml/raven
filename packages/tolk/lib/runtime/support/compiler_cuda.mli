(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The compiler of CUDA C++ for NVIDIA GPUs.

    NVRTC, NVIDIA's runtime compilation library, compiles the source in the
    process. The library, [libnvrtc], is loaded at the first compile, once per
    process, from where {!C.findlib} finds [nvrtc]: the file [NVRTC_PATH] names,
    then the system's library directories and those of the CUDA toolkit. *)

val nvrtc : ?ptx:bool -> ?cache_key:string -> string -> Renderer.Compiler.t
(** [nvrtc ~ptx ~cache_key arch] is the compiler of CUDA C++ source for the GPU
    architecture [arch], as in ["sm_89"]: to PTX if [ptx] is [true] (the
    default), which the driver translates for the GPU it loads it on, and else
    to a cubin, the binary of [arch] itself.

    Headers are searched in [include] of the directory that the variable
    [CUDA_PATH] names when the compiler is made, or if it is empty in
    [/usr/local/cuda/include], [/usr/include] and [/opt/cuda/include]. From
    version 12.4, NVRTC compiles faster without textures, surfaces and the
    device runtime API ([--minimal]), which rendered kernels do not use.

    With {!Setting.ccache}, binaries are cached in the table [compile_K_A_D],
    where [K] is [cache_key] (default ["cuda"]), [A] is [arch], and [D] the
    digest of NVRTC's library ({!C.identity}), whether it makes PTX, and its
    options: a binary is read back only for its source, compiled by the same
    NVRTC with the same options. {!Renderer.Compiler.compile} raises
    {!Renderer.Compiler.Compile_error} with NVRTC's error and log if NVRTC
    rejects the source, and with the reason if NVRTC cannot be loaded or reports
    no version. {!Renderer.Compiler.disassemble} prints the GPU's instructions,
    which the CUDA toolkit's [ptxas] and [nvdisasm] find, or else why they could
    not. *)
