(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** C-family language renderers.

    {!Renderer.t} values for C-style GPU and CPU backends (CUDA, HIP, Metal,
    OpenCL, Clang). Use {!Renderer.render} to convert {!Ir.Program.t} programs
    to backend-specific source code.

    Each renderer maps {!Ir.special_dim} values to backend-specific workitem
    expressions (e.g., [blockIdx]/[threadIdx] for CUDA, [get_global_id] for
    OpenCL). *)

(** {1:cpu CPU} *)

val clang : Renderer.t
(** Clang/CPU renderer with SIMD support.

    Generates C code for CPU execution using Clang extensions: [ext_vector_type]
    for SIMD, [__builtin_convertvector] for vector casts, [__builtin_sqrtf] etc.
    for math. No GPU thread support. Emits a fixed-ABI wrapper
    ([void kern(const unsigned long long *bufs, ...)]) around the kernel to
    avoid a libffi dependency at JIT time.

    {b Note.} Reads environment variables at module initialization:
    - [AMX]: set to [1] to enable Apple AMX tensor cores.
    - [THREADS]: set to [0] to disable host-side threading (default: enabled).
    - [CPU_COUNT]: override logical CPU count for thread pool size. *)

val clang_no_abi : Renderer.t
(** [clang_no_abi] is {!clang} without the fixed-ABI wrapper.

    Generates a plain [void name(...)] signature matching tinygrad's
    [ClangRenderer] output. Useful for parity testing. *)

(** {1:opencl OpenCL} *)

val opencl : Renderer.t
(** OpenCL renderer.

    Generates OpenCL C code using [get_global_id], [get_local_id], [__kernel],
    [__global], [__local]. *)

val intel : Renderer.t
(** Intel OpenCL renderer.

    {!opencl} variant with [intel_reqd_sub_group_size(8)] for sub-group WMMA
    operations. Uses Intel-specific bf16 conversion intrinsics. *)

val qcom : Renderer.t
(** Qualcomm OpenCL renderer.

    Standard {!opencl} for Qualcomm Adreno GPUs. *)

(** {1:metal Metal} *)

val metal : Renderer.t
(** Metal Shading Language renderer.

    Generates MSL code for Apple GPUs. Supports local thread IDs and up to 32KB
    of shared (threadgroup) memory. *)

(** {1:cuda CUDA} *)

(** The type for CUDA SM architecture tiers.

    Selects tensor core configurations for the CUDA renderer. *)
type cuda_arch =
  | SM75  (** SM 7.5 (Turing). *)
  | SM80  (** SM 8.0 (Ampere). *)
  | SM89  (** SM 8.9 (Ada Lovelace, includes fp8). *)

val cuda : ?arch:cuda_arch -> unit -> Renderer.t
(** [cuda ?arch ()] is a CUDA renderer for NVIDIA GPUs.

    Generates CUDA C code using [blockIdx], [threadIdx] for thread indexing and
    up to 48KB of shared memory. Uses half-precision intrinsics ([hexp2],
    [hlog2], etc.) for float16/bfloat16.

    [arch] selects tensor cores. When omitted, reads [CUDA_ARCH] (or [CUDA_SM]
    as fallback) from the environment and maps to the nearest tier; if neither
    is set, produces a renderer with no tensor cores. *)

(** {1:amd AMD} *)

(** The type for AMD GPU architecture families.

    Selects tensor core configurations and preamble builtins for the AMD HIP
    renderer. *)
type amd_arch =
  | RDNA3  (** RDNA3 (WMMA 16x16x16). *)
  | RDNA4  (** RDNA4 (WMMA 16x16x16, gfx12 builtins). *)
  | CDNA3  (** CDNA3 (MFMA fp8/bf16, 16x16x32/16). *)
  | CDNA4  (** CDNA4 (MFMA fp8/bf16/f16, 16x16x128/32/16). *)

val amd : ?arch:amd_arch -> unit -> Renderer.t
(** [amd ?arch ()] is an AMD HIP renderer.

    Generates HIP code using OCML math library and OCKL work item functions.
    Supports up to 64KB of shared memory. Uses [__ockl_get_group_id],
    [__ockl_get_local_id] for thread indexing and [__ocml_*_f\{16,32,64\}] for
    transcendentals.

    [arch] defaults to {!RDNA3}. *)
