(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** C-family language renderers.

    {!Renderer.t} values for C-style GPU and CPU backends: CUDA, HIP, Metal,
    OpenCL, and Clang. Each renderer converts an {!Ir.Program.t} into
    backend-specific source code via {!Renderer.render}.

    GPU renderers map {!Ir.special_dim} values to backend-specific workitem
    expressions (e.g., [blockIdx]/[threadIdx] for CUDA, [get_global_id] for
    OpenCL, [gid]/[lid] for Metal). The CPU renderer ({!clang}) has no GPU
    thread support.

    See {!Renderer} for the renderer interface. *)

(** {1:cpu CPU} *)

val clang : Renderer.t
(** [clang] is a Clang/CPU renderer with SIMD support.

    Generates C code for host CPU execution using Clang extensions:
    - [ext_vector_type] for SIMD vector types.
    - [__builtin_convertvector] for vector casts.
    - [__builtin_sqrtf], [__builtin_truncf], etc. for math.

    Emits a fixed-ABI wrapper
    ([void name(const unsigned long long *bufs, const long long *vals)]) around
    the kernel to avoid a libffi dependency at JIT time.

    Device is ["CPU"]. No GPU thread support ({!Renderer.has_local} is [false]).
    No shared memory.

    {b Note.} Reads environment variables at module initialization:
    - [AMX]: set to [1] to enable Apple AMX tensor cores.
    - [THREADS]: set to [0] to disable host-side threading (default: enabled).
    - [CPU_COUNT]: override logical CPU count for thread pool size.

    See also {!clang_no_abi}. *)

val clang_no_abi : Renderer.t
(** [clang_no_abi] is {!clang} without the fixed-ABI wrapper.

    Generates a plain [void name(...)] signature with individual typed
    parameters. Useful for testing and integration with runtimes that use native
    calling conventions.

    See also {!clang}. *)

(** {1:opencl OpenCL} *)

val opencl : Renderer.t
(** [opencl] is an OpenCL renderer.

    Generates OpenCL C code using [get_group_id], [get_local_id],
    [get_global_id] for thread indexing. Kernel functions are annotated with
    [__kernel], buffers with [__global], and shared memory with [__local].

    Device is ["CL"]. Shared memory limit is 32KB. Bfloat16 is emulated via
    promotion to float32.

    See also {!intel} and {!qcom}. *)

val intel : Renderer.t
(** [intel] is an Intel OpenCL renderer.

    {!opencl} variant with [intel_reqd_sub_group_size(8)] for sub-group WMMA
    operations. Uses Intel-specific bf16 conversion intrinsics
    ([intel_convert_bfloat16_as_ushort], [intel_convert_as_bfloat16_float])
    instead of manual bit manipulation.

    Device is ["CL"]. Shared memory limit is 32KB. Tensor cores use 8x8x16 tiles
    with 8 threads.

    See also {!opencl}. *)

val qcom : Renderer.t
(** [qcom] is a Qualcomm OpenCL renderer for Adreno GPUs.

    Identical to {!opencl} in code generation. The separate renderer allows
    device-specific scheduling in codegen passes.

    Device is ["QCOM"]. Shared memory limit is 32KB. *)

(** {1:metal Metal} *)

val metal : Renderer.t
(** [metal] is a Metal Shading Language renderer for Apple GPUs.

    Generates MSL code with [threadgroup_position_in_grid] and
    [thread_position_in_threadgroup] attributes for thread indexing. Uses
    [threadgroup] storage for shared memory and [threadgroup_barrier] for
    synchronization.

    Device is ["METAL"]. Shared memory limit is 32KB.

    {b Note.} Tensor cores (simdgroup 8x8 matrix operations) are only available
    on arm64 Apple Silicon. On Intel Macs, no tensor cores are configured. *)

(** {1:cuda CUDA} *)

(** The type for CUDA SM architecture tiers.

    Selects tensor core (MMA) configurations for the CUDA renderer:
    - {!SM75}: 8x16x8 tiles, f16 input.
    - {!SM80}: 8x16x16 tiles (f16, bf16) + 8x16x8 (f16, tf32).
    - {!SM89}: {!SM80} + 8x16x32 tiles for fp8 (e4m3, e5m2). *)
type cuda_arch =
  | SM75  (** SM 7.5 (Turing). *)
  | SM80  (** SM 8.0 (Ampere). *)
  | SM89  (** SM 8.9 (Ada Lovelace, includes fp8). *)

val cuda : ?arch:cuda_arch -> unit -> Renderer.t
(** [cuda ?arch ()] is a CUDA renderer for NVIDIA GPUs.

    Generates CUDA C code using [blockIdx]/[threadIdx] for thread indexing. Uses
    [extern "C" __global__] with [__launch_bounds__] when local dimensions are
    known. Half-precision intrinsics ([hexp2], [hlog2], [hsqrt], etc.) are used
    for float16 and bfloat16 transcendentals.

    Device is ["CUDA"]. Shared memory limit is 48KB. Global grid max is
    \[2{^ 31}-1, 65535, 65535\]. Local block max is \[1024, 1024, 64\].

    [arch] selects tensor cores. When omitted, reads [CUDA_ARCH] (or [CUDA_SM]
    as fallback) from the environment and maps to the nearest tier; if neither
    is set, produces a renderer with no tensor cores. *)

(** {1:amd AMD} *)

(** The type for AMD GPU architecture families.

    Selects tensor core configurations and preamble builtins for the AMD HIP
    renderer:
    - RDNA (gaming): WMMA instructions, 32-thread wavefronts.
    - CDNA (compute): MFMA instructions, 64-thread wavefronts. *)
type amd_arch =
  | RDNA3  (** RDNA3 (WMMA 16x16x16, 32-thread wavefront). *)
  | RDNA4  (** RDNA4 (WMMA 16x16x16, gfx12 builtins). *)
  | CDNA3  (** CDNA3 (MFMA fp8/bf16, 16x16x32/16, 64-thread wavefront). *)
  | CDNA4  (** CDNA4 (MFMA fp8/bf16/f16, 16x16x128/32/16). *)

val amd : ?arch:amd_arch -> unit -> Renderer.t
(** [amd ?arch ()] is an AMD HIP renderer.

    Generates HIP code using OCKL work item functions ([__ockl_get_group_id],
    [__ockl_get_local_id]) for thread indexing and OCML transcendentals
    ([__ocml_*_f\{16,32,64\}]) for math. Uses [__builtin_amdgcn_fence] for
    release-acquire barriers (unlike CUDA's [__syncthreads], AMD barriers do not
    imply a fence). Bfloat16 is emulated via software bit manipulation. Fp8 uses
    [__builtin_amdgcn_cvt_*] intrinsics.

    Device is ["AMD"]. Shared memory limit is 64KB. Global grid max is
    \[2{^ 31}-1, 65535, 65535\].

    [arch] defaults to {!RDNA3}. *)
