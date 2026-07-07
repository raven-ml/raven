(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** C-family language renderers.

    {!Renderer.t} values for C-style GPU and CPU backends: CUDA, HIP, Metal,
    OpenCL, and Clang. Each renderer converts a {!Program_spec.program} into
    backend-specific source code via {!Renderer.render}.

    GPU renderers map canonical SPECIAL names ([gidxN], [lidxN], [idxN]) to
    backend-specific workitem expressions (e.g., [blockIdx]/[threadIdx] for
    CUDA, [get_global_id] for OpenCL, [gid]/[lid] for Metal). The CPU renderer
    ({!clang}) has no GPU thread support.

    See {!Renderer} for the renderer interface. *)

(** {1:cpu CPU} *)

val clang : ?native_bf16:bool -> Gpu_target.cpu -> Renderer.t
(** [clang arch] is a Clang/CPU renderer with SIMD support.

    Generates C code for host CPU execution using Clang extensions:
    - [__builtin_convertvector] for vector casts.
    - [__builtin_sqrtf], [__builtin_truncf], etc. for math.

    Emits a fixed-ABI wrapper
    ([void name(const unsigned long long *bufs, const long long *vals)]) around
    the kernel to avoid a libffi dependency at JIT time.

    Device is ["CPU"]. No GPU thread support ({!Renderer.has_local} is [false]).
    No shared memory. [arch] selects dtype capabilities: bfloat16 is native on
    x86_64 and arm64 targets and storage-emulated on riscv64, matching
    tinygrad's Clang renderer policy.

    [native_bf16] (default [true]) states whether the host C compiler accepts
    the [__bf16] storage type (Clang gained it on x86-64 in version 15). When
    [false], bfloat16 is storage-emulated through float32 on every target,
    like riscv64. Runtimes should pass the result of a compiler probe such as
    [Compiler_cpu.supports_bf16].

    {b Note.} Reads environment variables at module initialization:
    - [THREADS]: set to [0] to disable host-side threading (default: enabled).
    - [CPU_COUNT]: override logical CPU count for thread pool size.

    See also {!clang_no_abi} for tests and runtimes that intentionally bypass
    the fixed ABI wrapper. *)

val clang_no_abi : ?native_bf16:bool -> Gpu_target.cpu -> Renderer.t
(** [clang_no_abi arch] is {!clang} without the fixed-ABI wrapper.

    Generates a plain [void name(...)] signature with individual typed
    parameters. This is a low-level renderer used by tests, golden generators,
    and integrations that provide their own native calling convention.

    See also {!clang}. *)

(** {1:opencl OpenCL} *)

val opencl : Gpu_target.opencl -> Renderer.t
(** [opencl arch] is an OpenCL renderer.

    Generates OpenCL C code using [get_group_id], [get_local_id],
    [get_global_id] for thread indexing. Kernel functions are annotated with
    [__kernel], buffers with [__global], and shared memory with [__local].

    Device is ["CL"]. Shared memory limit is 32KB.

    [arch] is tinygrad's comma-separated OpenCL target architecture string. It
    selects dtype capabilities: [cl_khr_fp16] enables float16, [cl_khr_fp64]
    enables float64, fp8 is unsupported, and bfloat16 is represented through
    the OpenCL bfloat16 rewrite path.

    See also {!intel} and {!qcom}. *)

val intel : Gpu_target.opencl -> Renderer.t
(** [intel arch] is an Intel OpenCL renderer.

    {!opencl} variant with [intel_reqd_sub_group_size(8)] for sub-group WMMA
    operations. Uses Intel-specific bf16 conversion intrinsics
    ([intel_convert_bfloat16_as_ushort], [intel_convert_as_bfloat16_float])
    instead of manual bit manipulation.

    Device is ["CL"]. Shared memory limit is 32KB. Tensor cores use 8x8x16 tiles
    with 8 threads. [arch] follows {!opencl}'s extension-based dtype policy.

    See also {!opencl}. *)

val qcom : Renderer.t
(** [qcom] is a Qualcomm OpenCL renderer for Adreno GPUs.

    Identical to {!opencl} in code generation. The separate renderer allows
    device-specific scheduling in codegen passes and carries a stricter dtype
    capability policy: no fp8, bfloat16, or float64. Float16 is enabled only
    when both [IMAGE] and [FLOAT16] are set, matching tinygrad's QCOM policy.

    Device is ["QCOM"]. Shared memory limit is 32KB. *)

(** {1:metal Metal} *)

val metal : Gpu_target.metal -> Renderer.t
(** [metal arch] is a Metal Shading Language renderer.

    Generates MSL code with [threadgroup_position_in_grid] and
    [thread_position_in_threadgroup] attributes for thread indexing. Uses
    [threadgroup] storage for shared memory and [threadgroup_barrier] for
    synchronization.

    Device is ["METAL"]. Shared memory limit is 32KB.

    [arch] selects tensor core capabilities, matching tinygrad's Metal target
    policy: Apple GPU family 7 and newer expose simdgroup matrix operations;
    older Apple families and Mac families do not. *)

(** {1:cuda CUDA} *)

val cuda : Gpu_target.cuda -> Renderer.t
(** [cuda arch] is a CUDA renderer for NVIDIA GPUs.

    Generates CUDA C code using [blockIdx]/[threadIdx] for thread indexing. Uses
    [extern "C" __global__] with [__launch_bounds__] when local dimensions are
    known. Half-precision intrinsics ([hexp2], [hlog2], [hsqrt], etc.) are used
    for float16 and bfloat16 transcendentals.

    Device is ["CUDA"]. Shared memory limit is 48KB. Global grid max is
    \[2{^ 31}-1, 65535, 65535\]. Local block max is \[1024, 1024, 64\].

    [arch] selects tensor core and dtype capabilities:
    - {!Gpu_target.SM75}: 8x16x8 tiles, f16 input.
    - {!Gpu_target.SM80}: 8x16x16 tiles (f16, bf16) + 8x16x8 (f16, tf32).
    - {!Gpu_target.SM89}: {!Gpu_target.SM80} + 8x16x32 tiles for fp8.
    - {!Gpu_target.SM90}: same tiles as {!Gpu_target.SM89}. *)

(** {1:amd AMD} *)

val amd : Gpu_target.amd -> Renderer.t
(** [amd arch] is an AMD HIP renderer.

    Generates HIP code using OCKL work item functions ([__ockl_get_group_id],
    [__ockl_get_local_id]) for thread indexing and OCML transcendentals
    ([__ocml_*_f\{16,32,64\}]) for math. Uses [__builtin_amdgcn_fence] for
    release-acquire barriers (unlike CUDA's [__syncthreads], AMD barriers do not
    imply a fence). Bfloat16 is emulated via software bit manipulation. Fp8 uses
    [__builtin_amdgcn_cvt_*] intrinsics.

    Device is ["AMD"]. Shared memory limit is 64KB. Global grid max is
    \[2{^ 31}-1, 65535, 65535\].

    [arch] selects tensor core and dtype capabilities:
    - {!Gpu_target.RDNA3}: WMMA 16x16x16, 32-thread wavefront.
    - {!Gpu_target.RDNA4}: WMMA 16x16x16, gfx12 builtins.
    - {!Gpu_target.CDNA3}: MFMA bf16, 16x16x16, 64-thread wavefront.
    - {!Gpu_target.CDNA4}: MFMA fp8/bf16/f16, 16x16x128/32/16. *)
