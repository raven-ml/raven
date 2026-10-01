(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Renderers of the C family: kernels as C, Metal, CUDA and HIP source.

    Each renderer writes a linearized kernel as one function of its language.
    The function takes the kernel's parameters in order, each named after its
    name, or [data] and its slot, followed by its shape: [data0_16]. Each loop
    is a [for] loop, and each value is a local variable, unless it is used once
    and cheap to repeat, in which case it is written into its user.

    A renderer's {!Renderer.t.render} raises [Invalid_argument] on a kernel that
    holds a node its language cannot write, such as an operation the target
    lacks natively ({!Renderer.t.code_for_op}): code generation decomposes those
    before rendering.

    A floating-point division ({!Op.Fdiv}) is the language's [/] on every
    target, whether or not the target lists it among its native operations: that
    list only decides whether code generation turns a reciprocal into a
    division.

    Two settings are read once, when they are first needed:
    - [EXPAND_SSA] (default [0]): when nonzero, every value is a local variable;
    - [ALIGNED] (default [1]): when zero, {!clang}'s vector types are aligned to
      one byte, so that buffers at any address can be passed. *)

val clang : Helpers.Target.t -> Renderer.t
(** [clang target] renders C for Clang, for a CPU. [target]'s architecture is
    written [ARCH,CPU[,FEATURES]], as ["x86_64,znver2"] or
    ["arm64,apple-m1,-neon"], and chooses the compiler's flags.

    The kernel is one thread with no workgroups. Vectors are Clang's vector
    extension. Half floats are [__fp16], and bfloat16 values are converted to
    and from floats with integer operations. The data types are those of
    {!Dtype.all} without the 8-bit floats, and without {!Dtype.Bfloat16} unless
    the architecture starts with [x86] or [arm]. On Windows the kernel uses
    Microsoft's calling convention.

    Raises [Invalid_argument] if the architecture has fewer than two fields or
    is not [x86_64], [arm64] or [riscv64]. *)

val metal : Helpers.Target.t -> Renderer.t
(** [metal target] renders Metal Shading Language for an Apple GPU. [target]'s
    architecture is the GPU family, as ["Apple9"] or ["Mac2"].

    The kernel's parameters are the fields of one argument buffer. Families
    [Apple7] and later have 8x8 tensor cores ({!Tc.metal}), and families
    [Apple6] and later have {!Dtype.Bfloat16}. There is no 8-bit float and no
    {!Dtype.Float64}.

    Raises [Invalid_argument] if the architecture starts with [Apple] and is not
    followed by an integer. *)

val cuda : Helpers.Target.t -> Renderer.t
(** [cuda target] renders CUDA C++ for an NVIDIA GPU, compiled with NVRTC.
    [target]'s architecture is the compute capability, as ["sm_89"]. The
    compiler produces a cubin, the binary of that architecture, which the
    driver loads without translating it. Its cache table is named after the
    device.

    A workgroup has at most 1024 by 1024 by 64 threads, a launch at most
    [2^31 - 1] by 65535 by 65535 workgroups, and a workgroup shares 48 KiB of
    memory. The tensor cores are those of the compute capability ({!Tc.cuda}).
    {!Dtype.Float16} needs capability 53, {!Dtype.Bfloat16} 80, and the 8-bit
    floats {!Dtype.fp8_ocp} 89. The [fnuz] floats are not supported. A
    conversion to an 8-bit float keeps an infinity special: it stays an infinity
    in {!Dtype.Fp8e5m2}, and becomes a NaN of its sign in {!Dtype.Fp8e4m3},
    which has no infinity. A finite value too large for the format becomes its
    greatest finite value of the same sign.

    Raises [Invalid_argument] if the architecture has no compute capability
    after its first three characters. *)

val hip : Helpers.Target.t -> Renderer.t
(** [hip target] renders HIP C++ for an AMD GPU, compiled with comgr. [target]'s
    architecture is the gfx target, as ["gfx1100"] or
    ["gfx942:sramecc+:xnack-"]; its part before the first [:] names the GPU.

    A launch has at most [2^31 - 1] by 65535 by 65535 workgroups and [2^32 - 1]
    threads on each axis, and a workgroup shares 64 KiB of memory. The tensor
    cores are those of the GPU ({!Tc.amd}). The 8-bit floats are the [fnuz] ones
    on ["gfx942"] and {!Dtype.fp8_ocp} on ["gfx950"], and none elsewhere. A load
    marked ["nontemporal"] bypasses the caches. *)
