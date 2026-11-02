(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Renderers of the C family: kernels as C, Metal, CUDA and HIP source.

    Each renderer writes a linearized kernel as one function of its language.
    The function takes the kernel's parameters in order, each named after its
    name, or [data] and its slot, followed by its shape: [data0_16]
    ({!param_name}). A {!clang} or {!metal} kernel takes its arguments as
    bytes ({!Renderer.t.packed_args}): code generation gives it one parameter,
    [args], from which it loads each argument, and a buffer
    ({!Op.Refparam}) is the address loaded, cast to the buffer's pointer type
    where it is used. Each loop is a [for] loop, and
    each value is a local variable, unless it is used once and cheap to
    repeat, in which case it is written into its user. That holds only while
    its expression nests at most 64 brackets, counting parentheses, square
    brackets and braces; a deeper one is a local variable, so a chain of any
    length stays below the 256 levels that Clang and Metal accept in a
    statement.

    A renderer's {!Renderer.t.render} raises [Invalid_argument] on a kernel that
    holds a node its language cannot write, such as an operation the target
    lacks natively ({!Renderer.t.code_for_op}): code generation decomposes those
    before rendering.

    A NaN constant is written from its bits, so a program stores its sign and
    payload, a signalling NaN staying one.

    A call ({!Op.Call}) of an external function ({!Op.Custom_function}) casts
    the function's symbol, which the program declares and is linked against, to
    the type its arguments and result give it. A call of a body of the program
    (an {!Op.Linear}, as code generation lowers a call of a sink or of a
    program) is a call of a function the program defines before the kernel,
    after the language's own definitions, under the name the body holds, which
    code generation gives it. The function is qualified
    as the language's device functions are ([__device__] in {!cuda}, a device
    attribute in {!hip}). Its parameters are the body's, by slot, and each
    argument is cast to its parameter's type.

    A floating-point division ({!Op.Fdiv}) is the language's [/] on every
    target, whether or not the target lists it among its native operations: that
    list only decides whether code generation turns a reciprocal into a
    division.

    {!Setting.expand_ssa} gives every value a local variable. *)

val param_name : Ops.t -> string
(** [param_name p] is the name a kernel gives its parameter [p].

    Raises [Invalid_argument] if [p] has no {!Ops.type-param_arg}. *)

val clang : Helpers.Target.t -> Renderer.t
(** [clang target] renders C for Clang, for a CPU. [target]'s architecture is
    written [ARCH[-SYSTEM],CPU[,FEATURES]], as ["x86_64,znver2"] or
    ["arm64,apple-m1,-neon"], and chooses the compiler and its flags
    ({!Compiler_cpu.clang}).

    The kernel is one thread with no workgroups. Vectors are Clang's vector
    extension. Half floats are [__fp16], and bfloat16 values are converted to
    and from floats with integer operations. The data types are those of
    {!Dtype.all} without the 8-bit floats, and without {!Dtype.Bfloat16} unless
    the architecture starts with [x86] or [arm].

    Under [x86_64-windows], whose objects call by Windows' convention and
    whose [long] has 32 bits, a 64-bit integer is a [long long], its literals
    end in [ll], and the kernel is declared [__attribute__((sysv_abi))].

    Raises [Invalid_argument] if the architecture has fewer than two fields or
    its machine is not [x86_64], [arm64] or [riscv64]. *)

val metal : Helpers.Target.t -> Renderer.t
(** [metal target] renders Metal Shading Language for an Apple GPU. [target]'s
    architecture is the GPU family, as ["Apple9"] or ["Mac2"].

    The kernel's parameters are the fields of one argument buffer. Families
    [Apple7] and later have 8x8 tensor cores ({!Tc.metal}), and families
    [Apple6] and later have {!Dtype.Bfloat16}. There is no 8-bit float and no
    {!Dtype.Float64}.

    Raises [Invalid_argument] if the architecture starts with [Apple] and is not
    followed by an integer, and [Failure] if MTLCompiler cannot be loaded
    ({!Compiler_metal.compiler}). *)

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
    conversion to an 8-bit float saturates: a value too large for the format,
    an infinity included, becomes its greatest finite value of the same sign,
    where a folded constant keeps an infinity in {!Dtype.Fp8e5m2} and makes it
    a NaN in {!Dtype.Fp8e4m3} ({!Dtype.truncate}).

    Raises [Invalid_argument] if the architecture has no compute capability
    after its first three characters, and [Failure] if no NVRTC is found
    ({!Compiler_cuda.nvrtc}). *)

val hip : Helpers.Target.t -> Renderer.t
(** [hip target] renders HIP C++ for an AMD GPU, compiled with comgr. [target]'s
    architecture is the gfx target, as ["gfx1100"] or
    ["gfx942:sramecc+:xnack-"]; its part before the first [:] names the GPU.

    A launch has at most [2^31 - 1] by 65535 by 65535 workgroups and [2^32 - 1]
    threads on each axis, and a workgroup shares 64 KiB of memory. The tensor
    cores are those of the GPU ({!Tc.amd}). The 8-bit floats are the [fnuz] ones
    on ["gfx942"] and {!Dtype.fp8_ocp} on ["gfx950"], and none elsewhere. A load
    marked ["nontemporal"] bypasses the caches.

    Raises [Failure] if no comgr is found ({!Compiler_amd.hip}). *)
