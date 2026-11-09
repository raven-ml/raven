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
    attribute in {!hip}), and inline unless a call of it is compiled on its
    own ({!Ops.type-call_info}, as a launch's calls are): such a function is
    kept out of line ([__noinline__] in {!cuda}, the [noinline] attribute
    elsewhere), so that the compiler compiles it once however many calls it
    has. Its parameters are the body's, by slot, and each argument is passed
    as written, converted to its parameter's type as the language converts an
    argument. The choices {!cuda} makes for a program, its
    asynchronous copies and matrix loads, read the bodies it calls too.

    A sum, difference, product, negation, left shift or multiply-add of a
    signed integer wraps at its width, as {!Ops.exec_alu} computes it. These
    languages leave the overflow of a signed integer undefined, and a left
    shift of a negative one, which a compiler may take for impossible: such an
    operation is computed in the unsigned integer of [int]'s width, or of
    [long long]'s for a 64-bit one, and converted back, unless its operands'
    bounds ({!Ops.operand_bounds}) keep each exact result it forms in that
    type and a shifted value from being negative, as an index's do.

    An integer division by 0, a signed division of the least value by -1 and
    a shift by a count at or past the width are what {!Ops.exec_alu} computes
    too: a quotient by 0 is 0 and a remainder by 0 the dividend; the least
    value's quotient by -1 is itself and its remainder 0; a left shift and an
    unsigned right shift give 0, and a signed right shift 0 or -1. The
    languages leave each undefined, so a quotient or remainder whose
    divisor's bounds reach 0 divides by 1 where the divisor is 0 and selects
    its value there; one whose operands' bounds reach the least value and -1
    divides their magnitudes in the unsigned integer of its width and takes
    its sign after; and a shift whose count's own bounds leave the width
    shifts by the count's low bits and selects the result past it.

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
    extension, and an operation computes at most 64 lanes of one at once
    ({!Renderer.t.vector_lanes}). Half floats are [__fp16], and bfloat16
    values are converted to and from floats with integer operations. The data
    types are those of {!Dtype.all} without the 8-bit floats, and without
    {!Dtype.Bfloat16} unless the architecture starts with [x86] or [arm]. Its
    vector registers are those of Clang's processor
    ({!Compiler_cpu.registers}). On [arm64] with the [FEATURE] [fp16],
    FEAT_FP16's half arithmetic, a half multiply-add, of one lane or many, is
    LLVM's fused multiply-add of halves, rounded once ({!Renderer.t.fma}).

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
    followed by an integer, and [Failure] if MTLCompiler is not found
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
    From capability 80 a store into shared memory of a 16-byte load from global
    memory is an asynchronous copy, PTX's [cp.async]
    ({!Renderer.t.async_copy}), and each barrier of a kernel holding one first
    waits for the thread's copies.
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

(**/**)

val cuda_sync : Helpers.Target.t -> Renderer.t
(** [cuda_sync target] is {!without_compiler}'s {!cuda} without asynchronous
    copies, named apart, as a capability below 80 renders: a test's kernel
    staged in one stage of copies through registers. *)

val without_compiler : Helpers.Target.t -> Renderer.t
(** [without_compiler target] is the renderer {!clang}, {!metal}, {!cuda} or
    {!hip} makes for [target], as its renderer names it, with a compiler that
    returns its source ({!Renderer.Compiler.v}): it is made without the
    target's toolchain, so a test lowers and renders for any target on any
    machine.

    Raises [Invalid_argument] as that renderer does, or if [target] names
    another renderer. *)
