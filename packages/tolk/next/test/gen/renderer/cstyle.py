"""Goldens of tinygrad/renderer/cstyle.py: the source that Clang, Metal, CUDA
and HIP write for linearized kernels, the rewrites each target needs last, and
what each renderer declares for its target.

A kernel is linearized as `to_program` linearizes it, and the input golden
`kernels` holds one `Ops.LINEAR` per case, whose arg is the case's name and
whose sources are its nodes in order. The table `cases` gives each case's
position among them, the target and the renderer, and the settings it is
rendered under; the source of case `c` is the text golden `c`.

The rewrites are a table `rewrites`, whose rows give the position of an input
in `rewrite_inputs` and of its result in `rewritten`. The table `declarations`
holds what each renderer declares for a target, and `written` how each writes
its native operations.
"""

from dataclasses import replace
import os

import tinygrad.runtime.support.compiler_amd as compiler_amd

# Making a HIP renderer makes its compiler, which asserts that comgr is loaded;
# comgr is absent where the goldens are generated, and rendering needs none.
compiler_amd.c.DLL._loaded_.add(compiler_amd.comgr.dll.nm)

import tinygrad.runtime.ops_metal as ops_metal

# Making a Metal renderer makes its compiler, which starts Metal's code
# generation service, whose threads make the processes forked for each golden
# crash at random; rendering needs no compiler.
ops_metal.MetalCompiler.support.MTLCodeGenServiceCreate = lambda name: None

from golden import graph, table, text
from tinygrad import Tensor, Variable, dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_alloc_to_buf, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, Target
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.uop.ops import AxisType, GroupOp, KernelInfo, Ops, UOp, graph_rewrite

RENDERERS = {"CLANG": ClangRenderer, "METAL": MetalRenderer, "CUDA": CUDARenderer, "HIP": HIPRenderer}

# Each target the suite renders for, by its name in the golden names. A table
# writes a target as its device, renderer and architecture.
TARGETS = {
    "clang": Target("CPU", "CLANG", "x86_64,x86-64"),
    "metal": Target("METAL", "METAL", "Apple9"),
    "cuda": Target("CUDA", "CUDA", "sm_89"),
    "hip": Target("AMD", "HIP", "gfx1100"),
    "hip_rdna4": Target("AMD", "HIP", "gfx1201"),
    "hip_cdna3": Target("AMD", "HIP", "gfx942"),
    "hip_cdna4": Target("AMD", "HIP", "gfx950"),
}
TARGET_COLUMNS = ["device", "renderer", "arch"]


def cells(t): return (t.device, t.renderer, t.arch)
def renderer_for(t): return RENDERERS[t.renderer](t)
def renderer(name): return renderer_for(TARGETS[name])


def empty(*shape, dtype=dtypes.float):
    return Tensor.empty(*shape, device="NULL", dtype=dtype)


def tensors(build, opts=None, index=None):
    """The one kernel that realizing the tensors `build()` returns compiles, or
    the kernel at `index` among them, linearized by the case's renderer, with
    `opts` in place of the heuristic's if given."""
    def make(ren):
        lin, _ = Tensor.linear_with_vars(*build())
        asts = [c.src[0] for c in lin.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
        (ast,) = asts if index is None else [asts[index]]
        if opts is not None: ast = ast.replace(arg=replace(ast.arg, opts_to_apply=tuple(opts)))
        return lowered(ast, ren, optimize=ast.tag is None)
    return make


def lowered(sink, ren, optimize=False):
    full = full_rewrite_to_sink(sink, ren, optimize=optimize)
    return line_rewrite(linearize(full), pm_linearize_cleanups + pm_alloc_to_buf)


def ast(make):
    """The kernel of the sink `make()` builds, lowered without optimisation."""
    return lambda ren: lowered(make(), ren)


def raw(make):
    """The nodes of the sink `make()` builds, in order, rendered as they are."""
    return lambda ren: make().toposort()


def raw_for(make):
    """The nodes of the sink `make(ren)` builds for the case's renderer."""
    return lambda ren: make(ren).toposort()


# Kernels

def up(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.UPCAST))
def local(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.LOCAL))


def mixed(dt):
    """Converts a float to `dt` and back, with arithmetic in `dt` between."""
    def build():
        x, y = empty(16), empty(16, dtype=dt)
        out = x.cast(dt) & y if dt == dtypes.bool else (x.cast(dt) + y) * 3
        return [out.cast(dtypes.float)]
    return tensors(build)


def transcendental(dt):
    def build():
        # One element, so that each decomposition is written once.
        x = empty(1, dtype=dt)
        return [x.exp2() + x.log2() + x.sin() + x.sqrt() + x.reciprocal() + x.trunc()]
    return tensors(build)


def specials(dt):
    def build():
        x = empty(16, dtype=dt)
        return [(x > 0).where(x, float("inf")) + (x < -1).where(float("nan"), x) + (x < 1).where(float("-inf"), x)]
    return tensors(build)


def chain(op):
    """Five operations of `op` in a row, each on the last's result."""
    dtype = dtypes.bool if op in ("or", "xor", "and") else dtypes.float
    def build():
        ret = empty(1, dtype=dtype)
        for _ in range(5):
            other = empty(1, dtype=dtype)
            ret = {"add": ret + other, "mul": ret * other, "sub": ret - other, "xor": ret ^ other, "or": ret | other,
                   "and": ret & other}[op]
        return [ret]
    return tensors(build)


def tensor_core(ren, i):
    """A matrix product of one tile of the renderer's `i`th tensor core, over
    two steps of its reduction."""
    core = ren.tensor_cores[i]
    n, m, k = core.dims
    def build():
        a, b = empty(m, 2 * k, dtype=core.dtype_in), empty(2 * k, n, dtype=core.dtype_in)
        return [a.matmul(b, dtype=core.dtype_out)]
    # Products of floats on tensor cores round their operands to tf32, which a
    # program asks for.
    with Context(ALLOW_TF32=1):
        return tensors(build, opts=[Opt(OptOps.TC, 0, (i, 0, 1))])(ren)


def atari(): return empty(210, 160, dtype=dtypes.uint8)


n = Variable("n", 1, 64)


def gated_store_on_threads():
    a = UOp.param(0, dtypes.int, 4)
    lidx0 = UOp.special(4, "lidx0")
    return a.index(lidx0.valid(lidx0.ne(0))).store(UOp.const(1).cast(dtypes.int)).sink(arg=KernelInfo())


def gated_store_in_loop():
    a = UOp.param(0, dtypes.int, 16)
    r = UOp.range(16, 0, AxisType.LOOP)
    return a.index(r.valid(r < 8)).store(r.cast(dtypes.int)).end(r).sink(arg=KernelInfo())


def inline_const_alu():
    """Ops.MAX of a load and the int one above the least, from
    test_renderer_failures.py."""
    a, b = UOp.param(0, dtypes.int, 1), UOp.param(1, dtypes.int, 1)
    idx = UOp.const(0)
    alu = b.index(idx).load().alu(Ops.MAX, UOp.const(dtypes.int.min + 1).cast(dtypes.int))
    return UOp.store(a.index(idx), alu).sink(arg=KernelInfo())


def call_out():
    """A call of the function pointer in F[0] that writes C[0], from
    device/cpu/test_call.py."""
    F, C = UOp.param(0, dtypes.uint64, 1), UOp.param(1, dtypes.int, 2)
    call = UOp.custom_function("callback", F[0].load()).call(UOp.const(3).cast(dtypes.int), C[0], ret_dtype=dtypes.void)
    return C.after(call)[1].store(C.after(call)[0].load() + 1).sink(arg=KernelInfo(name="call_out"))


def call_ret():
    F, C = UOp.param(0, dtypes.uint64, 1), UOp.param(1, dtypes.int, 1)
    val = UOp.custom_function("callback", F[0].load()).call(UOp.const(21).cast(dtypes.int), ret_dtype=dtypes.int)
    return C[0].store(val * 2).sink(arg=KernelInfo(name="call_ret"))


def call_stack():
    """A call whose argument is the address of a register, from
    null/test_call.py."""
    slot = UOp.placeholder((1,), dtypes.uint32, addrspace=AddrSpace.REG)
    return UOp.custom_function("callback", UOp.const(0, dtypes.uint64)).call(slot[0], ret_dtype=dtypes.void) \
        .sink(arg=KernelInfo("call_stack"))


def custom():
    a, b = UOp.param(0, dtypes.float, 4), UOp.param(1, dtypes.float, 4)
    r = UOp.range(4, 0, AxisType.LOOP)
    val = b.index(r).load()
    inline = UOp(Ops.CUSTOMI, src=(val,), arg=("__builtin_fabsf({0})", dtypes.float))
    named = UOp(Ops.CUSTOM, src=(inline, val), arg=("__builtin_fmaxf({0}, {1})", dtypes.float))
    return a.index(r).store(named).end(r).sink(arg=KernelInfo(name="custom"))


def volatile():
    a = UOp.param(0, dtypes.int, 4, volatile=True)
    b = UOp.param(1, dtypes.int, 4, volatile=True)
    r = UOp.range(4, 0, AxisType.LOOP)
    return a.index(r).store(b.index(r).load() + 1).end(r).sink(arg=KernelInfo(name="volatile_add"))


def scalar_params():
    """A buffer and variables of 32 and 64 bits."""
    a = UOp.param(0, dtypes.long, 4)
    v = UOp.variable("start", 0, 8, dtype=dtypes.int)
    s = UOp.variable("offset", 0, 2**40, dtype=dtypes.long)
    r = UOp.range(4, 0, AxisType.LOOP)
    return a.index(r).store((r + v).cast(dtypes.long) + s).end(r).sink(arg=KernelInfo(name="scalars"))


def nontemporal():
    a, b = UOp.param(0, dtypes.float, 64), UOp.param(1, dtypes.float, 64)
    r = UOp.range(16, 0, AxisType.LOOP)
    scalar = b.index(r).load(arg="nontemporal")
    return a.index(r).store(scalar * 2).end(r).sink(arg=KernelInfo(name="nontemporal"))


def binary():
    """A constant byte array, read at an index."""
    a = UOp.param(0, dtypes.uchar, 4)
    table = UOp(Ops.BINARY, arg=b"\x00\x7f\x80\xff")
    r = UOp.range(4, 0, AxisType.LOOP)
    return a.index(r).store(table.index(r).load()).end(r).sink(arg=KernelInfo(name="table"))


def int32(v): return UOp.const(v).cast(dtypes.int)


def unbounded_loop():
    """A loop with no trip count, left by its bottom test."""
    a = UOp.param(0, dtypes.int, 1)
    r = UOp(Ops.RANGE, src=(UOp(Ops.STACK),), arg=(0, AxisType.LOOP))
    cell = a.after(r).index(int32(0))
    step = cell.store(cell.load() + int32(1))
    done = a.after(step).index(int32(0)).load() < int32(10)
    return UOp(Ops.BACKEDGE, src=(step, r, done)).sink(arg=KernelInfo(name="unbounded"))


def reinterpreted(retype):
    """A load of four chars as one uint, through a cast or a bitcast of their
    address, from null/test_renderer_failures.py."""
    b = UOp.param(0, dtypes.char, 4)
    idx = b.index(int32(0))
    return idx.store(retype(idx).load() & UOp.const(0xffffff00).cast(dtypes.uint32)).sink(arg=KernelInfo(name="packed"))


def dynamic_lane():
    """A lane of a vector chosen by a variable."""
    a = UOp.param(0, dtypes.float, 1)
    lane = UOp.variable("lane", 0, 3, dtype=dtypes.int)
    lanes = UOp(Ops.STACK, src=tuple(UOp.const(float(i)).cast(dtypes.float) for i in (1, 2, 3, 4)))
    return a.index(int32(0)).store(lanes.index(lane)).sink(arg=KernelInfo(name="lane"))


def vector_cast():
    a, b = UOp.param(0, dtypes.int, 4), UOp.param(1, dtypes.float, 4)
    def lanes(buf): return UOp(Ops.SHRINK, src=(buf, int32(0), int32(4)))
    return lanes(a).store(lanes(b).load().cast(dtypes.int)).sink(arg=KernelInfo(name="vector_cast"))


def register_cast():
    """Two uint registers read as one ulong through a cast of their address,
    which C alone allows."""
    a = UOp.param(0, dtypes.uint64, 1)
    reg = UOp.placeholder((2,), dtypes.uint32, addrspace=AddrSpace.REG)
    filled = UOp.group(*[reg.index(int32(i)).store(UOp.const(i + 1).cast(dtypes.uint32)) for i in range(2)])
    wide = reg.after(filled).index(int32(0)).cast(dtypes.uint64).load()
    return a.index(int32(0)).store(wide).sink(arg=KernelInfo(name="register_cast"))


CONSTANTS = [(dtypes.bool, True), (dtypes.bool, False), (dtypes.char, -3), (dtypes.uchar, 200), (dtypes.short, -300),
             (dtypes.ushort, 60000), (dtypes.int, 42), (dtypes.uint, 42), (dtypes.uint, -1), (dtypes.long, 12345),
             (dtypes.ulong, 42), (dtypes.ulong, -1), (dtypes.half, 1.5), (dtypes.bfloat16, 1.5), (dtypes.float, 3.14),
             (dtypes.double, 3.14), *[(dt, 1.0) for dt in dtypes.fp8s]]


def constants(ren):
    """A store of a constant of each data type the renderer has, each to a
    parameter of its own."""
    # A boolean literal is committed already, so its cast is written out.
    stores = [UOp.param(i, dt, 1).index(int32(0)).store(UOp(Ops.CAST, src=(UOp.const(v),), arg=dt))
              for i, (dt, v) in enumerate(c for c in CONSTANTS if c[0] in ren.supported_dtypes())]
    return UOp.sink(*stores, arg=KernelInfo(name="constants"))


def named_params():
    """Parameters with names, one holding the separator of a device's name."""
    src = UOp.param(0, dtypes.float, 4, name="input:tile")
    dst = UOp.param(1, dtypes.float, 4, name="output:tile")
    v = UOp.variable("for", 0, 8, dtype=dtypes.int)
    r = UOp.range(4, 0, AxisType.LOOP)
    return dst.index(r).store(src.index(r).load() + v.cast(dtypes.float)).end(r).sink(arg=KernelInfo(name="named"))


# The cases of each target: its name and how its kernel is made from the
# renderer.

def dtype_name(dt): return dt.name.replace(" ", "_").replace("__", "")


def common(ren):
    cases = [
        ("add", tensors(lambda: [empty(64) + empty(64)])),
        ("sum", tensors(lambda: [empty(256).sum()])),
        ("matmul", tensors(lambda: [empty(16, 16) @ empty(16, 16)])),
        ("matmul_upcasted", tensors(lambda: [empty(16, 16) @ empty(16, 16)], opts=[up(0, 4), up(1, 4)])),
        ("padded", tensors(lambda: [empty(14).pad((1, 1)) + 1])),
        ("where_max", tensors(lambda: [((a := empty(16)) < (b := empty(16))).where(a, a.maximum(b))])),
        ("bitcast", tensors(lambda: [empty(16).bitcast(dtypes.int) + 1])),
        ("idiv", tensors(lambda: [(a := empty(16, dtype=dtypes.int)) // 7 + a % 7 + a // empty(16, dtype=dtypes.int)])),
        ("shrunk", tensors(lambda: [empty(64, 4)[:n.bind(10)] + 1])),
        ("rand", tensors(lambda: [Tensor.rand(16, device="NULL")], index=-1)),
        ("inline_const_alu", ast(inline_const_alu)),
        ("gated_store_in_loop", ast(gated_store_in_loop)),
        ("custom", ast(custom)),
        ("volatile", ast(volatile)),
        ("scalar_params", ast(scalar_params)),
        ("unbounded_loop", raw(unbounded_loop)),
        ("packed_cast", raw(lambda: reinterpreted(lambda idx: idx.cast(dtypes.uint32)))),
        ("packed_bitcast", raw(lambda: reinterpreted(lambda idx: idx.bitcast(dtypes.uint32)))),
        ("dynamic_lane", raw(dynamic_lane)),
        ("vector_cast", raw(vector_cast)),
        ("constants", raw_for(constants)),
        ("named_params", ast(named_params)),
    ]
    cases += [(f"dtype_{dtype_name(dt)}", mixed(dt)) for dt in dtypes.all if dt in ren.supported_dtypes()]
    floats = [dt for dt in (dtypes.half, dtypes.bfloat16, dtypes.float, dtypes.double) if dt in ren.supported_dtypes()]
    cases += [(f"transcendental_{dtype_name(dt)}", transcendental(dt)) for dt in floats]
    fp8s = [dt for dt in dtypes.fp8s if dt in ren.supported_dtypes()]
    cases += [(f"inf_nan_{dtype_name(dt)}", specials(dt)) for dt in floats + fp8s]
    return cases


def gpu():
    return [
        ("gated_store_on_threads", ast(gated_store_on_threads)),
        ("matmul_locals", tensors(lambda: [empty(64, 64) @ empty(64, 64)],
                                  opts=[up(0, 4), up(1, 4), local(0, 4), local(1, 4)])),
        ("group_reduce", tensors(lambda: [empty(64, 64) @ empty(64, 64)], opts=[Opt(OptOps.SPLIT, 2, (4, AxisType.LOCAL))])),
        ("group_reduce_upcasted", tensors(lambda: [empty(64, 64) @ empty(64, 64)],
                                          opts=[Opt(OptOps.SPLIT, 2, (16, AxisType.LOCAL)), up(0, 4)])),
    ]


def tensor_cores(ren):
    return [(f"tc_{dtype_name(core.dtype_in)}_{dtype_name(core.dtype_out)}_{'_'.join(map(str, core.dims))}",
             lambda ren, i=i: tensor_core(ren, i)) for i, core in enumerate(ren.tensor_cores)]


def cases_of(name):
    ren = renderer(name)
    if name == "clang":
        return common(ren) + [
            (f"chain_{op}", chain(op)) for op in ("add", "mul", "sub", "xor", "or", "and")] + [
            ("call_out", ast(call_out)), ("call_ret", ast(call_ret)), ("call_stack", ast(call_stack)),
            ("register_cast", raw(register_cast)),
            # The kernels of null/test_compile_failures.py
            *[(f"interpolate_atari_{i}", tensors(lambda: [atari().interpolate((64, 64))], index=i)) for i in range(2)],
            ("add_max_uchar", tensors(lambda: [(empty(1024, dtype=dtypes.uint8) + empty(1024, dtype=dtypes.uint8)).max()])),
            ("table", ast(binary))]
    if name in ("metal", "cuda", "hip"):
        extra = [("nontemporal", ast(nontemporal))] if name == "hip" else []
        return common(ren) + gpu() + tensor_cores(ren) + extra
    # The other AMD targets differ from gfx1100 in their tensor cores and 8-bit
    # and bfloat16 floats.
    kinds = [dt for dt in (dtypes.bfloat16, *dtypes.fp8s) if dt in ren.supported_dtypes()]
    return [(f"dtype_{dtype_name(dt)}", mixed(dt)) for dt in kinds] + \
        [(f"inf_nan_{dtype_name(dt)}", specials(dt)) for dt in kinds] + \
        [("constants", raw_for(constants))] + tensor_cores(ren)


# Settings read once per process: a case rendered under one is a case of its
# own, and the golden that renders it sets the variable first.
SETTINGS = [
    ("clang", "add", "EXPAND_SSA=1"),
    ("clang", "matmul_upcasted", "EXPAND_SSA=1"),
    ("clang", "add", "ALIGNED=0"),
    ("clang", "dtype_bool", "ALIGNED=0"),
    ("cuda", "matmul_locals", "EXPAND_SSA=1"),
]


def setting_suffix(setting):
    return {"EXPAND_SSA=1": "expand_ssa", "ALIGNED=0": "unaligned"}[setting]


def all_cases():
    """Every case: its golden name, its target name, the kernel's maker and
    its setting."""
    out = []
    for name in TARGETS:
        for case, make in cases_of(name):
            out.append((f"{name}_{case}", name, make, ""))
    makers = {c: m for c, _, m, _ in out}
    for name, case, setting in SETTINGS:
        out.append((f"{name}_{case}_{setting_suffix(setting)}", name, makers[f"{name}_{case}"], setting))
    return out


CASES = all_cases()


@graph
def kernels():
    return UOp.sink(*[UOp(Ops.LINEAR, src=tuple(make(renderer(name))), arg=case) for case, name, make, _ in CASES])


@table
def cases():
    rows = [(case, str(i), *cells(TARGETS[name]), setting or "-") for i, (case, name, _, setting) in enumerate(CASES)]
    return ["case", "kernel", *TARGET_COLUMNS, "setting"], rows


def source(case, name, make, setting):
    def body():
        if setting:
            key, value = setting.split("=")
            os.environ[key] = value
        ren = renderer(name)
        return ren.render(list(make(ren)))
    body.__name__ = case
    return body


for case, name, make, setting in CASES:
    text(source(case, name, make, setting))


# The rewrites each target needs last: every input is rewritten for every
# target, since a rewrite that leaves a graph alone is a fact too.

def stored(dtype, value):
    return UOp.param(0, dtype, 16).index(UOp.range(16, 0, AxisType.LOOP)).store(value)


def loaded(slot, dtype):
    return UOp.param(slot, dtype, 16).index(UOp.range(16, 0, AxisType.LOOP)).load()


def alu(dtype, op): return stored(dtype, loaded(1, dtype).alu(op, loaded(2, dtype)))
def cast(src, dst): return stored(dst, loaded(1, src).cast(dst))


def product(dtype):
    """A tensor core product of eight elements per operand."""
    def operand(slot): return UOp(Ops.STACK, src=tuple(loaded(slot, dtype).alu(Ops.ADD, UOp.const(i).cast(dtype))
                                                        for i in range(8)))
    acc = UOp(Ops.STACK, src=tuple(loaded(3, dtypes.float) for _ in range(4)))
    return UOp.wmma(operand(1), operand(2), acc, (16, 16, 32), 64)


REWRITE_INPUTS = [
    ("bf16_add", stored(dtypes.bfloat16, loaded(1, dtypes.bfloat16) + 1)),
    ("bf16_mul", alu(dtypes.bfloat16, Ops.MUL)),
    ("bf16_less", stored(dtypes.bool, loaded(1, dtypes.bfloat16) < loaded(2, dtypes.bfloat16))),
    ("bf16_where", stored(dtypes.bfloat16, (loaded(3, dtypes.float) < 0).where(loaded(1, dtypes.bfloat16),
                                                                             loaded(2, dtypes.bfloat16)))),
    ("bf16_sqrt", stored(dtypes.bfloat16, loaded(1, dtypes.bfloat16).sqrt())),
    ("bf16_exp2", stored(dtypes.bfloat16, loaded(1, dtypes.bfloat16).exp2())),
    ("bf16_to_float", cast(dtypes.bfloat16, dtypes.float)),
    ("float_to_bf16", cast(dtypes.float, dtypes.bfloat16)),
    ("half_to_bf16", cast(dtypes.half, dtypes.bfloat16)),
    ("bf16_to_int", cast(dtypes.bfloat16, dtypes.int)),
    ("double_to_half", cast(dtypes.double, dtypes.half)),
    ("double_to_float", cast(dtypes.double, dtypes.float)),
    ("half_add", alu(dtypes.half, Ops.ADD)),
    ("e4m3_add", stored(dtypes.fp8e4m3, loaded(1, dtypes.fp8e4m3) + 1)),
    ("e4m3_less", stored(dtypes.bool, loaded(1, dtypes.fp8e4m3) < loaded(2, dtypes.fp8e4m3))),
    ("e4m3_to_e5m2", cast(dtypes.fp8e4m3, dtypes.fp8e5m2)),
    ("e4m3_to_e4m3", stored(dtypes.fp8e4m3, UOp(Ops.CAST, src=(loaded(1, dtypes.fp8e4m3),), arg=dtypes.fp8e4m3))),
    ("e4m3_to_e4m3fnuz", cast(dtypes.fp8e4m3, dtypes.fp8e4m3fnuz)),
    ("float_to_e4m3", cast(dtypes.float, dtypes.fp8e4m3)),
    ("half_to_e5m2", cast(dtypes.half, dtypes.fp8e5m2)),
    ("e5m2_to_int", cast(dtypes.fp8e5m2, dtypes.int)),
    ("e4m3fnuz_mul", alu(dtypes.fp8e4m3fnuz, Ops.MUL)),
    ("e4m3_product", product(dtypes.fp8e4m3)),
    ("e5m2fnuz_product", product(dtypes.fp8e5m2fnuz)),
    ("half_product", product(dtypes.half)),
]

REWRITES = [(f"{name}_{case}", name, i) for name in TARGETS for i, (case, _) in enumerate(REWRITE_INPUTS)]


@graph
def rewrite_inputs(): return UOp.sink(*[u for _, u in REWRITE_INPUTS])


@table
def rewrites():
    return ["case", *TARGET_COLUMNS, "input", "output"], \
        [(case, *cells(TARGETS[name]), str(i), str(j)) for j, (case, name, i) in enumerate(REWRITES)]


@graph
def rewritten():
    renderers = {name: renderer(name) for name in TARGETS}
    return UOp.sink(*[graph_rewrite(REWRITE_INPUTS[i][1], renderers[name].extra_matcher) for _, name, i in REWRITES])


# What each renderer declares for its target

DECLARED = [
    *[Target("CPU", "CLANG", arch) for arch in ("x86_64,x86-64", "x86_64,znver2,avx", "arm64,apple-m1",
                                                 "arm64,generic,-neon", "riscv64,native")],
    *[Target("METAL", "METAL", arch) for arch in ("Apple5", "Apple6", "Apple7", "Apple9", "Mac2")],
    *[Target("CUDA", "CUDA", f"sm_{v}") for v in (52, 53, 75, 80, 86, 89, 90)],
    Target("NV", "CUDA", "sm_89"),
    *[Target("AMD", "HIP", arch) for arch in ("gfx90a", "gfx1100", "gfx1201", "gfx942", "gfx942:sramecc+:xnack-",
                                               "gfx950")],
]


def sizes(xs): return "-" if xs is None else " ".join(map(str, xs))


@table
def declarations():
    rows = []
    for t in DECLARED:
        r = renderer_for(t)
        rows.append((*cells(t), r.supports_float4, r.has_local, r.has_shared, sizes(r.global_max),
                     sizes(r.local_max), sizes(r.global_prod_max), r.shared_max,
                     " | ".join(map(repr, r.tensor_cores)) or "-",
                     " ".join(sorted(op.name for op in r.code_for_op)),
                     " ".join(repr(d) for d in dtypes.all if d in r.supported_dtypes()),
                     r.compiler.cachekey))
    return [*TARGET_COLUMNS, "supports_float4", "has_local", "has_shared", "global_max", "local_max",
            "global_prod_max", "shared_max", "tensor_cores", "code_for_op", "supported", "cachekey"], rows


# How each renderer writes the operations it has natively: the operands are
# written a, b and c.

WRITTEN_DTYPES = [dtypes.half, dtypes.bfloat16, dtypes.float, dtypes.double, dtypes.int]


@table
def written():
    rows = []
    for name in ("clang", "metal", "cuda", "hip"):
        r = renderer(name)
        for op, fn in r.code_for_op.items():
            arity = 1 if op in GroupOp.Unary else 3 if op in GroupOp.Ternary else 2
            for dt in WRITTEN_DTYPES:
                rows.append((*cells(TARGETS[name]), op, dt, fn(*"abc"[:arity], dt)))
    return [*TARGET_COLUMNS, "op", "dtype", "written"], rows
