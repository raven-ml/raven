"""Goldens of tinygrad/codegen/decomp/dtype.py: the kernels of `Tensor`
programs as the pass that emulates data types receives them, and what the pass
makes of them.

A golden `<type>_<kernel>.golden` holds a sink of two sources: the kernel that
the codegen pipeline for the CPU hands to its "decomp dtypes" pass when `<type>`
is emulated (EMULATED_DTYPES), and that kernel rewritten by the pass,
`pm_dtype_decomps + pm_commit_weak`. The pass reads the target only through its
data types, so it runs for two targets: one that has every data type and is
told to emulate `<type>`, and one that lacks `<type>`. The generator checks that
both make the same graph, which the golden holds.

The conversions of a narrow float are IEEE conversions in tolk
(DIVERGENCES D9), so for the narrow floats the pass runs with `f2f` and
`f2f_clamp` held opaque: each call is a CUSTOM node of the type the call
returns, over the call's operand, whose code names the call and its arguments,
`f2f <from> <to> <sat>` or `f2f_clamp <type> <sat>`. Everything else in the
graph is tinygrad's. A cast to a narrow float from a type more precise than a
float32 narrows its operand otherwise in tolk (D9), so does a cast of an
emulated 64-bit integer to a float32 or narrower (D9), and a cast of a float to
an emulated 64-bit integer splits it otherwise (D22), so no golden holds one.
"""

from golden import graph
from graph import kernels, stage
from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEV, Context, Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.uop.weak import pm_commit_weak
import tinygrad.codegen.decomp.dtype as decomp

DEV.value = "CPU"
CPU = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))


class Lacking(Renderer):
    """A renderer without the data types `lacks`."""
    def __init__(self, lacks):
        super().__init__(Target())
        self.lacks = lacks
    def supported_dtypes(self): return super().supported_dtypes() - self.lacks


def name(dt): return repr(dt)[len("dtypes."):]


def opaque_f2f(v, fr, to, sat=True):
    dt = to if fr.bitsize < to.bitsize else decomp.f2f_dt[to]
    return UOp(Ops.CUSTOM, src=(v,), arg=(f"f2f {name(fr)} {name(to)} {sat}", dt))


def opaque_f2f_clamp(val, dt, sat=True):
    return UOp(Ops.CUSTOM, src=(val,), arg=(f"f2f_clamp {name(dt)} {sat}", val.dtype))


def emulate(kernel, renderer, floats):
    f2f, f2f_clamp = decomp.f2f, decomp.f2f_clamp
    if floats: decomp.f2f, decomp.f2f_clamp = opaque_f2f, opaque_f2f_clamp
    try:
        return graph_rewrite(kernel, decomp.pm_dtype_decomps + pm_commit_weak, ctx=(set(), renderer))
    finally:
        decomp.f2f, decomp.f2f_clamp = f2f, f2f_clamp


def declare(golden, emulated, program):
    """Declare the golden `golden` of the kernel of `program()` when the data
    types `emulated` are emulated."""
    floats = any(dt in dtypes.floats for dt in emulated)
    lacks = set(emulated) | ({dtypes.ulong} if dtypes.long in emulated else set())

    def fn():
        with Context(EMULATED_DTYPES=",".join(name(dt) for dt in emulated)):
            kernel = stage("decomp dtypes", kernels(program())[-1], CPU)
            told = emulate(kernel, Renderer(Target()), floats)
        lacking = emulate(kernel, Lacking(lacks), floats)
        if told is not lacking: raise RuntimeError(f"{golden}: the two targets emulate differently")
        return UOp.sink(kernel, told)

    fn.__name__ = golden
    graph(fn)


def empty(n, dt): return Tensor.empty(n, dtype=dt)


NARROW = [dtypes.half, dtypes.bfloat16, dtypes.fp8e4m3, dtypes.fp8e5m2, dtypes.fp8e4m3fnuz, dtypes.fp8e5m2fnuz]


def narrow_kernels(dt):
    storage = dtypes.uint8 if dt.itemsize == 1 else dtypes.uint16
    return {
        # arithmetic in float32 on loaded values, stored back, over vectors of 4
        "add": lambda: empty(16, dt) + empty(16, dt),
        "mulsub": lambda: empty(8, dt) * empty(8, dt) - empty(8, dt),
        "maximum": lambda: empty(8, dt).maximum(empty(8, dt)),
        "where": lambda: (empty(8, dtypes.bool)).where(empty(8, dt), empty(8, dt)),
        "exp2": lambda: empty(16, dt).exp2(),
        "sqrt": lambda: empty(16, dt).sqrt() + 1,
        # a reduction, which accumulates in float32
        "sum": lambda: Tensor.empty(4, 8, dtype=dt).sum(1),
        # a comparison, stored as a boolean
        "lt": lambda: empty(8, dt) < empty(8, dt),
        # casts from and to the type
        "from_float": lambda: empty(16, dtypes.float).cast(dt),
        "to_float": lambda: empty(16, dt).cast(dtypes.float) * 2,
        "from_char": lambda: empty(8, dtypes.int8).cast(dt),
        "to_char": lambda: empty(8, dt).cast(dtypes.int8),
        # copies: a reversal, a gather by an index tensor, and a padding, whose
        # load is masked and whose fill is a constant of the type
        "flip": lambda: empty(16, dt).flip(0).contiguous(),
        "gather": lambda: empty(16, dt)[empty(4, dtypes.int32)],
        "pad": lambda: empty(8, dt).pad(((2, 2),), value=1.5),
        # bit reinterpretations from and to the unsigned integer of its width
        "bitcast_to_storage": lambda: empty(16, dt).bitcast(storage),
        "bitcast_from_storage": lambda: empty(16, storage).bitcast(dt),
        "bitcast_sum": lambda: (empty(16, dt) + 1).bitcast(storage),
    }


for dt in NARROW:
    for kernel, program in narrow_kernels(dt).items():
        declare(f"{name(dt)}_{kernel}", [dt], program)

# two emulated floats in one kernel, rewritten in the promotion order
declare("half_fp8e4m3_cast", [dtypes.half, dtypes.fp8e4m3],
        lambda: empty(16, dtypes.half).cast(dtypes.fp8e4m3))
declare("bfloat16_half_cast", [dtypes.half, dtypes.bfloat16],
        lambda: empty(16, dtypes.bfloat16).cast(dtypes.half))


def long_kernels(dt):
    return {
        "add": lambda: empty(8, dt) + empty(8, dt),
        "sub": lambda: empty(8, dt) - empty(8, dt),
        "neg": lambda: -empty(8, dt),
        "mul": lambda: empty(8, dt) * empty(8, dt),
        # a division unrolls its 64 steps: the golden of the signed one is the
        # quotient, of the unsigned one the remainder
        **({"div": lambda: empty(8, dt) // empty(8, dt)} if dt is dtypes.long else
           {"mod": lambda: empty(8, dt) % empty(8, dt)}),
        "shl": lambda: empty(8, dt) << 3,
        "shr": lambda: empty(8, dt) >> 5,
        "shl_by": lambda: empty(8, dt) << empty(8, dt),
        "xor": lambda: empty(8, dt) ^ empty(8, dt),
        "and_or": lambda: (empty(8, dt) & empty(8, dt)) | empty(8, dt),
        "lt": lambda: empty(8, dt) < empty(8, dt),
        "eq": lambda: empty(8, dt) == empty(8, dt),
        "maximum": lambda: empty(8, dt).maximum(empty(8, dt)),
        "where": lambda: empty(8, dtypes.bool).where(empty(8, dt), empty(8, dt)),
        "sum": lambda: Tensor.empty(4, 8, dtype=dt).sum(1),
        # a padding, whose load is masked
        "pad": lambda: empty(8, dt).pad(((2, 2),), value=7),
        # a constant splits into its two words by value
        "add_const": lambda: empty(8, dt) + (2**40 + 5),
        "from_int": lambda: empty(8, dtypes.int32).cast(dt),
        "from_uint": lambda: empty(8, dtypes.uint32).cast(dt),
        "from_bool": lambda: empty(8, dtypes.bool).cast(dt),
        "to_char": lambda: empty(8, dt).cast(dtypes.int8),
        "bitcast": lambda: empty(8, dt).bitcast(dtypes.ulong if dt is dtypes.long else dtypes.long),
    }


for dt in (dtypes.long, dtypes.ulong):
    for kernel, program in long_kernels(dt).items():
        declare(f"{name(dt)}_{kernel}", [dtypes.long], program)

# an unsigned 64-bit integer is emulated with the signed one, so naming it alone
# emulates nothing
declare("ulong_named_alone", [dtypes.ulong], lambda: empty(8, dtypes.ulong) + empty(8, dtypes.ulong))
