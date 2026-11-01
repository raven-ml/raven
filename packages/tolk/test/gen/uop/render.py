"""Goldens of tinygrad/uop/render.py: nodes written as expressions by
`UOp.render`, and graphs written one node a line by `render_uir`.

An input golden is a sink in the graph format of test/README.md. For the
expressions, the sink's sources are the cases, in order, and a table golden
gives each case's position among them and what `render` writes for it: with
`simplify=False` (`render`) and with the default (`simplified`). A wire format
golden is what `render_uir` writes for the toposort of a graph golden's sink.
"""

import ast
import functools

from golden import graph, table, text
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo, ParamArg, PatternMatcher, UOp
from tinygrad.uop.render import render_uir
import dataclasses


def named(u, name):
    """`u`, a define, with the name `name`."""
    return u.replace(arg=dataclasses.replace(u.arg, name=name))


def attempt(fn):
    try:
        return fn()
    except Exception as e:
        return f"raises {type(e).__name__}"


def rendered(cases):
    return ["case", "src", "render", "simplified"], [
        (name, str(i), u.render(simplify=False), attempt(u.render)) for i, (name, u) in enumerate(cases)]


a, b, c = UOp.variable("a", 1, 5), UOp.variable("b", 0, 9), UOp.variable("c", 0, 9)
x, y = UOp.variable("x", -1.0, 1.0, dtypes.float), UOp.variable("y", -1.0, 1.0, dtypes.float)
flag = UOp.variable("g", False, True, dtypes.bool)
p = UOp.param(0, dtypes.float, (4, 8))
buf = UOp.param(1, dtypes.float, 32)
r0, r1 = UOp.range(4, 0), UOp.range(8, 1)


def binary(op, l, r): return UOp(op, src=(l, r))


# Expressions

EXPRESSIONS = [
    # storage and variables are their names, or a letter and their slot
    ("variable", a),
    ("bound_variable", a.bind(3)),
    ("param", UOp.param(0, dtypes.float)),
    ("named_param", UOp.param(2, dtypes.int, name="w")),
    ("buffer", UOp.new_buffer("CPU", 4, dtypes.float, 3)),
    ("local_buffer", UOp.placeholder((4,), dtypes.float, slot=5, addrspace=AddrSpace.LOCAL)),
    ("named_buffer", named(UOp.placeholder((4,), dtypes.float, slot=6, addrspace=AddrSpace.LOCAL), "smem")),
    ("alloc", UOp.alloc((4,), dtypes.float, slot=2)),
    ("named_alloc", UOp(Ops.ALLOC, src=(UOp.const(4),), arg=ParamArg(8, dtypes.float, name="scratch"))),
    ("after", UOp.new_buffer("CPU", 4, dtypes.float, 0).after(buf.index(r0).store(1.0))),
    ("special", UOp.special(4, "gidx0")),
    # ranges are their identity, and a loop without bound its first part
    ("range", r0),
    ("range_of_parts", UOp(Ops.RANGE, src=(UOp.const(4),), arg=(AxisType.LOOP, 1, 2))),
    ("range_negative", UOp(Ops.RANGE, src=(UOp.const(4),), arg=(AxisType.WEAK, -1))),
    ("range_typed", UOp.range(4, 3, AxisType.GLOBAL, dtype=dtypes.int)),
    ("loop", UOp.loop(3)),
    ("loop_of_parts", UOp(Ops.RANGE, src=(UOp(Ops.NOOP),), arg=(AxisType.LOOP, 5, 6))),
    # constants are their value
    ("int", UOp.const(3)),
    ("negative_int", UOp.const(-3)),
    ("big_int", UOp.const(2**70)),
    ("float", UOp.const(1.5)),
    ("whole_float", UOp.const(3.0)),
    ("negative_zero", UOp.const(-0.0)),
    ("tenth", UOp.const(0.1)),
    ("large_float", UOp.const(1e16)),
    ("small_float", UOp.const(1e-5)),
    ("inf", UOp.const(float("inf"))),
    ("negative_inf", UOp.const(float("-inf"))),
    ("nan", UOp.const(float("nan"))),
    ("true", UOp.const(True)),
    ("false", UOp.const(False)),
    ("invalid", UOp.const(Invalid)),
    # a cast of a constant is the constant's value, whatever the cast's type
    ("typed_int", UOp.const(3, dtypes.int)),
    ("typed_half", UOp.const(1.5, dtypes.half)),
    ("typed_bool_to_int", UOp.const(True, dtypes.int)),
    ("forced_cast_of_float", UOp.cconst(3.7, dtypes.int)),
    ("forced_cast_of_bool", UOp.cconst(True, dtypes.int)),
    ("forced_cast_to_bool", UOp.cconst(0, dtypes.bool)),
    ("cast_of_typed_constant", UOp(Ops.CAST, src=(UOp.const(3, dtypes.int),), arg=dtypes.long)),
    # any other cast names its type
    ("cast_to_int", a.cast(dtypes.int)),
    ("cast_to_long", a.cast(dtypes.long)),
    ("cast_to_uchar", a.cast(dtypes.uchar)),
    ("cast_to_bool", a.cast(dtypes.bool)),
    ("cast_to_half", x.cast(dtypes.half)),
    ("cast_to_bfloat16", x.cast(dtypes.bfloat16)),
    ("cast_to_fp8", x.cast(dtypes.fp8e4m3)),
    ("cast_to_weakint", UOp(Ops.CAST, src=(x,), arg=dtypes.weakint)),
    ("cast_of_sum", (a + b).cast(dtypes.int)),
    # the other operations with a form of their own
    ("neg", UOp(Ops.NEG, src=(a,))),
    ("neg_of_negative", UOp(Ops.NEG, src=(UOp.const(-3),))),
    ("neg_of_sum", UOp(Ops.NEG, src=(a + b,))),
    ("reciprocal", UOp(Ops.RECIPROCAL, src=(x,))),
    ("max", binary(Ops.MAX, a, b)),
    ("max_of_sums", binary(Ops.MAX, a + b, b + c)),
    ("mulacc", UOp(Ops.MULACC, src=(a, b, c))),
    ("mulacc_of_sums", UOp(Ops.MULACC, src=(a + b, b, b + c))),
    ("where", flag.where(a, b)),
    ("where_of_comparison", binary(Ops.CMPLT, a, b).where(a + c, b)),
    ("cdiv", binary(Ops.CDIV, a, b)),
    ("cmod", binary(Ops.CMOD, a, b)),
    # the binary operations are infix
    ("add", binary(Ops.ADD, a, b)),
    ("sub", binary(Ops.SUB, a, b)),
    ("mul", binary(Ops.MUL, a, b)),
    ("floordiv", binary(Ops.FLOORDIV, a, b)),
    ("floormod", binary(Ops.FLOORMOD, a, b)),
    ("shl", binary(Ops.SHL, a, b)),
    ("shr", binary(Ops.SHR, a, b)),
    ("and", binary(Ops.AND, a, b)),
    ("or", binary(Ops.OR, a, b)),
    ("xor", binary(Ops.XOR, a, b)),
    ("cmplt", binary(Ops.CMPLT, a, b)),
    ("cmpne", binary(Ops.CMPNE, a, b)),
    ("add_of_negative", binary(Ops.ADD, a, UOp.const(-3))),
    ("sub_of_negative", binary(Ops.SUB, a, UOp.const(-3))),
    # equal precedence associates to the left
    ("add_left", binary(Ops.ADD, binary(Ops.ADD, a, b), c)),
    ("add_right", binary(Ops.ADD, a, binary(Ops.ADD, b, c))),
    ("sub_left", binary(Ops.SUB, binary(Ops.SUB, a, b), c)),
    ("sub_right", binary(Ops.SUB, a, binary(Ops.SUB, b, c))),
    ("sub_of_add", binary(Ops.SUB, a, binary(Ops.ADD, b, c))),
    ("add_of_sub", binary(Ops.ADD, binary(Ops.SUB, a, b), c)),
    ("mul_of_floordiv", binary(Ops.MUL, binary(Ops.FLOORDIV, a, b), c)),
    ("floordiv_of_mul", binary(Ops.FLOORDIV, a, binary(Ops.MUL, b, c))),
    ("floormod_left", binary(Ops.FLOORMOD, binary(Ops.FLOORMOD, a, b), c)),
    ("shl_of_shr", binary(Ops.SHL, binary(Ops.SHR, a, b), c)),
    ("shr_right", binary(Ops.SHR, a, binary(Ops.SHL, b, c))),
    # tighter operands lose their parentheses, looser ones keep them
    ("add_of_mul_left", binary(Ops.ADD, binary(Ops.MUL, a, b), c)),
    ("add_of_mul_right", binary(Ops.ADD, a, binary(Ops.MUL, b, c))),
    ("mul_of_add_left", binary(Ops.MUL, binary(Ops.ADD, a, b), c)),
    ("mul_of_add_right", binary(Ops.MUL, a, binary(Ops.ADD, b, c))),
    ("shl_of_add", binary(Ops.SHL, binary(Ops.ADD, a, b), binary(Ops.SUB, b, c))),
    ("add_of_shl", binary(Ops.ADD, binary(Ops.SHL, a, b), c)),
    ("and_of_shift", binary(Ops.AND, binary(Ops.SHR, a, b), binary(Ops.SHL, b, c))),
    ("shift_of_and", binary(Ops.SHR, binary(Ops.AND, a, b), c)),
    ("xor_of_and", binary(Ops.XOR, binary(Ops.AND, a, b), binary(Ops.AND, b, c))),
    ("and_of_xor", binary(Ops.AND, binary(Ops.XOR, a, b), c)),
    ("or_of_xor", binary(Ops.OR, binary(Ops.XOR, a, b), binary(Ops.XOR, b, c))),
    ("xor_of_or", binary(Ops.XOR, a, binary(Ops.OR, b, c))),
    ("or_of_everything", binary(Ops.OR, binary(Ops.XOR, binary(Ops.AND, binary(Ops.SHL, binary(Ops.ADD, binary(
        Ops.MUL, a, b), c), a), b), c), a)),
    # comparisons keep their operands' parentheses and their own
    ("cmplt_of_add", binary(Ops.CMPLT, binary(Ops.ADD, a, b), c)),
    ("cmpne_of_mul", binary(Ops.CMPNE, a, binary(Ops.MUL, b, c))),
    ("and_of_comparisons", binary(Ops.AND, binary(Ops.CMPLT, a, b), binary(Ops.CMPNE, b, c))),
    ("add_of_comparison", binary(Ops.ADD, binary(Ops.CMPLT, a, b), c)),
    ("comparison_of_comparison", binary(Ops.CMPNE, binary(Ops.CMPLT, a, b), flag)),
    ("mul_of_max", binary(Ops.MUL, binary(Ops.MAX, a, b), c)),
    ("mul_of_where", binary(Ops.MUL, flag.where(a, b), c)),
    ("add_of_cast", binary(Ops.ADD, a.cast(dtypes.int), b.cast(dtypes.int))),
    ("float_arithmetic", binary(Ops.ADD, binary(Ops.MUL, x, UOp.const(2.5)), UOp(Ops.RECIPROCAL, src=(y,)))),
    # movements write their source and their argument
    ("reshape", p),
    ("reshape_flat", p.reshape((32,))),
    ("reshape_to_scalar", UOp.param(3, dtypes.float, 1).reshape(())),
    ("expand", UOp.param(4, dtypes.float, (4, 1)).expand((4, 8))),
    ("permute", p.permute((1, 0))),
    ("permute_three", UOp.param(5, dtypes.float, (2, 3, 4)).permute((2, 0, 1))),
    ("permute_one", buf._mop(Ops.PERMUTE, (0,))),
    ("flip", p.flip((1,))),
    ("flip_all", p.flip((0, 1))),
    ("pad", p.pad(((0, 2), (1, 3)))),
    ("pad_one", buf.pad(((1, 1),))),
    ("shrink", p.shrink(((0, 2), (1, 3)))),
    ("shrink_one", buf.shrink(((2, 30),))),
    ("movements_chained", p.permute((1, 0)).reshape((32,)).shrink(((0, 16),))),
    # an index is its indices, each without its outer parentheses
    ("index", buf.index(r0)),
    ("index_of_sum", buf.index(binary(Ops.ADD, binary(Ops.MUL, r0, UOp.const(8)), r1))),
    ("index_twice", p.index(r0, r1)),
    ("index_of_where", buf.index(flag.where(a, b))),
    ("index_of_cast", buf.index(a.cast(dtypes.int))),
    ("index_of_neg", buf.index(UOp(Ops.NEG, src=(a,)))),
    ("index_of_max", buf.index(binary(Ops.MAX, a, b))),
    ("index_of_comparison", buf.index(binary(Ops.CMPLT, a, b))),
    ("index_of_stack", buf.index(UOp.stack(a, b))),
    ("stage", UOp(Ops.STAGE, src=(binary(Ops.ADD, x, y), r0, binary(Ops.ADD, r1, UOp.const(1))))),
    # a load through an index is the storage indexed
    ("load", buf.index(r0).load()),
    ("load_of_sum", buf.index(binary(Ops.ADD, r0, r1)).load()),
    ("load_of_movement", p.index(r0, r1).load()),
    ("load_after", UOp.new_buffer("CPU", 4, dtypes.float, 0).after(buf.index(r0).store(1.0)).index(r1).load()),
    ("gated_load", UOp(Ops.LOAD, src=(buf.index(r0), UOp.const(0.0, dtypes.float), binary(Ops.CMPLT, r0, a)))),
    ("add_of_loads", binary(Ops.ADD, buf.index(r0).load(), p.index(r0, r1).load())),
    # a stack is its sources between braces
    ("stack", UOp.stack(a, b)),
    ("stack_empty", UOp(Ops.STACK, src=())),
    ("stack_same", UOp(Ops.STACK, src=(UOp.const(0),) * 3)),
    ("stack_different", UOp(Ops.STACK, src=tuple(UOp.const(i) for i in range(3)))),
    ("stack_of_sums", UOp.stack(binary(Ops.ADD, a, b), binary(Ops.MUL, b, c))),
    # shared operands are written each time they are used
    ("shared", binary(Ops.MUL, binary(Ops.ADD, a, b), binary(Ops.ADD, a, b))),
]

# Operations without a form of their own are written as their repr, over
# several lines, even inside an expression.
UNRENDERED = [
    ("sqrt", UOp(Ops.SQRT, src=(x,))),
    ("cmpeq", binary(Ops.CMPEQ, a, b)),
    ("add_of_exp2", binary(Ops.ADD, a, UOp(Ops.EXP2, src=(x,)).cast(dtypes.weakint))),
    ("load_without_index", UOp(Ops.LOAD, src=(buf,))),
    ("load_of_two", UOp(Ops.LOAD, src=(buf.index(r0), UOp.const(0.0, dtypes.float)))),
    ("sink", UOp.sink(a, b)),
]

# The sizes of these movements are nodes, which a movement's argument holds
# simplified.
r2 = UOp.range(UOp.const(16, dtypes.int), 2, AxisType.WEAK, dtype=dtypes.int)
marg_shrink = UOp(Ops.SHRINK, src=(UOp.param(0, dtypes.uint, 32), (r2 * 2) + (r2 * 2), UOp.const(2, dtypes.int)))
SYMBOLIC = [
    ("param_of_variable_size", UOp.param(0, dtypes.float, (a,))),
    ("reshape_of_variable_size", UOp.param(1, dtypes.float, (4, a))),
    ("shrink_of_simplified_offset", marg_shrink),
    ("range_after_shrink", UOp.range(UOp.const(1, dtypes.int).after(marg_shrink), 0, dtype=dtypes.int)),
    ("pad_of_variable", buf.pad(((0, a),))),
    ("expand_of_variable", UOp.param(2, dtypes.float, (1,)).expand((b,))),
    ("product_of_variable", a * 3 * 4),
    ("product_of_variable_last", UOp.const(3) * 4 * a),
]


@graph
def expressions(): return UOp.sink(*[u for _, u in EXPRESSIONS])


@table
def expressions_rendered(): return rendered(EXPRESSIONS)


@graph
def unrendered(): return UOp.sink(*[u for _, u in UNRENDERED])


@text
def unrendered_rendered(): return "\n".join(u.render(simplify=False) for _, u in UNRENDERED)


@graph
def symbolic(): return UOp.sink(*[u for _, u in SYMBOLIC])


@table
def symbolic_rendered(): return rendered(SYMBOLIC)


# Simplifying trees of the binary operations, drawn as the suite's law draws
# them. CPython raises ValueError on a shift by a negative count, where
# tolk gives its bounds the type's and declines to fold it (the README's
# negative-shift row): `simplified` is tinygrad with that rule applied, and
# `tinygrad` is tinygrad as it is.

SYMBOLS = {ast.Add: Ops.ADD, ast.Sub: Ops.SUB, ast.Mult: Ops.MUL, ast.FloorDiv: Ops.FLOORDIV, ast.Mod: Ops.FLOORMOD,
           ast.LShift: Ops.SHL, ast.RShift: Ops.SHR, ast.BitAnd: Ops.AND, ast.BitOr: Ops.OR, ast.BitXor: Ops.XOR,
           ast.Lt: Ops.CMPLT, ast.NotEq: Ops.CMPNE}
TREE_VARIABLES = {n: UOp.variable(n, 0, 9) for n in "abc"}


def tree(text):
    def build(n):
        if isinstance(n, ast.Name): return TREE_VARIABLES[n.id]
        if isinstance(n, ast.Constant): return UOp.const(n.value)
        if isinstance(n, ast.UnaryOp): return UOp.const(-n.operand.value)
        if isinstance(n, ast.Compare): return binary(SYMBOLS[type(n.ops[0])], build(n.left), build(n.comparators[0]))
        return binary(SYMBOLS[type(n.op)], build(n.left), build(n.right))
    return build(ast.parse(text, mode="eval").body)


TREES = [
    # the bounds of a negative shift
    "(4<(b>>-2^9))", "((b>>-2|-2)!=0)", "(-1|(b<<-3)%a&9)", "(a!=((c|5)>>-3))", "(2<(c+a+(c>>-2)))",
    # folding a negative shift
    "(b<(8<<-2))", "(c!=(0>>-3&5>>a))", "(b<(-2+-3<<-3-4))", "(4-((9>>-3&c)+-3))", "(7>>(6>>4)-4&c<<b)",
    # the rule for (x&mask)>>k with a negative k
    "(((a-c)*(b%c)&(5&c)>>-2)//b)", "((2&-1)%(c>>a)//(c>>(a|1))%((5&5<<b)>>-3))",
    # a zero divisor
    "(a%0)", "(0&0-(b&c//0))", "(b-(c//(2//5)^a))", "((7<<(c&-1)&b)%0)", "(6//(c%(3//9))|9)",
    # a zero divisor behind a negative shift
    "(b%(-1<<-2&(0&c))-((1<<b)-b//9)//b)", "(((0>>8)//(-3+a)>>(8<<-3|8//6))%(1>>1%(6&6)))",
    "((a<(b&(a^a)))^((9//(c>>-1))!=((a<<a)//(5&b)))|(((5//1<<5//c)!=(6|b//0))|(b<(4&1&c<<c))))",
]


def declining_negative_shifts():
    """Apply the negative-shift row to tinygrad, in this process."""
    min_max = UOp.__dict__["_min_max"].func
    def bounded(self):
        if self.op in (Ops.SHL, Ops.SHR) and not dtypes.is_float(self.dtype):
            lo, hi = self.src[1]._min_max
            if lo == hi and type(lo) is int and lo < 0: return self.dtype.min, self.dtype.max
        return min_max(self)
    prop = functools.cached_property(bounded)
    prop.__set_name__(UOp, "_min_max")
    UOp._min_max = prop
    def rewrite(self, uop, ctx=None):
        if len(pats := self.pdict.get(uop.op, [])):
            if (ler := uop.__dict__.get("_src_ops")) is None: uop.__dict__["_src_ops"] = ler = {u.op for u in uop.src}
            for _, match, early_reject in pats:
                if not early_reject.issubset(ler): continue
                try: ret = match(uop, ctx)
                except ValueError as e:
                    if str(e) != "negative shift count": raise
                    continue
                if ret is not None and ret is not uop: return ret
        return None
    PatternMatcher.rewrite = rewrite


@graph
def trees(): return UOp.sink(*[tree(t) for t in TREES])


@table
def trees_rendered():
    # tinygrad as it is first: a raise caches nothing, and the rule's bounds
    # would stay cached on the nodes
    uops = [tree(t) for t in TREES]
    rows = [[t, str(i), attempt(u.render)] for i, (t, u) in enumerate(zip(TREES, uops))]
    declining_negative_shifts()
    for row, u in zip(rows, uops): row.append(attempt(u.render))
    return ["case", "src", "tinygrad", "simplified"], rows


# Wire format

# render_uir writes a graph one node a line: `%i = op srcs : arg`. The inputs
# are sinks; each output is what render_uir writes for the sink's toposort.

def kernel():
    out, inp = UOp.param(0, dtypes.float, 16), UOp.param(1, dtypes.float, 16)
    n = UOp.variable("n", 1, 4)
    gidx, r = UOp.special(4, "gidx0"), UOp.range(4, 0, AxisType.LOOP)
    i = gidx * 4 + r
    value = inp.index(i).load() * UOp.const(1.5, dtypes.float) + UOp.const(0.5)
    guarded = UOp(Ops.CMPLT, src=(r, n)).where(value, UOp.const(0.0, dtypes.float))
    return UOp.sink(out.index(i).store(guarded).end(r), arg=KernelInfo(name="k"))


def uir(sink): return render_uir(sink) + "\n"


@graph
def program(): return kernel()


@text
def program_uir(): return uir(kernel())


@text
def program_uir_colored():
    with Context(NO_COLOR=0): return uir(kernel())


# Every form of argument: constants of each kind, inline in their users;
# storage and variables with each option; ranges, kernels, calls, reductions,
# casts, copies and the arguments written as their repr.
def arguments():
    floats = [1.5, -0.0, 0.1, 1e16, float("inf"), float("-inf"), float("nan"), 5e-324, 2.0 ** -1022, -1e-300]
    consts = [UOp.const(True), UOp.const(False), UOp.const(3), UOp.const(-5), UOp.const(2**70), UOp.const(1.5),
              UOp.const(0.1), UOp.const(float("inf")), UOp.const(float("nan")), UOp.const(-0.0), UOp.const(Invalid),
              UOp.const(-5, dtypes.int32), UOp.const(12345678901234, dtypes.int64), UOp.const(200, dtypes.uint8),
              UOp.const(1.5, dtypes.half), UOp.const(2.0, dtypes.bfloat16),
              *[UOp.const(v, dtypes.float32) for v in floats], *[UOp.const(v, dtypes.float64) for v in floats]]
    a = UOp.variable("a", 1, 5)
    storage = [
        UOp.param(0, dtypes.float, 16),
        UOp.param(1, dtypes.int, 4, name="w"),
        UOp.param(2, dtypes.half, 8, device="CPU"),
        UOp.param(4, dtypes.int, 4, volatile=True),
        UOp.param(5, dtypes.int, vmin_vmax=(0, 10), multiple_of=2, addrspace=AddrSpace.ALU),
        UOp.variable("x", -1.0, 1.0, dtypes.float),
        UOp.variable("g", False, True, dtypes.bool),
        UOp.placeholder((4,), dtypes.float, slot=6, addrspace=AddrSpace.LOCAL),
        UOp.placeholder((4,), dtypes.int, slot=7, addrspace=AddrSpace.REG),
        UOp.alloc((4,), dtypes.float, slot=8),
        a, a.bind(3),
    ]
    # tolk has no device axis: it compiles for one device (see the README)
    ranges = [UOp.range(4, 0, t) for t in AxisType if t is not AxisType.DEVICE] + [UOp(Ops.RANGE, src=(UOp.const(4),), arg=(AxisType.LOOP, 1, 2))]
    body = UOp.param(0, dtypes.float, 4).index(UOp.const(0)).store(1.0).sink()
    buf = UOp.param(9, dtypes.float, 4)
    others = [
        UOp.sink(UOp.const(1), arg=KernelInfo(name="named")),
        body.call(buf),
        body.call(buf, name="fill"),
        body.call(buf, name="fill", precompile=True),
        UOp.custom_function("sqrt", dtype=dtypes.float64).call(UOp.const(16.0, dtypes.float64)),
        UOp(Ops.REDUCE, src=(buf, ranges[0]), arg=(Ops.ADD, 0)),
        UOp(Ops.REDUCE, src=(buf, ranges[0]), arg=(Ops.MAX, 1)),
        a.cast(dtypes.long), UOp(Ops.BITCAST, src=(a.cast(dtypes.int),), arg=dtypes.float),
        buf.copy_to_device("CPU"),
        UOp.special(4, "lidx1"),
        UOp(Ops.CUSTOM, src=(a,), arg=("abs({0})", dtypes.int)),
        UOp(Ops.SOURCE, arg="int main() { return 0; }"),
        UOp.stack(UOp.const(1), UOp.const(2)), UOp(Ops.STACK, src=()), UOp.stack(a, UOp.const(2)),
        a + UOp.stack(UOp.const(1), UOp.const(2)),
    ]
    return UOp.sink(*consts, *[c + 0 for c in consts[:3]], *storage, *ranges, *others)


@graph
def args(): return arguments()


@text
def args_uir(): return uir(arguments())


# A symbolic variable and the arithmetic of bounds and Invalid, from
# test_uir_render.py.
def symbolic_graph():
    n = UOp.variable("n", 1, 64)
    return UOp.sink(((n + 1) * 2) % 7, (n % 5) + UOp.const(Invalid))


@graph
def symbolic_vars(): return symbolic_graph()


@text
def symbolic_vars_uir(): return uir(symbolic_graph())
