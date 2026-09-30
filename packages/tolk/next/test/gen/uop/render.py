"""Goldens of tinygrad/uop/render.py: nodes written as expressions by
`UOp.render`, and linear programs listed by `print_uops`.

An input golden is a sink in the graph format of test/README.md. For the
expressions, the sink's sources are the cases, in order, and a table golden
gives each case's position among them and what `render` writes for it: with
`simplify=False` (`render`) and with the default (`simplified`). A listing's
input is a sink whose sources are the list printed, in order.
"""

from golden import graph, listing, table, text
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo, ParamArg, UOp


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
    ("named_buffer", UOp.placeholder((4,), dtypes.float, slot=6, addrspace=AddrSpace.LOCAL, tag="smem")),
    ("alloc", UOp.alloc((4,), dtypes.float, slot=2)),
    ("named_alloc", UOp(Ops.ALLOC, arg=ParamArg(8, dtypes.float, 4, name="scratch"))),
    ("after", UOp.new_buffer("CPU", 4, dtypes.float, 0).after(buf.index(r0).store(1.0))),
    ("special", UOp.special(4, "gidx0")),
    # ranges are their identity, and a loop without bound its first part
    ("range", r0),
    ("range_of_parts", UOp(Ops.RANGE, src=(UOp.const(4),), arg=(1, 2, AxisType.LOOP))),
    ("range_negative", UOp(Ops.RANGE, src=(UOp.const(4),), arg=(-1, AxisType.REDUCE))),
    ("range_typed", UOp.range(4, 3, AxisType.GLOBAL, dtype=dtypes.int)),
    ("loop", UOp.loop(3)),
    ("loop_of_parts", UOp(Ops.RANGE, src=(UOp(Ops.NOOP),), arg=(5, 6, AxisType.LOOP))),
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
    ("permute_one", UOp(Ops.PERMUTE, src=(buf,), arg=(0,))),
    ("flip", p.flip((1,))),
    ("flip_all", p.flip((0, 1))),
    ("flip_none", UOp(Ops.FLIP, src=(p,), arg=(False, False))),
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
    # tags are not written
    ("tagged", binary(Ops.ADD, a.rtag(1), UOp.const(2).rtag("t")).rtag(True)),
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
    ("tagged_sqrt", UOp(Ops.SQRT, src=(x,)).rtag("t")),
]

# The sizes of these movements are nodes, which a movement's argument holds
# simplified.
r2 = UOp.range(UOp.const(16, dtypes.int), 2, AxisType.WEAK, dtype=dtypes.int)
marg_shrink = UOp(Ops.SHRINK, src=(UOp.param(0, dtypes.uint, 32), (r2 * 2) + (r2 * 2), UOp.const(2, dtypes.int)))
SYMBOLIC = [
    ("param_of_variable_size", UOp.param(0, dtypes.float, (a,))),
    ("reshape_of_variable_size", UOp.param(1, dtypes.float, (4, a))),
    ("shrink_of_simplified_offset", marg_shrink),
    ("range_after_shrink", UOp.range(1, 0, src=(marg_shrink,), dtype=dtypes.int)),
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


# Listings

def kernel():
    out, inp = UOp.param(0, dtypes.float, 16), UOp.param(1, dtypes.float, 16)
    n = UOp.variable("n", 1, 4)
    gidx, r = UOp.special(4, "gidx0"), UOp.range(4, 0, AxisType.LOOP)
    i = gidx * 4 + r
    value = inp.index(i).load() * UOp.const(1.5, dtypes.float) + UOp.const(0.5)
    guarded = UOp(Ops.CMPLT, src=(r, n)).where(value, UOp.const(0.0, dtypes.float))
    return UOp.sink(out.index(i).store(guarded).end(r), arg=KernelInfo(name="k"))


def listed(uops): return UOp.sink(*uops)


PROGRAM = list(kernel().toposort())


@graph
def program(): return listed(PROGRAM)


@text
def program_listing(): return listing(PROGRAM)


@text
def program_listing_colored():
    with Context(NO_COLOR=0): return listing(PROGRAM)


# A list that leaves out some sources prints them as `--`, constants included.
def partial_list():
    r = UOp.range(4, 0)
    add = r + 1
    typed = UOp.const(2, dtypes.int)
    return [add, r, typed, typed.cast(dtypes.long) + a.cast(dtypes.long), a]


PARTIAL = partial_list()


@graph
def partial(): return listed(PARTIAL)


@text
def partial_listing(): return listing(PARTIAL)


# Columns wider than their width are not cut, and arguments print as tinygrad
# prints them.
def wide_list():
    ranges = [UOp.range(2, i, t) for i, t in [(12, AxisType.REDUCE), (10, AxisType.GLOBAL), (11, AxisType.LOCAL),
                                              (13, AxisType.UPCAST)]]
    ranges.append(UOp(Ops.RANGE, src=(UOp.const(2),), arg=(0, 1, AxisType.UNROLL)))
    ranges.append(UOp(Ops.RANGE, src=(UOp.const(2),), arg=(-1, AxisType.LOOP)))
    total = ranges[0]
    for r in ranges[1:]: total = total + r
    wide = UOp.sink(*[UOp.const(i) for i in range(12)])
    consts = [UOp.const(v) for v in (1e16, -0.0, float("nan"), float("inf"), 0.1, True, Invalid, 2**70)]
    tail = [
        UOp(Ops.CONTIGUOUS_BACKWARD, src=(total,)),
        UOp.const(3, dtypes.int).copy_to_device("CPU"),
        UOp.const(3, dtypes.int).copy_to_device(("CPU:0", "CPU:1")),
        UOp(Ops.SOURCE, arg="int main() { return 0; }"),
        UOp(Ops.CUSTOM, src=(a,), arg=("abs({0})", dtypes.int)),
        UOp(Ops.REDUCE, src=(UOp.param(6, dtypes.float, 8), ranges[0]), arg=(Ops.ADD, 0)),
        UOp.param(7, dtypes.half, 8, name="named"),
        a,
        a.bind(2),
        UOp.const(2).rtag("hidden"),
        UOp(Ops.STACK, src=tuple(consts[:5])),
        binary(Ops.AND, consts[5], flag),
        a.valid(flag),
        binary(Ops.ADD, a, consts[7]),
        wide,
    ]
    return [*ranges, total, *consts, *tail]


WIDE = wide_list()


@graph
def wide(): return listed(WIDE)


@text
def wide_listing(): return listing(WIDE)


@text
def wide_listing_colored():
    with Context(NO_COLOR=0): return listing(WIDE)
