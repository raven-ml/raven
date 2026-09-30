"""Goldens of tinygrad/codegen/decomp/op.py: fast_idiv over a grid of
dividends and divisors, the Threefry graph, and the rewrites of the simplifying
and late patterns.

A graph golden holds a sink of the input and its rewrite, applied to a fixed
point with the patterns alone, so that it pins the input the suite builds as
well as the result. A dividend is a variable, whose bounds fast_idiv reads.
"""

from golden import graph, table
from tinygrad.codegen.decomp.op import (fast_idiv, get_late_rewrite_patterns, get_simplifying_rewrite_patterns,
                                        threefry2x32)
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


class Narrow(Renderer):
    """A renderer with no data type to widen to."""
    def supported_dtypes(self): return set()


RENDERERS = {"all": Renderer(Target()), "none": Narrow(Target())}


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


def v(name, lo, hi, dtype=dtypes.int32):
    return UOp.variable(name, lo, hi, dtype=dtype)


# fast_idiv

INTS = {"char": dtypes.int8, "uchar": dtypes.uint8, "short": dtypes.int16, "ushort": dtypes.uint16,
        "int": dtypes.int32, "uint": dtypes.uint32, "long": dtypes.int64, "ulong": dtypes.uint64,
        "weakint": dtypes.weakint}
DIVISORS = [-3, 0, 1, 2, 3, 5, 6, 7, 10, 12, 25, 100, 127, 255, 448, 641, 1000, 2**15 - 1, 2**16 + 1, 2**31 - 1,
            2**31 + 11, 2**32 + 1, 2**40 + 3, 2**63 - 1, 2**63 + 1, 2**64 - 1, 2**64, 2**70 + 1]
BOUNDS = [1, 6, 7, 100, 1000, 2**15 - 1, 2**16 - 1, 90000, 2**31 - 1, 2**32 - 1, 2**40, 2**63 - 1, 2**64 - 1]


@table
def fast_idiv_grid():
    rows = []
    for name, dt in INTS.items():
        top = 2**64 - 1 if dt is dtypes.weakint else dt.max
        for vmin, vmax in [(0, b) for b in BOUNDS if b <= top] + [(-1, 100)]:
            if dt in dtypes.uints and vmin < 0: continue
            x = v("x", vmin, vmax, dt)
            for d in DIVISORS:
                for ren in RENDERERS:
                    r = fast_idiv(RENDERERS[ren], x, d)
                    rows.append((name, vmin, vmax, d, ren, "None" if r is None else r.render(simplify=False)))
    return ["dtype", "vmin", "vmax", "d", "renderer", "result"], rows


@graph
def fast_idiv_multiplies_and_shifts():
    x = v("x", 0, 1000)
    return UOp.sink(x, fast_idiv(RENDERERS["all"], x, 7))


@graph
def fast_idiv_shifts_out_powers_of_two():
    x = v("x", 0, 2**20)
    return UOp.sink(x, fast_idiv(RENDERERS["all"], x, 7 * 64))


@graph
def fast_idiv_widens():
    x = v("x", 0, 1000, dtypes.int16)
    return UOp.sink(x, fast_idiv(RENDERERS["all"], x, 3))


@graph
def fast_idiv_folds_a_small_dividend():
    x = v("x", 0, 6)
    return UOp.sink(x, fast_idiv(RENDERERS["all"], x, 7))


# Threefry

@graph
def threefry():
    return threefry2x32(UOp.param(0, dtypes.uint64), UOp.param(1, dtypes.uint64)).sink()


# Simplifying patterns

OPS = {"none": (), "shr_and": (Ops.SHR, Ops.AND), "shr_and_threefry": (Ops.SHR, Ops.AND, Ops.THREEFRY)}


def floordivs():
    pos, neg, mixed = v("p", 0, 100), v("n", -100, 0), v("m", -10, 10)
    wide = v("w", 0, 2**64 - 1, dtypes.uint64)
    return [pos // 3, neg // v("q", -7, -1), mixed // 3, mixed // v("d", 1, 7), pos // v("d", 1, 7),
            mixed // 8, mixed // 1, mixed // -4, wide // UOp.const(2**63, dtypes.uint64), wide // 2**63, wide // 2**64,
            v("k", -10, 10, dtypes.weakint) // 8,
            v("a", 0, 3) // v("b", 0, 3), v("c", -3, 0) // v("e", -3, 0),
            (v("x", -5, 5).maximum(0) + 1) // 3]


def floormods():
    pos, neg, mixed = v("p", 0, 100), v("n", -100, 0), v("m", -10, 10)
    wide = v("w", 0, 2**64 - 1, dtypes.uint64)
    return [pos % 3, neg % v("q", -7, -1), mixed % 3, mixed % v("d", 1, 7), mixed % 4, mixed % 1, mixed % -4,
            wide % UOp.const(2**63, dtypes.uint64), wide % 2**63, wide % 2**64, v("k", -10, 10, dtypes.weakint) % 8,
            v("a", 0, 3) % v("b", 0, 3), v("c", -3, 0) % v("e", -3, 0)]


def threefries():
    return [UOp.param(0, dtypes.uint64).threefry(UOp.param(1, dtypes.uint64))]


FAMILIES = {"floordiv": floordivs, "floormod": floormods, "threefry": threefries}

for ops_name, ops in OPS.items():
    for family, exprs in FAMILIES.items():
        declare(f"simplifying_{family}_{ops_name}", lambda ops=ops, exprs=exprs: (
            lambda s: UOp.sink(s, graph_rewrite(s, get_simplifying_rewrite_patterns(ops))))(UOp.sink(*exprs())))


# Late patterns

def late(exprs, ops, disable_fast_idiv=True, renderer="all"):
    s = UOp.sink(*exprs)
    return UOp.sink(s, graph_rewrite(s, get_late_rewrite_patterns(ops, disable_fast_idiv), ctx=RENDERERS[renderer]))


def maxes():
    return [v("x", -5, 5).maximum(v("y", -5, 5)), UOp.param(0, dtypes.float).maximum(UOp.param(1, dtypes.float)),
            v("s", -5, 5, dtypes.int16).maximum(v("y", -5, 5))]


def logical():
    a, b = UOp.param(0, dtypes.bool), UOp.param(1, dtypes.bool)
    return [a.logical_not() & b.logical_not()]


def muls():
    x, w = v("x", -100, 100), v("k", -100, 100, dtypes.weakint)
    return [x * 8, x * 1, x * 3, x * -8, w * 16, v("u", 0, 100, dtypes.uint32) * 4, UOp.param(0, dtypes.float) * 4.0]


def cdivs():
    u, pos, mixed = v("u", 0, 1000, dtypes.uint32), v("p", 0, 1000), v("m", -1000, 1000)
    return [u.alu(Ops.CDIV, UOp.const(8)), u.alu(Ops.CDIV, UOp.const(1)),
            pos.alu(Ops.CDIV, UOp.const(8)), mixed.alu(Ops.CDIV, UOp.const(8)),
            mixed.alu(Ops.CDIV, UOp.const(1)), pos.alu(Ops.CDIV, UOp.const(7)),
            pos.alu(Ops.CMOD, UOp.const(7)), mixed.alu(Ops.CDIV, UOp.const(7)),
            mixed.alu(Ops.CMOD, UOp.const(7)), pos.alu(Ops.CMOD, v("d", 1, 9)),
            pos.alu(Ops.CDIV, UOp.const(-3)), pos.alu(Ops.CMOD, UOp.const(0)),
            v("w", 0, 2**64 - 1, dtypes.uint64).alu(Ops.CMOD, UOp.const(3)),
            v("x", 0, 2**32 - 1, dtypes.uint32).alu(Ops.CDIV, UOp.const(7)),
            v("x", 0, 2**32 - 1, dtypes.uint32).alu(Ops.CMOD, UOp.const(7)),
            v("k", 0, 100, dtypes.weakint).alu(Ops.CDIV, UOp.const(8)), mixed.alu(Ops.CMOD, UOp.const(4)),
            v("l", 0, 100, dtypes.int64).alu(Ops.CDIV, UOp.const(2**63 - 1))]


def negations():
    x, y = v("x", -5, 5), v("y", -5, 5)
    f, g = UOp.param(0, dtypes.float), UOp.param(1, dtypes.float)
    return [x * -1, f * -1.0, x + y.alu(Ops.NEG), y.alu(Ops.NEG) + x, f + g.alu(Ops.NEG)]


def comparisons():
    x, y, u = v("x", -10, 10), v("y", -10, 10), v("u", 0, 10, dtypes.uint32)
    return [(x < 5).logical_not(), (5 < x).logical_not(), (u < 5).logical_not(), x * -1 < y * 3, x * -1 < 5,
            (3 < x) & (x < 5), (x < 5) & (3 < x), (3 < x) & (x < 6), (3 < u) & (u < 5),
            (UOp.const(3, dtypes.int32) < x) & (x < UOp.const(5, dtypes.int32)),
            x.ne(y).logical_not(), UOp.param(0, dtypes.float).ne(UOp.param(1, dtypes.float)).logical_not()]


def extremes():
    x, y = UOp.param(0, dtypes.int64), UOp.param(1, dtypes.int64)
    lo, hi = dtypes.int64.min, dtypes.int64.max
    return [(x < lo).logical_not(), (hi < x).logical_not(), x * -1 < lo, x * -1 < y * lo,
            (hi < x) & (x < lo + 1), (lo < x) & (x < lo + 2), (hi - 2 < x) & (x < hi), (2**80 - 1 < x) & (x < 2**80 + 1)]


def mulaccs():
    a, b, c = v("a", -5, 5), v("b", -5, 5), v("c", -5, 5)
    f, g, h = UOp.param(0, dtypes.float), UOp.param(1, dtypes.float), UOp.param(2, dtypes.float)
    return [a * b + c, c + a * b, f * g + h, a.alu(Ops.SHL, UOp.const(3, dtypes.int32)) + c, a * 4 + c]


def divisions():
    a, b = UOp.param(0, dtypes.float), UOp.param(1, dtypes.float)
    return [b.reciprocal(), a * UOp.const(1.0, dtypes.float).alu(Ops.FDIV, b),
            UOp.const(1.0, dtypes.float).alu(Ops.FDIV, b) * a, a * b.reciprocal()]


LATE = {
    "max": (maxes, {"none": (), "cmplt": (Ops.CMPLT,), "max_cmplt": (Ops.MAX, Ops.CMPLT)}),
    "logical": (logical, {"none": (), "or": (Ops.OR,)}),
    "mul": (muls, {"none": (), "shl": (Ops.SHL,)}),
    "cdiv": (cdivs, {"none": (), "shr": (Ops.SHR,)}),
    "negation": (negations, {"none": (), "neg": (Ops.NEG,), "neg_sub": (Ops.NEG, Ops.SUB)}),
    "comparison": (comparisons, {"none": (), "cmplt": (Ops.CMPLT,), "cmpeq": (Ops.CMPEQ,)}),
    "extremes": (extremes, {"cmplt": (Ops.CMPLT,)}),
    "mulacc": (mulaccs, {"none": (), "mulacc": (Ops.MULACC,), "mulacc_shl": (Ops.MULACC, Ops.SHL)}),
    "division": (divisions, {"none": (), "fdiv": (Ops.FDIV,)}),
}

for family, (exprs, op_sets) in LATE.items():
    for ops_name, ops in op_sets.items():
        declare(f"late_{family}_{ops_name}", lambda exprs=exprs, ops=ops: late(exprs(), ops))

declare("late_cdiv_shr_fast_idiv", lambda: late(cdivs(), (Ops.SHR,), disable_fast_idiv=False))
declare("late_cdiv_shr_fast_idiv_narrow", lambda: late(cdivs(), (Ops.SHR,), disable_fast_idiv=False, renderer="none"))
