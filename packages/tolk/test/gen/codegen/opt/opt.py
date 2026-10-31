"""Goldens of tinygrad/codegen/opt/__init__.py: how an Opt prints, and how two
Opts compare, a split's target by the order of AxisType.
"""

from golden import table
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import AxisType

OPTS = [
    Opt(OptOps.TC, 0, (-1, 0, 1)),
    Opt(OptOps.TC, 0, (0, 2, 2)),
    Opt(OptOps.TC, 2, (1, 1, 1)),
    Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST)),
    Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),
    Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)),
    Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),
    Opt(OptOps.SPLIT, 1, (16, AxisType.LOCAL)),
    Opt(OptOps.SPLIT, 1, (16, AxisType.LOCAL, True)),
    Opt(OptOps.SPLIT, 1, (2, AxisType.LOCAL, True)),
    Opt(OptOps.PADTO, 0, 32),
    Opt(OptOps.PADTO, 2, 2),
    Opt(OptOps.SWAP, 0, 1),
    Opt(OptOps.SWAP, 1, 0),
]


@table
def reprs():
    return ["opt", "axis"], [(repr(o), o.axis) for o in OPTS]


def order(a, b): return "<" if a < b else ">" if b < a else "="


@table
def comparisons():
    return ["a", "b", "order"], [(repr(a), repr(b), order(a, b)) for a in OPTS for b in OPTS]
