"""Goldens of tinygrad/uop/__init__.py: the operations and their groups."""

from golden import table
from tinygrad.uop import GroupOp, Ops

GROUPS = [name for name, value in vars(GroupOp).items() if isinstance(value, set)]


@table
def ops():
    return ["op", "value", *GROUPS], [(str(op), op.value, *(op in getattr(GroupOp, group) for group in GROUPS))
                                      for op in Ops]
