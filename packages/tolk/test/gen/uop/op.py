"""Goldens of tinygrad/uop/__init__.py: the operations and their groups."""

from golden import table
from tinygrad.uop import GroupOp, Ops

# The groups tolk declares, and those it excludes (README). A group tinygrad
# adds fails the generation, so that it is ported or excluded, never missed.
GROUPS = ["Unary", "Binary", "Ternary", "ALU", "Broadcastable", "Elementwise", "Defines", "Irreducible",
          "Movement", "Effect", "Commutative", "Associative", "Idempotent", "Reduce", "Comparison", "All"]
EXCLUDED = ["UnaryElementwise", "ShapeChanging", "PassthroughDType", "PromoteDType"]
NAMES = [name for name, value in vars(GroupOp).items() if isinstance(value, set)]
assert GROUPS == [name for name in NAMES if name not in EXCLUDED] and set(EXCLUDED) <= set(NAMES)


@table
def ops():
    return ["op", "value", *GROUPS], [(str(op), op.value, *(op in getattr(GroupOp, group) for group in GROUPS))
                                      for op in Ops]
