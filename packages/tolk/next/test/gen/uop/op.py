"""Goldens of tinygrad/uop/__init__.py: the operations and their groups."""

from golden import table
from tinygrad.uop import GroupOp, Ops

# The groups tolk.next declares. A group tinygrad adds fails the generation,
# so that it is ported or excluded, never missed.
GROUPS = ["Unary", "Binary", "Ternary", "ALU", "Broadcastable", "Elementwise", "Defines", "Irreducible",
          "Movement", "Commutative", "Associative", "Idempotent", "Reduce", "Comparison", "All"]
assert GROUPS == [name for name, value in vars(GroupOp).items() if isinstance(value, set)]


@table
def ops():
    return ["op", "value", *GROUPS], [(str(op), op.value, *(op in getattr(GroupOp, group) for group in GROUPS))
                                      for op in Ops]
