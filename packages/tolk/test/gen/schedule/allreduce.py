"""Goldens of tinygrad/schedule/allreduce.py: reductions across devices.

For each case, an allreduce of a buffer on several CPU devices under settings of
its own: `<case>.golden` is the sink of the allreduce, `<case>_handled.golden`
the sink of what `handle_allreduce` makes of it, and, for some cases,
`<case>_function.golden` the sink of `create_allreduce_function`'s value and
`<case>_debug.golden` what `handle_allreduce` prints at DEBUG=2.
"""

import contextlib
import io

from golden import graph, text
from tinygrad import dtypes
from tinygrad.helpers import Context
from tinygrad.schedule.allreduce import create_allreduce_function, handle_allreduce
from tinygrad.uop.ops import Ops, UOp


def devices(n): return tuple(f"CPU:{k}" for k in range(n))


def allreduce(n, shape, op=Ops.ADD, single=False, dtype=dtypes.float):
    numel = 1
    for s in shape: numel *= s
    buf = UOp.new_buffer(devices(n), numel, dtype).reshape(shape)
    return buf.allreduce(op, "CPU:0" if single else devices(n))


def symbolic():
    v = UOp.variable("v", 1, 16)
    buf = UOp.new_buffer(devices(4), 64, dtypes.float).reshape((16, 4)).shrink(((0, v), (0, 4)))
    return buf.allreduce(Ops.ADD, devices(4))


BIG = (300, 1000)

# case: (the allreduce, its settings, whether to record the function, whether
# to record the printout)
CASES = {
    # naive: two devices, few elements, or a symbolic shape
    "naive_two_devices": (lambda: allreduce(2, (4, 8)), {}, True, True),
    "naive_to_one_device": (lambda: allreduce(3, (3, 5), Ops.MAX, single=True), {}, False, False),
    "naive_four_devices": (lambda: allreduce(4, (16,)), {}, False, False),
    "naive_int": (lambda: allreduce(3, (6,), dtype=dtypes.int32), {}, False, False),
    "naive_two_devices_many_elements": (lambda: allreduce(2, BIG), {}, False, False),
    "naive_symbolic": (symbolic, {"RING": 2, "ALL2ALL": 2, "ALLREDUCE_NODE_NDEVS": 2}, True, True),
    # ring: forced, or past the threshold on three devices or more
    "ring_two_devices": (lambda: allreduce(2, (16,)), {"RING": 2}, False, False),
    "ring_uneven_chunks": (lambda: allreduce(3, (3, 5)), {"RING": 2}, True, True),
    "ring_chunks_of_eight": (lambda: allreduce(3, (1000,), Ops.MAX), {"RING": 2}, False, False),
    "ring_to_one_device": (lambda: allreduce(4, (4, 8), single=True), {"RING": 2}, True, False),
    "ring_many_elements": (lambda: allreduce(3, BIG), {}, False, False),
    "ring_off": (lambda: allreduce(3, BIG), {"RING": 0}, False, False),
    # all-to-all: forced, or past the threshold; it takes precedence over the ring
    "all2all": (lambda: allreduce(3, (3, 5)), {"ALL2ALL": 2, "RING": 2}, False, True),
    "all2all_to_one_device": (lambda: allreduce(4, (4, 8), Ops.MAX, single=True), {"ALL2ALL": 2}, False, False),
    "all2all_many_elements": (lambda: allreduce(3, BIG), {"ALL2ALL": 1}, False, False),
    # hierarchical: nodes of devices, when they divide the devices
    "nodes_of_two": (lambda: allreduce(4, (4, 8)), {"ALLREDUCE_NODE_NDEVS": 2}, True, False),
    "nodes_of_three": (lambda: allreduce(6, (12,), Ops.MAX), {"ALLREDUCE_NODE_NDEVS": 3}, False, False),
    "nodes_to_one_device": (lambda: allreduce(4, (8,), single=True), {"ALLREDUCE_NODE_NDEVS": 2}, False, False),
    "nodes_not_dividing": (lambda: allreduce(3, (3, 5)), {"ALLREDUCE_NODE_NDEVS": 2}, False, False),
}


def declare(name, case, settings, function, printed):
    def given(): return UOp.sink(case())
    def handled():
        red = case()
        with Context(**settings): return UOp.sink(handle_allreduce(red.src[0], red))
    def created():
        red = case()
        with Context(**settings): return UOp.sink(create_allreduce_function(red.src[0], red))
    def debug():
        red, out = case(), io.StringIO()
        with Context(DEBUG=2, **settings), contextlib.redirect_stdout(out): handle_allreduce(red.src[0], red)
        return out.getvalue().rstrip("\n")
    given.__name__, handled.__name__ = name, f"{name}_handled"
    created.__name__, debug.__name__ = f"{name}_function", f"{name}_debug"
    graph(given)
    graph(handled)
    if function: graph(created)
    if printed: text(debug)


for name, (case, settings, function, printed) in CASES.items():
    declare(name, case, settings, function, printed)
