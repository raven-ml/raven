"""Goldens of tinygrad/schedule/memory.py: memory planning.

For each case, a schedule and the buffers it holds: `<case>.golden` is the
schedule, a LINEAR of calls, `<case>_held.golden` the sink of its held
buffers, and `<case>_planned.golden` what `memory_plan_rewrite` makes of it.
Synthetic schedules are those of tinygrad's test_memory_planner.py, calls of
buffers or copies between two, and random ones; real schedules are those
`linear_with_vars` plans for `Tensor` programs on CPU devices.
`<case>_debug.golden` is what planning prints at DEBUG=1.
"""

import contextlib
import io
import random

from golden import graph, text
from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEV, Context
from tinygrad.uop.ops import Ops, UOp
import tinygrad.schedule
from tinygrad.schedule.memory import memory_plan_rewrite

DEV.value = "CPU"


def synthetic(lists, copies=(), held=(), sizes=None, device="CPU"):
    """The schedule of a call per list of buffer numbers, a copy from the second
    to the first where the list is in `copies`, and its held buffers."""
    bufs = {}

    def buf(i):
        if i not in bufs: bufs[i] = UOp.new_buffer(device, (sizes or {}).get(i, 16), dtypes.int8 if i % 3 else dtypes.float)
        return bufs[i]

    calls = []
    for l in lists:
        bs = [buf(i) for i in l]
        calls.append(bs[0].store_call(bs[1]) if tuple(l) in copies else UOp.sink(*bs).call(*bs))
    return UOp(Ops.LINEAR, src=tuple(calls)), {bufs[i] for i in held if i in bufs}


def chain():
    lists = [l for i in range(6) for l in ([2 * i + 1, 0], [2 * i + 2, 2 * i + 1])] + [[100]]
    return synthetic(lists, copies={(2 * i + 1, 0) for i in range(6)}, held=(0, 100))


def randomly(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 30)
    lists = [rng.sample(range(n), rng.randint(1, min(4, n))) for _ in range(rng.randint(2, 25))]
    sizes = {i: rng.choice([1, 16, 255, 256, 257, 1000, 4096, 65536, 1 << 20, rng.randint(1, 1 << 22)]) for i in range(n)}
    copies = {tuple(l) for l in lists if len(l) == 2 and rng.random() < 0.4}
    held = tuple(i for i in range(n) if rng.random() < 0.15)
    return synthetic(lists, copies=copies, held=held, sizes=sizes)


def planned(program):
    """The schedule and held buffers that realizing `program()` plans."""
    seen, plan = [], tinygrad.schedule.memory_plan_rewrite

    def capture(linear, held_bufs=None):
        seen.append((linear, held_bufs or set()))
        return plan(linear, held_bufs)

    tinygrad.schedule.memory_plan_rewrite = capture
    try:
        program().linear_with_vars()
    finally:
        tinygrad.schedule.memory_plan_rewrite = plan
    return seen[0]


def empty(*shape): return Tensor.empty(*shape)


CASES = {
    # tinygrad's test_memory_planner.py
    "simple": lambda: synthetic([[0, 1, 2], [1, 2, 3], [4, 3], [5, 2]]),
    "some_held": lambda: synthetic([[0, 1, 2], [1, 2, 3], [4, 3], [5, 2]], held=(0, 2)),
    "all_held": lambda: synthetic([[0, 1], [1, 2], [4, 3]], held=(0, 1, 2, 3, 4)),
    "reused": lambda: synthetic([[0, 1, 2], [1, 2, 3], [4, 3], [5, 3], [6, 5, 0], [7, 8], [8, 9], [9, 3, 5],
                                 [11, 0], [11, 10, 5], [12, 11, 0], [6, 12, 7], [13, 6, 11]], held=(0, 8)),
    "very_small": lambda: synthetic([[0, 1], [3, 4]], held=(0,), sizes={1: 32, 3: 4, 4: 6}),
    "big": lambda: synthetic([[0, 1], [3, 4]], held=(0,), sizes={1: 1 << 40, 3: 1 << 30, 4: 1 << 50}),
    "copy_apart_from_compute": lambda: synthetic([[0, 1], [1, 2], [3, 2]], copies={(1, 0)}),
    "copies_share": lambda: synthetic([[0, 1], [2, 1], [3, 2]], copies={(1, 0), (2, 1)}),
    "computes_share": lambda: synthetic([[0, 1], [2, 1], [3, 2], [4, 3]], copies={(1, 0)}),
    "copies_held_mixed": lambda: synthetic([[0, 1, 2], [1, 3, 2], [4, 3], [5, 4, 0]], copies={(1, 0), (3, 1)}, held=(0,)),
    "copy_chain": chain,
    "sizes": lambda: synthetic([[0, 1, 2], [3, 1], [4, 3, 2], [5, 4], [6, 5, 1]],
                               sizes={0: 1000, 1: 70000, 2: 33, 3: 4096, 4: 257, 5: 9, 6: 1 << 20}),
    "disk": lambda: synthetic([[0, 1], [2, 1]], device="DISK:/tmp/tolk"),
    "two_devices": lambda: synthetic([[0, 1], [2, 1], [3, 2]], device=("CPU:0", "CPU:1")),
    **{f"random_{k}": (lambda k=k: randomly(k)) for k in range(12)},
    # real schedules
    "softmax": lambda: planned(lambda: empty(64, 64).softmax(-1)),
    "attention": lambda: planned(lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4))),
    "matmul_chain": lambda: planned(lambda: ((empty(32, 32) @ empty(32, 32)).relu().contiguous() @ empty(32, 32)).exp()
                                    .contiguous().sum(0)),
    "convs": lambda: planned(lambda: empty(1, 2, 8, 8).conv2d(empty(4, 2, 3, 3)).relu().conv2d(empty(4, 4, 3, 3)).sum()),
    "sharded": lambda: planned(lambda: empty(4, 64).shard(("CPU:0", "CPU:1", "CPU:2", "CPU:3"), axis=0).exp().sum(0) + 1),
    "copies": lambda: planned(lambda: (empty(64, 64).to("CPU:1").exp() + 1).to("CPU:2").sum()),
    "sort": lambda: planned(lambda: empty(64).sort()[0]),
}

DEBUG = ["simple", "sizes", "copies"]


def declare(name, case):
    def given(): return case()[0]
    def held(): return UOp.sink(*sorted(case()[1], key=lambda b: b.arg.slot))
    def made():
        linear, held_bufs = case()
        return memory_plan_rewrite(linear, held_bufs)
    given.__name__, held.__name__, made.__name__ = name, f"{name}_held", f"{name}_planned"
    graph(given)
    graph(held)
    graph(made)


def declare_debug(name, case):
    def printed():
        linear, held_bufs = case()
        out = io.StringIO()
        with Context(DEBUG=1), contextlib.redirect_stdout(out): memory_plan_rewrite(linear, held_bufs)
        return out.getvalue().rstrip("\n")
    printed.__name__ = f"{name}_debug"
    text(printed)


for name, case in CASES.items():
    declare(name, case)
for name in DEBUG:
    declare_debug(name, CASES[name])
