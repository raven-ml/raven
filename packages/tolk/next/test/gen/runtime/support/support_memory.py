"""Goldens of tinygrad/runtime/support/memory.py: the TLSF allocator.

`traces.golden` is a table of random traces, one allocator each: its creation,
then allocations and frees, and what each allocation returns (`None` where
tinygrad raises MemoryError). The traces draw sizes around the block size and
the subdivisions, alignments, bases and the defaults of the planner's
allocator.
"""

import random

from golden import table
from tinygrad.runtime.support.memory import TLSFAllocator

# (size, base, block_size, lv2_cnt) choices, the planner's (256, 32) among them
SHAPES = [(16, 16), (256, 32), (64, 8), (16, 4), (32, 32), (16, 1)]


@table
def traces():
    rng = random.Random(1)
    rows = []
    for trial in range(40):
        size = rng.choice([0, 1024, 4096, 1 << 16, rng.randint(64, 1 << 20)])
        base = rng.choice([0, 0, 1000, 0x200000])
        block_size, lv2_cnt = rng.choice(SHAPES)
        a = TLSFAllocator(size, base=base, block_size=block_size, lv2_cnt=lv2_cnt)
        rows.append((trial, 0, f"create {size} {base} {block_size} {lv2_cnt}", ""))
        live = []
        for step in range(1, 61):
            if live and rng.random() < 0.45:
                x = live.pop(rng.randrange(len(live)))
                a.free(x)
                rows.append((trial, step, f"free {x}", ""))
            else:
                n = rng.choice([1, 15, 16, 17, 100, 255, 256, 257, 1000, 4096, rng.randint(1, size // 8 + 1)])
                align = rng.choice([1, 1, 1, 8, 64, 256, 4096])
                try:
                    x = a.alloc(n, align)
                    live.append(x)
                except MemoryError:
                    x = None
                rows.append((trial, step, f"alloc {n} {align}", str(x)))
    return ["trial", "step", "call", "result"], rows
