"""Goldens of tinygrad/engine/jit.py: lowering captured schedules.

Each case is a function that tinygrad's tests jit, captured by `TinyJit` on
its second call over realized buffers, on the devices of hcq2_null.py: the CPU
host, and CPU:1 to CPU:3 with the NULL device's queues, a copy queue among
them. For each case:
- `<case>.golden` is the captured schedule, every call of the function;
- `<case>_held.golden` the sink of the buffers the capture holds that the
  schedule reaches, by slot, and `<case>_inputs.golden` the sink of its input
  buffers, in the order of their parameters;
- `<case>_lowered.golden` is what `jit_lower` makes of them.

`captures.golden` gives each case's variables with the values the capture
binds.
"""

from golden import graph, table
import hcq2_null  # noqa: F401
from tinygrad import Tensor, TinyJit, Variable, dtypes, nn
from tinygrad.runtime.ops_cpu import CPUDevice
from tinygrad.uop.ops import Ops, UOp
import tinygrad.engine.jit as jit

# The NULL devices have copy queues, as the Hcq2 suite's do.
CPUDevice.has_copy_queue = property(lambda _: True)


class Captured(Exception):
    pass


def capture(case):
    """(linear, held, inputs, var_vals, lowered) of the case's capture."""
    fn, args = CASES[case]()
    seen, lower = [], jit.jit_lower

    def record(linear, held_bufs, input_uops):
        seen.append((linear, held_bufs, input_uops, lower(linear, held_bufs, input_uops)))
        raise Captured

    jf = TinyJit(fn)
    jf.cnt = 1
    var_vals = jit._prepare_jit_inputs(args, {})[1]
    jit.jit_lower = record
    try:
        jf(*args)
    except Captured:
        pass
    finally:
        jit.jit_lower = lower
    (linear, held, inputs, lowered), = seen
    reached = set(linear.toposort())
    return linear, sorted((b for b in held if b in reached), key=lambda b: b.arg.slot), inputs, var_vals, lowered


# Buffers

def realized(*shape, dtype=dtypes.float, device="CPU"):
    """A tensor of new storage, allocated as a realized tensor's is."""
    t = Tensor.empty(*shape, dtype=dtype, device=device).realize()
    for u in t.uop.toposort():
        if u.op is Ops.BUFFER: u.buffer.ensure_allocated()
    return t


def i(n=3): return Variable("i", 1, 10).bind(n)
def j(n=2): return Variable("j", 1, 10).bind(n)


# Cases: test_jit.py

def chain_of_three():
    def f(a, b):
        c = (a + b).realize()
        d = (c * 2).realize()
        return (d - a).realize()
    return f, (realized(8, 8), realized(8, 8))


def assign(dtype):
    def f(a):
        a += 1
        a.realize()
    return lambda: (f, (realized(1, dtype=dtype),))


def copyin(): return (lambda a: a + Tensor([1, 2, 3])), (realized(3),)
def clone(): return (lambda a: a.clone().realize()), (realized(4, 4),)
def transfers(): return (lambda a, b: (a.to("CPU:1").realize(), b.to("CPU:1").realize())), (realized(4, 4), realized(4, 4))


def several_devs():
    def f(a, b):
        x, y = a.to("CPU:1").realize(), b.to("CPU:1").realize()
        return x + y.realize(), x * y.realize()
    return f, (realized(4, 4), realized(4, 4))


def view_bitcast():
    def f(a): return ((a.sum(axis=(1,)) + 5).bitcast(dtypes.int32)).to("CPU:2").realize()
    return f, (realized(4, 16, device="CPU:1"),)


def multiple_outputs():
    return (lambda a, b: ((a + b).realize(), (a - b).realize(), (a * b).realize())), (realized(4, 4), realized(4, 4))


def weight():
    w = realized(5, 5)
    return (lambda x: w.dot(x).realize()), (realized(5),)


def assign_input(): return (lambda a, b: b.assign(a + 1)), (realized(1), realized(1))


def lazy_grad():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    conv.weight, conv.bias = realized(4, 3, 3, 3), realized(4)

    def step(x, y):
        out = conv(x.permute(0, 3, 1, 2).contiguous()).relu().flatten(1)
        loss = (out * y).sum(axis=1)
        loss.sum().backward()
        conv.weight.grad = None
        (loss * 0.5).sum().backward()
        return loss.mean().realize()
    return step, (realized(2, 4, 4, 3), realized(2, 4 * 4 * 4))


def copy_inside(): return (lambda x, y: x.to("CPU:1") + y), (realized(4, 4), realized(4, 4, device="CPU:1"))


def weights_copy():
    weights = realized(16, device="CPU:1")
    return (lambda x: (weights * 2).contiguous() + x.to("CPU:1")), (realized(16),)


def weights_independent_copy():
    weights = realized(16)
    return (lambda x: (weights * 2).contiguous().to("CPU:1") + x), (realized(16, device="CPU:1"),)


def weights_kernel():
    weights = realized(16)
    return (lambda x: (weights * 2).contiguous() + x), (realized(16),)


def held_constant():
    ext = Tensor([1, 24, 23, 45, 1])

    def f(x):
        t1 = (x * 2).contiguous().realize()
        t2 = (t1 + ext).contiguous().realize()
        return t2.sum().contiguous().realize()
    return f, (realized(5, dtype=dtypes.int32),)


def accumulator():
    x = realized(1, dtype=dtypes.int32)

    def f(y):
        nonlocal x
        x += y
        return x
    return f, (realized(1, dtype=dtypes.int32),)


def compute(device, t): return (t + 1.0).contiguous().realize()
def copy(device, t): return t.to(device).realize()


def split_simple():
    def f(inp): return compute("CPU:1", compute("CPU:1", compute("CPU:1", inp)))
    return f, (realized(4, 4, device="CPU:1"),)


def split_cpu():
    def f(inp, inp_cpu):
        op1 = compute("CPU:1", compute("CPU:1", inp))
        op2 = compute("CPU", inp_cpu)
        return op2, compute("CPU:1", op1)
    return f, (realized(4, 4, device="CPU:1"), realized(4, 4))


def split_cpu_several():
    def f(inp, inp_cpu):
        op1 = compute("CPU:1", compute("CPU:1", inp))
        op3 = compute("CPU", compute("CPU", inp_cpu))
        return op3, compute("CPU:1", op1)
    return f, (realized(4, 4, device="CPU:1"), realized(4, 4))


def split_multidev():
    def f(inp, inp_d1):
        op1 = compute("CPU:1", compute("CPU:1", inp))
        op3 = compute("CPU:2", compute("CPU:2", inp_d1))
        return op3, compute("CPU:1", op1)
    return f, (realized(4, 4, device="CPU:1"), realized(4, 4, device="CPU:2"))


def split_multidev_xfer():
    def f(inp, inp_d1):
        op1 = compute("CPU:1", compute("CPU:1", inp))
        op2 = compute("CPU:2", inp_d1)
        op3 = copy("CPU:1", op2)
        return op1, compute("CPU:2", op2), compute("CPU:1", op3)
    return f, (realized(4, 4, device="CPU:1"), realized(4, 4, device="CPU:2"))


def split_multidev_copy():
    def f(inp): return compute("CPU", copy("CPU", compute("CPU:1", compute("CPU:1", inp))))
    return f, (realized(4, 4, device="CPU:1"),)


# Cases: test_jit_cases.py

def implicit_input():
    x = realized(1)
    return (lambda: (x * 2).realize()), ()


def implicit_output():
    out = realized(1)
    return (lambda x: out.assign(x * 2).realize()), (realized(1),)


def implicit_io():
    x, out = realized(1), realized(1)
    return (lambda: out.assign(x * 2).realize()), ()


# Cases: test_jit_footguns.py

def slice_assign():
    cache = realized(4, 4)

    def f(pos):
        cache[pos:pos + 1, :].assign(Tensor.ones(1, 4))
        return cache.sum().realize()
    return f, (Variable("pos", 0, 3).bind(2),)


def two_kernels():
    def f(x):
        y = (x + 1).realize()
        return (y * 2).realize()
    return f, (realized(1),)


def cat_window():
    def f(buf, frame):
        new_buf = buf[1:].cat(frame, dim=0)
        return new_buf.contiguous(), new_buf[:1].contiguous()
    return f, (realized(3, 1), realized(1, 1))


def shift_window(): return (lambda buf, new: buf[8:].cat(new)), (realized(16, dtype=dtypes.int32), realized(8, dtype=dtypes.int32))


# Cases: test_symbolic_jit.py

def unary(f, *shape, view):
    return lambda: (f, (view(realized(*shape)),))


def binary(f, sa, sb, va, vb):
    return lambda: (f, (va(realized(*sa)), vb(realized(*sb))))


def plus1(a): return (a + 1).realize()
def padded(a): return (a + 1).pad((None, (0, 10 - a.shape[1])))


CASES = {
    # test_jit.py
    "input_view": lambda: ((lambda x: (x[2:5].contiguous() + 1).realize()), (realized(10),)),
    "chain_of_three": chain_of_three,
    "add": lambda: ((lambda a, b: (a + b).realize()), (realized(4, 4), realized(4, 4))),
    "assign": assign(dtypes.float),
    "assign_int8": assign(dtypes.int8),
    "copyin": copyin,
    "clone": clone,
    "transfers": transfers,
    "several_devs": several_devs,
    "view_bitcast": view_bitcast,
    "multiple_outputs": multiple_outputs,
    "weight": weight,
    "assign_input": assign_input,
    "lazy_grad": lazy_grad,
    "copy_inside": copy_inside,
    "weights_copy": weights_copy,
    "weights_independent_copy": weights_independent_copy,
    "weights_kernel": weights_kernel,
    "held_constant": held_constant,
    "accumulator": accumulator,
    "split_simple": split_simple,
    "split_cpu": split_cpu,
    "split_cpu_several": split_cpu_several,
    "split_multidev": split_multidev,
    "split_multidev_xfer": split_multidev_xfer,
    "split_multidev_copy": split_multidev_copy,
    # test_jit_cases.py
    "explicit": lambda: ((lambda x: x * 2), (realized(1),)),
    "implicit_input": implicit_input,
    "implicit_output": implicit_output,
    "implicit_io": implicit_io,
    # test_jit_footguns.py
    "sum": lambda: ((lambda x: x.sum().realize()), (realized(2),)),
    "two_kernels": two_kernels,
    "cat_window": cat_window,
    "shift_window": shift_window,
    "slice_assign": slice_assign,
    "masked_select": lambda: ((lambda x, m: x.masked_select(m, size=4, fill_value=-1).realize()),
                              (realized(4, dtype=dtypes.int32), realized(4, dtype=dtypes.bool))),
    "nonzero": lambda: ((lambda x: x.nonzero(size=3, fill_value=-1).realize()), (realized(5, dtype=dtypes.int32),)),
    # test_symbolic_jit.py
    "plus1": unary(plus1, 3, 10, view=lambda a: a[:, :i()]),
    "inner_bound_var_view": lambda: ((lambda a: (a + 1)[:Variable("k", 1, 10).bind(3)].realize()), (realized(10),)),
    "plus1_pad_view": unary(lambda a: padded(a).realize(), 3, 10, view=lambda a: a[:, :i()]),
    "plus1_pad": unary(lambda a: padded(a).contiguous().realize(), 3, 10, view=lambda a: a[:, :i()]),
    "symbolic_add": binary(lambda a, b: (a + b).realize(), (3, 10), (3, 10), lambda a: a[:, :i()], lambda b: b[:, :i()]),
    "symbolic_matmul": binary(lambda a, b: (a @ b).realize(), (3, 10), (10, 5), lambda a: a[:, :i()], lambda b: b[:i(), :]),
    "mixed_with_no_symbol_kernel": binary(lambda a, b: ((s := (a @ b).realize()) + s).realize(), (3, 10), (10, 5),
                                          lambda a: a[:, :i()], lambda b: b[:i(), :]),
    "symbolic_attention": lambda: ((lambda q, k, v: Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                                                         v.transpose(1, 2)).realize()),
                                   (realized(2, 1, 4, 8), realized(2, 10, 4, 8)[:, :i()], realized(2, 10, 4, 8)[:, :i()])),
    "cat_dim0": binary(lambda a, b: a.cat(b, dim=0).realize(), (10, 3), (2, 3), lambda a: a[:i()], lambda b: b),
    "cat_dim1": binary(lambda a, b: a.cat(b, dim=1).realize(), (3, 10), (3, 2), lambda a: a[:, :i()], lambda b: b),
    "cat_dim0_two_vars": binary(lambda a, b: a.cat(b, dim=0).realize(), (10, 3), (10, 3), lambda a: a[:i()], lambda b: b[:j()]),
    "cat_dim1_two_vars": binary(lambda a, b: a.cat(b, dim=1).realize(), (3, 10), (3, 10), lambda a: a[:, :i()], lambda b: b[:, :j()]),
    "two_vars_plus1_ij": binary(lambda a, b: (a @ b + 1).realize(), (10, 3), (3, 10), lambda a: a[:i(), :], lambda b: b[:, :j()]),
    "two_vars_plus1_ji": binary(lambda a, b: (a @ b + 1).realize(), (10, 3), (3, 10), lambda a: a[:j(), :], lambda b: b[:, :i()]),
    "symbolic_shrink": unary(plus1, 7, 11, view=lambda a: a.shrink(((3, 5), (i(), i() + 2)))),
    "symbolic_slice": unary(plus1, 7, 11, view=lambda a: a[3:5, i():i() + 2]),
    "slice_var_shape": lambda: (plus1, (realized(i(), 11)[:, 1:2],)),
    "ones_sum": unary(lambda a: a.sum().realize(), 10, view=lambda a: a[:i()]),
    **{f"mean{axis}": unary(lambda a, axis=axis: a.mean(axis).realize(), 10, 3, view=lambda a: a[:i()]) for axis in ("", 0, 1)},
    **{f"mean_2d{axis}": unary(lambda a, axis=axis: a.mean(axis).realize(), 10, 10, view=lambda a: a[:i(), :j()])
       for axis in ("", 0, 1)},
    **{f"var{axis}": unary(lambda a, axis=axis: a.var(axis).realize(), 10, 3, view=lambda a: a[:i()]) for axis in ("", 0, 1)},
    **{f"var_2d{axis}": unary(lambda a, axis=axis: a.var(axis).realize(), 10, 10, view=lambda a: a[:i(), :j()])
       for axis in ("", 0, 1)},
}


def declare(case):
    def given(): return capture(case)[0]
    def held(): return UOp.sink(*capture(case)[1])
    def inputs(): return UOp.sink(*capture(case)[2])
    def lowered(): return capture(case)[4]
    given.__name__, held.__name__, inputs.__name__, lowered.__name__ = case, f"{case}_held", f"{case}_inputs", f"{case}_lowered"
    for fn in (given, held, inputs, lowered): graph(fn)


for case in CASES: declare(case)


@table
def captures():
    rows = [(case, " ".join(f"{k}={v}" for k, v in sorted(capture(case)[3].items())) or "-") for case in CASES]
    return ["case", "vars"], rows
