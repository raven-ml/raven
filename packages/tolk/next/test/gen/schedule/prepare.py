"""Goldens of tinygrad/schedule/prepare.py: the rewrites that prepare a tensor
graph for its ranges.

Recorded programs: `<program>.golden` is the graph that `prepare_rangeify`
receives when the tensors of a `Tensor` program are scheduled on the CPU, and
`<program>_prepared.golden` what it returns. A program that schedules several
graphs, such as an allreduce compiled as a function of its own, records the
later ones as `<program>_<n>.golden` and `<program>_<n>_prepared.golden`.
"""

from golden import graph
from tinygrad import Tensor, Variable, dtypes, function, nn
from tinygrad.helpers import DEV, SPLIT_REDUCEOP
from tinygrad.llm.model import ExpertGating, TransformerBlock, TransformerConfig
from tinygrad.uop.ops import KernelInfo, Ops, ParamArg, UOp
import os
import tinygrad.schedule
import tinygrad.schedule.prepare

DEV.value = "CPU"
D2 = ("CPU:0", "CPU:1")
DISK = "DISK:/tmp/tolk-next-prepare"


def scheduled(program):
    """The (input, output) pairs of each `prepare_rangeify` that scheduling the
    tensors of `program()` runs, in order."""
    seen, prepare = [], tinygrad.schedule.prepare_rangeify

    def capture(sink):
        out = prepare(sink)
        seen.append((sink, out))
        return out

    tinygrad.schedule.prepare_rangeify = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.schedule.prepare_rangeify = prepare
    if not seen: raise RuntimeError("scheduling the tensors runs no prepare_rangeify")
    return seen


def empty(*shape, dtype=dtypes.float, device=None): return Tensor.empty(*shape, dtype=dtype, device=device)


def sharded(*shape, axis, dtype=dtypes.float):
    shard = tuple(n // 2 if a == axis else n for a, n in enumerate(shape))
    return Tensor(empty(*shard, dtype=dtype, device=D2).uop.unshard(axis), device=D2)


def view_of_self(view):
    # an assign whose value reads its own destination through a view
    a = empty(4, 4).contiguous()
    return a.assign(view(a) + empty(4, 4))


def assign_twice():
    a, v = empty(4, 4).contiguous(), empty(4, 4) + 1
    a.assign(v)
    return a.assign(v)


def assign_own_contents():
    base = empty(3, dtype=dtypes.int64)
    contig = base.contiguous()
    contig.assign(Tensor([1, 4, 3], dtype=dtypes.int64))
    return base.assign(contig)


def assign_bitcast():
    a = empty(4).contiguous()
    return a.bitcast(dtypes.uint32).assign(empty(4, dtype=dtypes.uint32) + 1)


def assign_to_function_output():
    @function
    def f(x): return x * 2
    out = f(empty(4))
    return out.assign(empty(4) + 9)


def setitem(index, value):
    t = empty(8, 8).contiguous()
    t[index] = value
    return t


def add_one(C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def custom_kernel():
    return Tensor.custom_kernel(empty(4, 4), empty(4, 4) * 2, fxn=add_one)[0] + 1


def inline_function():
    @function
    def f(a, b): return a * 2 + b
    return f(empty(4, 8), empty(4, 8)).sum(1)


def inline_sharded():
    @function
    def f(a): return a * 2 + 1
    return f(sharded(4, 8, axis=0)).sum(1)


def inline_symbolic():
    @function
    def f(a): return a * 2
    return f(empty(10)[:Variable("v", 1, 10).bind(3)]) + 1


def precompiled_function():
    @function(precompile=True)
    def f(a): return (a * 2).sum(0)
    return f(empty(4, 8) + 1) + 1


def no_split():
    SPLIT_REDUCEOP.value = 0
    return empty(65536).sum()


def gpt_oss_block(tokens, start_pos):
    """One transformer block of gpt-oss, at a small width: attention with sinks
    over a sliding window and a cache, rotary embeddings, and a mixture of four
    experts, two per token, as tinygrad's model runs it. Its weights, rotary
    table and cache are buffers, as a loaded model's are."""
    config = TransformerConfig(
        num_blocks=1, dim=64, hidden_dim=32, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=1, head_dim=16,
        rope_theta=150000.0, rope_dim=16, v_head_dim=16, max_context=16, num_experts=4, num_experts_per_tok=2,
        expert_gating_func=ExpertGating.SOFTMAX_WEIGHT, qkv_bias=True, expert_proj_bias=True, attn_output_bias=True,
        attn_sinks=True, swiglu_alpha=1.702, swiglu_clamp_exp=7.0, swiglu_up_bias=1.0, sliding_window=8)
    block = TransformerBlock(config)
    for t in nn.state.get_state_dict(block).values(): t.replace(empty(*t.shape, dtype=t.dtype))
    block.cache_kv = empty(2, 1, config.n_kv_heads, config.max_context, config.head_dim, dtype=dtypes.half)
    block.freqs_cis = empty(config.max_context, config.rope_dim)
    return block(empty(1, tokens, config.dim), start_pos)


PROGRAMS = {
    # elementwise, reductions and matmuls
    "add": lambda: empty(4, 4) + empty(4, 4),
    "sum": lambda: empty(4, 8).sum(1),
    "matmul": lambda: empty(4, 8) @ empty(8, 3),
    "softmax": lambda: empty(4, 8).softmax(-1),
    "attention": lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4)),
    "conv": lambda: empty(1, 2, 8, 8).conv2d(empty(3, 2, 3, 3)),
    "cat": lambda: Tensor.cat(empty(2, 4), empty(3, 4)),
    "stack": lambda: Tensor.stack(empty(4), empty(4), empty(4)),
    "pad": lambda: empty(4, 4).pad(((1, 2), (0, 3))) + 1,
    "arange": lambda: Tensor.arange(16).reshape(4, 4) + 1,
    "sort": lambda: empty(8).sort()[0],
    "embedding": lambda: empty(10, 4)[Tensor([1, 2, 3], dtype="int")],
    "cumsum": lambda: empty(8).cumsum(),
    "where": lambda: (empty(4, 4) > 0).where(empty(4, 4), 0),
    # reductions split in two kernels, or not
    "split_sum": lambda: empty(65536).sum(),
    "split_rows": lambda: empty(4, 65536).sum(1),
    "split_expanded": lambda: empty(1, 256).expand(256, 256).sum(),
    "split_max": lambda: empty(256, 256).max(),
    "no_split": no_split,
    "split_unit_axis": lambda: empty(1, 65536).sum(1),
    "split_prime": lambda: empty(65537).sum(),
    "split_at_threshold": lambda: empty(32768).sum(),
    "below_threshold": lambda: empty(16384).sum(),
    # empty tensors
    "sum_of_nothing": lambda: empty(0, 4).sum(0),
    "max_of_nothing": lambda: empty(4, 0).max(1),
    "add_nothing": lambda: empty(0) + 1,
    # detaches and gradient markers
    "detach": lambda: (empty(4) + 1).detach() * 2,
    "contiguous_backward": lambda: (empty(4) + 1).contiguous_backward() * 2,
    # stages
    "contiguous": lambda: (empty(4, 4) + 1).contiguous().sum(0),
    "contiguous_of_storage": lambda: empty(4, 4).contiguous() + 1,
    "contiguous_permuted": lambda: (empty(4, 4) + 1).permute(1, 0).contiguous() * 2,
    "clone": lambda: empty(4, 4).clone(),
    # copies
    "copy": lambda: empty(4, 4).to("CPU:1") * 2,
    "copy_view": lambda: empty(8, 8)[2:6].to("CPU:1") + 1,
    "copy_same_device": lambda: (empty(4) + 1).to("CPU") * 2,
    "copy_into_storage": lambda: empty(4, 4, device="CPU:1").assign(empty(4, 4).to("CPU:1")),
    "copy_computed": lambda: (empty(4, 4) + 1).to("CPU:1").sum(),
    "copy_from_disk": lambda: empty(16, dtype=dtypes.uint8, device=DISK)[4:12].to("CPU") + 1,
    "copy_staged_view_from_disk": lambda: empty(16, dtype=dtypes.uint8, device=DISK)[2:6].contiguous().to("CPU") + 1,
    "copy_permuted_from_disk": lambda: empty(4, 4, dtype=dtypes.uint8, device=DISK).permute(1, 0).to("CPU") + 1,
    # bitcasts that change the shape
    "bitcast_narrow": lambda: empty(8).bitcast(dtypes.uint8) + 1,
    "bitcast_wide": lambda: empty(4, 8, dtype=dtypes.uint8).bitcast(dtypes.float) * 2,
    "bitcast_half": lambda: empty(8).bitcast(dtypes.half) + 1,
    "bitcast_long": lambda: empty(4, dtype=dtypes.int64).bitcast(dtypes.int32) + 1,
    "bitcast_on_disk": lambda: empty(16, dtype=dtypes.uint8, device=DISK).bitcast(dtypes.float).to("CPU") + 1,
    # stores
    "assign": lambda: empty(4, 4).assign(empty(4, 4) + 1),
    "assign_permuted_self": lambda: view_of_self(lambda a: a.permute(1, 0)),
    "assign_flipped_self": lambda: view_of_self(lambda a: a.flip(0)),
    "assign_reshaped_self": lambda: view_of_self(lambda a: a.reshape(16).reshape(4, 4)),
    "assign_shrunk_in_place": lambda: (lambda a: a[1:3].assign(a[1:3] + 1))(empty(4, 4).contiguous()),
    "assign_shrunk_self": lambda: (lambda a: a[1:3].assign(a[0:2] + 1))(empty(4, 4).contiguous()),
    "assign_twice": assign_twice,
    "assign_own_contents": assign_own_contents,
    "assign_bitcast": assign_bitcast,
    "assign_reshaped": lambda: empty(16).reshape(4, 4).assign(empty(4, 4) * 2),
    "assign_cross_device": lambda: empty(3).assign(empty(3, device="CPU:1") + 1),
    "assign_shifted_self": lambda: (lambda a: a[0:12].assign(a[4:16]))(empty(16).contiguous()),
    "assign_disjoint_self": lambda: (lambda a: a[0:4].assign(a[8:12]))(empty(16).contiguous()),
    "assign_double_bitcast": lambda: empty(4).contiguous().bitcast(dtypes.uint32).bitcast(dtypes.int32).assign(
        empty(4, dtype=dtypes.int32) + 1),
    "assign_shrink_then_bitcast": lambda: empty(4).contiguous()[0:2].bitcast(dtypes.uint32).assign(
        empty(2, dtype=dtypes.uint32) + 1),
    "assign_bitcast_wider": lambda: empty(8, dtype=dtypes.uint8).contiguous().bitcast(dtypes.int64).assign(
        empty(1, dtype=dtypes.int64) + 1),
    "assign_broadcast": lambda: empty(3, 5).contiguous().assign(empty(5) + 1),
    "assign_to_disk": lambda: empty(5, device=DISK).assign(empty(5) + 1),
    "assign_deviceless_const": lambda: empty(4, device="CPU:1").assign(Tensor(UOp.const(2.0).cast(dtypes.float))),
    "assign_to_function_output": assign_to_function_output,
    "setitem": lambda: setitem(slice(2, 4), 1.0),
    "setitem_tensor": lambda: setitem((slice(None), slice(2, 4)), empty(8, 2)),
    # models
    "gpt_oss_prefill": lambda: gpt_oss_block(8, 0),
    "gpt_oss_decode": lambda: gpt_oss_block(1, Variable("start_pos", 0, 15).bind(7)),
    # calls
    "custom_kernel": custom_kernel,
    "inline_function": inline_function,
    "precompiled_function": precompiled_function,
    "inline_sharded": inline_sharded,
    "inline_symbolic": inline_symbolic,
    # symbolic shapes
    "variable_shrink": lambda: empty(10)[:Variable("v", 1, 10).bind(3)] * 2,
    # several devices
    "shard_add": lambda: sharded(4, 8, axis=0) + 1,
    "shard_sum": lambda: sharded(4, 8, axis=0).sum(0),
    "shard_gather": lambda: sharded(4, 8, axis=0).to("CPU:0") + 1,
}


def forked(fn):
    """`fn()`, computed in a child process, so that the generator's state stays
    the one each golden starts from."""
    read, write = os.pipe()
    if (pid := os.fork()) == 0:
        os.close(read)
        with os.fdopen(write, "w") as out: out.write(repr(fn()))
        os._exit(0)
    os.close(write)
    with os.fdopen(read) as out: text = out.read()
    if os.waitpid(pid, 0)[1] != 0: raise RuntimeError("the child failed")
    return eval(text)


def declare(name, program):
    for n in range(forked(lambda: len(scheduled(program)))):
        stem = name if n == 0 else f"{name}_{n}"
        def given(n=n): return scheduled(program)[n][0]
        def made(n=n): return scheduled(program)[n][1]
        given.__name__, made.__name__ = stem, f"{stem}_prepared"
        graph(given)
        graph(made)


for name, program in PROGRAMS.items():
    declare(name, program)


# Hand-built graphs: `<graph>.golden` and `<graph>_prepared.golden`, as for
# programs, each a function's sink on the CPU with its output in slot 0.

def param(slot, *shape, dtype=dtypes.float, device="CPU"):
    n = 1
    for s in shape: n *= s
    u = UOp.param(slot, dtype, n, device)
    return u if len(shape) == 1 else u.reshape(shape)


def scratch(n, device="CPU"):
    # call-local storage, in a slot apart from the parameters'
    return UOp(Ops.ALLOC, arg=ParamArg(10, dtypes.float, n, device=device))


def alias_of_placed():
    # storage computed into an output is also read directly elsewhere
    out, other, s = param(0, 4, 4), param(2, 16), scratch(16)
    computed = s.after(s.store(param(1, 16).exp2())).reshape((4, 4))
    return UOp.sink(out.store(computed), other.after(other.store(s + 1)))


def placed_through_view():
    # storage stored into an output through a view of it, ordered after its store
    out, s = param(0, 4, 4), scratch(16)
    return UOp.sink(out.store(s.reshape((4, 4)).after(s.store(param(1, 16).exp2()))))


def placed_after_read():
    # an output ordered after its store is read by another output
    out, other, s = param(0, 16), param(2, 16), scratch(16)
    first = out.after(out.store(s.after(s.store(param(1, 16).exp2()))))
    return UOp.sink(first, other.after(other.store(first + 1)))


def store_ordered_before():
    # an output's store is an effect another output waits for
    out, other = param(0, 16), param(2, 16)
    st = out.store(param(1, 16).exp2().contiguous())
    return UOp.sink(st, other.after(other.store(param(3, 16)), st))


def sharded_output_of_whole_storage():
    # new storage of the whole value's size stored into a sharded output
    devices = ("CPU:0", "CPU:1")
    out = UOp.param(0, dtypes.float, 8, devices).unshard(0)
    s = scratch(16, devices)
    return UOp.sink(out.store(s.after(s.store(UOp.param(1, dtypes.float, 16, devices).exp2()))))


def hazard_behind_other_after():
    # the value reads its destination through a store that another storage waits for
    out, other = param(0, 4, 4), param(2, 4, 4)
    through = other.after(out.store(param(1, 4, 4)))
    return UOp.sink(out.after(out.store(through.permute((1, 0)) + 1)))


def disk_staged_view():
    # a materialised view of a disk buffer copied to the CPU
    out = param(0, 4, dtype=dtypes.uint8)
    disk = UOp.param(1, dtypes.uint8, 16, DISK).shrink(((2, 6),)).contiguous()
    return UOp.sink(out.after(out.store(disk.copy_to_device("CPU"))))


def flip_of_other():
    # the value flips other storage and reads its destination in place
    out = param(0, 16)
    return UOp.sink(out.after(out.store(param(1, 16).flip((0,)) + out)))


GRAPHS = {
    "alias_of_placed": alias_of_placed,
    "placed_through_view": placed_through_view,
    "placed_after_read": placed_after_read,
    "store_ordered_before": store_ordered_before,
    "sharded_output_of_whole_storage": sharded_output_of_whole_storage,
    "hazard_behind_other_after": hazard_behind_other_after,
    "flip_of_other": flip_of_other,
    "disk_staged_view": disk_staged_view,
}


def declare_graph(name, sink):
    def given(): return sink()
    def made(): return tinygrad.schedule.prepare.prepare_rangeify(sink())
    given.__name__, made.__name__ = name, f"{name}_prepared"
    graph(given)
    graph(made)


for name, sink in GRAPHS.items():
    declare_graph(name, sink)
