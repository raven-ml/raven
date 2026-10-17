"""Goldens of the stage benchmarks: `<program>.golden` is the first graph that
`prepare_rangeify` receives when the tensors of a `Tensor` program are
scheduled on the CPU.

The benchmarks name their cases after these files and keep baselines for
them, so their programs are recorded here, apart from the prepare suite's
cases, which hold the same ten programs today and change with that suite.
"""

from golden import graph
from tinygrad import Tensor, Variable, dtypes, nn
from tinygrad.helpers import DEV
from tinygrad.llm.model import ExpertGating, TransformerBlock, TransformerConfig
import tinygrad.schedule

DEV.value = "CPU"


def first_prepared(program):
    """The graph the first `prepare_rangeify` receives when scheduling the
    tensors of `program()`."""
    seen, prepare = [], tinygrad.schedule.prepare_rangeify

    def capture(sink):
        seen.append(sink)
        return prepare(sink)

    tinygrad.schedule.prepare_rangeify = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.schedule.prepare_rangeify = prepare
    if not seen: raise RuntimeError("scheduling the tensors runs no prepare_rangeify")
    return seen[0]


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype)


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
    "matmul": lambda: empty(4, 8) @ empty(8, 3),
    "softmax": lambda: empty(4, 8).softmax(-1),
    "attention": lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4)),
    "conv": lambda: empty(1, 2, 8, 8).conv2d(empty(3, 2, 3, 3)),
    "sort": lambda: empty(8).sort()[0],
    "embedding": lambda: empty(10, 4)[Tensor([1, 2, 3], dtype="int")],
    "cumsum": lambda: empty(8).cumsum(),
    "split_sum": lambda: empty(65536).sum(),
    "gpt_oss_prefill": lambda: gpt_oss_block(8, 0),
    "gpt_oss_decode": lambda: gpt_oss_block(1, Variable("start_pos", 0, 15).bind(7)),
}


for name, program in PROGRAMS.items():
    def given(program=program): return first_prepared(program)
    given.__name__ = name
    graph(given)
