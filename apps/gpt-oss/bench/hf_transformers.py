# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "accelerate==1.16.0",
#                 "kernels==0.18.3"]
# ///
"""Time bench_gpt_oss.ml's rows with HF transformers, on CUDA's first GPU.

    uv run apps/gpt-oss/bench/hf_transformers.py

The weights are the ones bench_gpt_oss.ml reads, openai/gpt-oss-20b in raven's
cache (README.md says how to fetch them). transformers keeps the experts as
MXFP4, through the triton kernels of kernels-community/gpt-oss-triton-kernels,
and the attention projections and the head at bfloat16, as the bench's rows
without a suffix do. Each call is a forward pass and an argmax read back on
the host, as raven's greedy step is:
- decode/512: one token at position 512, the cache copied from a 512-token
  prompt before each call; the median of 30 calls after 3;
- prefill/512: a 512-token prompt; the median of 10 calls after 3;
- first-token/12: a new process from its start to the token after a 12-token
  prompt, loading the weights included; the median of 3 processes after
  one that fills the compilation caches.
"""
import copy, os, statistics, subprocess, sys, time
import torch
from transformers import AutoModelForCausalLM, DynamicCache

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
WEIGHTS = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "huggingface", "openai-gpt-oss-20b", REVISION)


def load():
    return AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.bfloat16,
                                                device_map="cuda").eval()


def ids(n):
    return torch.arange(17, 17 + n, device="cuda")[None]


@torch.no_grad()
def step(m, x, cache):
    return m(input_ids=x, past_key_values=cache, use_cache=True).logits[0, -1].argmax().item()


def times(f, before, n, setup=lambda: None):
    for _ in range(before):
        f(setup())
    out = []
    for _ in range(n):
        s = setup()
        t0 = time.perf_counter(); f(s); out.append(time.perf_counter() - t0)
    return out


def row(name, tokens, ts):
    m = statistics.median(ts)
    print(f"{name:<16} {m * 1e3:8.2f} ms  {tokens / m:7.1f} tokens/s  "
          f"({min(ts) * 1e3:.2f}-{max(ts) * 1e3:.2f} ms, {len(ts)} calls)", flush=True)


def main():
    m = load()
    print(f"transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
          f"experts {'dequantized to bfloat16' if m.config.quantization_config.dequantize else 'MXFP4'}",
          flush=True)
    prompt = DynamicCache(config=m.config)
    step(m, ids(512), prompt)
    row("decode/512", 1, times(lambda c: step(m, ids(1), c), 3, 30,
                               setup=lambda: copy.deepcopy(prompt)))
    free, total = torch.cuda.mem_get_info()
    print(f"{'device memory':<16} {(total - free) / 1e9:8.2f} GB in use after decode/512", flush=True)
    row("prefill/512", 512, times(lambda c: step(m, ids(512), c), 3, 10,
                                  setup=lambda: DynamicCache(config=m.config)))
    del m, prompt
    torch.cuda.empty_cache()
    first = []
    for i in range(4):  # The first fills the compilation caches.
        t0 = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--first"], check=True)
        if i > 0:
            first.append(time.perf_counter() - t0)
    print(f"{'first-token/12':<16} {statistics.median(first):8.2f} s   "
          f"({min(first):.2f}-{max(first):.2f} s, 3 processes)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--first"]:
        m = load()
        step(m, ids(12), DynamicCache(config=m.config))
    else:
        main()
