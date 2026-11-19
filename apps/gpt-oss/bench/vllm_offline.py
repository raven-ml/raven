# /// script
# requires-python = "==3.12.*"
# dependencies = ["vllm==0.34.0"]
# ///
"""Time bench_gpt_oss.ml's rows with vLLM at batch 1, on CUDA's first GPU.

    uv run apps/gpt-oss/bench/vllm_offline.py

The weights are the ones bench_gpt_oss.ml reads, openai/gpt-oss-20b in raven's
cache (README.md says how to fetch them). vLLM keeps the experts as MXFP4 and
the attention projections and the head at bfloat16, as the bench's rows
without a suffix do. Each request is greedy, prefix caching off, through
vLLM's offline engine with its defaults otherwise:
- decode/512: a 512-token prompt generating 129 tokens, less the same prompt
  generating 1, over 128: a token at positions 512 to 639; the median of 10
  pairs after 3;
- prefill/512: the 512-token prompt generating 1 token; the median of 10 after 3;
- first-token/12: a new process from its start to the token after a 12-token
  prompt, loading the weights included; the median of 3 processes after
  one that fills the compilation caches.
"""
import os, statistics, subprocess, sys, time

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
WEIGHTS = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "huggingface", "openai-gpt-oss-20b", REVISION)


def engine():
    from vllm import LLM
    return LLM(model=WEIGHTS, skip_tokenizer_init=True, max_model_len=1024,
               enable_prefix_caching=False, seed=0)


def generate(llm, n, tokens):
    from vllm import SamplingParams
    p = SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True, detokenize=False)
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": list(range(17, 17 + n))}], p, use_tqdm=False)
    return time.perf_counter() - t0


def row(name, tokens, ts):
    m = statistics.median(ts)
    print(f"{name:<16} {m * 1e3:8.2f} ms  {tokens / m:7.1f} tokens/s  "
          f"({min(ts) * 1e3:.2f}-{max(ts) * 1e3:.2f} ms, {len(ts)} calls)", flush=True)


def main():
    # The new processes first, while this one holds no GPU memory.
    first = []
    for i in range(4):  # The first fills the compilation caches.
        t0 = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--first"], check=True)
        if i > 0:
            first.append(time.perf_counter() - t0)
    import torch, vllm
    llm = engine()
    print(f"vllm {vllm.__version__}, torch {torch.__version__}", flush=True)
    for _ in range(3):
        generate(llm, 512, 1), generate(llm, 512, 129)
    pairs = [(generate(llm, 512, 1), generate(llm, 512, 129)) for _ in range(10)]
    row("decode/512", 1, [(long - short) / 128 for short, long in pairs])
    free, total = torch.cuda.mem_get_info()
    print(f"{'device memory':<16} {(total - free) / 1e9:8.2f} GB in use after decode/512", flush=True)
    row("prefill/512", 512, [short for short, _ in pairs])
    print(f"{'first-token/12':<16} {statistics.median(first):8.2f} s   "
          f"({min(first):.2f}-{max(first):.2f} s, 3 processes)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--first"]:
        generate(engine(), 12, 1)
    else:
        main()
