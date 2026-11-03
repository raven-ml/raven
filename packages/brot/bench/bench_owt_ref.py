# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gigatoken==0.11.0",
#   "tokenizers==0.23.3",
#   "awkward==2.14.0",
#   "numpy==2.5.4",
# ]
# ///
"""The reference engines' side of bench_owt.exe: encode the documents of a
corpus split on <|endoftext|>, and print each run's time, MB/s, id count and
id sum, which equal bench_owt's when the outputs agree. Not on any alias.

usage: uv run packages/brot/bench/bench_owt_ref.py ENGINE TOKENIZER.json CORPUS MODE REPEAT

ENGINE is gt (gigatoken's encode_batch) or hf (tokenizers' encode_batch_fast,
without special tokens). MODE is st (one thread), mt (RAYON_NUM_THREADS
threads) or none, which stops before encoding: the baseline that, subtracted
from a callgrind count of a run with REPEAT passes, leaves the passes'. With
CHECK=0 the runs print nothing, so a count holds no summing.
"""

import os
import sys
import time

engine, tok_path, corpus, mode, repeat = sys.argv[1:6]
repeat = int(repeat)
data = open(corpus, "rb").read()
docs = [d for d in data.split(b"<|endoftext|>") if d]
nbytes = sum(len(d) for d in docs)
check = os.environ.get("CHECK", "1") == "1"

if engine == "gt":
    import gigatoken as gt

    tok = gt.Tokenizer(tok_path)
    parallel = mode == "mt"

    def encode(xs):
        return tok.encode_batch(xs, parallel=parallel)

    def summary(out):
        import awkward as ak
        import numpy as np

        return int(ak.count(out)), int(ak.sum(ak.values_astype(out, np.uint64)))

    # Warm what a first call sets up, as bench_owt does, without seeding the
    # pretoken caches with the corpus.
    encode([b"a"])
else:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    docs = [d.decode("utf-8", "surrogateescape") for d in docs]
    if mode == "st":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def encode(xs):
        return tok.encode_batch_fast(xs, add_special_tokens=False)

    def summary(out):
        return sum(len(e.ids) for e in out), sum(sum(e.ids) for e in out)

    encode(["a"])

if mode == "none":
    sys.exit(0)

best = float("inf")
for run in range(1, repeat + 1):
    t0 = time.perf_counter()
    out = encode(docs)
    dt = time.perf_counter() - t0
    best = min(best, dt)
    if check:
        n, s = summary(out)
        print(f"run {run}: {dt:.3f} s, {nbytes / 1e6 / dt:.1f} MB/s, {n} ids, sum {s}", flush=True)
if repeat > 0:
    print(f"best: {nbytes / 1e6 / best:.1f} MB/s")
