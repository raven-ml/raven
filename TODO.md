# todo

## beta (jit)

goalpost: jit-compiled gpt2 matching pytorch performance

- review outdated website docs

perf follow-ups:
- fp16 train step: per-leaf unscale/isfinite/where plumbing over 148 leaves
  adds ~750 ms/step (compute itself is ~95 ms with TC engaged) — needs a
  fused/tree-level formulation
- rune warm start is now trace-dominated (~3.6 s effect replay +
  transform_to_call)

rune/jit follow-ups:
- an empty output of a compiled function on Metal raises "Metal OOM while
  allocating buffer", over host leaves too (`jit' ~device:"METAL" (fun v ->
  Nx.mul_s (Nx.slice [R (1, 1)] v) 2.0)`); `test_zero_size_outputs` runs on
  the CPU only
- symbolic shapes through rune (inherit tolk's symbolic shrink/assign): one
  compiled kernel set for all positions, dissolves the fixed-shape kv-cache
  masks in kaun attention and per-prompt-length signatures
- donation: same-call buffer reuse via static last-read-before-first-write
  analysis on the linear schedule, lru pool watermark knob
- ptree product combinators (`Pair`/`List_of`/`Leaf` functors) to absorb
  hand-written jit2 step modules; dedupe jit.ml's two inline leaf modules

nx follow-ups:
- complex construction still assembles by rotation, so `complex ~re ~im` with
  a non-finite `im` leaves the real component NaN (`im * i` expands the real
  part as `im * 0`). reads are exact through the component view; closing this
  needs its dual, `float[s; 2] -> complex[s]` — the inverse isomorphism, and
  its own adjoint. it belongs in the movement family (same buffer, different
  view, dtype changes) rather than as a general bitcast: `complex[s]` *is*
  `float[s; 2]` structurally, so there is no punning to justify the wider op
- restore the `magnitude` gradient case in rune's test_ops once the component
  view lands — it needs the complex `abs` conjugate fix, now on main
- complex gradient convention is written down nowhere and is unaudited: 16
  conjugate-sensitive reverse rules are reachable on complex, only `abs` has
  ever met a finite-difference oracle, and it was wrong

decode contract follow-ups (rfc 0002):
- brot: load llama 3's tokenizer (cl100k-family split regex, one negative
  lookahead); then the llama example takes text
- the indexed store and the scatter-add over indices landed
  (`Op.scatter_indexed`, `packages/rune/bench/indexed_store`), kaun's cache
  write uses it and models have one fold; what it leaves: the cpu zero-copy path
  reuses no storage, so a cpu step still copies a written pool once per leaf;
  sharded traces keep the one-hot scatter until the custom kernel is exercised
  under multi; a chain of writes into one input copies once per write;
  propose the `split_reduceop` one-hot guard upstream with a 65536-row
  `test_index` case
- `Rune.remat` is an identity under jit: when a training run needs the memory
- `argsort` under jit recovers indices by matching every entry against every
  sorted entry, quadratic in the axis: carry the indices through the sort's
  swaps, comparing (value, position) pairs to stay stable; propose upstream
- `Nx.top_k` above 16 entries is a whole sort: a partial-selection kernel when
  a layer picks thousands of columns from a long axis
- `Kaun.Cache_index`: slots that hold a block of positions, windowed layers
  that free old columns, tree speculation (a caller-given write target and a
  token-to-token mask), packed sequences with position reset: when a model
  needs them
- storage reuse under `pmap` per shard: inside the placement rfc

loading weights follow-ups (rfc 0003):
- rune stages an upload in a `Bytes` it keeps per distinct leaf size for the
  life of the compiled function (`jit.ml` `scratch_bytes`): bound it to one
  chunk, which is stage 3's chunked path
- the chunked path must synchronize once the bytes copied since the last
  synchronize pass a bound, or cuda parks a pinned staging buffer per copy
  until the next one
- tolk's allocator cache is unbounded and keyed by size: a process that drops
  one model and loads another holds both until an allocation fails
- `Kaun_hf.load_checkpoint` probes `model.safetensors.index.json` over the
  network on every call for an unsharded repository (~0.25 s, and it needs the
  network); remember the miss in the cache
- `Kaun_hf` detects curl with `command -v` through `Unix.system`, which is
  `cmd.exe` on windows, so downloads cannot work there

next model targets:
- llama3 in kaun-models + tolk parity (rope, rmsnorm, gqa, sharded
  safetensors; llama.py-style per-weight model-parallel axis choices become
  expressible once `Tensor.shard` lands)
- quantized inference: gguf loading (tinygrad `gguf_load` parity; the tolk
  gpt2 example's gguf path), int8/int4 kernels (int4 currently rejected by
  rune's jit) — pairs with llama3

## v1 (production)

goalpost: end-to-end train -> deploy as unikernel or static binary

training:
- gradient accumulation
- flash attention (tolk kernel and/or kaun.fn primitive)
- parallel data loading (ocaml 5 domains, background prefetch; overlap host
  i/o with device compute — the host is idle during steps since residency)
- checkpoint hardening: optimizer-state save/resume, atomic/async writes
- layer completions: transposed conv, group norm, full conv2d stride/dilation/padding
- onnx import (onnx -> tolk ir adapter, cover resnet/bert/gpt2/llama/vit/whisper ops)

deployment:
- aot compilation: cpu (c via clang, musl static linking) and gpu (cuda/metal/opencl)
- mimir: kv cache, continuous batching, pagedattention
- mimir: http server (rest api, /health, /metrics, sigterm, structured logging)
- post-training quantization (int8/int4, tolk quantized kernels)
- mirageos unikernel deployment (raven-mirage package)
  - no blas dep (tolk aot generates all compute)
  - weight loading via network (mirage-http)
  - verify ocaml 5 effects on mirageos runtime
  - http server on mirageos network stack
