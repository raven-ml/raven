# todo

## next

goalpost: jit-compiled gpt-oss matching pytorch performance

perf follow-ups:
- fp16 train step: per-leaf unscale/isfinite/where plumbing over 148 leaves
  adds ~750 ms/step (compute itself is ~95 ms with TC engaged) — needs a
  fused/tree-level formulation
- rune warm start regressed: GPT-2's training step takes ~11 s on its first
  call with a warm compile cache on Metal (4.4 s when the cache landed,
  cd234a40e; 22 s cold). the cache hits, so the time is tracing, building the
  call and importing the cached program: profile by phase. one quadratic,
  `Uop.substitute`'s list lookup, is fixed (5cb655b6f)

rune/jit follow-ups:
- revisit the CPU's τ (rune quant.ml) if the block kernel comes to multiply
  bfloat16 at float32: over 32 experts float32's crossover puts τ in [254,
  573) and bfloat16's in [1020, 2298); rerun the grid at 128, 192, 256, 320
  and 384 routes, and lower τ if the two ranges meet, or give τ a dtype if
  they stay apart. drop this when RFC 0010 step 6 removes rune's quantised
  rules
- symbolic shapes through rune (inherit tolk's symbolic shrink/assign): one
  compiled kernel set for all positions, dissolves the fixed-shape kv-cache
  masks in kaun attention and per-prompt-length signatures
- tolk port onto nx.device: check that per-device budgets, `free_cache` and
  nx.device's bounded staging bound released storage, a dropped model's cached
  buffers, and cuda's pinned staging (tolk's allocator cache is unbounded and
  keyed by size today)

nx follow-ups:
- complex construction still assembles by rotation, so `complex ~re ~im` with
  a non-finite `im` leaves the real component NaN (`im * i` expands the real
  part as `im * 0`). reads are exact through the component view; closing this
  needs its dual, `float[s; 2] -> complex[s]` — the inverse isomorphism, and
  its own adjoint. it belongs in the movement family (same buffer, different
  view, dtype changes) rather than as a general bitcast: `complex[s]` *is*
  `float[s; 2]` structurally, so there is no punning to justify the wider op
- run the CUDA device suites (`packages/nx/test/device/cuda`) on an NVIDIA
  machine: they were rewritten on a Mac, where they build and skip

decode contract follow-ups (rfc 0002):
- the indexed store and the scatter-add over indices landed
  (`Op.scatter_indexed`, `packages/rune/bench/indexed_store`), kaun's cache
  write uses it and models have one fold; what it leaves: the cpu zero-copy path
  reuses no storage, so a cpu step still copies a written pool once per leaf;
  sharded traces keep the one-hot scatter until the custom kernel is exercised
  under multi; a chain of writes into one input copies once per write;
  propose the `split_reduceop` one-hot guard upstream with a 65536-row
  `test_index` case
- `Kaun.Cache_index`: slots that hold a block of positions, windowed layers
  that free old columns, tree speculation (a caller-given write target and a
  token-to-token mask), packed sequences with position reset: when a model
  needs them

next model targets:
- llama 3: tolk parity for the `05-llama` example
- quantized inference: gguf loading (tinygrad `gguf_load` parity; the tolk
  gpt2 example's gguf path), int8/int4 kernels (int4 currently rejected by
  rune's jit) — pairs with llama3
