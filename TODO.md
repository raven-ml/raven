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

## bugs

- rig: a copy can raise another device's pending fault through a shared
  staging half. Copy's `names_lost` replaces a half only for devices already
  lost; a device faulted but not yet seen lost leaves its unreached use on the
  half, and the next copy through it waits on that point, surfaces the fault
  and raises the other device's `Lost`. Seen on the Mac in rig.model, "every
  call answers as rig.mli says", seed s1:6946fc93e9c8e1e8 (case 229, a
  transfer between two Polled devices raising a lost device of case 78). It
  should vanish with commit 4a's per-device staging slots: check the seed then
- rig: `Rig.set_budget` on a device that faulted can raise `Lost` from
  `Memory.give_back` while a copy on another domain loses it. rig.model, "calls
  on two domains answer as some order of them", seed s1:42573910ee757b5b, fails
  about one run in three on the base before commit 9 as on commit 9 (budget
  1048576 on a faulted polled-copyless-itself device). Fix test-first from the
  seed

## perf

- rig pool, kimchi: launch/empty-performance-cores and launch/empty-all-cores
  time the same 6-thread job and read 0.51-0.71 us by process, each process
  steady within 0.4%, so a bless records one mode; find what sets the mode and
  make the rows hold one
- rig pool stall bound: a waiting thread parks once the job it waits on has
  been closed with a thread inside for 10-20 us. In a probe it cut the jobs
  that last a spin window when a preempted thread holds every other core
  (M1 claim job past 100 us: 1.2-2.7% to 0.28-0.36%; kimchi beside a busy
  process: 0.41-0.46% to 0.01-0.03%) at 2-3 times the parks; measure it on
  every compute, launch and claim row of both hosts before it lands
