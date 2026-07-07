# todo

## next (gpt2 parity + multi-device follow-ups)

production leverage (independent of the multi-device wave, parallelizable):
- mixed precision end-to-end: bf16 training + fp16 inference. the tensor-core
  wmma path is byte-parity-tested but no model has ever exercised it. needs
  dtype plumbing through kaun/rune (loss scaling for fp16), bf16 safetensors
  loading (currently f32-only), and a HALF gpt2 acceptance run against
  tinygrad's `HALF=1` mode
- rng + dropout: tolk has `Threefry` in the uop layer with codegen
  decomposition, but the rng frontend is missing (`rand`/`manual_seed`/
  `multinomial`, per-device seed scheme) and kaun's gpt2 omits dropout.
  blocks stochastic training and temperature sampling; well-scoped parity work
- differential fuzzing: generate random small programs, cross-check nx-eager
  vs rune-jit vs tolk-cpu vs tolk-cuda (optionally the tinygrad reference).
  the recurring bug class this cycle was silent wrongness (`Hashtbl.hash`
  cache collision, scatter-`Add` zeroing, `Mselect` first-child stub, graph
  stale addresses) — all caught incidentally; hunt it systematically
- persistent compilation cache for rune: tolk disk-caches compiled kernels,
  but rune re-schedules and re-lowers in-process every start (~16 s first
  train step, 1-2 s per decode signature). cache the scheduled/lowered
  program keyed by trace signature for near-zero warm starts

multi-device (single-gpu work):
- tolk frontend `Tensor.shard`/`shard_`/`to_` on device tuples (schedule and
  engine underneath are done and golden-pinned; move `shard`/`unshard` to uop
  ownership)
- re-audit multi parity when the tinygrad reference (`_tinygrad`, a330bfffc)
  is next bumped — its multi layer is actively evolving

multi-device (needs >= 2 gpus; land code-complete, hardware-gated):
- cuda peer access (`cuDeviceCanAccessPeer`/`cuCtxEnablePeerAccess` on open)
- eventful cross-context `_transfer` (allocator `transfer` gains src/dest
  device params — touches every backend record)
- multi-device graph batching (per-shard graph nodes, `MultiGraphRunner`
  same-backend rule; today multi calls run ungraphed — correct but unbatched)

rune/jit follow-ups:
- symbolic shapes through rune (inherit tolk's symbolic shrink/assign): one
  compiled kernel set for all positions, dissolves the fixed-shape kv-cache
  masks in kaun attention and per-prompt-length signatures
- donate phase 3: per-leaf donation mask (`in_axes`-style), same-call buffer
  reuse via static last-read-before-first-write analysis on the linear
  schedule, lru pool watermark knob
- reverse-mode kernel count: rune schedules 1469 kernels for the gpt2 train
  step vs tinygrad's 665 (gpu-time impact nil at this size; revisit at scale)
- ptree product combinators (`Pair`/`List_of`/`Leaf` functors) to absorb
  hand-written jit2 step modules; dedupe jit.ml's two inline leaf modules

next model targets:
- half/fp16 gpt2 (the acceptance vehicle for mixed precision above)
- llama3 in kaun-models + tolk parity (rope, rmsnorm, gqa, sharded
  safetensors; llama.py-style per-weight model-parallel axis choices become
  expressible once `Tensor.shard` lands)
- quantized inference: gguf loading (tinygrad `gguf_load` parity; the tolk
  gpt2 example's gguf path), int8/int4 kernels (int4 currently rejected by
  rune's jit) — pairs with llama3

note: remaining train-perf gap vs tinygrad (110 vs 89.7 ms/step) is gpu
kernel time + the fingerprint-read protocol; both understood, neither a bug.

## beta (jit)

goalpost: jit-compiled gpt2 matching pytorch performance

perf:
- close rune grad performance gap (within <2x of pytorch)
- close nx performance gaps (within <2x of numpy)

tolk:
- integrate tolk as rune jit transformation
- kernel fusion and optimization
- cpu, cuda, metal backends

## v1 (production)

goalpost: end-to-end train -> deploy as unikernel or static binary

training:
- gradient accumulation
- mixed precision (fp16/bf16 forward, fp32 master weights, loss scaling)
- gradient checkpointing (rune.checkpoint, recompute activations in backward)
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

observability:
- timeline profiler (tinygrad `VIZ=1` is the parity target)
- per-kernel timings surfaced through rune
- memory-stats api: resident bytes, arena sizes (extend `jit_stats`)

ci/release:
- green ci on cpu-only runners (`RUNE_JIT_FORCE_COPY` + cpu-tuple multi
  tests were designed for this); gpu runner for the cuda suites
- opam beta release once the api freeze list clears (capture rule done,
  `Attention.Cache` rename done)

docs/website:
- landing page rewrite with benchmarks
- deployment guide (aot, static binary, docker, mirageos, gpu)
- end-to-end examples (serving, onnx+deploy workflow)
