# todo

## beta (jit)

goalpost: jit-compiled gpt2 matching pytorch performance

- review outdated website docs

perf follow-ups:
- fp16 train step: per-leaf unscale/isfinite/where plumbing over 148 leaves
  adds ~750 ms/step (compute itself is ~95 ms with TC engaged) — needs a
  fused/tree-level formulation
- rune warm start is now trace-dominated (~3.6 s effect replay +
  transform_to_call); weight loading (safetensors, 2.8-9.4 s) dominates
  example warm starts

rune/jit follow-ups:
- symbolic shapes through rune (inherit tolk's symbolic shrink/assign): one
  compiled kernel set for all positions, dissolves the fixed-shape kv-cache
  masks in kaun attention and per-prompt-length signatures
- donate phase 3: per-leaf donation mask (`in_axes`-style), same-call buffer
  reuse via static last-read-before-first-write analysis on the linear
  schedule, lru pool watermark knob
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
