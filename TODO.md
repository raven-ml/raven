# todo

## alpha3

- bring in talon changes
- consider rename talon to talf
- explore renames (sowilo, talon, etc.)
- quill: fix bytecode+threads segfault
- hugin: fix plot3d
- hugin: fix contour

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
- parallel data loading (ocaml 5 domains, background prefetch)
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

docs/website:
- landing page rewrite with benchmarks
- deployment guide (aot, static binary, docker, mirageos, gpu)
- end-to-end examples (serving, onnx+deploy workflow)
