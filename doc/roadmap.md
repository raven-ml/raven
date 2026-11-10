# Roadmap

Raven is an ecosystem of OCaml libraries for machine learning and scientific computing: equivalents of NumPy, JAX, PyTorch, Pandas, Matplotlib, and their friends that compose as ordinary OCaml code.

Today, the stack works end-to-end: every layer runs in OCaml, compiled for the GPU. This page is our road from that demo to 1.0, the release where Raven can replace a Python stack in production.

## Where we are

A year ago we scoped our beta around JIT compilation, with training GPT-2 on GPU as the goal post. That bar is met: a GPT-2 training step compiles into a single GPU program and trains on an H100. Every layer of that run is Raven: Brot tokenizes the corpus, Kaun defines the model, Rune differentiates it, Tolk -- our port of tinygrad -- compiles it, Vega updates the weights, and Munin records the metrics. The whole pipeline is one OCaml program.

The next goal post on the road to 1.0 raises the bar to a mixture-of-experts model: gpt-oss-20b, jit-compiled, at least as fast as tinygrad, JAX, llama.cpp and vLLM on NVIDIA (through CUDA and through NVIDIA's kernel driver directly) and on AMD. [Performance](performance.md) tracks where we stand against it.

We're also pretty happy with where we landed on our design: in particular on an Nx whose every function issues a small set of primitive operations, and Rune on top interpreting them, through effect handlers, for automatic differentiation, JIT compilation, and vectorization. The APIs will keep changing until v1, but we like how well they compose and make difficult things simple without unnecessary abstractions.

Raven's ambition is bigger than LLM workflows: a complete numerical computing ecosystem for OCaml. That broader goal has progressed too, with Hugin for plotting, Talon for dataframes, and Quill for notebooks. Norn brings probabilistic inference to the core. Fehu for reinforcement learning and Sowilo for image processing are contrib packages, which live in the same repository and release on their own schedule.

## The road ahead: 1.0

Our next milestone from here is to release 1.0. Our broad acceptance bar for 1.0 is "production-ready" and being able to replace the typical Python stack with Raven in real-world scenarios. We define two workflows in particular that will drive the development and priorities:

1. **Training** -- fine-tuning open-weight LLMs: LoRA fine-tuning of Llama 3.1 70B on one 8×H100 node, and full fine-tuning across two.
2. **Deployment** -- serving open-weight LLMs behind an OpenAI-compatible API, from a single static binary: Llama 3.1 70B from bf16 on 4×H100 down to INT4 on one GPU, and DeepSeek-V4-Pro at its native mixed FP4/FP8 precision, expert-parallel across two 8×H200 nodes.

For these two workflows, the acceptance bar is performance on par with, or exceeding, the equivalent Python stack (PyTorch with FSDP or PEFT for training, vLLM for serving), meaning that beyond the change of language, there should be no friction to migrate from Python to Raven.

Llama 3.1 70B is the reference model because at that scale every hard problem must actually be solved (fine-tuning forces sharding and multi-node training; serving forces tensor parallelism and quantization) and because it's a scale real teams run in production. The mixture-of-experts model is there because that's the shape the open-weight frontier now ships in. When both workflows hold this bar, and every core package clears its readiness bar, we ship 1.0.

## Training

Training in Raven follows the architecture JAX proved out: a training step -- forward, backward, optimizer update -- is a single program, and everything that makes it scale is a transformation of that program. Our bet is that this functional design belongs in a functional language, and that algebraic effects are the right way to build the transformations: `grad`, `vmap`, and `jit` are effect handlers over the same step, and a step's values carry the devices they lie on. Getting to a 70B fine-tune means composing more transformations, and making them fast.

**Goal post.** We fine-tune Llama 3.1 70B at 90% or better of the throughput of the equivalent PyTorch stack (torch with FSDP for full fine-tuning, torch with PEFT for LoRA), as (1) LoRA fine-tuning on a single 8×H100 node and (2) full fine-tuning across two 8×H100 nodes.

**Milestones:**

- ✓ **Llama-class models on Kaun.** Kaun provides the modern decoder block (RoPE, grouped-query attention, RMSNorm, SwiGLU) and a Llama written as user code with validated checkpoint import: load the open weights from safetensors and match the reference implementation's logits, layer by layer. Done for Llama 3.2 1B and TinyLlama 1.1B in [`kaun/examples/05-llama`](../packages/kaun/examples/05-llama): every block, float32 and half precision, eager and compiled, whole sequences and ragged batches through the key-value caches.
- **Mixed-precision training.** bf16 compute with fp32 master weights in Vega's optimizers, and making `Rune.remat`'s memory savings hold through the compiled path. A 70B fine-tune doesn't fit in memory without both. `Rune.remat` already recomputes under `Rune.jit`, and Vega steps bf16 parameters at float32 with stochastic rounding.
- ✓ **Memory-efficient attention.** A flash-attention-class fused kernel generated through Tolk, with O(n) memory at real context lengths. Done in Rune's attention kernels: each query row folds over its keys in tiles with an online softmax, and the backward pass recomputes each weight from the row's log-sum-exp.
- **GEMM throughput at parity.** Most of a training step is plain bf16 matmuls, so Tolk's generated GEMMs must approach cuBLAS/optimized CUDA throughput at Llama shapes.
- **JIT-compiled linear algebra and FFT.** Completing the machine-learning story beyond neural networks: common linear algebra operations (QR, SVD, linear solves) and FFTs compile through Tolk like any other op, so models that lean on them -- Gaussian processes, Kalman filters, spectral methods -- get the same compiled path as deep learning. Cholesky, LU, QR, SVD, symmetric eigendecomposition, linear solves and FFTs compile today at float32 and float64; general eigendecomposition and complex matrices remain.
- **LoRA in Kaun.** Adapters over any parameter tree, with merged-weight export.
- **The data pipeline.** Brot at Gigatoken-level throughput, and streaming tokenized datasets fast enough that the GPUs never wait for data.
- **Distributed training, in three steps.** First, single-node data parallelism: a step over a batch cut across devices compiles under `Rune.jit` into one program today, summing the devices' partial gradients; it now needs to run across several GPUs, with collectives reliable on CUDA and overlapped with compute. Second, FSDP-style sharding of parameters, gradients, and optimizer state, cut across devices by their placements: `Rune.jit` compiles such a step today, gathering each layer's parameters where it runs, and Vega's optimizers and a real model have yet to use it. Third, multi-node execution: our own collectives over InfiniBand with GPUDirect RDMA, integrated into Tolk's scheduler so communication hides behind compute, plus a launcher to bring up the cluster -- our equivalent of torchrun. Full fine-tuning of a 70B model takes roughly 1.1 TB of training state, more than an 8×H100 node holds, so the goal post forces multi-node.
- **Checkpointing at scale.** Sharded save and resume of the full training state -- parameters, optimizer state, data position -- so a multi-day run survives a node failure.
- **The benchmark harness.** The goal post is comparative, so the comparison itself is a deliverable: pinned reference stacks, re-runnable benchmarks, published numbers, and automated loss-parity checks.

## Deployment

Raven aims at zero system dependencies by design. Today it builds and installs with nothing beyond OCaml, and compiling kernels at run time loads the platform's compiler: Clang on the CPU, NVRTC on NVIDIA, comgr on AMD. Ahead-of-time compilation, below, removes that last dependency. It's part of why we chose tinygrad as the model for our ML compiler: it cuts layers of complexity and aims to compile to GPU kernels and talk to the GPU without depending on the CUDA userspace. We're building on that to provide a much better ML deployment story: compile your whole application, model included, into one static binary that you can just ship, with nothing to install on the host beyond the GPU's kernel driver. The binary carries the weights, the tokenizer, the inference engine, and the GPU kernels. Small models start in milliseconds(!!); at 70B scale, cold start is bounded by how fast the weights stream from disk. As far as we know, no existing stack does this at production serving performance, which is why deployment is the pillar we're most excited about.

**Goal post.** We serve open-weight models behind an OpenAI-compatible API as a dependency-free static binary, at 90% or better of vLLM's throughput, in three configurations:

- Llama 3.1 70B at bf16, tensor-parallel on 4×H100. The full-precision reference.
- Llama 3.1 70B at INT4 on one H100, compiled ahead of time. The smallest machine a 70B runs on.
- DeepSeek-V4-Pro at its native mixed FP4/FP8 precision, expert-parallel across two 8×H200 nodes. The precision the frontier ships in, on the smallest machine that serves it at full context.

**Milestones:**

- **In-process inference.** Small and mid-size models -- embedders, classifiers, rerankers -- compiled ahead of time and linked into an OCaml service: running a model is a function call, and the separate model server is gone. This depends only on the ahead-of-time compilation work, so it ships well before the inference engine, and we expect Raven's first production deployments to look like this.
- **The inference engine.** A new library in the ecosystem: KV cache, continuous batching, and PagedAttention, following vLLM's design.
- **The serving layer.** A demo of an OpenAI-compatible API with token streaming backed by Raven-compiled frontier open-weight models.
- **Tensor-parallel inference.** A 70B model at bf16 doesn't fit on one GPU, so the engine needs to shard it across several. The Llama example compiles a tensor-parallel decode step today.
- **Post-training quantization.** Weight-only INT8 first, then calibrated INT4, which is what fits 70B on a single GPU. Weight-only INT8 is done: `Nx_quant.quantize_q8_0`.
- **MoE serving.** The DeepSeek-V4 family in Kaun: validated import of the 1.6T checkpoint and its compressed-attention kernels. Then expert routing, expert-parallel execution alongside tensor parallelism, and FP8 and FP4 execution paths, the precision the model ships in. The gpt-oss example already routes tokens to experts, runs them expert-parallel within a node, and multiplies MXFP4 weights in compiled kernels.
- **Multi-node serving.** The engine spans nodes, reusing the collectives and launcher from distributed training. A trillion-parameter MoE at native precision and full context doesn't fit one node. Our parity claims stop at two nodes.
- **Ahead-of-time compilation.** On CPU, compile the model to native code and link it statically; on GPU, precompile kernels for the target architecture and embed them. Nothing gets JIT-compiled at startup.
- **Weight streaming.** Checkpoints load at disk bandwidth: mmap, direct I/O, and host-to-device copies overlapped per GPU. Rig maps files, reads them on several threads, and overlaps a copy's disk reads with its transfer to the GPU; direct I/O and loading several GPUs at once remain.
- **The driver-less CUDA path.** Talking to the GPU directly, without any CUDA userspace. We're building it for 1.0, but 1.0 doesn't gate on it; the standard CUDA driver path ships either way. Rig drives NVIDIA GPUs through the kernel driver with no CUDA library, and gpt-oss-20b decodes that way; kernels still compile through NVRTC.
- **The serving benchmark.** Throughput and inter-token latency against the pinned vLLM, published and re-runnable.

## Support us

If your team wants to take part, as an early adopter, a production pilot, or a sponsor, see [Support Raven](support-raven.md).
