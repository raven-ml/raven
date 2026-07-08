# Raven in Mid-2026: A Compiler, GPUs, and the Road to Beta

[Raven](https://github.com/raven-ml/raven) is an ecosystem of OCaml libraries for machine learning and scientific computing — the equivalents of NumPy, JAX, PyTorch, and their friends, rebuilt with type safety. This is a progress update covering the last eight months, and it's a big one: **Raven now runs on GPUs**. We've completed the port of tinygrad to OCaml, integrated it as Rune's JIT compiler, and validated it end-to-end by training and running GPT-2 on an H100 — with performance on par with tinygrad itself. With this, the scope of our first beta is complete.

## TL;DR

Since last November, we've shipped alpha3 (March 2026) and completed the full scope of beta1:

- **Tolk**, our new ML compiler, is a complete port of tinygrad to OCaml: it schedules tensor programs into fused kernels, optimizes them, and compiles them for CPU, CUDA, and Metal. Generated kernels are tested byte-identical to tinygrad's.
- **Rune's JIT is powered by Tolk**: a whole training step — forward, backward, and optimizer update — compiles into one program and runs on GPU. GPT-2 (124M) trains at ~110 ms/step on an H100 versus tinygrad's ~90 ms, and decodes at ~380 tokens/s.
- **Rune and Kaun were rewritten** on typed parameter trees, taking advantage of OCaml 5.5's modular explicits for a dramatically more ergonomic and flexible API.
- **Both Outreachy internships completed successfully**: nx-oxcaml, a pure-OCaml SIMD backend approaching C performance, and a training dashboard that has since evolved into Munin, a full experiment-tracking library.
- **Three new libraries**: Munin (experiment tracking, W&B/MLflow equivalent), Vega (optimizers, Optax equivalent), and Norn (MCMC sampling, BlackJAX/PyMC equivalent).

The beta scope is complete. We're not releasing beta1 immediately: it's being battle-tested by users who were waiting on JIT and related features to migrate to Raven, and we'll release once that validation settles.

## The Milestone: Raven on GPU

The centerpiece of this period is the completion of our JIT compilation pipeline, the goal we set for beta last year.

Back then, we were leaning toward porting tinygrad to OCaml rather than designing a new compiler. We committed to that path, and it paid off. The result is [**Tolk**](https://github.com/raven-ml/raven/tree/main/packages/tolk), a minimal ML compiler for tensor computation: a tensor program is a graph of micro-operations; Tolk schedules it into kernels, lowers and optimizes them, renders C-style source, compiles, and executes the result — entirely from OCaml.

What Tolk covers today:

- **Backends**: CPU (via Clang), CUDA (via NVRTC, with the driver loaded dynamically so builds need no CUDA toolkit), and Metal. Renderers additionally target AMD/HIP and OpenCL.
- **Tensor cores**: WMMA kernels lower end-to-end, with targets up to Hopper (SM90) and FP8 support.
- **Symbolic shapes**: one compiled kernel set serves every sequence position of a transformer decode step.
- **Multi-device execution**: sharded buffers, per-device kernel launches, and ring allreduce.
- **Capture-and-replay JIT** with CUDA-graph batching: whole kernel sequences replay as a single graph launch.

Because Tolk is a faithful port, we can verify it against tinygrad directly: for our GPT-2 validation workloads, every compiled kernel is byte-identical to the reference, and generated text reproduces the reference token stream exactly.

And the performance is there:

| GPT-2 124M on an H100 | Raven eager | Raven jitted         | tinygrad |
| --------------------- | ----------- | -------------------- | -------- |
| Training step         | —           | **110 ms**           | 90 ms    |
| Decoding (KV cache)   | ~4 tok/s    | **380 tok/s** (~85×) | —        |

The remaining training gap versus tinygrad is understood and not structural.

**Raven is now genuinely usable on GPU.**

## The Architecture Has Converged

Once Tolk was ready, we rewrote Rune and Kaun to take full advantage of it — and of OCaml 5.5's new modular explicits. Autodiff transformations (`grad`, `vjp`, `vmap`, ...) now operate over *typed* parameter structures: your model is a plain record of tensors, and gradients come back with the same structure and types, with no workarounds and no dynamic typing anywhere. Kaun shed its framework machinery: a layer is a record with a pure apply function, and a training step is code you own.

This is what a GPT-2 training step looks like today — forward, backward, and optimizer update, compiled into a single program (the full example lives in [`packages/kaun/examples/04-gpt2`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/04-gpt2)):

```ocaml
let train_step { Step_in.params; inputs; targets } =
  let loss, grads =
    Rune.value_and_grad (module Gpt2.Params) (objective inputs targets) params
  in
  let params, _ = Vega.sgd_step (module Gpt2.Params) ~lr state ~params ~grads in
  { Step_out.params; loss }

let step =
  Rune.jit2 ~device:"CUDA" (module Step_in) (module Step_out) train_step
```

Swap `jit2` for `pmap2 ~devices` and the *same step* is data-parallel across GPUs.

On top of the rewrite, Rune gained the features that make GPU training practical: `pmap` for data-parallel training across devices, JAX-style explicit RNG keys that compile under JIT, mixed-precision training (float16/bfloat16 with float32 master weights and loss scaling via Vega), a persistent compilation cache (warm starts skip compilation entirely), and device-resident outputs so training loops never round-trip through the host.

What struck us most is how *straightforward* this integration was. Once the compiler existed, wiring Rune and Kaun onto it took a fraction of the effort — the effect-based backend design we've been refining since alpha0 absorbed a whole ML compiler without friction. We take this as a strong signal that Raven's architecture has converged to the right one.

## Outreachy: Both Internships Shipped

Both internships from the December 2025 cohort completed successfully, and both projects shipped in alpha3 (March 2026):

- **nx-oxcaml** (Nirnay Roy): a pure-OCaml tensor backend using OxCaml's unboxed types and SIMD intrinsics, with BLIS-style parallel matmul kernels. Performance approaches our C backend — in pure OCaml. This gives the OxCaml team a real-world community use case for their extensions, as we hoped.
- **Training dashboard** (Arsalaan Alam): the terminal dashboard for monitoring training runs shipped as kaun-board in alpha3 — and has since grown well beyond the original scope (see Munin below).

## New Libraries

The library table keeps growing. Alongside the compiler work, Raven gained three new packages:

- **Munin** — local experiment tracking, the equivalent of Weights & Biases or MLflow without a server. The Outreachy dashboard evolved into a full tracking library: log metrics and artifacts from your training script, watch runs live in the terminal with `munin watch`, compare them with `munin compare`. Data is plain JSON on disk, so `jq` and shell scripts work out of the box. Useful for AI experiments, but equally for any kind of scientific experiment.

<figure>
<img src="/docs/assets/munin.png" alt="munin watch dashboard">
<figcaption><code>munin watch</code> monitoring a live training run — a CNN compiled with <code>Rune.jit</code> on Metal (~25 ms/step, ~5,100 images/s), with loss curves, hyperparameters, and system panels.</figcaption>
</figure>

- **Vega** — gradient-based optimizers and learning-rate schedules (SGD, Adam, AdamW, Lion, LAMB, Adafactor, ...), the equivalent of Optax. Optimizer state is shaped like your parameters, and everything threads through `jit` and `pmap`.
- **Norn** — Markov chain Monte Carlo with automatic gradients via Rune: HMC and NUTS with Stan-style adaptation and convergence diagnostics. The equivalent of BlackJAX/PyMC.

| Library | Description                                  | Equivalent    |
| ------- | -------------------------------------------- | ------------- |
| Nx      | N-dimensional arrays with pluggable backends | NumPy         |
| Tolk    | Minimal ML compiler (CPU, CUDA, Metal)       | tinygrad      |
| Rune    | Autodiff and JIT compilation                 | JAX           |
| Kaun    | Deep learning on Rune                        | Flax/PyTorch  |
| Vega    | Optimizers and LR schedules                  | Optax         |
| Munin   | Local experiment tracking                    | W&B/MLflow    |
| Norn    | MCMC sampling                                | BlackJAX/PyMC |
| Fehu    | RL environments and algorithms               | Gymnasium/SB3 |
| Brot    | Tokenization (formerly Saga)                 | HF Tokenizers |
| Talon   | DataFrames                                   | Pandas/Polars |
| Hugin   | Data visualization                           | Matplotlib    |
| Quill   | Notebooks and REPL                           | Jupyter       |
| Sowilo  | Differentiable computer vision               | OpenCV        |

Alpha3 also brought ground-up rewrites of Quill (terminal + web notebook interfaces) and Hugin (declarative plotting API), and Brot — the tokenization library formerly known as Saga — was rewritten and now outperforms HuggingFace Tokenizers' native Rust implementation on most workloads ([methodology and full results](https://github.com/raven-ml/raven/tree/main/packages/brot/bench)):

| Tokenization (Apple M3 Pro) | Brot (OCaml) | HF Tokenizers (Rust) | Speedup  |
| --------------------------- | ------------ | -------------------- | -------- |
| GPT-2 encode, 1 KB          | 46 µs        | 209 µs               | **4.5×** |
| GPT-2 encode, batch of 32   | 1.38 ms      | 3.05 ms              | **2.2×** |
| BERT decode, 64 KB          | 1.25 ms      | 7.63 ms              | **6.1×** |

## Road to Beta1 and Beyond

As stated above, the beta scope — a JIT-compiled GPT-2 with competitive performance — is complete. Rather than releasing immediately, we've put beta1 in the hands of the users who were waiting on it, and we'll cut the release once their migrations validate the design. The beta releases themselves will focus on smoothing the rough edges this testing surfaces: correctness, operator and layer coverage, and UX.

Beyond beta, we're opening our v1 scope, with one goal: **making Raven production-ready, for both training and deployment.**

**Training.** We're almost there. The tinygrad port gave us multi-device support essentially for free — `Rune.pmap` compiles data-parallel training across GPUs today, and the compiler's sharding layer already supports the model-parallel patterns larger models need. The remaining scope is deliberately small: operator and layer coverage, UX, and hardening Raven by training progressively larger models — Llama 3 is next. We're consciously scoping v1 to single-node training: production fine-tuning today is overwhelmingly a single-node, multi-GPU workload, and that's the ground we intend to cover exceptionally well before considering multi-node.

**Deployment.** This is the more exploratory arc, and the one we're most excited about. The goal is for Raven to have a great runtime for running models: ahead-of-time compilation (Tolk generates all compute, so a deployed model needs no BLAS, no Python, no JIT at startup), quantized inference (int8/int4, GGUF loading), and an inference runtime with the features that make serving fast — KV-cache management, prefix caching, continuous batching — behind an OpenAI-compatible endpoint. We won't build observability and ops layers; the focus is the runtime itself, competitive with modern inference engines.

The idea we keep coming back to is **compiling models as unikernels**: a model AOT-compiled into a self-contained binary running directly on the hypervisor, with no operating system underneath. For companies deploying models, this has very real value — in security (a minimal attack surface, no shell, no userland) and in cost (millisecond boot times make scale-to-zero and horizontal scaling practical). We'll get there stepwise: static binaries first, then CPU unikernels, then the GPU story.

## Thanks

None of this happens without support. Thank you to our sponsors — [Ahrefs](https://ahrefs.com), [Tarides](https://tarides.com), and our individual sponsors on [GitHub Sponsors](https://github.com/sponsors/tmattio) — whose backing carried us through the most technically ambitious stretch of the project so far: a complete ML compiler in OCaml, GPU support at tinygrad parity, and the mentoring of two successful Outreachy interns whose work now ships in Raven. Thank you as well to everyone who contributed code, filed issues, and tested the alphas.

A year ago, Raven was a NumPy equivalent with promising foundations. Today it trains transformers on H100s at the performance of the Python ML stack — in OCaml, with typed APIs throughout. We're closing in on beta1.

If Raven's direction resonates with you or your company, you can [sponsor the project](https://github.com/sponsors/tmattio) or [get in touch](mailto:thibaut.mattio@gmail.com).
