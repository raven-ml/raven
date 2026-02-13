# mimir

Experimental inference engine for raven.

The gap between "I can run a forward pass" and "I can serve a model in production" is large. mimir is where we figure out what the OCaml answer to that gap looks like.

## Current state

The sampling layer: composable logits processors (temperature, top-k, top-p, repetition penalty, n-gram blocking), stopping criteria, and the autoregressive generation loop operating on nx tensors.

This is the outermost piece of the inference puzzle — the part that turns model logits into actual token sequences. Everything below is open.

## What we want to explore

**Memory management for KV cache.** The attention mechanism produces intermediate state (keys and values) that grows linearly with sequence length. Naive allocation wastes memory; the interesting question is whether we can apply OS-style virtual memory ideas — fixed-size blocks, deferred allocation, reference-counted sharing — to make long sequences and shared prefixes cheap. This is the core idea behind PagedAttention.

**Request scheduling.** A single request is simple. Thousands of concurrent requests with different prompt lengths, generation limits, and priority levels is a scheduling problem. Batching amortizes GPU overhead but introduces latency trade-offs. Continuous batching (letting new requests join mid-batch as others finish) changes the calculus further. OCaml's algebraic types and pattern matching may give us a cleaner expression of scheduling policies than the typical mutable-state approach.

**Prefill/decode asymmetry.** The two phases of autoregressive generation have opposite performance characteristics — one is compute-bound, the other memory-bound. An engine that treats them identically leaves performance on the table.

**JIT compilation of decode steps.** The decode phase repeats the same computation graph with different inputs. If rune's JIT can capture and replay these graphs, we avoid per-step compilation overhead — similar in spirit to CUDA graph capture.

**Structured generation.** Constraining the sampling step so that output conforms to a grammar, regex, or JSON schema. This means masking logits at each step based on what the constraint automaton allows, which interacts with the sampling pipeline we already have.

**Tensor parallelism.** Splitting a model across multiple devices. This is a rune-level concern more than a mimir concern, but the inference engine needs to coordinate it.

## References

- [Nano-vLLM](https://github.com/GeeeekExplworker/nano-vllm) — minimal (~1,200 lines) inference engine by a DeepSeek contributor, good for understanding the essential moving parts
- [vLLM: PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [SGLang](https://github.com/sgl-project/sglang) — alternative engine with RadixAttention for prefix sharing
