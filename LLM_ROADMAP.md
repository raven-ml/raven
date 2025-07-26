# roadmap to llms with raven

1. attention mechanisms
- [ ] multi-head attention layer
- [ ] scaled dot-product attention
- [ ] flash attention 2 (as an optimized backend)
- [ ] positional encodings (learned or sinusoidal)
- [ ] rotary positional embeddings (RoPE)

2. transformer Building Blocks
- [ ] layer normalization
- [ ] embedding layers (token + positional)
- [ ] feed-forward network (FFN/MLP block)
- [ ] transformer encoder/decoder blocks
- [ ] causal masking for autoregressive models

3. activation functions
- [ ] gelu activation (standard for transformers)
- [ ] silu/swish activation

4. training
- [ ] gradient clipping (by global norm)
- [ ] learning rate schedulers (cosine decay with warmup)
- [ ] mixed precision training support (bfloat16/float16 with loss scaling)
- [ ] gradient accumulation
- [ ] adamw optimizer (the standard for transformers)
- [ ] loss functions (softmax cross-entropy)

5. inference
- [ ] checkpoint saving/loading (including for sharded models)
- [ ] parameter counting
- [ ] tensor parallelism layers (e.g., `RowParallelLinear`, `ColParallelLinear`)
- [ ] kv-cache for efficient autoregressive inference
- [ ] generation/sampling methods (greedy, top-k, top-p, temperature)

6. data pipeline
- [ ] tokenizer integration (bindings to a library like Hugging Face tokenizers)
- [ ] streaming/lazy dataset support for terabyte-scale corpora
- [ ] sequence packing/batching strategies

7. parallelism
- [ ] data parallelism (DP)
- [ ] fully sharded data parallelism (FSDP)
- [ ] tensor parallelism (TP)
- [ ] pipeline parallelism (PP)
- [ ] collective communication primitives (e.g., `all_reduce`, `all_gather`)

1. interoperability
- [ ] evaluation metrics (perplexity)
- [ ] logging hooks (integrate with kaun-dashboard, built with mosaic)
- [ ] hugging face hub integration (downloading/uploading models)
- [ ] model quantization support (e.g., GPTQ, AWQ)
