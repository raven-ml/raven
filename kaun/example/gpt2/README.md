# GPT-2 Implementation with Kaun

This directory contains a GPT-2 implementation using the Kaun neural network library. The implementation follows the architecture from the original GPT-2 paper and is inspired by the Flax implementation structure.

## Files

- `config.ml` - Model configurations for different GPT-2 sizes (small, medium, large, XL)
- `attention.ml` - Multi-head self-attention implementation with causal masking
- `mlp.ml` - Feed-forward network with GELU activation
- `block.ml` - Transformer block combining attention and MLP with residual connections
- `gpt2.ml` - Main GPT-2 model and language model head
- `main.ml` - Example usage and training loop
- `kaun_extensions.ml` - Documentation of missing features needed in Kaun/Rune

## Architecture

The GPT-2 model consists of:

1. **Token and Position Embeddings**: Learned embeddings for input tokens and positions
2. **Transformer Blocks**: Stacked blocks each containing:
   - Layer normalization
   - Multi-head self-attention with causal masking
   - Residual connection
   - Layer normalization
   - Feed-forward network (MLP)
   - Residual connection
3. **Final Layer Norm**: Applied before the output layer
4. **Language Model Head**: Linear projection to vocabulary size

## Model Sizes

| Model | Parameters | Layers | Hidden Size | Heads |
|-------|-----------|--------|-------------|-------|
| GPT-2 Small | 124M | 12 | 768 | 12 |
| GPT-2 Medium | 355M | 24 | 1024 | 16 |
| GPT-2 Large | 774M | 36 | 1280 | 20 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 |

## Missing Features in Kaun/Rune

The implementation identifies several features that need to be added to Kaun/Rune:

### Essential Tensor Operations
- **Slicing**: Multi-dimensional tensor slicing with start/end indices
- **Split**: Splitting tensors along an axis
- **Where**: Conditional element selection
- **Triangular operations**: `tril`, `triu` for masking
- **Concatenation**: Joining tensors along an axis
- **Range creation**: `arange` for position indices

### Layers & Modules
- **Dropout**: Random dropout during training (may exist)
- **Embedding**: Token/position embeddings (may exist)
- **Layer Normalization**: Per-layer normalization (may exist)

### Training Infrastructure
- **Cross-entropy loss**: With ignore index for padding
- **Optimizers**: AdamW with weight decay
- **Learning rate scheduling**: Cosine schedule with warmup
- **Mixed precision training**: For efficiency

### Model Features
- **Weight tying**: Sharing weights between embedding and output
- **KV caching**: For efficient autoregressive generation
- **Checkpointing**: For large model training

### Generation Utilities
- **Sampling strategies**: Top-k, Top-p (nucleus), temperature
- **Beam search**: For better quality generation

## Usage

```ocaml
(* Initialize model *)
let config = Config.gpt2_small in
let model = Gpt2.gpt2_lm_head ~config () in
let params = Kaun.init model ~rngs ~device ~dtype in

(* Forward pass *)
let output = Kaun.apply model params ~training:false input_ids in

(* Training step *)
let loss, grads = Kaun.value_and_grad loss_fn params in
let new_params = optimizer_step params grads in
```

## Next Steps

1. **Implement missing operations**: Add the essential tensor operations to Rune
2. **Complete layers**: Ensure all required layers exist in Kaun
3. **Add training utilities**: Implement proper loss functions and optimizers
4. **Load pretrained weights**: Support loading weights from HuggingFace or other sources
5. **Generation pipeline**: Implement proper text generation with sampling
6. **Benchmarking**: Compare performance with PyTorch/JAX implementations

## Notes

This is a work-in-progress implementation designed to:
1. Explore the Kaun API and identify missing features
2. Provide a blueprint for transformer implementations in OCaml
3. Guide the development of necessary extensions to the Kaun/Rune ecosystem

Many operations are currently placeholders that will fail at runtime. See `kaun_extensions.ml` for detailed documentation of what needs to be implemented.