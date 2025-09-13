# Performance Comparison: PyTorch vs OCaml/Rune

## Configuration
- Batch size: 256
- Sequence length: 16  
- Hidden dimension: 64
- Vocabulary size: 27
- Number of heads: 4

## Benchmark Results

| Operation | PyTorch (CPU) | OCaml/Rune | Slowdown | Notes |
|-----------|---------------|------------|----------|-------|
| Embedding | 0.070ms | 5.03ms | **71.9x** | Major bottleneck |
| MatMul | 0.048ms | 9.48ms | **197.5x** | Critical for all operations |
| LayerNorm | 0.145ms | 15.66ms | **108.0x** | Significant overhead |
| Attention | 4.891ms | 69.37ms | **14.2x** | Compound effect of matmul |
| GRU Cell | 0.254ms | 5.43ms | **21.4x** | Multiple matmuls |
| GRU Sequence | 5.044ms | 299.08ms | **59.3x** | Sequential bottleneck |
| Transformer Block | - | 389.83ms | - | Not measured in Python |

## Key Findings

### Critical Bottlenecks

1. **Matrix Multiplication (197.5x slower)**
   - Core operation used everywhere
   - Every attention head, FFN, and RNN gate uses matmul

2. **Embedding Lookup (71.9x slower)**
   - Should be a simple indexing operation
   - Suggests inefficient implementation

3. **GRU Sequence (59.3x slower)**
   - Compounds the matmul inefficiency over 16 timesteps
   - Sequential nature prevents parallelization

### Performance Impact on Training

For the makemore MLP model:
- Python: ~11ms per training step
- OCaml: ~26.6ms per training step
- **2.4x slower overall**

For GRU/LSTM/Transformer models, the slowdown is much worse due to:
- More matrix multiplications per step
- Sequential dependencies in RNNs
- Multiple attention heads in transformers

## Optimization Priorities

1. **Immediate: Matmul**
   - This alone could give 10-100x speedup on matmul

2. **High Priority: Optimize embedding**
   - Should be a simple gather operation
   - Current implementation seems inefficient

3. **Medium Priority: Batch operations**
   - Reduce overhead by batching small operations
   - Fuse operations where possible

4. **Consider: GPU support**
   - PyTorch benefits from years of CUDA optimization
   - CPU-only comparison is inherently limiting

## Next Steps

2. Profile the matmul implementation
3. Investigate embedding implementation
5. Implement operation fusion for common patterns