# Convolution Optimization

Working on speeding up the extremely slow convolution in nx.

## Quick Start

```bash
# Run tests to ensure correctness
dune build @dev/conv2d

# Measure performance  
dune exec dev/conv2d/bench_conv.exe
```

## Structure

- `nx_conv.ml` - Optimized convolution implementation
- `test_conv.ml` - Tests using alcotest using nx_conv.ml
- `bench_conv.ml` - Performance benchmarks using ubench
- `optimization_log.md` - Track what we tried and results

## The Problem

Convolution is ~100x slower than it should be because:
1. The `pool` function creates too many intermediate tensors
2. Many forced `contiguous` calls copy memory unnecessarily  
3. Complex reshape/permute sequences
