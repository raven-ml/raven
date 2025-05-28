# Convolution Optimization

Working on speeding up the extremely slow convolution in nx.

## Quick Start

```bash
# Run tests to ensure correctness
dune test src/

# Measure performance  
dune exec src/bench_conv.exe

# After making changes to src/conv_optimized.ml:
dune test src/  # Make sure nothing breaks
dune exec src/bench_conv.exe  # See if it's faster
```

## Structure

All files are in `src/`:
- `conv_optimized.ml` - Optimized convolution implementation
- `test_conv_optimized.ml` - Tests using alcotest (non-trivial cases)
- `bench_conv.ml` - Performance benchmarks using ubench
- `dune` - Build configuration

Other files:
- `optimization_log.md` - Track what we tried and results
- `patches/` - Save working optimizations as patches

## The Problem

Convolution is ~100x slower than it should be because:
1. The `pool` function creates too many intermediate tensors
2. Many forced `contiguous` calls copy memory unnecessarily  
3. Complex reshape/permute sequences

## Current Status

Setting up to test optimizations...