# Convolution Optimization Log

## Baseline Performance

```
Tiny (1,1,4,4): 0.000225s
Small (1,1,8,8): 0.000450s  
Medium (1,4,16,16): 0.008s
Large (1,8,32,32): 0.64s  🔥 Way too slow!
```

## Method

1. Copy function to optimize from nx/lib/core/frontend.ml
2. Make changes in src/conv_optimized.ml
3. Test: `dune build @runtest` (must pass!)
4. Measure: `dune exec benchmarks/simple_test.exe`
5. If better: save as patch, if not: revert

## Next Ideas
- [ ] Fix the pool function to use fewer operations
- [ ] Combine multiple reshapes into one
- [ ] Avoid padding when not needed

## Attempts

### 1. Reduce contiguous calls in correlate_nd_general
- **What**: Try reshape without calling contiguous first
- **Result**: TODO - need to test
- **Patch**: `patches/01_correlate_nd_reduce_contiguous.patch`

### 2. Fast path for 3x3 stride=1 (FAILED)
- **What**: Special case for most common convolution
- **Result**: ❌ Broke avg_pool1d tests - reshape logic was wrong
- **Learning**: Need to be careful about dimension handling

