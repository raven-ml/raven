# Convolution Optimization Log

## Baseline Performance

┌───────────────────┬────────────────┬─────────────┬────────────┐
│ Name              │       Time/Run │     mWd/Run │ vs Fastest │
├───────────────────┼────────────────┼─────────────┼────────────┤
│ orig_tiny_4x4     │   147_536.04ns │   15228.93w │    100.00% │
│ opt_tiny_4x4      │   152_219.94ns │   16740.15w │    103.17% │
│ orig_small_8x8    │   261_287.84ns │   62179.17w │    177.10% │
│ opt_small_8x8     │   266_763.90ns │   62179.17w │    180.81% │
│ orig_medium_16x16 │ 4_357_020.06ns │ 3599291.44w │   2953.19% │
│ opt_medium_16x16  │ 4_497_289.66ns │ 3589479.56w │   3048.27% │
└───────────────────┴────────────────┴─────────────┴────────────┘

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

### 1. Reduce contiguous calls
- **What**: Extend View.shape to support more non-contiguous cases and remove unnecessary `contiguous` calls
- **Result**: TODO - need to test

### 2. Optimize convolution flow
- **What**: Reshape input tensors BEFORE pooling rather than after. When groups > 1, reshape from `(bs, cin_total, *spatial)` to `(bs, groups, cin_per_group, *spatial)` before the pool operation. This avoids the problematic reshape of `(bs, cin_total, *output_spatial, *kernel_spatial)` to `(bs, groups, cin_per_group, *output_spatial, *kernel_spatial)` which often fails with strided layouts from pooling. Also simplified the groups=1 case to avoid group-related reshapes entirely.
- **Expected Result**: Significant reduction in `contiguous` calls during convolution, especially for grouped convolutions. Should see performance closer to the raw pool operation (0.31ms) rather than 250x slower (78.5ms).

┌───────────────────┬────────────────┬─────────────┬────────────┐
│ Name              │       Time/Run │     mWd/Run │ vs Fastest │
├───────────────────┼────────────────┼─────────────┼────────────┤
│ orig_tiny_4x4     │   153_457.09ns │   14548.29w │    100.00% │
│ opt_tiny_4x4      │   156_873.79ns │   14548.29w │    102.23% │
│ orig_small_8x8    │   241_079.03ns │   58212.89w │    157.10% │
│ opt_small_8x8     │   250_014.05ns │   58212.89w │    162.92% │
│ opt_medium_16x16  │ 3_749_291.10ns │ 3434090.44w │   2443.22% │
│ orig_medium_16x16 │ 3_936_370.21ns │ 3333021.44w │   2565.13% │
└───────────────────┴────────────────┴─────────────┴────────────┘
