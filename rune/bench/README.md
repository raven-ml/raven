# Rune Performance Benchmark Suite

This comprehensive benchmark suite provides systematic performance testing for Rune's tensor operations using the **ubench** library for precise measurements. It includes **forward pass**, **backward pass (gradients)**, and **JIT compilation** benchmarks, enabling performance comparisons against mature frameworks like PyTorch and JAX.

## Quick Start

```bash
# Run Rune benchmarks
dune exec rune/bench/bench_rune.exe

# Run PyTorch comparison (requires PyTorch installation)
uv run rune/bench/bench_pytorch.py

# Run JAX comparison (requires JAX installation)
uv run rune/bench/bench_jax.py
```

## System Specifications

Benchmarks were conducted on the following system:

- **OS**: Ubuntu 22.04 LTS (WSL2 on Windows)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- **RAM**: 5.8GB available
- **OCaml**: 5.3.0
- **Python**: 3.12.3
- **PyTorch**: Not installed (benchmarks use system Python)
- **JAX**: Not installed (benchmarks use system Python)

*Note: Performance results are hardware-dependent. PyTorch and JAX benchmarks require separate installation of these frameworks.*

**JIT Backend Limitation**: The Metal JIT backend is only available on macOS. On Linux/WSL2, JIT benchmarks fall back to eager execution mode. This is expected behavior and does not affect forward/backward pass benchmarks.

## Benchmark Structure

Following the `nx/bench/` pattern with:
- **Test sizes**: [50, 100, 200] for systematic scaling analysis
- **Data types**: [Float32] initially
- **ubench configuration**: 10 measurements, 3 warmup iterations, 2s time limit
- **Precise timing**: Mean and standard deviation in nanoseconds

## Operation Coverage

### 1. Forward Pass Operations

#### Element-wise Operations
- **add, mul, square, sqrt, exp, log, sin, cos**
- Tests fundamental tensor arithmetic and mathematical functions

#### Reduction Operations  
- **sum, mean, max**
- Tests aggregation performance across tensor dimensions

#### Linear Algebra
- **matmul**: Standard matrix multiplication
- **batched_matmul**: Batched matrix operations for ML workloads

#### Shape Operations
- **reshape, transpose, slice, broadcast**
- Tests memory layout and tensor manipulation efficiency

#### Neural Network Operations
- **relu, sigmoid, tanh, softmax**
- Tests activation functions critical for ML applications

### 2. Backward Pass (Automatic Differentiation)
- All forward operations using **`Rune.grad`**
- Systematic comparison of AD performance vs forward pass
- Tests gradient computation efficiency for optimization

### 3. JIT Compilation
- All operations using **`Rune.jit`**
- Measures compilation overhead vs execution speedup  
- Tests Metal/LLVM backend performance

## Usage

### Running Individual Benchmarks

```bash
# Rune benchmarks (outputs markdown table)
dune exec rune/bench/bench_rune.exe

# PyTorch comparison benchmarks (requires PyTorch installation)
uv run rune/bench/bench_pytorch.py

# JAX comparison benchmarks (requires JAX installation)
uv run rune/bench/bench_jax.py
```

### Saving Results

```bash
# Save Rune results
dune exec rune/bench/bench_rune.exe > results.rune.md

# Save PyTorch results
uv run rune/bench/bench_pytorch.py > results.pytorch.md

# Save JAX results  
uv run rune/bench/bench_jax.py > results.jax.md
```

## Benchmark Implementation

The benchmarks use **ubench** library for precise measurements with:

- **Warmup**: 3 iterations to stabilize JIT compilation
- **Measurements**: 10 iterations for statistical accuracy  
- **Time limit**: 2 seconds per benchmark maximum
- **GC stabilization**: Garbage collection between measurements
- **Statistical output**: Mean and standard deviation in nanoseconds

### Code Organization

- `bench_elementwise()` - Element-wise operations (add, mul, square, sqrt, exp, log, sin, cos)
- `bench_reductions()` - Reduction operations (sum, mean, max)
- `bench_linalg()` - Linear algebra (matmul, batched_matmul)
- `bench_shape_ops()` - Shape operations (reshape, transpose, slice, broadcast)  
- `bench_neural_ops()` - Neural network operations (relu, sigmoid, tanh, softmax)

Each function tests the same operations in three modes:
- **Forward pass**: Standard operation execution
- **Backward pass**: Using `Rune.grad` for gradient computation  
- **JIT**: Using `Rune.jit` for compiled execution

## Output Format

Results are output in **markdown table format** for easy analysis:

```markdown
| Operation | Size | Data Type | Pass Type | Time (ns) | Std Dev (ns) |
|-----------|------|-----------|-----------|-----------|--------------|
| Add | 50x50 | f32 | forward | 1250 | 95 |
| Add | 50x50 | f32 | backward | 2100 | 150 |
| Add | 50x50 | f32 | jit | 890 | 45 |
...
```

## System Requirements

### OCaml Dependencies
- **rune**: Tensor operations and automatic differentiation
- **ubench**: Precise benchmarking framework  
- **str**: String processing for output
- **unix**: System timing functions

### Python Dependencies (for comparison)
- **PyTorch**: `torch`, `torch.nn.functional`
- **JAX**: `jax`, `jax.numpy`

## Framework Comparison Analysis

### Performance Metrics
- **Time per operation**: Mean execution time in nanoseconds
- **Standard deviation**: Measurement consistency
- **Scaling behavior**: Performance across tensor sizes (50x50, 100x100, 200x200)
- **Pass type comparison**: Forward vs backward vs JIT performance ratios

### Expected Performance Characteristics

#### Rune Advantages
- **Low overhead**: Minimal runtime overhead for small operations
- **OCaml efficiency**: Native compiled performance
- **JIT compilation**: Metal/LLVM optimization potential

#### Framework Advantages  
- **PyTorch**: Mature CUDA kernels, extensive optimization
- **JAX**: XLA compilation, advanced fusion, TPU support

### Analysis Goals
- **Identify bottlenecks**: Operations needing optimization in Rune
- **Guide development**: Focus optimization efforts on slow operations
- **Validate improvements**: Track performance changes over time
- **Benchmark JIT effectiveness**: Compilation overhead vs speedup analysis

## Extending Benchmarks

### Adding New Operations

1. **Update Rune benchmark**:
   ```ocaml
   let new_op_bench = bench (make_name "NewOp" size dtype "forward") 
     (fun () -> ignore (Rune.new_operation a)) in
   ```

2. **Add to PyTorch benchmark**:
   ```python
   mean_time, std_time = time_operation(f"NewOp {size}x{size} f32 (forward)", 
       lambda: torch.new_operation(a))
   ```

3. **Add to JAX benchmark**:
   ```python
   mean_time, std_time = time_operation(f"NewOp {size}x{size} f32 (forward)", 
       lambda: jnp.new_operation(a))
   ```

2. Add the function call to the main entry point

3. Update this README with the new category description

## Framework Comparison: Rune vs PyTorch vs JAX

This benchmark suite enables systematic comparison with mature frameworks through comprehensive benchmark scripts:

### PyTorch Comparison (`benchmark_pytorch.py`)
- **Forward pass**: Compare `torch.add`, `torch.matmul`, etc. vs Rune equivalents
- **Backward pass**: Compare `torch.autograd` and `loss.backward()` vs `Rune.grad`
- **JIT**: Compare `torch.jit.script` compilation vs `Rune.jit`
- **Memory efficiency**: Tensor allocation and GPU/CPU performance patterns
- **Comprehensive coverage**: All operations matching Rune benchmark structure

### JAX Comparison (`bench_jax.py`)
- **Forward pass**: Compare `jax.numpy` operations vs Rune
- **Backward pass**: Compare `jax.grad` automatic differentiation vs `Rune.grad`
- **JIT**: Compare `jax.jit` XLA compilation vs `Rune.jit`
- **Composition**: Compare `jax.jit(jax.grad(f))` vs `Rune.jit(Rune.grad(f))`
- **Functional transformations**: JAX's functional programming approach vs Rune

### Automated Comparison Analysis (`compare_results.py`)
- **Side-by-side tables**: Direct performance comparison across all operations
- **Speedup calculations**: Automatic computation of performance ratios
- **Statistical analysis**: Summary statistics and performance insights
- **Identification of strengths/weaknesses**: Areas where Rune excels or needs improvement
- **Markdown report generation**: Professional comparison reports

### Performance Analysis Goals
- **Identify performance bottlenecks** in Rune operations vs established frameworks
- **Guide optimization efforts** by highlighting operations that need improvement
- **Track performance regression** across Rune versions with baseline comparisons
- **Validate JIT compilation** effectiveness compared to PyTorch JIT and JAX XLA
- **Benchmark automatic differentiation** efficiency against PyTorch autograd and JAX grad
- **Measure compilation overhead** vs runtime performance gains across frameworks
- **Assess ecosystem maturity** through comprehensive operation coverage

## Notes

- The benchmark includes both forward and backward pass operations
- JIT compilation warnings may appear during first runs but don't affect timing
- Memory usage and allocation patterns are important for performance
- Results may vary based on hardware and system load
