#!/usr/bin/env python3
"""
JAX Comparative Benchmark Script
Matches Rune benchmark structure exactly for direct comparison
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

def time_operation(name, operation, *args, runs=1):
    """Time an operation"""
    times = []
    for _ in range(runs):
        start = time.time()
        result = operation(*args)
        # Block until computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append(time.time() - start)
    
    if runs == 1:
        print(f"{name}: {times[0]:.6f} s")
    else:
        times = sorted(times)
        min_time = times[0]
        max_time = times[-1]
        mean_time = sum(times) / len(times)
        median_time = times[len(times) // 2]
        print(f"{name}: min={min_time:.6f} max={max_time:.6f} mean={mean_time:.6f} median={median_time:.6f} s")

def time_gradient(name, forward_fn, input_tensor):
    """Time gradient computation using jax.grad"""
    grad_fn = jax.grad(lambda x: jnp.sum(forward_fn(x)))
    start = time.time()
    result = grad_fn(input_tensor)
    result.block_until_ready()
    elapsed = time.time() - start
    print(f"grad({name}): {elapsed:.6f} s")

def benchmark_jax():
    print("=== JAX Performance Benchmark ===")
    print("Testing forward pass, backward pass (jax.grad), and JIT compilation")
    print("")
    
    # Check for GPU
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Default device: {jax.devices()[0]}")
    print("")
    
    # Generate random keys
    key = jax.random.PRNGKey(42)
    
    # Forward pass benchmarks
    bench_elementwise(key)
    bench_reductions(key)
    bench_linear_algebra(key)
    bench_shape_ops(key)
    bench_nn_ops(key)
    
    # Backward pass benchmarks
    bench_gradients(key)
    
    # JIT compilation benchmarks
    bench_jit(key)
    bench_jit_grad(key)
    
    # Performance comparisons
    bench_comparison(key)
    
    # Additional benchmarks
    bench_memory_patterns(key)
    bench_detailed_stats(key)
    
    print("\n=== JAX Comprehensive Benchmarking Complete ===")

def bench_elementwise(key):
    print("--- Element-wise Operations ---")
    n = 1_000_000
    key, subkey1, subkey2 = jax.random.split(key, 3)
    a = jax.random.normal(subkey1, (n,))
    b = jax.random.normal(subkey2, (n,))
    ones_a = jnp.ones_like(a)
    positive_a = a + ones_a
    
    # Basic arithmetic
    time_operation("add", lambda x, y: jnp.add(x, y), a, b)
    time_operation("sub", lambda x, y: jnp.subtract(x, y), a, b)
    time_operation("mul", lambda x, y: jnp.multiply(x, y), a, b)
    time_operation("div", lambda x, y: jnp.divide(x, y), a, b)
    
    # Power operations
    time_operation("square", lambda x: jnp.square(x), a)
    time_operation("sqrt", lambda x: jnp.sqrt(x), positive_a)
    
    # Exponential and logarithmic
    time_operation("exp", lambda x: jnp.exp(x), a)
    time_operation("log", lambda x: jnp.log(x), positive_a)
    
    # Trigonometric
    time_operation("sin", lambda x: jnp.sin(x), a)
    time_operation("cos", lambda x: jnp.cos(x), a)
    time_operation("tan", lambda x: jnp.tan(x), a)
    
    # Additional operations
    time_operation("abs", lambda x: jnp.abs(x), a)
    time_operation("neg", lambda x: jnp.negative(x), a)

def bench_reductions(key):
    print("\n--- Reduction Operations ---")
    key, subkey = jax.random.split(key)
    n = 1_000_000
    a = jax.random.normal(subkey, (n,))
    
    time_operation("sum", lambda x: jnp.sum(x), a)
    time_operation("mean", lambda x: jnp.mean(x), a)
    time_operation("max", lambda x: jnp.max(x), a)

def bench_linear_algebra(key):
    print("\n--- Linear Algebra ---")
    
    # Different matrix sizes
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    for m, n in sizes:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = jax.random.normal(subkey1, (m, n))
        y = jax.random.normal(subkey2, (n, m))
        time_operation(f"matmul {m}x{n}", lambda a, b: jnp.matmul(a, b), x, y)
    
    # Vector operations
    key, subkey1, subkey2 = jax.random.split(key, 3)
    v1 = jax.random.normal(subkey1, (10000,))
    v2 = jax.random.normal(subkey2, (10000,))
    time_operation("dot product", lambda x, y: jnp.sum(jnp.multiply(x, y)), v1, v2)
    
    # Batched matrix multiplication
    key, subkey1, subkey2 = jax.random.split(key, 3)
    batch = 10
    m, p = 256, 256
    a = jax.random.normal(subkey1, (batch, m, p))
    b = jax.random.normal(subkey2, (batch, p, m))
    time_operation("batched matmul 10x256x256", lambda x, y: jnp.matmul(x, y), a, b)

def bench_shape_ops(key):
    print("\n--- Shape Operations ---")
    n = 1024
    a = jnp.linspace(0.0, 1.0, n * n)
    
    time_operation("reshape", lambda x: x.reshape(n, n), a)
    
    m = a.reshape(n, n)
    time_operation("transpose", lambda x: x.transpose(), m)
    time_operation("slice", lambda x: x[:100, :], m)
    
    # Broadcast
    key, subkey = jax.random.split(key)
    b = jnp.linspace(0.0, 1.0, n)
    m_broadcast = b.reshape(n, 1)
    time_operation("broadcast", lambda x: jnp.broadcast_to(x, (n, n)), m_broadcast)

def bench_nn_ops(key):
    print("\n--- Neural Network Operations ---")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    n = 1_000_000
    a = jax.random.normal(subkey1, (n,))
    small_a = jax.random.normal(subkey2, (1000,))
    
    # Activation functions
    time_operation("relu", lambda x: jnp.maximum(0, x), a)
    time_operation("sigmoid", lambda x: jax.nn.sigmoid(x), a)
    time_operation("tanh", lambda x: jnp.tanh(x), a)
    
    # Softmax (expensive, use smaller tensor)
    time_operation("softmax", lambda x: jax.nn.softmax(x), small_a)
    
    # Additional activations
    time_operation("abs (activation)", lambda x: jnp.abs(x), a)
    
    # Common neural network patterns
    key, subkey1, subkey2 = jax.random.split(key, 3)
    weights = jax.random.normal(subkey1, (1000, 1000))
    input_tensor = jax.random.normal(subkey2, (1000,))
    time_operation("dense layer (matmul + bias)", 
                  lambda w, x: jnp.maximum(0, jnp.sum(jnp.multiply(w, x))), weights, input_tensor)
def bench_gradients(key):
    print("\n--- Gradient Computation (Backward Pass) ---")
    
    # Element-wise operation gradients
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (10000,))
    ones_v = jnp.ones_like(v)
    positive_v = v + ones_v
    
    time_gradient("add", lambda t: jnp.add(t, positive_v), v)
    time_gradient("mul", lambda t: jnp.multiply(t, positive_v), v)
    time_gradient("square", lambda t: jnp.square(t), v)
    time_gradient("sqrt", lambda t: jnp.sqrt(t + ones_v), v)
    time_gradient("exp", lambda t: jnp.exp(t), v)
    time_gradient("log", lambda t: jnp.log(t + ones_v), v)
    time_gradient("sin", lambda t: jnp.sin(t), v)
    time_gradient("cos", lambda t: jnp.cos(t), v)
    
    # Reduction operation gradients
    time_gradient("sum", lambda t: t, v) # sum gradient is identity
    time_gradient("mean", lambda t: t, v) # mean gradient is identity/n
    
    # Linear algebra gradients
    key, subkey1, subkey2 = jax.random.split(key, 3)
    m1 = jax.random.normal(subkey1, (100, 100))
    m2 = jax.random.normal(subkey2, (100, 100))
    time_gradient("matmul", lambda t: jnp.matmul(t, m2), m1)
    
    # Shape operation gradients
    time_gradient("reshape", lambda t: t.reshape(100, 100), v)
    time_gradient("transpose", lambda t: t.transpose(), m1)
    
    # Neural network activation gradients
    time_gradient("relu", lambda t: jnp.maximum(0, t), v)
    time_gradient("sigmoid", lambda t: jax.nn.sigmoid(t), v)
    time_gradient("tanh", lambda t: jnp.tanh(t), v)

def bench_jit(key):
    print("\n--- JIT Compilation ---")
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    v = jax.random.normal(subkey1, (10000,))
    ones_v = jnp.ones_like(v)
    positive_v = v + ones_v
    
    # Element-wise JIT operations
    @jax.jit
    def jit_add(x, y):
        return jnp.add(x, y)
    
    @jax.jit
    def jit_mul(x, y):
        return jnp.multiply(x, y)
    
    @jax.jit
    def jit_square(x):
        return jnp.square(x)
    
    @jax.jit
    def jit_sqrt(x):
        return jnp.sqrt(x)
    
    @jax.jit
    def jit_exp(x):
        return jnp.exp(x)
    
    @jax.jit
    def jit_log(x):
        return jnp.log(x)
    
    @jax.jit
    def jit_sin(x):
        return jnp.sin(x)
    
    @jax.jit
    def jit_cos(x):
        return jnp.cos(x)
    
    # Reduction JIT operations
    @jax.jit
    def jit_sum(x):
        return jnp.sum(x)
    
    @jax.jit
    def jit_mean(x):
        return jnp.mean(x)
    
    @jax.jit
    def jit_max(x):
        return jnp.max(x)
    
    # Warm up JIT functions
    _ = jit_add(v[:100], positive_v[:100]).block_until_ready()
    _ = jit_square(v[:100]).block_until_ready()
    _ = jit_exp(v[:100]).block_until_ready()
    
    time_operation("jit(add)", jit_add, v, positive_v)
    time_operation("jit(mul)", jit_mul, v, positive_v)
    time_operation("jit(square)", jit_square, v)
    time_operation("jit(sqrt)", jit_sqrt, positive_v)
    time_operation("jit(exp)", jit_exp, v)
    time_operation("jit(log)", jit_log, positive_v)
    time_operation("jit(sin)", jit_sin, v)
    time_operation("jit(cos)", jit_cos, v)
    
    time_operation("jit(sum)", jit_sum, v)
    time_operation("jit(mean)", jit_mean, v)
    time_operation("jit(max)", jit_max, v)
    
    # Linear algebra JIT
    key, subkey1, subkey2 = jax.random.split(key, 3)
    m1 = jax.random.normal(subkey1, (256, 256))
    m2 = jax.random.normal(subkey2, (256, 256))
    
    @jax.jit
    def jit_matmul(x, y):
        return jnp.matmul(x, y)
    
    _ = jit_matmul(m1[:100, :100], m2[:100, :100]).block_until_ready()
    time_operation("jit(matmul 256x256)", jit_matmul, m1, m2)
    
    # Neural network JIT operations
    @jax.jit
    def jit_relu(x):
        return jnp.maximum(0, x)
    
    @jax.jit
    def jit_sigmoid(x):
        return jax.nn.sigmoid(x)
    
    @jax.jit
    def jit_tanh(x):
        return jnp.tanh(x)
    
    _ = jit_relu(v[:100]).block_until_ready()
    _ = jit_sigmoid(v[:100]).block_until_ready()
    _ = jit_tanh(v[:100]).block_until_ready()
    
    time_operation("jit(relu)", jit_relu, v)
    time_operation("jit(sigmoid)", jit_sigmoid, v)
    time_operation("jit(tanh)", jit_tanh, v)

def bench_jit_grad(key):
    print("\n--- JIT + Gradient Composition ---")
    
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (5000,))
    
    # JIT of gradient functions
    @jax.jit
    def jit_grad_square(x):
        return jax.grad(lambda t: jnp.sum(jnp.square(t)))(x)
    
    @jax.jit
    def jit_grad_exp(x):
        return jax.grad(lambda t: jnp.sum(jnp.exp(t)))(x)
    
    @jax.jit
    def jit_grad_sin(x):
        return jax.grad(lambda t: jnp.sum(jnp.sin(t)))(x)
    
    # Warm up
    _ = jit_grad_square(v[:100]).block_until_ready()
    _ = jit_grad_exp(v[:100]).block_until_ready()
    _ = jit_grad_sin(v[:100]).block_until_ready()
    
    time_operation("jit(grad(square))", jit_grad_square, v)
    time_operation("jit(grad(exp))", jit_grad_exp, v)
    time_operation("jit(grad(sin))", jit_grad_sin, v)
    
    # Gradient of JIT functions
    @jax.jit
    def jit_square_fn(x):
        return jnp.square(x)
    
    @jax.jit
    def jit_exp_fn(x):
        return jnp.exp(x)
    
    _ = jit_square_fn(v[:100]).block_until_ready()
    _ = jit_exp_fn(v[:100]).block_until_ready()
    
    time_gradient("jit(square)", jit_square_fn, v)
    time_gradient("jit(exp)", jit_exp_fn, v)

def bench_comparison(key):
    print("\n--- Performance Comparison: Regular vs Grad vs JIT vs JIT+Grad ---")
    
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (50000,))
    
    # Square operation comparison
    print("\nSquare operation comparison:")
    time_operation("  regular square", lambda x: jnp.sum(jnp.square(x)), v)
    time_gradient("  grad(square)", lambda t: jnp.square(t), v)
    
    @jax.jit
    def jit_square_sum(x):
        return jnp.sum(jnp.square(x))
    
    _ = jit_square_sum(v[:100]).block_until_ready()
    time_operation("  jit(square)", jit_square_sum, v)
    
    @jax.jit
    def jit_grad_square_fn(x):
        return jax.grad(lambda t: jnp.sum(jnp.square(t)))(x)
    
    _ = jit_grad_square_fn(v[:100]).block_until_ready()
    time_operation("  jit(grad(square))", jit_grad_square_fn, v)
    
    # Exp operation comparison
    print("\nExp operation comparison:")
    time_operation("  regular exp", lambda x: jnp.sum(jnp.exp(x)), v)
    time_gradient("  grad(exp)", lambda t: jnp.exp(t), v)
    
    @jax.jit
    def jit_exp_sum(x):
        return jnp.sum(jnp.exp(x))
    
    _ = jit_exp_sum(v[:100]).block_until_ready()
    time_operation("  jit(exp)", jit_exp_sum, v)
    
    @jax.jit
    def jit_grad_exp_fn(x):
        return jax.grad(lambda t: jnp.sum(jnp.exp(t)))(x)
    
    _ = jit_grad_exp_fn(v[:100]).block_until_ready()
    time_operation("  jit(grad(exp))", jit_grad_exp_fn, v)
    
    # Matrix multiplication comparison
    print("\nMatrix multiplication comparison (128x128):")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    m1 = jax.random.normal(subkey1, (128, 128))
    m2 = jax.random.normal(subkey2, (128, 128))
    
    time_operation("  regular matmul", lambda x, y: jnp.sum(jnp.matmul(x, y)), m1, m2)
    time_gradient("  grad(matmul)", lambda t: jnp.matmul(t, m2), m1)
    
    @jax.jit
    def jit_matmul_sum(x, y):
        return jnp.sum(jnp.matmul(x, y))
    
    _ = jit_matmul_sum(m1[:50, :50], m2[:50, :50]).block_until_ready()
    time_operation("  jit(matmul)", jit_matmul_sum, m1, m2)
    
    @jax.jit
    def jit_grad_matmul_fn(x, y):
        return jax.grad(lambda t: jnp.sum(jnp.matmul(t, y)))(x)
    
    _ = jit_grad_matmul_fn(m1[:50, :50], m2[:50, :50]).block_until_ready()
    time_operation("  jit(grad(matmul))", jit_grad_matmul_fn, m1, m2)

def bench_memory_patterns(key):
    print("\n--- Memory and Performance Patterns ---")
    
    # Contiguous vs non-contiguous memory access
    key, subkey = jax.random.split(key)
    m = jax.random.normal(subkey, (1000, 1000))
    time_operation("sum (contiguous)", lambda x: jnp.sum(x), m)
    
    m_t = m.transpose()
    time_operation("sum (after transpose)", lambda x: jnp.sum(x), m_t)
    
    # Different tensor sizes  
    sizes = [1000, 10000, 100000, 1000000]
    for n in sizes:
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, (n,))
        time_operation(f"element-wise add (n={n})", lambda x: jnp.add(x, x), a)

def bench_detailed_stats(key):
    print("\n--- Detailed Statistics (5 runs each) ---")
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    n = 100000
    a = jax.random.normal(subkey1, (n,))
    b = jax.random.normal(subkey2, (n,))
    
    time_operation("add (detailed)", lambda x, y: jnp.add(x, y), a, b, runs=5)
    time_operation("mul (detailed)", lambda x, y: jnp.multiply(x, y), a, b, runs=5) 
    time_operation("exp (detailed)", lambda x: jnp.exp(x), a, runs=5)

if __name__ == "__main__":
    benchmark_jax()