#!/usr/bin/env python3
"""
PyTorch Comparative Benchmark Script
Matches Rune benchmark structure exactly for direct comparison
"""

import torch
import torch.nn.functional as F
import time
import numpy as np

def time_operation(name, operation, *args, runs=1):
    """Time an operation"""
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = operation(*args)
        if torch.is_tensor(result):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
    """Time gradient computation using torch.autograd"""
    input_tensor.grad = None
    start = time.time()
    loss = forward_fn(input_tensor)
    loss.backward()
    elapsed = time.time() - start
    print(f"grad({name}): {elapsed:.6f} s")

def benchmark_pytorch():
    print("=== PyTorch Performance Benchmark ===")
    print("Testing forward pass, backward pass (torch.autograd), and JIT compilation")
    print("")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("")
    
    # Forward pass benchmarks
    bench_elementwise(device)
    bench_reductions(device)
    bench_linear_algebra(device)
    bench_shape_ops(device)
    bench_nn_ops(device)
    
    # Backward pass benchmarks
    bench_gradients(device)
    
    # JIT compilation benchmarks
    bench_jit(device)
    bench_jit_grad(device)
    
    # Performance comparisons
    bench_comparison(device)
    
    # Additional benchmarks
    bench_memory_patterns(device)
    bench_detailed_stats(device)
    
    print("\n=== PyTorch Comprehensive Benchmarking Complete ===")

def bench_elementwise(device):
    print("--- Element-wise Operations ---")
    n = 1_000_000
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    ones_a = torch.ones_like(a)
    positive_a = a + ones_a
    
    # Basic arithmetic
    time_operation("add", lambda x, y: torch.add(x, y), a, b)
    time_operation("sub", lambda x, y: torch.sub(x, y), a, b)
    time_operation("mul", lambda x, y: torch.mul(x, y), a, b)
    time_operation("div", lambda x, y: torch.div(x, y), a, b)
    
    # Power operations
    time_operation("square", lambda x: torch.mul(x, x), a)
    time_operation("sqrt", lambda x: torch.sqrt(x), positive_a)
    
    # Exponential and logarithmic
    time_operation("exp", lambda x: torch.exp(x), a)
    time_operation("log", lambda x: torch.log(x), positive_a)
    
    # Trigonometric
    time_operation("sin", lambda x: torch.sin(x), a)
    time_operation("cos", lambda x: torch.cos(x), a)
    time_operation("tan", lambda x: torch.tan(x), a)
    
    # Additional operations
    time_operation("abs", lambda x: torch.abs(x), a)
    time_operation("neg", lambda x: torch.neg(x), a)

def bench_reductions(device):
    print("\n--- Reduction Operations ---")
    n = 1_000_000
    a = torch.randn(n, device=device)
    
    time_operation("sum", lambda x: torch.sum(x), a)
    time_operation("mean", lambda x: torch.mean(x), a)
    time_operation("max", lambda x: torch.max(x).values, a)

def bench_linear_algebra(device):
    print("\n--- Linear Algebra ---")
    
    # Different matrix sizes
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    for m, n in sizes:
        x = torch.randn(m, n, device=device)
        y = torch.randn(n, m, device=device)
        time_operation(f"matmul {m}x{n}", lambda a, b: torch.matmul(a, b), x, y)
    
    # Vector operations
    v1 = torch.randn(10000, device=device)
    v2 = torch.randn(10000, device=device)
    time_operation("dot product", lambda x, y: torch.sum(torch.mul(x, y)), v1, v2)
    
    # Batched matrix multiplication
    batch = 10
    m, p = 256, 256
    a = torch.randn(batch, m, p, device=device)
    b = torch.randn(batch, p, m, device=device)
    time_operation("batched matmul 10x256x256", lambda x, y: torch.bmm(x, y), a, b)

def bench_shape_ops(device):
    print("\n--- Shape Operations ---")
    n = 1024
    a = torch.linspace(0.0, 1.0, n * n, device=device)
    
    time_operation("reshape", lambda x: x.reshape(n, n), a)
    
    m = a.reshape(n, n)
    time_operation("transpose", lambda x: x.transpose(0, 1), m)
    time_operation("slice", lambda x: x[:100, :], m)
    
    # Broadcast
    b = torch.linspace(0.0, 1.0, n, device=device)
    m_broadcast = b.reshape(n, 1)
    time_operation("broadcast", lambda x: x.expand(n, n), m_broadcast)

def bench_nn_ops(device):
    print("\n--- Neural Network Operations ---")
    n = 1_000_000
    a = torch.randn(n, device=device)
    small_a = torch.randn(1000, device=device)
    
    # Activation functions
    time_operation("relu", lambda x: F.relu(x), a)
    time_operation("sigmoid", lambda x: torch.sigmoid(x), a)
    time_operation("tanh", lambda x: torch.tanh(x), a)
    
    # Softmax (expensive, use smaller tensor)
    time_operation("softmax", lambda x: F.softmax(x, dim=0), small_a)
    
    # Additional activations
    time_operation("abs (activation)", lambda x: torch.abs(x), a)
    
    # Common neural network patterns
    weights = torch.randn(1000, 1000, device=device)
    input_tensor = torch.randn(1000, device=device)
    time_operation("dense layer (matmul + bias)", 
                  lambda w, x: F.relu(torch.sum(torch.mul(w, x))), weights, input_tensor)

def bench_gradients(device):
    print("\n--- Gradient Computation (Backward Pass) ---")
    
    # Element-wise operation gradients
    v = torch.randn(10000, device=device, requires_grad=True)
    ones_v = torch.ones_like(v)
    positive_v = v + ones_v
    
    time_gradient("add", lambda t: torch.sum(torch.add(t, positive_v.detach())), v)
    time_gradient("mul", lambda t: torch.sum(torch.mul(t, positive_v.detach())), v)
    time_gradient("square", lambda t: torch.sum(torch.mul(t, t)), v)
    time_gradient("sqrt", lambda t: torch.sum(torch.sqrt(t + ones_v.detach())), v)
    time_gradient("exp", lambda t: torch.sum(torch.exp(t)), v)
    time_gradient("log", lambda t: torch.sum(torch.log(t + ones_v.detach())), v)
    time_gradient("sin", lambda t: torch.sum(torch.sin(t)), v)
    time_gradient("cos", lambda t: torch.sum(torch.cos(t)), v)
    
    # Reduction operation gradients
    time_gradient("sum", lambda t: torch.sum(t), v)
    time_gradient("mean", lambda t: torch.mean(t), v)
    
    # Linear algebra gradients
    m1 = torch.randn(100, 100, device=device, requires_grad=True)
    m2 = torch.randn(100, 100, device=device)
    time_gradient("matmul", lambda t: torch.sum(torch.matmul(t, m2)), m1)
    
    # Shape operation gradients
    time_gradient("reshape", lambda t: torch.sum(t.reshape(100, 100)), v)
    time_gradient("transpose", lambda t: torch.sum(t.transpose(0, 1)), m1)
    
    # Neural network activation gradients
    time_gradient("relu", lambda t: torch.sum(F.relu(t)), v)
    time_gradient("sigmoid", lambda t: torch.sum(torch.sigmoid(t)), v)
    time_gradient("tanh", lambda t: torch.sum(torch.tanh(t)), v)

def bench_jit(device):
    print("\n--- JIT Compilation ---")
    
    v = torch.randn(10000, device=device)
    ones_v = torch.ones_like(v)
    positive_v = v + ones_v
    
    # Element-wise JIT operations
    @torch.jit.script
    def jit_add(x, y):
        return torch.add(x, y)
    
    @torch.jit.script
    def jit_mul(x, y):
        return torch.mul(x, y)
    
    @torch.jit.script
    def jit_square(x):
        return torch.mul(x, x)
    
    @torch.jit.script
    def jit_sqrt(x):
        return torch.sqrt(x)
    
    @torch.jit.script
    def jit_exp(x):
        return torch.exp(x)
    
    @torch.jit.script
    def jit_log(x):
        return torch.log(x)
    
    @torch.jit.script
    def jit_sin(x):
        return torch.sin(x)
    
    @torch.jit.script
    def jit_cos(x):
        return torch.cos(x)
    
    # Reduction JIT operations
    @torch.jit.script
    def jit_sum(x):
        return torch.sum(x)
    
    @torch.jit.script
    def jit_mean(x):
        return torch.mean(x)
    
    @torch.jit.script
    def jit_max(x):
        return torch.max(x)
    
    # Warm up JIT functions
    _ = jit_add(v[:100], positive_v[:100])
    _ = jit_square(v[:100])
    _ = jit_exp(v[:100])
    
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
    m1 = torch.randn(256, 256, device=device)
    m2 = torch.randn(256, 256, device=device)
    
    @torch.jit.script
    def jit_matmul(x, y):
        return torch.matmul(x, y)
    
    _ = jit_matmul(m1[:100, :100], m2[:100, :100])
    time_operation("jit(matmul 256x256)", jit_matmul, m1, m2)
    
    # Neural network JIT operations
    @torch.jit.script
    def jit_relu(x):
        return F.relu(x)
    
    @torch.jit.script
    def jit_sigmoid(x):
        return torch.sigmoid(x)
    
    @torch.jit.script
    def jit_tanh(x):
        return torch.tanh(x)
    
    _ = jit_relu(v[:100])
    _ = jit_sigmoid(v[:100])
    _ = jit_tanh(v[:100])
    
    time_operation("jit(relu)", jit_relu, v)
    time_operation("jit(sigmoid)", jit_sigmoid, v)
    time_operation("jit(tanh)", jit_tanh, v)

def bench_jit_grad(device):
    print("\n--- JIT + Gradient Composition ---")
    
    v = torch.randn(5000, device=device, requires_grad=True)
    
    # Note: PyTorch JIT + autograd composition is complex
    # We'll benchmark regular gradients of JIT functions
    
    @torch.jit.script
    def jit_square_fn(x):
        return torch.mul(x, x)
    
    @torch.jit.script  
    def jit_exp_fn(x):
        return torch.exp(x)
    
    # Warm up
    _ = jit_square_fn(v[:100])
    _ = jit_exp_fn(v[:100])
    
    time_gradient("jit(square)", lambda t: torch.sum(jit_square_fn(t)), v)
    time_gradient("jit(exp)", lambda t: torch.sum(jit_exp_fn(t)), v)

def bench_comparison(device):
    print("\n--- Performance Comparison: Regular vs Grad vs JIT vs JIT+Grad ---")
    
    v = torch.randn(50000, device=device)
    v_grad = torch.randn(50000, device=device, requires_grad=True)
    
    # Square operation comparison
    print("\nSquare operation comparison:")
    time_operation("  regular square", lambda x: torch.sum(torch.mul(x, x)), v)
    time_gradient("  grad(square)", lambda t: torch.sum(torch.mul(t, t)), v_grad.clone().detach().requires_grad_(True))
    
    @torch.jit.script
    def jit_square_sum(x):
        return torch.sum(torch.mul(x, x))
    
    _ = jit_square_sum(v[:100])
    time_operation("  jit(square)", jit_square_sum, v)
    time_gradient("  jit(grad(square))", lambda t: jit_square_sum(t), v_grad.clone().detach().requires_grad_(True))
    
    # Exp operation comparison
    print("\nExp operation comparison:")
    time_operation("  regular exp", lambda x: torch.sum(torch.exp(x)), v)
    time_gradient("  grad(exp)", lambda t: torch.sum(torch.exp(t)), v_grad.clone().detach().requires_grad_(True))
    
    @torch.jit.script
    def jit_exp_sum(x):
        return torch.sum(torch.exp(x))
    
    _ = jit_exp_sum(v[:100])
    time_operation("  jit(exp)", jit_exp_sum, v)
    time_gradient("  jit(grad(exp))", lambda t: jit_exp_sum(t), v_grad.clone().detach().requires_grad_(True))
    
    # Matrix multiplication comparison
    print("\nMatrix multiplication comparison (128x128):")
    m1 = torch.randn(128, 128, device=device)
    m2 = torch.randn(128, 128, device=device)
    m1_grad = torch.randn(128, 128, device=device, requires_grad=True)
    
    time_operation("  regular matmul", lambda x, y: torch.sum(torch.matmul(x, y)), m1, m2)
    time_gradient("  grad(matmul)", lambda t: torch.sum(torch.matmul(t, m2)), m1_grad)
    
    @torch.jit.script
    def jit_matmul_sum(x, y):
        return torch.sum(torch.matmul(x, y))
    
    _ = jit_matmul_sum(m1[:50, :50], m2[:50, :50])
    time_operation("  jit(matmul)", jit_matmul_sum, m1, m2)
    time_gradient("  jit(grad(matmul))", lambda t: jit_matmul_sum(t, m2), m1_grad.clone().detach().requires_grad_(True))

def bench_memory_patterns(device):
    print("\n--- Memory and Performance Patterns ---")
    
    # Contiguous vs non-contiguous memory access
    m = torch.randn(1000, 1000, device=device)
    time_operation("sum (contiguous)", lambda x: torch.sum(x), m)
    
    m_t = m.transpose(0, 1)
    time_operation("sum (after transpose)", lambda x: torch.sum(x), m_t)
    
    # Different tensor sizes  
    sizes = [1000, 10000, 100000, 1000000]
    for n in sizes:
        a = torch.randn(n, device=device)
        time_operation(f"element-wise add (n={n})", lambda x: torch.add(x, x), a)

def bench_detailed_stats(device):
    print("\n--- Detailed Statistics (5 runs each) ---")
    
    n = 100000
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    
    time_operation("add (detailed)", lambda x, y: torch.add(x, y), a, b, runs=5)
    time_operation("mul (detailed)", lambda x, y: torch.mul(x, y), a, b, runs=5) 
    time_operation("exp (detailed)", lambda x: torch.exp(x), a, runs=5)

if __name__ == "__main__":
    benchmark_pytorch()