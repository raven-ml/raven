=== JAX Performance Benchmark ===
Testing forward pass, backward pass (jax.grad), and JIT compilation

Available devices: [CpuDevice(id=0)]
Default device: TFRT_CPU_0

--- Element-wise Operations ---
add: 0.001410 s
sub: 0.029517 s
mul: 0.030482 s
div: 0.030894 s
square: 0.029702 s
sqrt: 0.026673 s
exp: 0.026781 s
log: 0.028226 s
sin: 0.026546 s
cos: 0.024916 s
tan: 0.026039 s
abs: 0.019657 s
neg: 0.018360 s

--- Reduction Operations ---
sum: 0.034811 s
mean: 0.030369 s
max: 0.026548 s

--- Linear Algebra ---
matmul 128x128: 0.036720 s
matmul 256x256: 0.012780 s
matmul 512x512: 0.021621 s
matmul 1024x1024: 0.050852 s
dot product: 0.048872 s
batched matmul 10x256x256: 0.022263 s

--- Shape Operations ---
reshape: 0.012395 s
transpose: 0.030750 s
slice: 0.021120 s
broadcast: 0.029216 s

--- Neural Network Operations ---
relu: 0.028196 s
sigmoid: 0.030712 s
tanh: 0.027276 s
softmax: 0.115071 s
abs (activation): 0.000631 s
dense layer (matmul + bias): 0.073506 s

--- Gradient Computation (Backward Pass) ---
grad(add): 0.059305 s
grad(mul): 0.036049 s
grad(square): 0.028967 s
grad(sqrt): 0.036457 s
grad(exp): 0.031792 s
grad(log): 0.067319 s
grad(sin): 0.029474 s
grad(cos): 0.057177 s
grad(sum): 0.001634 s
grad(mean): 0.001146 s
grad(matmul): 0.065090 s
grad(reshape): 0.022608 s
grad(transpose): 0.054688 s
grad(relu): 0.045541 s
grad(sigmoid): 0.030919 s
grad(tanh): 0.053224 s

--- JIT Compilation ---
jit(add): 0.018902 s
jit(mul): 0.017156 s
jit(square): 0.017531 s
jit(sqrt): 0.016825 s
jit(exp): 0.021091 s
jit(log): 0.022537 s
jit(sin): 0.015825 s
jit(cos): 0.018253 s
jit(sum): 0.020858 s
jit(mean): 0.028244 s
jit(max): 0.023313 s
jit(matmul 256x256): 0.010692 s
jit(relu): 0.024391 s
jit(sigmoid): 0.031248 s
jit(tanh): 0.018682 s

--- JIT + Gradient Composition ---
jit(grad(square)): 0.024157 s
jit(grad(exp)): 0.021884 s
jit(grad(sin)): 0.018831 s
grad(jit(square)): 0.097512 s
grad(jit(exp)): 0.040195 s

--- Performance Comparison: Regular vs Grad vs JIT vs JIT+Grad ---

Square operation comparison:
  regular square: 0.044896 s
grad(  grad(square)): 0.144783 s
  jit(square): 0.032517 s
  jit(grad(square)): 0.036278 s

Exp operation comparison:
  regular exp: 0.025163 s
grad(  grad(exp)): 0.026507 s
  jit(exp): 0.027951 s
  jit(grad(exp)): 0.022259 s

Matrix multiplication comparison (128x128):
  regular matmul: 0.021125 s
grad(  grad(matmul)): 0.064648 s
  jit(matmul): 0.023561 s
  jit(grad(matmul)): 0.020544 s

--- Memory and Performance Patterns ---
sum (contiguous): 0.017022 s
sum (after transpose): 0.000763 s
element-wise add (n=1000): 0.019309 s
element-wise add (n=10000): 0.000528 s
element-wise add (n=100000): 0.022868 s
element-wise add (n=1000000): 0.019048 s

--- Detailed Statistics (5 runs each) ---
add (detailed): min=0.000218 max=0.002965 mean=0.000866 median=0.000449 s
mul (detailed): min=0.000097 max=0.020038 mean=0.004116 median=0.000110 s
exp (detailed): min=0.000078 max=0.024256 mean=0.004960 median=0.000137 s

=== JAX Comprehensive Benchmarking Complete ===
