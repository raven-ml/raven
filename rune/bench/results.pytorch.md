=== PyTorch Performance Benchmark ===
Testing forward pass, backward pass (torch.autograd), and JIT compilation

Using device: cpu

--- Element-wise Operations ---
add: 0.001801 s
sub: 0.002450 s
mul: 0.000700 s
div: 0.001536 s
square: 0.001334 s
sqrt: 0.002657 s
exp: 0.000497 s
log: 0.003721 s
sin: 0.000491 s
cos: 0.000524 s
tan: 0.000663 s
abs: 0.000504 s
neg: 0.000350 s

--- Reduction Operations ---
sum: 0.000337 s
mean: 0.000247 s
max: 0.000254 s

--- Linear Algebra ---
matmul 128x128: 0.000342 s
matmul 256x256: 0.000322 s
matmul 512x512: 0.001747 s
matmul 1024x1024: 0.011184 s
dot product: 0.000059 s
batched matmul 10x256x256: 0.003093 s

--- Shape Operations ---
reshape: 0.000113 s
transpose: 0.000042 s
slice: 0.000100 s
broadcast: 0.000025 s

--- Neural Network Operations ---
relu: 0.000429 s
sigmoid: 0.000447 s
tanh: 0.000561 s
softmax: 0.000085 s
abs (activation): 0.000381 s
dense layer (matmul + bias): 0.001261 s

--- Gradient Computation (Backward Pass) ---
grad(add): 0.000410 s
grad(mul): 0.000120 s
grad(square): 0.000090 s
grad(sqrt): 0.000183 s
grad(exp): 0.000068 s
grad(log): 0.000104 s
grad(sin): 0.000077 s
grad(cos): 0.000091 s
grad(sum): 0.000046 s
grad(mean): 0.000079 s
grad(matmul): 0.000233 s
grad(reshape): 0.000084 s
grad(transpose): 0.000058 s
grad(relu): 0.000125 s
grad(sigmoid): 0.000106 s
grad(tanh): 0.000089 s

--- JIT Compilation ---
jit(add): 0.000313 s
jit(mul): 0.000129 s
jit(square): 0.000175 s
jit(sqrt): 0.000158 s
jit(exp): 0.000151 s
jit(log): 0.000147 s
jit(sin): 0.000098 s
jit(cos): 0.000104 s
jit(sum): 0.024715 s
jit(mean): 0.000179 s
jit(max): 0.000100 s
jit(matmul 256x256): 0.000448 s
jit(relu): 0.000108 s
jit(sigmoid): 0.000094 s
jit(tanh): 0.000104 s

--- JIT + Gradient Composition ---
grad(jit(square)): 0.000354 s
grad(jit(exp)): 0.000195 s

--- Performance Comparison: Regular vs Grad vs JIT vs JIT+Grad ---

Square operation comparison:
  regular square: 0.000050 s
grad(  grad(square)): 0.000203 s
  jit(square): 0.000309 s
grad(  jit(grad(square))): 0.000115 s

Exp operation comparison:
  regular exp: 0.000049 s
grad(  grad(exp)): 0.000068 s
  jit(exp): 0.000189 s
grad(  jit(grad(exp))): 0.000093 s

Matrix multiplication comparison (128x128):
  regular matmul: 0.000074 s
grad(  grad(matmul)): 0.000126 s
  jit(matmul): 0.000231 s
grad(  jit(grad(matmul))): 0.000137 s

--- Memory and Performance Patterns ---
sum (contiguous): 0.000188 s
sum (after transpose): 0.000113 s
element-wise add (n=1000): 0.000013 s
element-wise add (n=10000): 0.000006 s
element-wise add (n=100000): 0.000131 s
element-wise add (n=1000000): 0.000911 s

--- Detailed Statistics (5 runs each) ---
add (detailed): min=0.000014 max=0.000124 mean=0.000071 median=0.000101 s
mul (detailed): min=0.000011 max=0.000020 mean=0.000013 median=0.000011 s
exp (detailed): min=0.000016 max=0.000045 mean=0.000023 median=0.000017 s

=== PyTorch Comprehensive Benchmarking Complete ===
