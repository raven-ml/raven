  $ ./bug_matmul_gradient_values.exe
  Testing matmul gradient computation
  a = 
  Nx Info:
    Shape: [2x3]
    Dtype: float32
    Strides: [3; 1]
    Offset: 0
    Size: 6
    Data: [[1, 2, 3],
           [4, 5, 6]]
  
  
  b = 
  Nx Info:
    Shape: [3x2]
    Dtype: float32
    Strides: [2; 1]
    Offset: 0
    Size: 6
    Data: [[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6]]
  
  
  c = a @ b = 
  Nx Info:
    Shape: [2x2]
    Dtype: float32
    Strides: [2; 1]
    Offset: 0
    Size: 4
    Data: [[2.2, 2.8],
           [4.9, 6.4]]
  
  
  Gradients:
  grad_a = 
  Nx Info:
    Shape: [2x3]
    Dtype: float32
    Strides: [3; 1]
    Offset: 0
    Size: 6
    Data: [[0.5, 0.7, 0.9],
           [0.5, 0.7, 0.9]]
  
  
  grad_b = 
  Nx Info:
    Shape: [3x2]
    Dtype: float32
    Strides: [2; 1]
    Offset: 0
    Size: 6
    Data: [[3, 3],
           [7, 7],
           [11, 11]]
  
  
  Expected grad_a (from test): [[0.3, 0.7, 1.1], [0.3, 0.7, 1.1]]
  This matches the manual calculation!
  
  But we got: Nx Info:
    Shape: [2x3]
    Dtype: float32
    Strides: [3; 1]
    Offset: 0
    Size: 6
    Data: [[0.5, 0.7, 0.9],
           [0.5, 0.7, 0.9]]
  
  
  This suggests the gradient computation might be using a different formula.