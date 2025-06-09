  $ ./bug_metal_cast_complex.exe
  Testing Metal cast operation with complex types...
  Created float32 tensor:
  Nx Info:
    Shape: [2x2]
    Dtype: float32
    Strides: [2; 1]
    Offset: 0
    Size: 4
    Data: [[1, 2],
           [3, 4]]
  
  
  Trying to cast float32 to complex64...
  Expected failure: dtype_to_metal_type: complex types not supported
  
  Trying to create complex64 tensor directly...
  Created complex tensor:
  Nx Info:
    Shape: [2x2]
    Dtype: complex64
    Strides: [2; 1]
    Offset: 0
    Size: 4
    Data: [[(0+0i), (1+0i)],
           [(2+0i), (3+0i)]]
  
  FAIL: Metal should not support complex types