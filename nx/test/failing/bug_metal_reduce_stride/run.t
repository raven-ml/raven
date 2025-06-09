  $ ./bug_metal_reduce_stride.exe
  Testing Metal reduce operations with non-contiguous views
  Original matrix shape: [3, 4]
  Transposed view shape: [4, 3]
  Transposed is contiguous: false
  
  Sum along axis 0 of transposed view:
  Nx Info:
    Shape: [3]
    Dtype: float32
    Strides: [1]
    Offset: 0
    Size: 3
    Data: [22, 26, 30]
  
  FAIL: Index 0: expected 18.0, got 22.0
  Fatal error: exception Invalid_argument("index out of bounds")
  [2]
