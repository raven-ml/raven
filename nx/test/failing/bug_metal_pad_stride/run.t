  $ ./bug_metal_pad_stride.exe
  Testing Metal pad operation with non-contiguous views
  Original matrix shape: [3, 2]
  Transposed view shape: [2, 3]
  Transposed is contiguous: false
  
  Padded transposed view:
  Nx Info:
    Shape: [4x5]
    Dtype: float32
    Strides: [5; 1]
    Offset: 0
    Size: 20
    Data: [[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 0],
           [0, 4, 5, 6, 0],
           [0, 0, 0, 0, 0]]
  
  
  Expected padded result:
  Nx Info:
    Shape: [4x5]
    Dtype: float32
    Strides: [5; 1]
    Offset: 0
    Size: 20
    Data: [[0, 0, 0, 0, 0],
           [0, 1, 3, 5, 0],
           [0, 2, 4, 6, 0],
           [0, 0, 0, 0, 0]]
  
  
  This test demonstrates that Metal pad needs stride support for non-contiguous views