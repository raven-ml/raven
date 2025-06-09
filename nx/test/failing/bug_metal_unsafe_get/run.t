  $ ./bug_metal_unsafe_get.exe
  Testing Metal unsafe_get operation...
  Created matrix:
  Nx Info:
    Shape: [2x2]
    Dtype: float32
    Strides: [2; 1]
    Offset: 0
    Size: 4
    Data: [[1, 2],
           [3, 4]]
  
  
  Trying to get element at [0, 1]...
  FAIL: Invalid_argument("index out of bounds")
  Metal backend missing implementation for get/unsafe_get