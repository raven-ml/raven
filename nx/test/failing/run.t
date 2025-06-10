  $ ./bug_blit_overlapping.exe
  Original array: [1 2 3 4 5 ]
  
  view1 = slice [R [0; 3]] (indices 0-2): [1 2 3 ]
  view2 = slice [R [2; 5]] (indices 2-4): [3 4 5 ]
  
  Attempting blit view1 -> view2...
  Result: [1 2 1 2 1 ]
  Expected: [1 2 1 2 3]

  $ ./bug_conv_memory_simple.exe
  Simple convolution memory corruption test...
  30913 Abort trap: 6           ./bug_conv_memory_simple.exe
  [134]

  $ ./bug_gather_index_shape.exe
  Testing gather operation index shape bug
  Data shape: [2; 3; 4]
  FAIL: Gather raised exception: op_gather: data rank (3) and indices rank (1) must match
  This is the bug - gather creates multi-dimensional index tensor instead of 1D

  $ ./bug_logspace_division_by_zero.exe
  Testing logspace division by zero bug
  Unexpected exception: Failure("BUG: logspace should have raised Division_by_zero exception")
  Fatal error: exception Failure("BUG: logspace should have raised Division_by_zero exception")
  [2]

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
    Data: [10, 26, 42]
  
  FAIL: Index 0: expected 18.0, got 10.0
  Fatal error: exception Invalid_argument("index out of bounds")
  [2]

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

  $ ./bug_slice_batch_process_bounds.exe
  Testing slice batch processing bounds bug
  Input shape: [3; 4; 5]
  Unexpected exception: Invalid_argument("op_gather: data rank (3) and indices rank (1) must match")
  Fatal error: exception Invalid_argument("op_gather: data rank (3) and indices rank (1) must match")
  [2]
