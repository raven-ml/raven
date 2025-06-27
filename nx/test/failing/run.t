  $ ./bug_blit_overlapping.exe
  Original array: [1 2 3 4 5 ]
  
  view1 = slice [R [0; 3]] (indices 0-2): [1 2 3 ]
  view2 = slice [R [2; 5]] (indices 2-4): [3 4 5 ]
  
  Attempting blit view1 -> view2...
  Result: [1 2 1 2 1 ]
  Expected: [1 2 1 2 3]

  $ ./bug_gather_index_shape.exe
  Testing gather operation index shape bug
  Data shape: [2; 3; 4]
  Result shape: 2 3 4 
  Expected shape: [2; 2; 4]
  FAIL: Gather produced incorrect shape
  Got: [2 3 4 ]

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
  FAIL: Index 1: expected 22.0, got 26.0
  FAIL: Index 2: expected 26.0, got 42.0
  
  FAIL: Reduce operation produces incorrect results on non-contiguous views
  This is a Metal-specific bug - reduce kernels don't handle strides/offsets

  $ ./bug_slice_batch_process_bounds.exe
  Testing slice batch processing bounds bug
  Input shape: [3; 4; 5]
  Unexpected exception: Invalid_argument("op_shrink: shrink: invalid bounds array (length 1 != ndim 3)")
  Fatal error: exception Invalid_argument("op_shrink: shrink: invalid bounds array (length 1 != ndim 3)")
  [2]
