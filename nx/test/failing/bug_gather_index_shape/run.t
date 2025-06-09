  $ ./bug_gather_index_shape.exe
  Testing gather operation index shape bug
  Data shape: [2; 3; 4]
  FAIL: Gather raised exception: op_gather: data rank (3) and indices rank (1) must match
  This is the bug - gather creates multi-dimensional index tensor instead of 1D
