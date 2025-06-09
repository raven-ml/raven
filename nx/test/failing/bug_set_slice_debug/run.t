  $ ./bug_set_slice_debug.exe
  Testing set_slice behavior:
  Original array:
  [[0, 0, 0],
   [0, 0, 0]]
  
  Value to set: [7, 8, 9]
  
  
  1. Using set_slice [I 1] (should set row 1):
  [[0, 0, 0],
   [7, 8, 9]]
  
  
  2. Using set_slice [I 1; R []] (explicit full range for columns):
  Fatal error: exception Invalid_argument("set_slice: cannot reshape [3] to [1,3] (incompatible ranks 1 and 2)")
  [2]
