  $ ./bug_blit_overlapping.exe
  Original array: [1 2 3 4 5 ]
  
  view1 = slice [R [0; 3]] (indices 0-2): [1 2 3 ]
  view2 = slice [R [2; 5]] (indices 2-4): [3 4 5 ]
  
  Attempting blit view1 -> view2...
  Result: [1 2 1 2 1 ]
  Expected: [1 2 1 2 3]
