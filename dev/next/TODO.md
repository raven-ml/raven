# to do

refactor nx
- complete frontend
  - create
  - rand functions
  - sort functions
  - functions that realize tensors (array_equal, element access, etc.)
- allocate buffers outside of the backend?
- not sure we need non-polymorphic functions for perf of where, we should benchmark

poc jit
- placeholder? not in tinygrad
- const_scalar? not in tinygrad (there's const and vconst)
- test for kernelize (see kernelize.md)
- integrate with rune jit

next
- symbolic shapes for jit
- memory planner during lowering (scheduling)
