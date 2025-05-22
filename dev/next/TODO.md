# to do

refactor nx
- allocate buffers outside of the backend?
- how do we implement create?
- how do we do slicing and element access now?
- access to buffer as bigarray (data)
- [ ] cast
- [ ] matmul and dot
- [ ] convolve1d
- [ ] convolve2d
- not sure we need non-polymorphic functions for perf of where, we should benchmark

poc jit
- placeholder? not in tinygrad
- const_scalar? not in tinygrad (there's const and vconst)
- test for kernelize (see kernelize.md)
- integrate with rune jit

next
- symbolic shapes for jit
- memory planner during lowering (scheduling)
