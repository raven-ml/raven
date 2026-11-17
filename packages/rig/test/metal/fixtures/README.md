# Metal fixtures

`gen.sh` makes each library from its source. `fill.metallib` holds the
kernels of `fill.metal`, `launch.metallib` those of `launch.metal`,
`vertex.metallib` the vertex function of `vertex.metal`,
`threadgroup.metallib` the kernels of `threadgroup.metal`, and
`constants.metallib` those of `constants.metal` and `narrow.metal`, linked
into one library. Each source says what its functions do.
