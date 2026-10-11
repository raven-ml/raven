# Shared fixtures

Files that more than one suite opens, each made from its source by
`gen.sh`.

- `<f>_x86_64.o` and `<f>_aarch64.o` for each `<f>` of `affine`,
  `calling` and `waiting`, the host and program suites' host code, and
  `ready`, the program and remote suites' host code that writes a rail's
  area and calls its ready function. Each source says what it does.
- `simple_add_sm89.cubin`, an NVIDIA executable of `simple_add.cu` that the
  ELF and NV ABI suites read.
