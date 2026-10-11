# Host code fixtures

`gen.sh` makes `<f>_x86_64.o` and `<f>_aarch64.o` from `<f>.c` for each
`<f>`:

- `spin`, the bench's fill that counts to the word its argument points at,
  as work that takes a device a known while.
- `fill`, which stores its second parameter at the address in its first;
  `fill_indirect` follows that address once more; `fill_empty` accepts no
  parameters; `fill_call` reads a host entry from the container in its first
  parameter and calls it on the buffer in its second.
