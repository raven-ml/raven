# Byte-copy kernel

`copy.cu` adapts NVIDIA nvbandwidth's `kernels.cu` at commit
`82fc4e8c6afa0babb8687793678f615b3b8d793e` under Apache-2.0 (`LICENSE`).
It adds arbitrary byte lengths, an unaligned byte loop, and one copy per
launch. Aligned pointers use cached 16-byte loads and streaming stores.

Regenerate the private OCaml module with:

```
uv run gen.py --nvcc /usr/local/cuda/bin/nvcc
uv run gen.py --check
```

Both refuse an object whose `copy_bytes` takes stack or shared memory: the
driver launches it with no local memory.

The compiler is needed only to regenerate the source-derived object.
Ordinary builds and device opening read its embedded ELF bytes through the
driver's existing cubin reader. No CUDA library is linked.

The committed sm80 object was made with NVCC 13.4.92. Its metadata reports
16 registers and zero stack, local and shared bytes. It has passed the full
offset, guard and source-preservation checks plus 256 MiB on sm89, using
Pinned code and zero local-memory backing. Other sm8x devices are untested.
Other architectures currently use the copy engine.
