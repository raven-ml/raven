# Private AMD byte copy

`copy.cl` derives from ROCm clr's MIT-licensed `__amd_rocclr_copyBuffer`
in `rocclr/device/blitcl.cpp`, commit
`2007bc5482af4d67b402016d13a9c98b4170a881`. `LICENSE` preserves its license.
It takes destination, source, and byte count. Aligned addresses use 16-byte
vectors and a byte tail; other addresses use byte loads and stores.

Regenerate `../copy_kernel.ml` with Clang and AMDGPU LLD 22:

```sh
uv run python generate.py --clang /path/to/clang --lld /path/to/ld.lld
```

It refuses an object whose `copy_bytes` is not what the driver dispatches:
no private or group memory, 24 bytes of arguments, groups of 256
work-items.

The checked-in objects were compiled by Homebrew Clang 22.1.7 and linked
by Triton's LLD 22.0.0, LLVM revision
`f58b06dce1f9c15707c5f808fd002e18c2accf7e`. They use code object V6.
The runtime uses its existing ELF reader and launch descriptors. It has no
compiler or linker dependency.

| Target | ELF bytes | Image bytes | SHA256 |
|---|---:|---:|---|
| gfx9-4-generic | 6152 | 14336 | 62ca0208baa1276b8821811de8c241b14c3e3bc583768aecdde4aad22f504f30 |
| gfx11-generic | 5816 | 10240 | 1e3d8c55025d3cdbf1024a5c4229c18d279fdbac8af7363ed69c54f735e38f90 |
| gfx12-generic | 5816 | 10240 | 7fa73234c9da7ed91211dfd830982d0dd611aa9a164c3beaa073790fb361e0ed |

All three objects have zero private and group segments. Only gfx12-generic
has run on hardware: gfx1201, through ordinary amdgpu/KFD with PM4, including
Pinned instruction storage and the complete 256 MiB byte oracle. The other
targets, AQL, and driverless paths remain untested.
