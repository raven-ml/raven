# AMD loader fixture

`simple_add_gfx1100.hsaco` is a linked AMDGPU HSA code object, compiled from
`simple_add.cpp` by LLVM clang 22.1.7 and linked by Ubuntu LLD 18.1.3. It uses
code-object v5 explicitly: that version is understood by both tools. The
kernel uses compiler builtins and has no dependency on ROCm device libraries.
A fixed compilation-unit ID avoids embedding a hash of the checkout path.
Its indexing assumes 64 work-items per group.

`simple_add_gfx1100.fields` and `.image` come from `_amd_program_image` in the
frozen tinygrad target `471a3aeb6924257d5e9bf321f5ff0a519163f18e`. The required
host test compares resource fields and the entire image against those outputs,
with the documented NOBITS layout adjustment: Tolk reserves `.relro_padding`
and the zero-filled `__hip_cuid` symbol, moving the unchanged comment section
from offset 6272 to 14577. Both images end on a four-byte boundary. See
`packages/tolk/DIVERGENCES.md` for the CPU custom-kernel consumer that requires
zero-filled ELF storage. This test does not open a GPU or establish hardware
execution. No compiler or Python is needed to run it.

To regenerate with an AMDGPU-capable LLVM installation and that target
extracted into `_tinygrad_target`, run from the repository root:

```sh
python3 packages/tolk/test/fixtures/amd/generate_fixture.py \
  --tinygrad-root _tinygrad_target --clang clang --lld ld.lld
```

The JSON sidecar records complete tool versions, options, source/binary/image
hashes and reference parser hashes. Review those together with the regenerated
fields. Generation here used Homebrew clang on macOS and LLD in an ARM64
Ubuntu 24.04 container (`lld-18`): the relocatable object was copied into the
container, linked with `ld.lld-18 -shared`, and the result copied back. The
same two tools can run natively; the container is not an application dependency.
