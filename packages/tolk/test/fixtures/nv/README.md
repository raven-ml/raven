# NVRTC loader fixture

`simple_add_sm89.cubin` is a real NVRTC 12.8 executable for `sm_89`.
`simple_add_sm89.fields` comes from `NVProgramData` in the frozen tinygrad
migration target `471a3aeb6924257d5e9bf321f5ff0a519163f18e`, without opening a GPU.
The JSON sidecar records compiler options, source/binary hashes and hashes of
the reference compiler and parser. The test requires the committed fixture;
neither the toolkit nor Python is needed to run it.

From the repository root, with that target extracted into `_tinygrad_target`:

```sh
docker run --rm --network none \
  -v "$PWD/_tinygrad_target:/target:ro" \
  -v "$PWD/packages/tolk/test/fixtures/nv:/fixtures" \
  ghcr.io/tinygrad/cuda-arm64@sha256:350801c5bc2fd3cf7a0c59e88e10483419a97a039e39bcb304cb9598d995349a \
  /fixtures/generate_fixture.py --tinygrad-root /target \
  --compiler-image ghcr.io/tinygrad/cuda-arm64@sha256:350801c5bc2fd3cf7a0c59e88e10483419a97a039e39bcb304cb9598d995349a
```

This is the CUDA compiler image used by the target on macOS. Its generation
command requires an ARM64 Docker host or emulation. The generator can also run
on Linux with libnvrtc; supply an accurate `--compiler-image` identifier for
that environment and review the changed provenance and expectations.

The fields describe the target's argument blob. Tolk's direct dispatch test
also accounts for its eight trailing QMD slots. This fixture checks loading
and metadata, not GPU execution or hardware compatibility.
