# rig firmware fixtures

Both files are AMD's, under AMD's licence in `LICENSE.amdgpu`. Neither is
under raven's ISC licence.

- `smu_13_0_14.bin`: linux-firmware's `amdgpu/smu_13_0_14.bin` at commit
  `0a6871b19abf5d6e024b5d208b101ae53e7fa0de`, unmodified. It is 1,420
  bytes, and its BLAKE2b-256 digest is
  `03efa4555167f71cb1cb6057a67fdf0ea08dbbe9712e3e53c0a13dccdd7fd37c`, the
  one rig.amd.pci pins.
- `LICENSE.amdgpu`: that image's licence, linux-firmware's
  `LICENSES/LICENSE.amdgpu` at the same commit, whole.

The licence permits redistributing the image in binary form with its
notice, and forbids reverse engineering, decompiling and disassembling it.
The tests of `rig firmware` only serve the image, hash it and compare it:
it is the one image whose real pin they can show fetched and kept.
