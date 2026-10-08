# Rig

Rig drives compute devices from OCaml: the host's CPU, GPUs and stores of
bytes, their memory, and the order of work across them. The library `rig`
holds devices, buffers and submissions; each driver is a library of its own:
`rig.host` runs ELF programs on the host's threads (`rig.elf`, `rig.pool`),
`rig.metal` drives the Mac's GPU, `rig.cuda` CUDA GPUs, `rig.amd` AMD GPUs
through amdgpu (`rig.amd.amdgpu`) or over PCI (`rig.amd.pci`), `rig.nv`
NVIDIA GPUs through their kernel driver (`rig.nv.nvidia`) or over PCI
(`rig.nv.pci`), `rig.disk` files, and `rig.remote` another machine. The
`.abi` libraries describe each vendor's formats, `rig.pci` reaches a GPU's
PCI function, and `rig.edge` is the C interface by which work reaches a
driver. Each `.mli` documents its library; `doc/` holds the GPU hardware
notes and the testing guide.
