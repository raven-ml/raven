# amd.pci's tests

`test_amd_pci.ml` checks amd.pci through its public interface:

- boot reports of the R9700's discovery tables (`../fixtures`), the lookup
  of its images by pinned digest, and the tables of other GPUs, made from
  the R9700's with other block versions: those this library boots are
  reported, the others refused, naming the block and its version;
- GPU numbering and names on fixture trees;
- the refusal of an open of a GPU amdgpu has not let go of.

`test_root.ml` opens this machine's GPU 0 driver-less, as root, with the
firmware `RIG_AMD_PCI_FIRMWARE` lists, and skips elsewhere: two domains
closing one device, memory freed after its GPU opened again, and a GPU
whose holder was killed, reset by the next open.

`test_discovery.ml`, `test_firmware.ml`, `test_psp.ml`, `test_smu.ml`,
`test_gmc.ml`, `test_gfx.ml` and `test_ih.ml` check the formats the boot
reads and writes, through amd.pci's hidden section: the discovery table,
the firmware images and their pieces, the security processor's commands,
the power manager's messages, the page-table entries, the queue
descriptors and the interrupt entries.

What nothing here checks, since no interface reaches it without a GPU:

- how an open boots a GPU from the marks the last session left: partially,
  in full, or after a reset;
- the order in which a reset restores the function's configuration;
- a flush the hubs did not confirm turning bus mastering off at once;
- a stop disabling a copy queue a session that died left enabled;
- the registers' addresses, the GPU's shape and compiler target read from
  its discovery table, and the ranges a virtual function guards.

A driver-less open on a detached GPU (`test_root.ml`) goes through all of
them. Its killed holder checks what `rig_amd_pci.mli` promises of them: a
GPU whose state rig cannot vouch for is reset before rig uses it.
