# GPUs without a driver

On Linux, rig opens an AMD or NVIDIA GPU that no kernel driver holds. The
process takes the GPU's PCI function and boots the GPU itself. This page
follows one such GPU from its kernel driver to a running device and back.
[`Rig_pci`](../lib/pci/rig_pci.mli) states what any such open needs, in its
section "What an open needs", and each opener states its vendor's facts:
[`Rig_amd_pci`](../lib/amd/pci/rig_amd_pci.mli) and
[`Rig_nv_pci`](../lib/nv/pci/rig_nv_pci.mli). Once open, the device is an
ordinary `Rig_amd` or `Rig_nv` device, and every rule of the other pages
holds for it.

## Preparing the machine

Take a machine with one AMD GPU, `0000:03:00.0`, driven by `amdgpu` and
showing a desktop. The GPU is one PCI function of its card, beside its audio
function, and it is GPU 0 of its vendor whichever driver holds it
([`Rig_pci`](../lib/pci/rig_pci.mli), "What an open needs").

An administrator detaches it once, outside the program:
`Rig_amd_pci.detach` unbinds `amdgpu`, keeps every kernel driver off the
function, removes the audio function and grows the GPU's memory window
([`Rig_pci.Driver`](../lib/pci/driver.mli), "Changes to the machine"). The
desktop loses the GPU until it is attached again or the machine reboots.

The machine's state then decides how a process will take the function
([`Function`](../lib/pci/function.mli)):

- **Behind an IOMMU**, through VFIO. The administrator binds the function to
  `vfio-pci` and grants the user its group's file, and the program needs no
  root. The GPU reaches only the memory the process maps for it.
- **Physically**, through the function's files under `/sys/bus/pci`. The GPU
  reaches system memory at its physical addresses, and nothing confines it.

## The GPU's states

From here the GPU moves between three states
([`Rig_pci.Driver`](../lib/pci/driver.mli)):

```
                 detach                 Rig.open_
kernel driver ----------> detached -----------------> held
      ^                    |  ^  ^                      |
      +------- attach -----+  |  +------- close --------+
                              |                         | unclean stop,
                              |                         v killed process
                              +--------- reset ------- lost
```

A program's `Rig.open_` holds the GPU until its device stops, and a close
leaves it detached again. A fault, a hang or a killed process leaves it
lost, and its next open resets it first. Each opener states what its next
open does in each state.

## Firmware

The GPU runs signed firmware from its vendor, which the open loads at boot
from directories its caller names, then from rig's firmware directory,
`$XDG_DATA_HOME/rig/firmware` or `~/.local/share/rig/firmware`, then from
`/lib/firmware`, each image pinned by its digest
([`Firmware`](../lib/pci/firmware.mli)). The `rig firmware` command fetches
the pinned images into rig's directory
([`Rig_pci`](../lib/pci/rig_pci.mli), "What an open needs", and
`rig firmware --help`). The directory is in the home of the user the open
runs as. A program that takes the GPU through VFIO runs as its user, who
fetches the images once:

<!-- $MDX skip -->
```sh
$ rig firmware amd
```

The opener then needs no directory:

<!-- $MDX skip -->
```ocaml
let gpu = Rig.open_ (Rig_amd_pci.gpu 0)
```

A program run as root, as a physical takeover usually is, searches root's
home instead, so its images come from `sudo rig firmware amd`. A program
that keeps the images elsewhere names their directory to the opener, as
`Rig_amd_pci.gpu ~firmware:[ "/opt/rig-firmware" ] 0`, and keeps that
directory apart from `/lib/firmware`.

An image missing from all of them makes the open return `Error`, naming the
image, before the boot writes to the GPU
([`Rig_amd_pci.gpu`](../lib/amd/pci/rig_amd_pci.mli),
[`Rig_nv_pci.gpu`](../lib/nv/pci/rig_nv_pci.mli)).

## Booting

On AMD, the open starts the GPU's security processor, which loads the other
firmware, then its power manager, its compute engine and its copy engines,
and lets the GPU master the bus only once its memory hubs translate
addresses ([`Rig_amd_pci`](../lib/amd/pci/rig_amd_pci.mli), "Opening").
On NVIDIA, the open boots the GPU's system processor (GSP) and then acts as
the kernel's half of the resource manager that the GSP runs. Each opener
states what its boot needs and what it refuses.

Once booted, the GPU runs submissions as it would under its kernel driver.
The process stops it when the device closes, and at the process's exit
([`Rig_pci.Driver`](../lib/pci/driver.mli), "Opening").

## Memory the GPU may write

Behind an IOMMU, the GPU's access ends when the process's files close, at
its death too. Taken physically, the GPU keeps writing its pages after its
process dies of a signal or a crash, so the library gives it huge pages of a
file under `/dev/hugepages`, which outlive the process until the GPU is next
reset ([`Function`](../lib/pci/function.mli), "System memory").

For the same reason, a device whose GPU is taken physically maps no other
host memory: its driver declares no `maps_host`
([`Rig_amd_pci`](../lib/amd/pci/rig_amd_pci.mli), "What an open needs";
[`Rig_nv_pci`](../lib/nv/pci/rig_nv_pci.mli), "Requirements"). Copies
between it and host memory then go through staging memory
([`Rig.Buffer`](../lib/rig.mli), "Copying").

## Safety

A GPU taken physically can write any memory of the machine that an address
in its work or its page tables names, and the kernel can move even huge
pages while a GPU whose process died still writes them. An IOMMU is the only
protection from both ([`Rig_pci`](../lib/pci/rig_pci.mli), "What a function
can reach"). No kernel bounds a driver-less GPU's work either, so its device
declares a hang bound, and rig loses the device when its work stops
advancing ([`Rig_pci.Driver.hang_ms`](../lib/pci/driver.mli), and
[`Rig`](../lib/rig.mli), "Loss").

## Status

Which of these paths have run on hardware is stated by
[`Rig_amd_pci`](../lib/amd/pci/rig_amd_pci.mli) and
[`Rig_nv_pci`](../lib/nv/pci/rig_nv_pci.mli).
