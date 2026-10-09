# nv.pci's tests

`test_nv_pci.ml` checks nv.pci through its public interface:

- GPU numbering and names on fixture trees;
- boot reports: the chip table, each family's firmware files, their lookup
  by pinned digest, and the VBIOS walk on the fixtures' images;
- opens of a GPU whose GSP runs, which reset it, and of a GPU another
  process holds, which never do;
- with NVIDIA's firmware listed in `RIG_NV_PCI_FIRMWARE` (`rig firmware nv
  DIR` fetches it), the lookup of found files and an open that fails before
  the boot takes memory. Without it these skip.

What nothing here checks, since no host that runs the suite gives root on
an NVIDIA GPU:

- the bytes of the boot: the GSP's queues and messages, the WPR metadata,
  the radix-3 table, the libos arguments, FWSEC's FRTS patch, the COT
  payload and the FSP's messages, the falcons' sequences and the GSP's CPU
  sequences, and the page-table entries;
- `detach`'s refusal of a GPU a process holds a device file of: `detach`
  acts on this machine only, and the files it reads (`/dev/nvidia*`,
  `/proc/driver/nvidia`, debugfs) cannot be stood in for.

- whether Linux's `rom` file of a GPU's function holds the FWSEC that
  `report` needs: it reads the ROM through the expansion ROM BAR, which may
  show less than the first MiB the open reads through BAR 0.

An open, a dispatch and a stop on a root NVIDIA host check them, and a read
of `rom` there settles the last.
