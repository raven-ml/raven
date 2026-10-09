# nv.pci's tests

`test_nv_pci.ml` checks nv.pci through its public interface:

- GPU numbering and names on fixture trees;
- boot reports: the chip table, each family's firmware files, their lookup
  by pinned digest, and the VBIOS walk on the fixtures' images;
- opens of a GPU whose GSP runs, which reset it, and of a GPU another
  process holds, which never do;
- with NVIDIA's firmware listed in `RIG_NV_PCI_FIRMWARE` (`rig firmware nv
  DIR` fetches it):
  - the lookup of found files;
  - an open that fails before the boot takes memory, which resets nothing;
  - a boot on a fixture tree that fails at the falcons' first DMA, after
    taking its memory: FWSEC in the GPU's memory set up for the memory's
    size, bus mastering off, the system memory given back, the GPU lost.

  Without the firmware these skip: it cannot be in the repository, and an
  open reaches nothing past its report without it.

`test_rpc.ml`, `test_fb_layout.ml` and `test_page_entry.ml` check the
library's private encodings of NVIDIA's formats, copied into the test
(`dune`): the GSP's records, checksums, messages and fault reports, the RM's
call bodies and answers, the CPU sequences and their refusals, the radix-3
table, the WPR metadata, the region the GSP reserves and its check, and the
page-table entries.

What nothing here checks, since no host that runs the suite gives root on
an NVIDIA GPU:

- the GSP's queues as rings over a window: wrap-around and a full queue;
- the falcons' interpreter and its sequences past FWSEC's DMA, the CPU
  sequences' steps among them;
- the order of a stop (bus mastering off before the memory is given back),
  of which a fixture shows only the end;
- `detach`'s refusal of a GPU a process holds a device file of: `detach`
  acts on this machine only, and the files it reads (`/dev/nvidia*`,
  `/proc/driver/nvidia`, debugfs) cannot be stood in for;
- whether Linux's `rom` file of a GPU's function holds the FWSEC that
  `report` needs: it reads the ROM through the expansion ROM BAR, which may
  show less than the first MiB the open reads through BAR 0.

An open, a dispatch and a stop on a root NVIDIA host check them, and a read
of `rom` there settles the last.
