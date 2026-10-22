# nv.pci's tests

`test_nv_pci.ml` checks nv.pci through its public interface:

- GPU numbering and names on fixture trees;
- boot reports: the chip table, each family's firmware files, their lookup
  by pinned digest, and the VBIOS walk on the fixtures' images;
- opens of a GPU whose GSP runs, which reset it, and of a GPU another
  process holds, which never do; a reset that leaves WPR2 up, which fails;
- the order an open searches for firmware in: its caller's directories,
  rig's, then `/lib/firmware`;
- with NVIDIA's firmware in a directory `RIG_NV_PCI_FIRMWARE` lists, rig's
  firmware directory or `/lib/firmware` (`rig firmware nv DIR` fetches it),
  where the NV suites look too:
  - the lookup of found files;
  - an open that fails before the boot takes memory, which resets nothing;
  - a boot on a fixture tree that fails at the falcons' first DMA, after
    taking its memory: FWSEC in the GPU's memory set up for the memory's
    size, bus mastering off, the system memory given back, the GPU lost.

  Without the firmware these skip: it cannot be in the repository, and an
  open reaches nothing past its report without it.

`test_rpc.ml`, `test_fb_layout.ml` and `test_page_entry.ml` check the
formats the boot writes and reads, through the library's hidden section:
the GSP's records, calls and error reports, the tables of its addresses
it is handed, the register accesses its CPU sequences become, the region
it reserves and the page-table entries.

What nothing here checks, since no host that runs the suite gives root on
an NVIDIA GPU:

- the GSP's queues as rings over a window: wrap-around and a full queue;
- the GPU's page tables as the GPU walks them, translating from address 0,
  and the golden context's tables handed to the GSP: a fixture's boot
  stops at the falcons' first DMA, before the tables hold an entry;
- the error of a call the GSP leaves unanswered, which names the GPU's
  first fault and the GSP's reports since the call began;
- the falcons' interpreter and its sequences past FWSEC's DMA, the CPU
  sequences' steps among them;
- the order of a stop (bus mastering off before the memory is given back),
  of which a fixture shows only the end, and its teardown: the GSP's
  suspension, FWSEC's SB command and the unloading booter taking WPR2 down;
- `detach`'s refusal of a GPU a process holds a device file of: `detach`
  acts on this machine only, and the files it reads (`/dev/nvidia*`,
  `/proc/driver/nvidia`, debugfs) cannot be stood in for;
- whether Linux's `rom` file of a GPU's function holds the FWSEC that
  `report` needs: it reads the ROM through the expansion ROM BAR, which may
  show less than the first MiB the open reads through BAR 0.

An open, a dispatch and a stop on a root NVIDIA host check them, and a read
of `rom` there settles the last.
