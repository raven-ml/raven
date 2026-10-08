# `x-pci`

**Needs Linux.** Elsewhere this machine lists no PCI functions, and the example
says so.

The drivers that boot a GPU over PCI, with no kernel driver, start from the
machine's PCI functions. This example lists them, marks the GPUs, and counts the
GPUs each kernel-driver path numbers. Listing needs no privilege and changes
nothing; taking a function needs both, and is left out.

```bash
dune exec dev/rig/examples/x-pci/main.exe
```

## What You'll Learn

- This machine: `Rig_pci.Machine.this`, `functions`, `page`
- A function's identity: its bus address, vendor, device and class code
- Which functions are GPUs: `Rig_amd.is_gpu`, `Rig_nv.is_gpu`
- Every path numbers a vendor's GPUs in bus order, so GPU `i` is the same GPU
  whichever path opens it

## Key Functions

| Function                           | Purpose                                 |
| ---------------------------------- | --------------------------------------- |
| `Rig_pci.Machine.functions m`      | The PCI functions of `m`, in bus order  |
| `Rig_amd.is_gpu ~vendor ~class_`   | Whether a function is an AMD GPU        |
| `Rig_nv.is_gpu ~vendor ~class_`    | Whether it is an NVIDIA GPU             |
