# Rig Examples

Each example is a program that teaches one idea of rig. Start with `01-add`,
which runs a kernel on a GPU. The examples after it take that program apart,
in order: memory, timelines and order, steps, the lives of buffers and
devices, and host code. `14-machines` uses another machine's devices.

| Example | Shows | Key functions |
|---|---|---|
| [`01-add`](./01-add/) | A kernel on the first GPU here, or on the host's cores | `Rig_gpus.openers`, `Image.load`, `Submission.launch`, `submit` |
| [`02-buffers`](./02-buffers/) | Bytes of a device's memory, on the host | `Buffer.create`, `of_bigarray`, `read`, `write`, `view`, `copy` |
| [`03-files`](./03-files/) | Files as the disk's memory, copied or borrowed | `Rig_disk.create_file`, `of_file`, `barrier`, `Buffer.borrow` |
| [`04-timelines`](./04-timelines/) | Submissions, values and points | `memory_device`, `Submission.make`, `Submission.Run.make`, `submit` |
| [`05-stamps`](./05-stamps/) | Work across devices, ordered by the memory it touches | `reaches`, `Buffer.borrow`, `Buffer.read`, `Buffer.write` |
| [`06-steps`](./06-steps/) | A step made once and run many times | `Submission.fill`, `Submission.make ~keep ~fixed`, `Submission.address` |
| [`07-reclamation`](./07-reclamation/) | Memory that returns without a free | `budget`, `set_budget`, `allocated`, `Out_of_memory` |
| [`08-claims`](./08-claims/) | Writing a donated input in place | `Claim.with_`, `exclusive`, `consume`, `Buffer.export` |
| [`09-loss`](./09-loss/) | A device that fails, and what goes on | `Lost`, `lost`, `signaled` |
| [`10-profiles`](./10-profiles/) | What every device did while a function ran | `Profile.take`, `span`, `output_chrome_trace` |
| [`11-elf`](./11-elf/) | An ELF object read into its image | `Rig_elf.of_string`, `symbol` |
| [`12-host-programs`](./12-host-programs/) | A C function loaded and launched on the host's cores | `Image.load host`, `Submission.launch`, `submit` |
| [`13-pool`](./13-pool/) | A job of one's own on the host's threads | `rig_pool_run`, `rig_pool_cores` |
| [`14-machines`](./14-machines/) | Another machine's devices, through an agent | `Rig_remote.listen`, `serve`, `connect`, `devices` |

The numbered examples run on any machine, in the test suite, and print the
same on each, which `main.expected` holds. Where an idea needs a device with
a timeline, they use memory devices, which take the same calls as a GPU.
A memory device's memory lies in the host's memory, and its work runs
before `submit` returns.
`01-add` uses a GPU where rig has a binary of its kernel for one, and the
host's cores otherwise, and prints the device on stderr.

Run an example from its directory, since some read or write files there:

```bash
cd packages/rig/examples/04-timelines
dune exec ./main.exe
```

`01-add`'s kernel binaries are committed beside their sources, in
`kernels/`. Its `gen.sh` makes them again, and neither dune nor the test
suite runs it.

[`x-pci`](./x-pci/) needs Linux. It builds on every system and never runs in
the test suite. Elsewhere it prints one line and exits.

| Example | Needs | Shows |
|---|---|---|
| [`x-pci`](./x-pci/) | Linux | The machine's PCI functions and which are GPUs |
