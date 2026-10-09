# Rig Examples

Each example is a program that teaches one idea of rig, from buffers on the host
to kernels on a GPU. Start with `01-buffers` and work through them in order.

The numbered examples run on any machine. Where an idea needs a device with a
timeline, they use memory devices: devices whose memory is the host's and whose
work runs before `submit` returns, which show the same calls a GPU takes. Each
builds and runs with the test suite, and `main.expected` holds what it prints.
Run one from its directory, since some read or write files there:

```bash
cd dev/rig/examples/03-timelines
dune exec ./main.exe
```

| Example | Shows | Key functions |
|---|---|---|
| [`01-buffers`](./01-buffers/) | Bytes of a device's memory, on the host | `Buffer.create`, `of_bigarray`, `bigarray`, `view`, `copy` |
| [`02-files`](./02-files/) | Files as the disk's memory: copy or borrow | `Rig_disk.create_file`, `of_file`, `barrier`, `Buffer.borrow` |
| [`03-timelines`](./03-timelines/) | Submissions, values and points | `memory_device`, `Submission.make`, `Submission.Run.make`, `submit`, `wait` |
| [`04-stamps`](./04-stamps/) | Work across devices, ordered by the memory it touches | `Buffer.borrow`, `submit ~waits`, `Buffer.wait` |
| [`05-steps`](./05-steps/) | A step prepared once and run many times | `Submission.Fill`, `Hold.make`, `submit ~run ~reads ~writes` |
| [`06-reclamation`](./06-reclamation/) | Memory that returns without a free | `budget`, `set_budget`, `Out_of_memory` |
| [`07-claims`](./07-claims/) | Writing a donated input in place | `Claim.with_`, `exclusive`, `consume` |
| [`08-loss`](./08-loss/) | A device that fails, and what goes on | `Lost`, `lost`, `signaled` |
| [`09-profiles`](./09-profiles/) | What every device did while a function ran | `Profile.take`, `span`, `output_chrome_trace` |
| [`10-elf`](./10-elf/) | An ELF object read into its image | `Rig_elf.of_string`, `symbol` |
| [`11-host-programs`](./11-host-programs/) | A C function linked and called on the host's cores | `Rig_host.link`, `call`, `split` |
| [`12-pool`](./12-pool/) | A job of one's own on the host's threads | `rig_pool_run`, `rig_pool_cores` |

The `x-` examples need hardware. They build on every system and never run in
the test suite; run them by hand. Without what they need, they print one line
and exit.

| Example | Needs | Shows |
|---|---|---|
| [`x-gpu`](./x-gpu/) | Any GPU rig drives | Opening a GPU through its driver and path; copies; work in flight |
| [`x-metal-kernel`](./x-metal-kernel/) | A Mac on macOS 15 or later | A kernel from an indirect command buffer, run by an Objective-C fill |
| [`x-cuda-kernel`](./x-cuda-kernel/) | An NVIDIA GPU and `libcuda` | A PTX kernel launched as a part, its parameters in a run |
| [`x-amd-kernel`](./x-amd-kernel/) | An AMD gfx1201 GPU under `amdgpu` | A dispatch written as PM4 words |
| [`x-nv-kernel`](./x-nv-kernel/) | An NVIDIA sm_89 GPU under NVIDIA's kernel driver | A launch descriptor scheduled by channel words |
| [`x-pci`](./x-pci/) | Linux | The machine's PCI functions and which are GPUs |
