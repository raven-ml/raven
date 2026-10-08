# GPU hardware notes

Writing the drivers in `dev/device` taught us facts about Apple, NVIDIA and
AMD GPUs, and the software around them, that the vendors document poorly or
not at all. Each note states one fact, how it was established, and what a
driver does about it. Paths are relative to `dev/device/`; a commit hash names
the commit in this repository that carries the test, probe or measurement. A
fact seen on one machine is stated with that machine and may differ on
another. "Read in source" marks a fact taken from vendor code that no run here
has confirmed.

The machines:

| Name here | Hardware | Software |
|---|---|---|
| M1 Max | Apple M1 Max, GPU family Apple7, 32 GB | macOS 26.3.1 |
| RTX 5000 Ada | NVIDIA RTX 5000 Ada Generation, PCIe 4.0 x16 | Linux 6.12, NVIDIA kernel driver 615.71.09, CUDA 13.4 |
| R9700 | AMD Radeon AI PRO R9700 (gfx1201, GFX12), resizable BAR off, behind a PCIe 3.0 x16 root port | Linux 6.12, KFD 1.17 |

## NVIDIA

### A ring entry names one contiguous run of commands

A GPFIFO entry holds a pushbuffer segment's address and its length in words
(`NVC56F_GP_ENTRY0_GET`, `NVC56F_GP_ENTRY1_GET_HI` and
`NVC56F_GP_ENTRY1_LENGTH` in `clc56f.h`). The channel's DMA engine (PBDMA)
reads that many words from that address, in address order. It knows nothing
of the buffer the driver writes its segments into, so a segment that runs to
that buffer's end continues past it.

On the RTX 5000 Ada, an entry that claimed 16 words starting 36 bytes before
the end of a 1 MiB segment buffer ended both of the device's channels with
the RM's channel error 32 (`PBDMA_ERROR`). A loop of 16-byte copies, each
taking 64 bytes of segment, reached that position at exactly the 32,768th
copy; the copies before it, and single copies of every memory kind, ran.

A driver that keeps its segments in a ring closes the open segment into its
entry whenever its write position returns to the ring's start, also when the
last command ended exactly at the ring's end, and opens a new segment there.

### CUDA's compiled code reads the launch sizes from constant bank 0

Code that NVCC compiles reads `blockDim` and `gridDim` from the start of
constant bank 0, an area the CUDA driver fills and no NVIDIA header
describes. nvdisasm of NVCC 13.4's output shows where: for sm_89, `blockDim`
x, y, z at `c[0x0][0x0]`, `[0x4]`, `[0x8]` and `gridDim` at `[0xc]` to
`[0x14]`; for sm_120 (Blackwell), `blockDim` from `c[0x0][0x360]` and
`gridDim` from `[0x370]`. On Ada the shared and local memory windows follow
at `0x18` and `0x20`, 64 bits each, and the stack limit at `0x28`; kernels
that use them run on the RTX 5000 Ada with these words.

A driver that launches through its own launch descriptors writes these words
itself. Left at zero, a kernel that indexes by `blockDim.x` computes wrong
addresses and does not fault: on the RTX 5000 Ada a 16-block launch wrote one
block's elements, and 936 of 1,000 sums were missing. With the words written,
a kernel that reads them saw the sizes set (956264bda). The Blackwell offsets
come from the disassembly alone; no Blackwell GPU ran them.

### The kernel driver's structures change between releases

The parameters of the RM's escape ioctls and of the unified memory driver's
ioctls change layout between release branches. Between 580, 610 and 615 these
changed: the GPFIFO allocation (610), the channel group allocation (615), the
schedule control (580), the framebuffer information control, the DMA map and
the VA space allocation (580), and the unified memory driver's free and
channel unregistration, which lost a field (610). Release 615's headers
comment out two status codes. Established by generating the layouts from each
release's headers, pinned by SHA-256, and comparing them (848a0af91,
b383e3c2e). The headers also define some names twice under conditionals (the
unified memory driver's ioctl base differs on Windows), so a reader evaluates
them as a 64-bit Linux build does (cf659ef9f).

A driver reads the release from `NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2`,
uses that release's layouts, and refuses a release it has none for.

### Unified memory needs a second file per process

`UVM_MM_INITIALIZE` ties the unified memory driver to the process's address
space. It runs on a second open file of `/dev/nvidia-uvm` and names the
first. Run on the first file, it is refused; registering a GPU's VA space then
fails with `NV_ERR_PAGE_TABLE_NOT_AVAIL`, and no GPU opens (RTX 5000 Ada,
125150f84). The call is made once per process: a second one, such as CUDA's in
the same process, is refused, and the first serves.

The unified memory driver also registers a GPU once per process. A driver
keeps a GPU's RM device, subdevice, VA space and their registration for the
life of the process, and a device's end releases only its channels'
registrations (b782f91ef).

### A map into an empty region pays for its page tables

On the RTX 5000 Ada, `NV_ESC_RM_MAP_MEMORY_DMA` of a 64 KiB allocation takes
21.4 µs when nothing else is mapped in its region of the GPU's address space,
and 8.6 µs with the same parameters when one other allocation lives there.
Measured per ioctl with an `LD_PRELOAD` counter. We read the difference as the
RM building the region's page tables at its first map and freeing them at its
last unmap. A 64 KiB allocation and free took 73.5 µs alone and 46.0 µs beside
one live 4 KiB allocation.

A driver that allocates and frees small buffers in an otherwise empty region
pays this every time. Long-lived memory of the driver in the same region
keeps the tables; allocation benches hold one live allocation of the kind, as
a running program does.

### Rings in host memory cost less host time than rings in BAR1

The RM accepts a channel's GPFIFO ring and USERD in host memory (an
OS-descriptor allocation) as well as in GPU memory, and a one-page coherent
host buffer as the channel's error notifier. On the RTX 5000 Ada:

| | Rings in host memory | Rings in BAR1 |
|---|---|---|
| Empty submission, round trip | 3.45 µs | 3.24 µs |
| 16-byte copy, round trip | 4.64 µs | 4.15 µs |
| Host time per submission | 0.25 µs | 1.14 µs |

Rings in BAR1 save 0.2-0.5 µs of latency, since the GPU fetches from its own
memory, and cost 0.9 µs of host time per submission: uncached stores, and the
BAR1 read that makes them land before the doorbell. The NVIDIA driver keeps
its rings in host memory, and pays the BAR1 read only while memory the host
writes through BAR1 is live: 0.66 µs per submission.

### Device-to-host copies run at 60% of the link

On the RTX 5000 Ada (PCIe 4.0 x16, 31.5 GB/s), copy engine copies of 256 MiB
between pinned host memory and GPU memory run at 25.2 GB/s host to device and
18.9 GB/s device to host, through the RM and through CUDA alike (CUDA: 25.1
and 19.1 GB/s). CUDA stages pageable host memory and copies it to the device
at 18.5 GB/s. Within GPU memory CUDA copies at 241.8 GB/s, 84% of the
288 GB/s that 576 GB/s of DRAM bandwidth leaves for a read and a write. Copy
benches gate on these figures, measured from C over the vendor's interface.

### GSP firmware is an ELF file read by section

`gsp-570.144.bin`, the GSP-RM firmware of driver 570.144, is a 63.5 MB ELF
file of 14 sections whose addresses (`sh_addr`) are all 0. Its payload is the
`.fwimage` section, the only one allocated. Laid out as a loadable image, the
file makes a 63 MB copy that no boot step reads. A loader reads the sections
it needs by name and copies `.fwimage` once, into the memory the GSP boots
from. `fmc-570.144.bin` is a 32-bit ELF file of 6 sections, also all at
address 0. Read from the firmware files.

## CUDA

### After a sticky error, queued work does not run (driver 615)

CUDA's reference says that after a sticky error such as
`CUDA_ERROR_ILLEGAL_ADDRESS` every later call in the context returns it. It
does not say whether work queued behind the fault still runs. On the RTX 5000
Ada with driver 615.71.09 it does not. After a kernel stored to address 0, a
write of the timeline word queued behind it and a copy into a watched pinned
buffer queued behind that never ran; for 200 ms of sampling the word and the
buffer stayed still. `cuStreamQuery` reported the error within the driver's
1 ms poll, a new context on the GPU in the same process was refused with the
same error, and the GPU was back at its idle power state, with no memory in
use, 11 s after the process exited. Every run of the fault test showed this,
one of them under ASan and UBSan (cuda/test/test_fault.ml, 7b4d40034 and
36de95b6c).

A driver that reports a device stopped after a sticky error, promising that
its work writes no more memory, rests on this behaviour of one driver
release.

### Module loads and unloads wait for all of the GPU's work

`cuModuleUnload` returns only once every kernel running on the GPU has
finished, whichever module it belongs to: on the RTX 5000 Ada an unload took
950 ms behind a 1 s kernel of another module. `cuModuleLoadData` waits the
same way. A binding to a runtime with a global lock releases it around both,
or every other thread waits with them (861d4777d). Measured while a kernel
ran, `cuModuleGetFunction` took 31 µs and `cuMemHostGetDevicePointer` 1 µs:
neither waits for the GPU.

### A page-locked range may span unlocked pages

A driver that accepts host memory another owner page-locked must check that
the whole range is locked. Asking CUDA about the range's two ends is not
enough: on the RTX 5000 Ada, with pages 0 and 2 registered by two calls and
page 1 not, both ends answered and the middle page returned
`CUDA_ERROR_INVALID_VALUE`. `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` gives the
start of the allocation that holds a pointer, for registered memory and
`cuMemHostAlloc` memory alike, so the driver takes a range as locked only when
both ends report the same start (206559a62). The GPU reports
`CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM` as 1, so one pointer from
`cuMemHostGetDevicePointer` serves every device.

### A stream memory-operation wait costs 1.6 µs of host time

On the RTX 5000 Ada, `cuStreamWaitValue64` costs 1.62 µs of host time per
call, `cuStreamWaitEvent` 61 ns and `cuEventRecord` 69 ns, for events made
with `CU_EVENT_DISABLE_TIMING` and without `CU_EVENT_BLOCKING_SYNC`. A round
trip that switches streams takes 4.86-4.93 µs ordered by a memory-operation
wait and 3.64-3.71 µs ordered by an event. Four satisfied waits cost
6.46-6.54 µs as four calls and 1.85-1.87 µs as one `cuStreamBatchMemOp`,
whose count the reference requires below 256. Measured from C over libcuda
alone, medians of three runs.

The CUDA driver orders work within a device by events, and waits on other
devices' words in batches of at most 255.

## AMD under KFD

### KFD refuses a queue whose context save area is short of its rule

`AMDKFD_IOC_CREATE_QUEUE` refuses a compute queue whose context save area is
smaller than KFD's own size (`kfd_queue_ctx_save_restore_size` in
`kfd_queue.c`). That size is, for each die (XCC), the topology's `cwsr_size`
plus a debugger area of 32 bytes per wave rounded up to 64 bytes, the total
rounded up to a page. KFD counts 32 waves per compute unit from GFX 10.1, and
before it 40 per compute unit, up to 512 per shader engine, whatever the
compute unit runs: an area sized from the topology's `max_waves_per_simd` was
refused on the R9700 (b03b50421). The queue's control stack size is the
topology's `ctl_stack_size`.

### A host data path flush from a GFX12 compute queue hangs it

On the R9700 under amdgpu, a compute queue that flushed the host data path
(HDP) itself, writing `GPU_HDP_FLUSH_REQ` and waiting on `GPU_HDP_FLUSH_DONE`
with `WAIT_REG_MEM`, hung within a few hundred batches run back to back, and
the kernel driver then reset the GPU for every process on it. At the hang no
wave ran and the MEC waited on a register read
(`CP_CPC_STALLED_STAT1.MEC1_WAIT_ON_RCIU_READ`). The request sets every
client's bit at once, in a handshake the kernel driver's own engines use one
bit each. Of four runs of 1,000 back-to-back calls with the flush in the
queue, three hung; of five without it, none (main 025423194a). A driver
flushes the HDP from the host, as ROCr does.

### Host writes through the BAR land after an HDP flush read back

GPU memory that the host writes through the BAR passes the HDP, which can
hold the writes. amdgpu flushes it by writing the remapped
`HDP_MEM_FLUSH_CNTL` register and reading it back
(`amdgpu_hdp_generic_flush`); the read returns once the write has landed. The
AMD driver does the same before a doorbell, and only while memory the host
writes through the BAR is live: its rings, words and segments live in host
memory, which the HDP does not carry. Read in source; the R9700's host has no
host-visible GPU memory (below).

### Without resizable BAR no GPU memory is host-visible

With resizable BAR off, the R9700's BAR0 is 256 MiB and KFD's topology lists
no memory bank of the public heap type (`heap_type` 1), only private ones.
Memory the host writes in GPU memory through the BAR does not exist on that
machine, and a driver falls back to host memory: reading 1 MiB took 0.041 ms
from both, where reading it across the bus would take milliseconds. A driver
decides from the topology's heaps, and takes a BAR as small when it is smaller
than the GPU's memory.

### An AQL packet is live once its header is

An AQL queue's packet processor may take a packet as soon as its header type
is valid, before any doorbell (the HSA runtime specification's packet header
rule). A writer stores words 1 to 15 of each 64-byte packet, then word 0,
which holds the header, with a release store. A writer that drops packets it
already placed beyond the write position, after a failed fill, stores the
`INVALID` type (1) into each dropped header, since the processor may already
be reading them. The doorbell carries the index of the last packet written,
one less than the write index. Unverified on hardware: the R9700 runs PM4
compute queues, and no GPU here takes AQL packets.

### Write positions count dwords, bytes or packets

A queue's write position, and the value its doorbell takes, count dwords on a
PM4 queue, bytes on an SDMA queue and packets on an AQL queue. The PM4 and
SDMA units run on the R9700; the AQL unit is unverified there.

### The copy engine's fence writes 32 bits

SDMA's `FENCE` packet writes 32 bits, and SDMA's `POLL_REGMEM` and PM4's
`WAIT_REG_MEM` compare 32 bits. A 64-bit timeline word released by two fences,
the low half and then the high half when the low half wraps, reads lower
between the two writes: `0x0_ffffffff` becomes `0x0_00000000` before
`0x1_00000000`, and a waiter comparing 64 bits sees the word move backwards.
The AMD driver releases on the copy queue only values whose high half is
unchanged, and sends the one value in 2^32 whose low half is 0 to the compute
queue, whose `RELEASE_MEM` writes 64 bits. A wait through a 32-bit compare is
exact only if the value it waits for cannot recur within 2^32 values. On the
R9700, a device started near 2^32 released across it in order (8b362523b lets
a test start one there).

### A TRAP after the copy queue's fence makes it visible sooner

On the R9700, a copy-queue submission of a 16-byte copy and a `FENCE` of a
word the host spins on completed in 10.8 µs without a following SDMA `TRAP`
packet and in 3.2 µs with one, in the same run. The cause was not found. The
AMD driver ends every copy-queue release with `TRAP`.

### Dispatch and descriptor fields keep the low bits of what does not fit

PM4 dispatch registers and buffer descriptors take the low bits of a value
too large for their field, without error, and the kernel runs with the
truncated value. A driver bounds each before it writes:

- `COMPUTE_PGM_RSRC2.LDS_SIZE` is 9 bits, in units of 512 bytes on every
  generation the library encodes except GFX950, which counts units of 1280
  bytes (AMDGPUUsage's `LDS_SIZE` row; LLVM's
  `FeatureLDSEncodingGranularity1280`). A larger request dispatches with
  almost no LDS (7c85f2603, fc79eda60).
- `COMPUTE_TMPRING_SIZE.WAVESIZE` gives a wave's scratch in granules: 13 bits
  of 1 KiB on GFX9, 15 bits of 256 bytes on GFX11, 18 bits on GFX12, at most
  131,056, 131,068 and 1,048,572 bytes per lane. Past it the waves' scratch
  wraps and they overwrite each other. LLVM compiles no more than this
  (`getMaxWaveScratchSize` in `GCNSubtarget.h`) (2cf255980).
- A buffer descriptor's record count is 32 bits. A scratch share of 2^32
  bytes becomes 0, which faults the kernels that use it (810277172).

### Two AMD sources disagree on counter events

AMD publishes performance counter event numbers in two places: the SOC
enumerations that aqlprofile ships (`vega10_enum.h`, `soc21_enum.h`,
`soc24_enum.h`) and rocprofiler-compute's `counter_defs.yaml`. They disagree.
For gfx942 and gfx950, 60 of the 91 SQ event names the two share have
different numbers; for GFX12, `GL2C_HIT` is 39 in the enumeration and 41 in
the YAML. The AMD ABI library's counter table comes from `counter_defs.yaml`,
the one AMD's profiler counts with (85d0b6e56). The files were compared; no
gfx942 or gfx950 GPU ran the counters here.

### CDNA drops completion interrupts whose context id is 0

On gfx9, KFD discards an end-of-pipe interrupt with context id 0 whenever it
expects valid ids, which it does for every gfx9.4.x (MI200, MI300): a
workaround for firmware that sent bogus signals with id 0
(`kfd_int_process_v9.c:333-345`). A process sleeping in
`AMDKFD_IOC_WAIT_EVENTS` then wakes only at its timeout. On gfx11 and gfx12 an
id of 0 misses the fast lookup and makes KFD scan the process's signal page
(`kfd_int_process_v11.c:355`, `kfd_events.c:707-735`). A release that raises
an interrupt carries its KFD signal event's id in `INT_CTXID`, as ROCr does.
Read in source; no CDNA GPU here.

### Unmapping registered host memory stalls every queue

When a process unmaps host memory that KFD still tracks before freeing it in
KFD, the kernel invalidates the range, and KFD evicts every queue of the
process and restores them later. On the R9700 the next work on any queue
waited 5 to 10 ms (main b6963ba6c). A driver frees the KFD allocation and its
GPU mapping first, then unmaps. Memory that KFD allocates itself (GTT) is
never evicted this way, so the AMD driver keeps its rings and words there.

### KFD's wait timeouts round up to timer ticks

`AMDKFD_IOC_WAIT_EVENTS` converts its timeout in milliseconds to timer ticks
and rounds up (`user_timeout_to_jiffies` in `kfd_events.c`). At 250 Hz a 1 ms
timeout lasts at least a 4 ms tick when no event fires. A code load that
polled a copy's completion with 1 ms sleeps took 8 to 12 ms on the R9700 for a
copy of microseconds, since the copy raised no event (main b6963ba6c). A
driver spins on the word before it sleeps, and sleeps in KFD only on work that
raises an event. A wake through the interrupt path costs about 36 µs more
than spinning (`wake/driver` against `release/driver` in amd/bench).

### A process has one GPU address space per GPU

KFD keeps one GPU address space per GPU per process, bound to one render node
file by `AMDKFD_IOC_ACQUIRE_VM`; the process cannot hold a second one on the
same GPU through another file. A memory fault in that address space reaches
every user of the GPU in the process. The AMD bench runs its C floors, which
make their own queues, in worker processes for this reason.

### The stable power state has one holder

amdgpu holds a GPU in its stable power state
(`AMDGPU_CTX_OP_SET_STABLE_PSTATE`) for one context at a time and refuses
another with `EBUSY`. The AMD driver takes it for thread traces, which need
steady clocks and shader engines, once per process under a lock, so that two
threads cannot both create a context and ask (c27af1ddc).

## Apple GPUs under Metal

### Stores to unmapped GPU addresses do not fault

On the M1 Max, a compute kernel that stored 64 words at GPU address `0x10`, at
`0x7f0000000000`, which nothing maps, or at the GPU address of a buffer freed
and removed from the residency set before the submission, completed: its
command buffer reported no error, later work ran, and the GPU stayed usable.
Two guarded runs, one process each, on 2026-10-08. Metal tells a driver
nothing of a stray store, and a test cannot make a page fault this way; the
Metal driver's failure path is tested through its C completion ring without a
GPU.

### Completion handlers run in no documented order

Metal calls each command buffer's completed handler on a thread of its own and
documents no order between the handlers of one queue. A handler that writes a
timeline word directly can pass an earlier command buffer that then reports
failure. The Metal driver gives each command buffer a slot before making it;
a handler completes its slot, and the leading completed slots are released in
commit order, each writing its value unless a slot before it failed
(61855353e). There are as many slots as the queue holds command buffers, so
taking a slot is the back-pressure, and Metal's `commandBuffer` never blocks.

### An indirect command buffer's commands are autoreleased

`indirectComputeCommandAtIndex:` returns its command autoreleased. A thread
without an autorelease pool of its own, such as an OCaml domain, never drains
one, so each command lives until the thread ends. Code that records an
indirect command buffer wraps the calls in its own `@autoreleasepool`
(f847dcfb5).

### Buffers start at multiples of 256 bytes

Apple documents no alignment for a buffer's first byte. On the M1 Max, shared
buffers smaller than a page start at multiples of 256 bytes inside a 16 KiB
page (seen at offsets 7936, 8192, 8448 and 12544). The Metal driver promises
256 bytes and checks every buffer Metal returns (4b0f5f464).

### Mac GPU families publish no argument offset alignment

A dispatch's argument buffer offset must be a multiple of the GPU's minimum
constant buffer offset alignment. Apple's feature set tables give 4 bytes for
the Apple GPU families and nothing for the Mac families. The Metal driver uses
256 bytes on Mac families, a multiple of every smaller power of two, and the
record compiled code reads carries the value (0e4bd2ca5).

### A dispatch costs about 200 µs round trip, mostly outside the GPU

On the M1 Max, one dispatch committed and waited with `waitUntilCompleted`
took 206 µs at the median of 300 interleaved rounds: 98 µs from commit to GPU
start, 9.6 µs on the GPU, 97 µs from GPU end until the host saw completion. A
residency set, a fence, or a completion handler with the host spinning on a
word each changed it by under 5 µs. A submission costs about 5 µs of CPU, of
which creating the encoder takes 2.0 and the commit 1.8. Measured under load:
an empty command buffer completes in 15-46 µs; pipelined, a command buffer
costs about 25 µs; after 20 ms idle, commit to GPU start grows to about
390 µs.

A kernel can write a shared word that the host spins on, which the host sees
98-130 µs after the commit, earlier than the completion. The Metal driver does
not take it as completion: Metal reports a failed command buffer only through
its status.

### An indirect command buffer adds about 16 µs of GPU time

On the M1 Max, running dispatches from an indirect command buffer takes about
16 µs more GPU time per command buffer than encoding them directly, 10-20 µs
more latency on each side of the GPU, and 1.9 µs per dispatch that waits on
the one before. None of these moved it: the default inherit flags,
`optimizeIndirectCommandBuffer`, the buffer in the residency set or passed to
`useResource`, a concurrent encoder. Direct serial encoding costs 0.1 µs of
CPU per dispatch and wins in probes of 1 to 512 dependent dispatches.

On real steps of gpt-oss-20b the result depends on the size. Decode steps of
656 dispatches ran 3% faster encoded directly (31.3 against 32.2 ms); prefill
steps of 3,452 dispatches ran 12-15% slower (1.21 s against 1.04-1.09 s).
Encoding costs 0.5-0.7 µs per dispatch either way. The crossover lies between
those sizes and was not located.

### Memory is wired again after 1 to 3 seconds idle

After 1.2-3 s without work, the first submission on the M1 Max waits before
the GPU starts, in proportion to the residency set's memory: 20-76 ms for
1 GiB, 2.5-8.6 ms for 64 MiB, 1.4-3.2 ms for 16 KiB, against 80-220 µs warm.
The size-dependent part is macOS wiring the memory again; the 1-3 ms floor is
the GPU waking. `requestResidency` called just before the commit changes
nothing. Only work or a residency request well before the submission hides
it.

### Completion wakes are bimodal per process

On the M1 Max, a thread blocked on a condition that a completion handler
signals wakes about 10 µs after the handler in most processes; in about one
process in four, every wake comes 1.0-1.6 ms after it (p90 4 ms). GPU end to
handler stays 92 µs in both modes, and `waitUntilCompleted` shows both.
Raising the waiting thread to the user-interactive QoS class, by an override
or by `pthread_set_qos_class_self_np`, moved neither mode; a command-line
process's main thread already runs at that class. The cause was not found.

## Host code

### arm64 macOS writes JIT code through a listed callback

On arm64 macOS, code memory is `MAP_JIT` memory, mapped read-write-execute,
whose write protection each thread lifts for itself. Under the hardened
runtime's `com.apple.security.cs.jit-write-allowlist` entitlement,
`pthread_jit_write_protect_np` ends the process at its first call. Code is
written instead by a callback passed to `pthread_jit_write_with_callback_np`
and listed with `PTHREAD_JIT_WRITE_ALLOW_CALLBACKS_NP`; without the
entitlement the system lifts the protection around the callback as the pair
of calls did. The system reads only the lists of images present at start-up,
so a bytecode program that loads its C stubs at run time still ends at its
first write, unless it holds `jit-write-allowlist-freeze-late` and calls
`pthread_jit_write_freeze_callbacks_np` after loading them. Checked by hand on
the M1 Max with an executable signed with the hardened runtime, `allow-jit`
and `jit-write-allowlist`: killed with SIGKILL at the first link before the
change, passing after it (afa17b70d).

### arm64 cores need an isb to run code another core wrote

Cache maintenance after writing code (`__builtin___clear_cache`) reaches every
core, but a core that fetched instructions earlier, such as those of a
collected program once mapped at the same address, may run them until it takes
a context synchronization, such as `isb` (Arm Architecture Reference Manual,
instruction cache coherency). A pool worker running job after job takes none.
The host loader counts installs with a release increment after the
maintenance, and a thread executes `isb` before entering linked code when the
count moved since its last one, a single load when nothing changed. x86_64
keeps instruction fetch coherent with stores.

### Windows commits a thread's stack one page at a time

Windows grows a thread's stack through a guard page, so a function whose frame
exceeds 4 KiB must touch each page in order, which Windows compilers do with a
stack probe (`__chkstk`). Code compiled for an ELF target has no probes, so
host code linked on Windows keeps its frames under 4 KiB. Not run: no Windows
machine here.

## Memory ordering

### On x86, order streaming loads with mfence

A C11 sequentially consistent fence compiles to `mfence` with clang and to
`lock orq $0, (%rsp)` with gcc 14. A locked instruction orders ordinary loads
and stores. Intel's manual names `MFENCE` or `LFENCE` for the non-temporal
loads (`MOVNTDQA`) that fast reads of write-combining memory use, and Linux's
`mb()`, its barrier for device memory, is `mfence`. The barrier for device
memory is therefore `mfence` written out (0cfc4f48c). It costs about 4 ns more:
8.2 against 4.0 ns per fence on the RTX 5000 Ada's host, 7.4 against 3.9 ns on
the R9700's, beside bus round trips of hundreds of nanoseconds. On arm64 the
barrier is `dsb sy`.

### Before a doorbell: fence, publish, read back, ring

The GPU must see every store it will read before the doorbell. Uncached
stores through a BAR are posted, and PCIe keeps them in order. Write-combined
stores may reach the device in any order and after later stores, and AMD's HDP
can hold writes after they arrive. A read through the BAR returns only after
earlier posted writes to it have completed. The NVIDIA and AMD drivers submit
with these steps, after UVM's `uvm_hal_turing_host_write_gpu_put` and
amdgpu:

1. write the ring entries and command segments, then a fence: a store fence
   suffices (`sfence` on x86_64, `dsb st` on arm64: Linux's `wmb()`), and the
   AMD driver uses a full one;
2. publish the write position (NVIDIA: `GP_PUT` in USERD);
3. while memory the host writes through a BAR is live, a full fence and a
   read through that BAR (NVIDIA: BAR1, after step 2), or the HDP flush and
   its read back (AMD, before step 2);
4. ring the doorbell (NVIDIA: the channel's work submit token, written to
   `NVC361_NOTIFY_CHANNEL_PENDING`).

On the RTX 5000 Ada step 3 adds 0.66 µs per submission.

### arm64 Device memory faults on the accesses memcpy makes

On arm64, an unaligned access to Device memory raises an alignment fault, as
does `DC ZVA` (Arm Architecture Reference Manual). The C library's `memcpy`
and `memset` use unaligned and overlapping accesses, and `DC ZVA` to zero, so
they fault on a BAR mapped as Device memory. Accesses to register windows are
loops of 32- or 64-bit loads and stores aligned by address, since a window may
start off a word (aa4249658). Not run: no arm64 host with a discrete GPU here.

## Address spaces and sanitizers

### NVIDIA command segments lie below 2^40

A GPFIFO entry holds 40 bits of segment address: bits 31:2 in `GET` and bits
7:0 of the high word in `GET_HI` (`clc56f.h`). The NVIDIA kernel path maps
memory that the host also maps at the same address on both sides, and keeps
every GPU address it hands out below 2^40, reserving the range in the process
so that nothing else maps there. This caps a GPU's mapped memory at
1 TiB.

### x86_64 AddressSanitizer owns 2.25 GiB to 2 TiB

On x86_64 Linux, ASan's shadow gap covers `0x00008fff7000` to
`0x02008fff6fff`, about 2.25 GiB to 2 TiB (compiler-rt's `asan_mapping.h`). By
default ASan maps the gap inaccessible, so a process under ASan can reserve
nothing there. With `protect_shadow_gap=0` ASan maps the gap's own shadow
instead, from 2.3 to 258 GiB, and leaves the rest free. NVIDIA segment
addresses must lie below 2^40, inside the gap: with its GPU addresses from
64 GiB no GPU opened under ASan. The kernel path reserves them from 384 GiB,
and its suites set `protect_shadow_gap=0` in their own environment
(4a0726ad7).

## PCI functions on Linux

These matter to a driver that takes a GPU from its kernel driver. Our hosts
give no root, so none of them ran here.

### Bus mastering outlives the process

A function unbound from its kernel driver and driven through sysfs keeps bus
mastering on after the process exits. Its GPU can then still write system
memory the process gave back, which by then belongs to another process or to
the kernel. The PCI library clears the command register's bus master bit at
release and in an exit handler (5efba0e6f). Behind an IOMMU, closing VFIO's
device file clears it (`vfio_pci_core_disable`). Read in source.

### flock locks a function's sysfs config file

`flock` on a descriptor of `/sys/bus/pci/devices/<bus>/config` locks: kernfs
gives its files no flock operation, so the kernel's generic one locks the
inode, and kernfs keeps one inode per sysfs node. Every process that opens the
file sees the lock. Any user may read `config`, so any user can hold it
(486a71057). VFIO's group file admits one process at a time and needs no
lock. Read in source.

### A BAR resize refused for room may fit smaller

Writing a size to `resourceN_resize` fails with `ENOSPC` when the bridge
window above has no room for it, and a smaller size may then fit. Any other
refusal (`EBUSY`, a bridge that cannot move, no privilege) holds for every
size, and the BAR keeps its own (c19f11152). Read in source.

## Round trips on these machines

Medians of loops in C (Objective-C on the M1 Max) over each vendor's
interface: the floors the drivers' benches compare against (`release/floor`,
`launch/floor-1`, `launch/floor-64` and the copy floors in each driver's
bench).

| | RTX 5000 Ada, RM | RTX 5000 Ada, CUDA | R9700, KFD | M1 Max, Metal |
|---|---|---|---|---|
| Empty submission, round trip | 3.39 µs | 3.54-3.60 µs | 14.7 µs | 15-46 µs, under load |
| One empty kernel, round trip | 4.64 µs | 4.71-4.84 µs | 15.6-15.7 µs | 206 µs |
| Each further kernel of one submission | 0.6 µs | 1.2 µs | 0.47 µs | |
| Host to device, 256 MiB | 25.2 GB/s | 25.1 GB/s | 13.9 GB/s | |
| Device to host, 256 MiB | 18.9 GB/s | 19.1 GB/s | 14.0 GB/s | |
| Within GPU memory, 256 MiB | | 241.8 GB/s | 206 GB/s | |

On the R9700 the copy engine's 206 GB/s within GPU memory is 32% of its
640 GB/s, and one copy queue runs the two directions one after the other:
2 × 256 MiB, one each way, take 38.48 ms, the sum of each alone. A release at
agent scope costs the same 15 µs as one at system scope, so the round trip is
not the L2 write-back. The M1 Max has one memory, so its copies are the
host's.

## Measuring on these machines

- The RTX 5000 Ada boosts between 2550 and 2805 MHz from run to run and within
  one, and an unprivileged process cannot pin the clock. A GPU-bound row moves
  with it: 64 chained launches took 47.6 µs at 2805 MHz and 51.8 µs at 2550.
  Rows record the P-state and clocks.
- On the R9700 some rows are bimodal between processes and steady within one:
  a round trip that switches queues takes 14.8 or 18.8 µs, a 16-byte copy 3.9
  to 6.1 µs. One session ran every latency row, floors included, 1.6 times
  slower. The cause was not found.
- The R9700's own link reads 32 GT/s to its PCIe switch, but the root port
  above runs PCIe 3.0 x16, and host copies stop at its 15.75 GB/s. A link
  budget reads every hop up to the root port.
- On the M1 Max, launch timings drift 10-30% between processes. Variants
  compared for small differences run interleaved in one process.
