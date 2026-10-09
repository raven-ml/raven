# Testing the device libraries

This document is for a contributor who adds a driver or changes rig. It
says what the device libraries promise, which test pins each kind of promise,
and why the suites are built the way they are. Paths are relative to
`dev/rig/`.

## What a test states

Each library's `.mli` is its specification. A test states one promise of that
`.mli`, through the public interface, and its name is the promise: "a fault two
domains' sleeps find loses the device once", "an allocation past the GPU's
memory is None". A reader should be able to check the suite against the `.mli`
by reading test names alone.

The promises fall into a few kinds, each tested its own way:

| Promise | Example | Test |
|---|---|---|
| A value is computed from a spec | a GPFIFO entry's fields, a QMD's bits | `cases` from the vendor header, `prop` over the field ranges |
| A reader agrees with a writer | an ELF object reads back as written | `prop` round trip |
| A refusal is total | a corrupted object reads or is refused | `prop` over corruptions, `cases` at each bound |
| State across calls follows rules | stamps, rings, windows, page tables | `stateful` against a model |
| Calls from several domains are safe | free, unmap and unload end a value once | `stateful ~domains:2` |
| A known interleaving is handled | a worker parking as a job is published | a held schedule (hook points, gates) |
| A device that fails is lost cleanly | loss, faults, Unknown stops | rig's test driver, guarded runs on GPUs |
| A path costs what it says | a submit that does not wait allocates nothing | words counted with `Gc.minor_words` |

Layout mirrors `lib/`. A library's suites live in `test/<lib>/` as
`test_<module>.ml` (rig's in `test/`), helpers shared by two suites (or by
a suite and a bench) in `test/<lib>/support/`, and fixtures in
`test/<lib>/fixtures/` with the command that made them. Its benches live in
`bench/<lib>/` and read their fixtures from `test/<lib>/fixtures/`; the GPU
bench reads each vendor's. A suite reads only its own directory. What every
GPU suite shares lives once in `test/support/`: `rig_gpu_support`, host
memory by address and a GPU of one driver opened through rig (below). The
laws every GPU driver keeps run once, in `test/conformance/`, on each driver
the machine has ("A new driver's tests", below). The
machine's GPU lock is the library `rig.gpu_lock`, which links no rig, so
that suites outside rig take it too. A tool that
makes fixtures for several libraries lives in `test/gen/`, such as
`nvrtc.c`, which compiles the NV suites' cubins. Every top-level group sets
`~timeout`, and `dev/rig/dune` sets `WINDTRAP_TIMEOUT` to 60 s for any test
without one, because a test that never ends blocks every build on the shared
watch server.

A test names only what an `.mli` documents or an installed C header
declares. A hidden section (`(**/**)`) holds only what modules of the same
library need from each other; no test, bench or other library names it.

## Laws, cases and models

The suites use windtrap; its `.mli` (under
`_build/_private/default/.pkg/windtrap.*/source/lib/windtrap.mli`) is the
contract for every verb below.

**Laws.** A `prop` states something true of every input. Its strength comes
from the generator and its `cover`s. Inputs are drawn at the edges: zero,
one, the largest legal value and its neighbours, `min_int` and `max_int`,
addresses at the top of their range. `test/nv/abi/test_gpfifo.ml` draws
segments whose word count is 0, 1 or `Gpfifo.max_words` one time in four, and
"an entry names its segment's address and words, and waits for nothing" covers
"the most words" and "the last address". A cover that a random draw reaches
only by luck fails some seeds: uniform sizes reach the largest SDMA copy in
about one draw in fifty, which fails a law requiring it about one seed in
thirty. So a size at a bound is drawn by construction, at a fixed weight
(`test/amd/abi/test_sdma.ml`). The space model in `test/pci/test_space.ml`
draws, for the largest free range of `g` addresses, each size `g / 2 - a` at
its alignment `a`, the fit bound exactly. A `cover` names an outcome the
generator decides. An outcome the scheduler decides may not occur on a given
machine, so it is no cover.

**Cases.** A `cases` test lists values a spec states. Expected values come from
the spec: the NV ABI suites read descriptors back field by field at the
`MW(hi:lo)` positions of NVIDIA's QMD headers (`test/nv/abi/test_qmd.ml`); the
GPFIFO refusals are `min_int`, `-1`, `max_words + 1` and `max_int`. A value
learned by running the code is a baseline and is written as `expect`
(the Chrome trace in `test/test_profile.ml`, the cubin listings in
`test/nv/abi/test_cubin.ml`).

**Models.** A `stateful` test runs random programs of calls against a model
that says what each call returns. Use one whenever a promise spans calls.
`test/metal/test_metal.ml` models the ring of a device's command buffers:
slots taken in commit order, completed in any order, released in commit order,
the word stopped at the first failed slot. Its covers ("a slot completes before
an earlier one", "a value completes after a failed slot", "the last slot
completes after stop") show the programs reached the cases the rules are about.
`test/test_stamps.ml` models what a submit and a `Buffer.wait` wait for
across two devices that share memory. `test/pci/test_window.ml` holds a mapped
window, a combining window and a window through a transport against a model of
plain bytes. windtrap fails a stateful test whose commands were never called
over a passing run, so a `~pre` that is never true shows up.

**Cost.** Where an `.mli` promises an allocation bound, a test counts words:
"a submit that does not wait allocates nothing" (`test/test_submit.ml`),
"a host buffer of 64 bytes costs at most 61 words" (`test/test_buffer.ml`).
`test/amd/abi/test_init.ml` measures the AMD ABI library's own initialisation
between probes linked before and after it. Time is asserted as CPU time or as
a ratio between two sizes. "70,000 symbols at extended indexes, in linear
time" (`test/elf/test_elf.ml`) reads an object of `n` and of `n / 4` symbols
and requires a ratio below 8, about 4 if reading is linear and 16 if it is
quadratic. A ratio holds on a slow or sanitized build where seconds would not.

## Concurrency

Every module with state says in its preamble which domains may call it. The
suites test that sentence.

**Two domains against a model.** `stateful ~domains:2` runs a prefix of calls,
then two branches at once on two domains, then a suffix, and passes when some
order of the calls explains every result. Ending a value once is rig's rule:
rig frees each region once and unloads each image once, so a driver checks
neither (`Rig_edge.Driver`), and the two-domain model of rig
(`test/test_model.ml`, "calls on two domains answer as some order of them")
drops buffers and submissions and collects on both domains, so either may
free their memory. A flag that is a plain
mutable bool, read and then written, lets two domains both pass the check and
end a value twice, which a driver answers by crashing the process; an `Atomic`
taken with `compare_and_set` lets exactly one call end it. "stamps raised from
two domains are their maxima" runs the stamp model on two domains, and "two
domains submit one submission at once" (`test/test_submit.ml`) submits with a
run per domain. In pci, "two domains allocate at once" and "pins
and DMA memory are counted the same from two domains".

A process whose test spawned domains cannot fork afterwards, so a test that
forks lives in a suite of its own (`test/test_fork.ml`: "a forked child's
devices are lost for good").

**Held interleavings.** Random schedules rarely land in the window a protocol
bug needs. When the window is known, the test holds it. The pool's protocol
(`lib/pool/rig_pool.c`) has named hook points, `RIG_POOL_HOOK(published)`,
`RIG_POOL_HOOK(parking)` and others, which compile to nothing. The suite compiles
`rig_pool.c` a second time with the hooks defined
(`test/pool/rig_pool_hooked_probe_stubs.c`), and each scenario makes the caller
and chosen workers wait for each other's events, so the timeline in its comment
is the one that runs. `test/pool/test_interleavings.ml` holds eleven of them,
among them "a worker that decided to park before a publication, and set its bit
after, runs the job" and "a child forked while a worker holds the parking
mutex runs a job on every core". Every wait gives up after 10 s and the
scenario reports which event it was waiting for.

Rig's test driver (below) gives the same control over a wait. Its sleeps
check, in order, a gate, a fault, an interruption and a stall:
`Polled.gate` blocks every sleep until `open_gate`, `sleepers` says how many
are blocked, `interrupt` raises SIGINT in the next sleep's thread, and
`stall n` makes the next `n` sleeps return without running the queue, as over
work that runs long. "a fault while a submit waits for room commits nothing"
(`test/test_turn.ml`) opens a device of capacity 1, gates it, starts a
submit that must wait for room, waits until it sleeps at the gate, faults the
device, opens the gate, and checks that the submit raised `Lost` and that no
value was assigned to it.

**The turn.** `Rig.submit` takes the device's turn, the right to be its
one submission between the room check and the hand-over, and releases it while
it waits for room. A driver's `room` and `submit` are called one at a time per
device under the turn; every other call may come from any domain. The tests in
`test/test_turn.ml` pin each part: "a submit waiting for room lets a
submission that fits through", "a submit blocked in its driver holds only its
device's turn", "two threads of one domain submit to a device that blocks", and
"Sys.Break while waiting for room assigns no value". The last two use
systhreads of one domain, so a turn or wait that kept the domain lock would
stop the test. A driver relies on the turn and does not check it; its suite
need not test concurrent submits.

**Signals.** A test synchronizes on a signal from its own helpers in
`support/`. It does not wait through a function of the library under test, or
sleep for a fixed time. `Rig_support.await` polls a condition with a
10 s watchdog. Where a correct library gives no signal, such as a call held
back on a mutex, the test waits for the other side's signal, samples a short
window, and its name says "(sampled)": "fork waits for a running job of more
than one thread to end, sampled for 50 ms" (`test/pool/test_threads.ml`). A
fixed delay fails a correct run under load, or ends before the contending
domain arrives and leaves the race unexercised.

## Loss and faults

A device whose driver reports a fault, or whose hand-over fails, is lost once
and for good (`lib/rig.mli`, "Loss"); so is a closed device, and every device
of a failed process. Every later use of it, of its memory, and of other memory
that waits for a point it did not reach raises `Lost`; other devices go on;
its facts still answer; its memory returns only once its word shows its last
value reached.

**Rig's test driver.** Rig's job is to keep that contract over any
driver, so its suites run on `Rig_support.Polled`
(`test/support/`), a driver over host memory whose queue runs only when
the test runs it or a wait sleeps. A wait that returns before it slept leaves
work unrun, which is how the stamp tests see that a submit waited for the
right point. Every driver call is logged, so a test can say which calls reached
the driver: "a stopped device is only freed and unmapped". Its options select
the driver behaviours rig must handle:

| Option | Driver it stands for | Test |
|---|---|---|
| `~capacity`, `~may_block` | a queue that answers `Later`, or blocks in `submit` | the room and turn tests |
| ``~answer:`Unknown`` | a stop with work still in flight | "an Unknown answer keeps memory until the word drains" |
| `~transport:true` | a word the host does not address, read through `signaled` | "a lost device behind a transport answers its last reading" |
| ``~completion:`Object``, `~waits_on` | completion through a driver object, and queues that wait on another device's | "a queue that waits on objects waits on the producer's object" |
| `~host_visible`, `~peers`, `~memory`, `~window`, `~budget` | memory the host or peers cannot address, and limits | the budget, borrow and copy-route tests |

`Polled.fault`, `fail`, `interrupt` and `stall` inject the failures. The loss
laws in `test/test_loss.ml` pin each sentence of the contract: "a failed
hand-over loses the device once", "a fault two domains' sleeps find loses the
device once", "a queue waiting on a lost device's value is lost", "a lost
device answers its facts and values", "memory whose points a lost device
reached is ordinary", "a close waits for the work, then ends the device". A
failed process stays failed, so `test/test_fail.ml` runs each of its cases in
a forked child. Polled tests rig; it is never used to test a driver.

**Failure walks.** A failure path is reached by few tests, so a walk reaches
every one. `test/test_walk.ml` runs each operation of rig over Polled
once for each fallible call it makes, with that call failing.
`Polled.fail_at d n` makes the `n`-th of the driver's facts, counted calls and
hand-overs fault, and every one after it, as a faulted device's do; or refuses
the `n`-th, or every one from it, where a call can answer `None` or `Error`, as
a device out of memory does. An operation that gives rig a function (a hold's
release, a profile's reader, an opener) runs once more with that function
raising. After each failure the walk checks that:

- the outcome is one the `.mli` states;
- the driver holds no region of the operation's (`Polled.outstanding`), and a
  lost device was stopped once;
- the C heap and the open descriptors are back where they were;
- the operation runs again, on a new device of the same name if the failure
  lost the first.

Each failure runs four times, on a device that ran the operation once, which
made what the device keeps for good. The heap is the allocator's own count
(`Rig_support.heap_bytes`: the sanitizer's, glibc's `mallinfo2` or the macOS
malloc zones), the descriptors are `/dev/fd`'s entries, and both are read
after the collector ran and the devices drained. A leak shows in every run,
while an allocator's own caches (glibc's per-thread ones) move single runs
either way, so the walk checks the least growth of the four against 128 bytes:
the one thing a failure may keep, a lost device's reason, and the allocator's
rounding.

A driver walks its real resources: the walk counts what an operation
takes and makes the `k`-th acquisition fail, for every `k`, under a limit
that a forked child alone carries, so nothing shared is exhausted. NVIDIA's
kernel path (`test/nv/nvidia/test_walk.ml`) runs each operation with `k`
more files than it holds, so the open of each file it takes fails once, and
a process's first open with a few bytes of address space, so the reservation
of the GPU's addresses fails. After each failure the walk checks that the
outcome is one the `.mli` states, that the child holds no more files or
mappings than before (`/proc/self/fd`, `/proc/self/maps`), that the
operation succeeds once the limit is lifted, and that the GPU opens again.
The same limits reach pinned memory (`RLIMIT_MEMLOCK` against a page
range's pin), and on AMD the KFD and render-node files and the doorbell
mappings, which no walk takes yet. The disk (`test/disk/test_walk.ml`) walks under limits of a
file's size and of open files, and host programs (`test/host/test_walk.ml`)
under a limit of the address space, on Linux, where it bounds mappings.

**Drivers on their hardware.** A driver's fault path is tested on its GPU. A
test that faults or hangs a GPU joins a suite only after one guarded run: one
process, nothing else using the GPU, killed after 60 s, and the GPU checked
idle with no memory in use afterwards. kimchi gives no root, so a wedged GPU
cannot be reset there; nonnormal's root serves only the driver-less AMD path
(below).

`test/cuda/test_fault.ml` is such a test. Value 1 stores to address 0; value 2,
queued behind it, would copy into a watched host buffer. The test checks that
`sleep` raises `Fault` with `CUDA_ERROR_ILLEGAL_ADDRESS`, that `alloc` and
`image` raise it afterwards and `submit` answers `Failed` with it, that the
word stays and the watched buffer stays zero, that `stop` answers `Stopped`,
and that reopening the GPU is refused with the same error. A CUDA fault fails
its context for the rest of the process, so the test is a suite of its own and
takes the GPU lock like the main suite.

On Metal no kernel makes the GPU fault on demand. On the M1 Max (macOS 26.3.1)
a kernel's stores to GPU address `0x10`, to `0x7f0000000000` and to a freed
buffer's address all complete without a command-buffer error. The handler's
failure path is therefore tested through the ring's C seam
(`test/metal/support/`, `rig_metal_ring.h`) by the ring model above, which
completes slots as failed in any order. On the GPU, "a failed fill stops the
word before its value" covers a submission that fails as it is made.

pci injects failure only at a window's far transport, which stands for no
device: test_window's `break` and `break_at` fail an access at a chosen count.
A driver's step over a failed machine ("a transport failing at access k ends a
step in Error, never in bytes") left with pci's fake machines; a link's
failure is tested over a real connection to a real agent once one exists.

Long work is no fault: a device is lost only on its driver's report, and no
timeout decides it. The CUDA suite holds it as "long work is no fault, and a
stale seen returns at once", the AMD suite as "long work is no fault", and the
rig as "a word still for three intervals loses nothing".

## Rings

A ring writer fails at its boundaries. A law over a ring draws the writes that
end exactly at the ring's end, one unit short of it and one unit past it,
because a fit check such as `at + bytes > size` changes its answer at exactly
those points, and random sizes almost never land there. The failure looks like
this: a write that ends exactly at the end leaves the write position at 0, the
next write fits at offset 0, and the segment still open at the end is never
closed into a ring entry, so the engine reads past the ring. In the NV driver's
1 MiB segment ring, a stream of 16-byte copies reaches that case at value
32,768; the first crossing, at value 16,384, wraps through the closing path. A
stream test of other sizes can pass every crossing it makes.

What each ring suite holds:

- Room. A driver's `room` answers `Never` for parts that do not fit its empty
  rings or exceed the part bound, and `Rig.submit` raises `Invalid_argument`
  for them: "a submission the rings have no room for raises"
  (`test/nv/test_nv.ml`) submits a ring entry cut in half, more words than a
  segment holds and more parts than its rings hold, and "a fill that declares
  ring room raises" (`test/cuda/test_cuda.ml`) ring units and segment bytes
  CUDA's streams do not have; each checks the refusal is the room's and no
  value was assigned. Kinds of work a queue does not run are refused earlier,
  by `Submission.make`, on every driver (`test/conformance/`). `Later` is the
  core's to wait out; Polled's `~capacity` holds it in rig's own suites. Parts
  whose declared sizes are near `max_int` must answer `Never`: summing them as
  64-bit integers can wrap and answer `Fits` for work no ring holds.
- Streams longer than the rings, with every byte of every copy checked at the
  end.
- Counters past their width. A counter that wraps only after billions of
  values is reached through a hidden seam that renumbers an idle device
  (`Rig_amd.renumber`), so the test makes a handful of submissions.
- Order. The Metal ring model above: releases in commit order whatever order
  slots complete in.

## Real hardware

Drivers are tested on their GPUs. There are no mock drivers: a mock confirms
the words a driver writes, while the bugs are in what the hardware does with
them (ordering, visibility, completion). A test that acts on a GPU takes only
the class of device its library drives, and skips only when the machine has
none (or no vendor library to reach it), so a suite that passes on a machine
with the GPU has run its GPU tests. No test sweeps every device of the host.

Suites and benches take turns on a machine's GPUs through one lock: `flock`
on `/tmp/raven-rig-gpu.lock`, shared by every checkout and user of the
machine, taken by `Rig_gpu_lock.hold`. The caller decides whether the
machine has its GPU, from files alone (`/dev/nvidiactl`, the Metal
framework, `/sys`), and takes the lock only then; deciding starts no vendor
library, which a bench must not start before it forks. A suite takes the
lock before `Windtrap.run`, so the wait counts against no test's timeout; a
bench takes it before `Thumper.run`, so the workers it forks measure under
the lock, and no GPU row runs beside a GPU test. The process holds the lock
until it exits; the holder writes its executable and process id into the
file. While another process holds it, `hold` prints the file's note once,
which names an earlier holder when the holder took the lock with the
shell's `flock`. A process still waiting after 300 s fails. Under `dune
runtest` the GPU suites of a machine therefore run one after another.

A suite reaches its GPU through `Rig_gpu_support.Make`, applied once in its
support to the driver and a line that says whether the machine has the GPU
(`present`). `open_` opens the GPU, hands the driver's device to rig under
one name of the GPU's and gives back both (`{ d; g }`); `close` is
`Rig.close`; `with_` brackets the two; `submit` and `wait` go through rig.
Before it opens, `open_` ends what the last open made, as a failed test
leaves it, so no test stops a driver rig owns and one name serves every
test. A test of the driver alone, which rig never takes, opens it with
`driver` and stops it with `stop_driver`; `release` ends either kind, for a
test that opens the GPU's driver itself on a GPU that has one device at a
time. A test of the driver's stop states it through `Rig.close` where the
statement survives, and through a loss where work must still run when the
device stops, since a close waits for the work first.

One lock order holds everywhere: the GPU lock first, then the hosts' timing
locks, so nothing waits for the GPU while it holds a timing lock. A timing
run of a GPU bench on kimchi or nonnormal takes the GPU lock in the shell,
then the timing locks, and tells the bench it holds it with
`RIG_GPU_LOCK_HELD=1`, which makes `Rig_gpu_lock.hold` return at once:

```
flock -w 1800 /tmp/raven-rig-gpu.lock env RIG_GPU_LOCK_HELD=1 \
  flock -w 1800 ~/benchwork/TIMING.lock flock -w 1800 ~/benchwork/TIMING-GPU.lock \
  taskset -c 0-5 ./bench_gpu.exe ...          # kimchi
flock -w 1800 /tmp/raven-rig-gpu.lock env RIG_GPU_LOCK_HELD=1 \
  flock -w 1800 ~/wt/TIMING-AMD.lock ./bench_gpu.exe ...   # nonnormal
```

The variable says the process that started the bench holds the lock for it;
set otherwise, nothing guards the GPU.

| Host | Hardware | What only it runs |
|---|---|---|
| Mac | Apple M1 Max, macOS | Metal on the GPU |
| kimchi | Linux x86_64, RTX 5000 Ada, NVIDIA driver 615 | CUDA, NV and its `nvidia` kernel path |
| nonnormal | Linux x86_64, Radeon AI PRO R9700 (gfx1201) | AMD and its `amdgpu` kernel path |

Every other suite (pool, elf, host, rig, pci, the ABI libraries) runs on all
three. The Linux hosts also run the pool's cgroup and thread tests, pci's live
reads of `/sys/bus/pci`, and the gcc build. A library is done when its suite
passes on the Mac, under the sanitize profile, and on Linux. Tests that need
two GPUs ("map each other's memory") skip on every host, since each has one.

Paths a library reads from the host (`/sys`, `/proc`, `/dev`, a firmware
directory) are data. Where the library's public entry takes the host's files
as data, a machine at a root or a firmware directory, the suite runs that
entry on fixture trees built under the test's own `_build` directory on
every machine. Otherwise its parsers are tested live, on the host's own
files, on the host that has the device; no public reader exists only so that
a test can reach a parser, and hardware no host has gets no fixture test.
pci's machine at a root (`Machine.at`) is public: pci's suites and the PCI
paths' numbering (`buses`) run on fixture trees through it. One suite still
reaches readers through a hidden section: amdgpu's descriptions of a GPU
(`gpu_at`, `machine_at`, `save_area_at`).

Tests never exhaust a shared resource: threads, processes, file descriptors,
memory, GPU memory or disk. A failure path that needs a limit is reached by a
limit on a forked child alone. "workers that cannot be made are missing from
every job; with none, the calling thread runs every chunk"
(`test/pool/test_threads.ml`) sets `RLIMIT_NPROC` to 0 in a child on Linux,
where it binds that process only. Without such a limit or a seam that injects
the failure, the case is not tested.

## No simulations, and root

No test simulates a kernel, a device, its firmware or its registers. A
component that answers a library's calls in place of the hardware proves
only that the library agrees with the component, and the bugs are in what
the hardware does (ordering, visibility, completion, cost). What a test may
give a library is:

- data it parses: sysfs and `/proc` trees, discovery tables, firmware and
  VBIOS headers, an ioctl's answer written as bytes, recorded from a machine
  or built from the vendor's layout;
- values it encodes, compared with the vendor's specification, such as an
  entry's bits or a message's checksum;
- the core's reference implementation of its own extension point: Polled
  implements `Rig.Driver` and stands for no vendor.

A state the public interface reaches too slowly, such as a counter past
2^32, is reached through a seam the library documents, in a section of its
`.mli` or in an installed header (`rig_metal_ring.h`), never through a
hidden one.

Two simulations remain, and their tests move to hardware: the host path of
`test/amd/test_amd.ml`, whose rings nothing runs; and the fake RM path of
`test/nv/test_nv.ml`, which still holds the local-memory handover. A test that hands a fake device's values to rig opens it
through `Rig.open_` and submits through `Rig.submit`, as a program does.

pci's suites use no fake machine. They take functions of fixture trees
through `Machine.at`, physically on Linux, with memory in the tree's hugetlbfs
and frames in its pagemap, and write page tables in the table-backed format
of `test/pci/support` (`Tables`), whose flush answers what the test sets.
What only a transport, an IOMMU host or a second GPU reaches has no pci test:
a failed machine and its accesses, failure at access k, a wait that fails
midway, the IOMMU paths of placement, pins and peers, and system memory
exhausted. The root suite on nonnormal (`test/amd/pci/test_root.ml`) states
the hold's laws on the R9700: a device two domains stop, frees after a reopen
that leave the new device's bytes, and the reset after a holder killed by
SIGKILL. It is to restate the rest where the R9700 reaches them; the others
wait for a host with an IOMMU, two GPUs or a transport.

kimchi gives no root. nonnormal gives root for one purpose: the driver-less
AMD path on its R9700, which takes the GPU from amdgpu, boots it with no
kernel driver and gives it back. Every such run takes the machine's GPU
lock first, unbinds the GPU, runs, and gives it back to amdgpu by its
reload protocol before it releases the lock. Deliberate faulting or hanging
runs on a shared GPU host need the maintainer's go, beyond the fault suites
already run on kimchi (CUDA's and NV's). The live tests elsewhere take only
what the process may take, such as a function behind an IOMMU, and skip
otherwise. Tests of file permissions skip when run as root, which opens a
file whatever its mode.

## Sanitizers

```
dune build --profile sanitize @dev/rig/test/<lib>/runtest
```

The `sanitize` profile in `dev/rig/dune` compiles the C with
AddressSanitizer and UndefinedBehaviorSanitizer, links their runtimes into every
executable, and sets CI's options, so a local run fails where CI does:

- `halt_on_error=1`: a report ends the process.
- `detect_leaks=0`: leaks are not errors.
- `use_sigaltstack=0`: an OCaml domain's thread exits on the signal stack the
  runtime allocated, which AddressSanitizer would unmap as its own.
- `allocator_may_return_null=1`: an allocation the allocator refuses returns
  NULL, as libc's does, so a test of a refused allocation sees the library's
  answer instead of the sanitizer's abort.

The profile also stresses the collector, so a value a stub holds unrooted, or
memory a finaliser frees while something still uses it, fails at once:

- `OCAMLRUNPARAM=s=4k,o=20,M=1,m=1,V=1`: a 32 KiB minor heap collects at almost
  every allocating stub; major cycles run far more often, and custom blocks
  with external memory push them at once, so finalisers run at many more
  points; the runtime verifies the heap at the end of each major cycle.
- `-runtime-variant d`: the debug runtime runs its own assertions. It prints
  a two-line banner on stderr at start.

A test meets the stress as it meets a slow machine: it keeps every value it
still uses reachable past its last use, waits for a finaliser by its signal
rather than by a count of collections, and bounds its work so its timeout
holds on a build an order of magnitude slower.

Constraints a new library meets:

- Bytecode runs are skipped under the profile: `ocamlrun` loads each
  library's stubs as a shared library and lacks the sanitizer runtime.
- `cuInit` maps memory over AddressSanitizer's shadow gap. `lib/cuda/dune` alone
  adds `protect_shadow_gap=0`; without it CUDA sees no GPU and every GPU test
  skips. The other libraries keep the gap protected.
- With the gap unprotected, AddressSanitizer on x86_64 maps the gap's shadow
  between about 2.3 and 258 GiB. The NV kernel path reserves its GPU addresses
  from 384 GiB, below 2^40, so a GPU opens in the profile.
- The sanitizer gate is Linux's. On the Mac, Metal keeps the pipelines it
  compiles in a cache under the user's cache directory. Every executable
  without a bundle shares one cache, which grows with each program that
  compiles Metal code: on the M1 Max on 2026-10-08, a 19 MB index
  (`functions.list`) over 757 MB of data. Loading a pipeline opens the cache,
  and the open grows a buffer with `realloc` while it reads the index.
  AddressSanitizer's `realloc` copies the block on every call, so under the
  profile the open ran for minutes (sampled: `fscache_open_worker` in
  `realloc` and `relocate_maps`) where the default build opens it at once.
  The suite's executable therefore carries a bundle identifier
  (`test/metal/rig_metal_test_bundle.c`), and Metal gives it a cache of its
  own that holds only the suite's pipelines. Any other executable that loads
  Metal code under the profile needs the same. Turning the cache off
  (`MTL_SHADER_CACHE_SIZE=0`) does not do: every load then compiles anew, and
  the two-domain image test outruns its timeout even without the sanitizers.

A GPU test that skips checks nothing, so read the skip count of a sanitize run
on a GPU host.

## Fixing a bug

The test comes first. Write the test that states the promise the bug breaks,
run it on the unfixed code and watch it fail, then fix. A bug found and not yet
fixed lands as an `xfail` whose reason states the wrong behaviour; the fix
removes the `xfail`. A generator withholds an input only for a bug that has its
own `xfail` test.

That run on the unfixed code is the only fail-first check. No mutation runs
and no hand-made mutants: a law's strength comes from its generator and its
covers.

Prefer extending a law or a model over adding a test. The ring case above is
an edge the ring law's generator must draw; the end-once crashes are commands
the two-domain model already had, run on more than one domain.

## Benches

Benches guard performance the way suites guard behaviour. Each library with a
bench keeps `<lib>/bench/` and a per-machine baseline (`<lib>.thumper`). Benches
are not part of `runtest`: build and run them with `dune build @bench`, which
takes a workspace lock so suites do not time each other. A bench links only
sandbox libraries.

Baselines are recorded in the release profile, which builds what users run:
the dev profile compiles with `-opaque`, which removes inlining across
modules, so its numbers describe no user's program. Build the bench with
`dune build --profile release` in a worktree whose build directory holds no
dev build in use (a watch server builds the dev profile), and record from
there; a dev build checks code, never baselines.

Each row times what a caller calls, beside the floor that bounds it where the
process can measure one: CUDA's release row beside a row that makes the same
CUDA calls directly, rig's submit rows beside Polled's own C room and
submit entries. The distance to the floor is the library's share. Rows also
record words allocated: a reached `Buffer.wait` that builds the closure of its
slow path before it takes its fast path shows there as 6 words a call.

A change runs every row of the modules it touches against the committed
baseline on every host it re-records; a row that moves is explained in the
commit before its baseline is re-recorded. The performance document covers the
rows and their numbers.

## A new driver's tests

The laws every driver keeps live once, in `test/conformance/`, and run on
each driver the machine has: its facts against `Rig.queues` and
`Submission.make`'s refusals, copies through any two memories on its copy
queues, every reader reading the last write, the order of values on every
queue, a workspace's launches, its images, the timeline (an idle close,
sleeps, work that runs with no further call, its own commits), a failed fill's
loss, waits on another device's point, and its peers. A law reads what it
needs from the device's facts: a device whose queues run no copy draws no
copy part. A driver joins with its support matching
`Rig_gpu_support.Conformance`: a binary of its fixtures, a second device
where the machine has one, and two works on its first queue, a copy of words
and a spin.

A driver's own suite opens its GPU through `Rig_gpu_support.Make` and holds,
on its GPU, what is its own:

- Its facts as its `.mli` states them, and its refusals, with misuse raising
  `Invalid_argument` at each bound the `.mli` states.
- Room: work its rings have no room for refused through `Rig.submit`, ring
  wraps drawn at exact and near fits, and streams longer than the rings.
- Its stop: what a loss does to running and queued work.
- Its fault path, after a guarded run, in a suite of its own if a fault
  outlives the device in its process.
- Its sanitize run, with any environment its vendor library needs set in its
  own `dune`.
