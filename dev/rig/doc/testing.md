# Testing the device libraries

This document is for a contributor who adds a driver or changes the core. It
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
| A device that fails is lost cleanly | loss, faults, Unknown stops | the core's test driver, guarded runs on GPUs |
| A path costs what it says | a submit that does not wait allocates nothing | words counted with `Gc.minor_words` |

Layout mirrors `lib/`. A library's suites live in `test/<lib>/` as
`test_<module>.ml` (the core's in `test/`), helpers shared by two suites (or by
a suite and a bench) in `test/<lib>/support/`, and fixtures in
`test/<lib>/fixtures/` with the command that made them. Its benches live in
`bench/<lib>/` and read their fixtures from `test/<lib>/fixtures/`. A suite
reads only its own directory. Every top-level group sets `~timeout`, and
`dev/rig/dune` sets `WINDTRAP_TIMEOUT` to 60 s for any test without one,
because a test that never ends blocks every build on the shared watch server.

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
order of the calls explains every result. A driver suite holds its end-once
rule this way: "an allocation freed from two domains is freed once", "a mapping
unmapped from two domains is unmapped once", "an image unloaded from two domains
is unloaded once" (`test/cuda/test_cuda.ml`; `test/metal/test_metal.ml` for
images). A value's flag that is a plain mutable bool, read and then written,
lets two domains both pass the check: two `cuMemFree` of one address, two
`cuModuleUnload` of one module, or a Metal pipeline released twice, each of
which crashes the process. The two-domain models reach that race on their first
cases; an `Atomic` taken with `compare_and_set` lets exactly one call end the
value. In the core, "stamps raised from two domains are their maxima" runs the
stamp model on two domains. In pci, "two domains allocate at once" and "pins
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

The core's test driver (below) gives the same control over a wait. Its sleeps
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
and for good (`lib/rig.mli`, "Loss"). Every later use of it, and of
memory whose stamps name it, raises `Lost`; other devices go on; its facts
still answer; its memory returns only once its word shows its last value
reached.

**The core's test driver.** The core's job is to keep that contract over any
driver, so its suites run on `Rig_support.Polled`
(`test/support/`), a driver over host memory whose queue runs only when
the test runs it or a wait sleeps. A wait that returns before it slept leaves
work unrun, which is how the stamp tests see that a submit waited for the
right point. Every driver call is logged, so a test can say which calls reached
the driver: "a stopped device is only freed and unmapped". Its options select
the driver behaviours the core must handle:

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
device answers its facts and values", "a loss leaves other devices and their
memory working". Polled tests the core; it is never used to test a driver.

**Failure walks.** A failure path is reached by few tests, so a walk reaches
every one. `test/test_walk.ml` runs each operation of the core over Polled
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
after the collector ran and the devices drained. Four runs make a leak of a
few dozen bytes per failure stand out beside the one thing a failure may keep:
a lost device's reason. A library whose driver has a seam of its own, such as a
path record or a transport, walks it the same way, with a countdown copied
into its own support.

**Drivers on their hardware.** A driver's fault path is tested on its GPU. A
test that faults or hangs a GPU joins a suite only after one guarded run: one
process, nothing else using the GPU, killed after 60 s, and the GPU checked
idle with no memory in use afterwards. The shared Linux hosts give no root, so
a wedged GPU cannot be reset there.

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

pci injects failure at its transport: "a transport failing at access k ends a
step in Error, never in bytes" (`test/pci/test_function.ml`) fails the k-th
access of a driver's step, for every k, and requires the step to end in the
machine's reason.

Long work is no fault: a device is lost only on its driver's report, and no
timeout decides it. The CUDA suite holds it as "long work is no fault, and a
stale seen returns at once", the AMD suite as "long work is no fault", and the
core as "a word still for three intervals loses nothing".

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

- Room. `room` answers `Fits`, `Later` or `Never`: `Later` only while the rings
  hold unreached work, `Fits` once the word reaches the last value, `Never` for
  parts that do not fit the empty rings or exceed the part bound. Parts whose
  declared sizes are near `max_int` must answer `Never`: summing them as 64-bit
  integers can wrap and answer `Fits` for work no ring holds.
- Streams longer than the rings, with every byte of every copy checked at the
  end.
- Counters past their width. "a Word wait holds the work across the 64-bit
  wrap (sampled)" (`test/cuda/test_cuda.ml`). A counter that wraps only after
  billions of values is reached through a hidden seam that renumbers an idle
  device (`Rig_amd.renumber`), so the test makes a handful of submissions.
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
machine. A suite that finds its GPU calls its support's `hold_gpu` before
`Windtrap.run`, so the wait counts against no test's timeout; a bench calls
it before `Thumper.run`, so the workers it forks measure under the lock, and
no GPU row runs beside a GPU test. `hold_gpu` decides from files alone
(`/dev/nvidiactl`, the Metal framework, `/sys`) and starts no vendor library,
which a bench must not start before it forks. The process holds the lock
until it exits; the holder writes its executable and process id into the
file. A process still waiting after 300 s fails, naming the holder. Under
`dune runtest` the GPU suites of a machine therefore run one after another.

One lock order holds everywhere: the GPU lock first, then the hosts' timing
locks, so nothing waits for the GPU while it holds a timing lock. A timing
run of a GPU bench on kimchi or nonnormal takes the GPU lock in the shell,
then the timing locks, and tells the bench it holds it with
`RIG_GPU_LOCK_HELD=1`, which makes `hold_gpu` and the core's GPU bench
return at once:

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

Every other suite (pool, elf, host, core, pci, the ABI libraries) runs on all
three. The Linux hosts also run the pool's cgroup and thread tests, pci's live
reads of `/sys/bus/pci`, and the gcc build. A library is done when its suite
passes on the Mac, under the sanitize profile, and on Linux. Tests that need
two GPUs ("map each other's memory") skip on every host, since each has one.

Paths a library reads from the host (`/sys`, `/proc`, `/dev`, a firmware
directory) are data its private module takes. The suite runs that code on a
fixture tree built under the test's own `_build` directory on every machine,
and checks the live machine with one read-only test.

Tests never exhaust a shared resource: threads, processes, file descriptors,
memory, GPU memory or disk. A failure path that needs a limit is reached by a
limit on a forked child alone. "workers that cannot be made are missing from
every job; with none, the calling thread runs every chunk"
(`test/pool/test_threads.ml`) sets `RLIMIT_NPROC` to 0 in a child on Linux,
where it binds that process only. Without such a limit or a seam that injects
the failure, the case is not tested.

## What cannot be tested without root

The hosts give no root. Taking a PCI function from its kernel driver, VFIO,
BAR windows and GPU page tables on real hardware, and the driver-less AMD and
NV paths built on them, cannot run there.

pci tests its own logic on fake machines. A transport is a machine the library
already supports, so a fake is a transport that keeps each operation's
contract, records each call, and records as `wrong` any call that breaks the
contract, which the library had to refuse before asking
(`test/pci/test_function.ml`, `test/pci/test_gpus.ml`). Machine files come
from fixture trees (`Machine.at root`). The fakes find bugs in pci's own
logic. "system memory is freed before its addresses are handed out again"
(`test/pci/test_memory.ml`) has the fake's `free_dma` ask the shared address
space for memory while it frees. A free that returns a region's addresses to
the space before it frees the system memory at them lets another GPU's owner,
on another domain, map over memory still being freed.

A fake cannot show hardware ordering, visibility or cost. The hardware paths
are not implemented against fakes and wait for a host with root. On the hosts,
the live tests take only what the process may take, such as a function behind
an IOMMU, which needs no root, and skip otherwise. Tests of file permissions
skip when run as root, which opens a file whatever its mode.

## Sanitizers

```
dune build --profile sanitize @dev/rig/lib/<lib>/runtest
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
- Metal tests that load a metallib do not finish within their timeout under
  AddressSanitizer on the M1 Max: Metal's shader cache reads its per-user cache
  through the sanitizer's `realloc`. The ring tests and the tests that open
  nothing pass. A stalled run holds the machine's GPU lock and blocks every
  other Metal run, so a sanitize run on the Mac leaves out
  `@dev/rig/lib/metal/runtest` until that stall is fixed.

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
CUDA calls directly, the core's submit rows beside Polled's own C room and
submit entries. The distance to the floor is the library's share. Rows also
record words allocated: a reached `Buffer.wait` that builds the closure of its
slow path before it takes its fast path shows there as 6 words a call.

A change runs every row of the modules it touches against the committed
baseline on every host it re-records; a row that moves is explained in the
commit before its baseline is re-recorded. The performance document covers the
rows and their numbers.

## A new driver's tests

A driver's suite holds, on its GPU:

- Its facts and its refusals, with misuse raising `Invalid_argument` at each
  bound the `.mli` states.
- Copies through any two kinds of its memory as the identity (`prop`).
- Room: `Later` only while a value is unreached, `Never` past the empty rings,
  ring wraps drawn at exact and near fits, and streams longer than the rings.
- The timeline: values complete in order, the word never moves backwards, a
  wait across its counter's wrap, long work that is no fault, `stop` of an
  idle device and of a running one.
- End-once for regions, mappings and images from two domains
  (`stateful ~domains:2`).
- Its fault path, after a guarded run, in a suite of its own if a fault
  outlives the device in its process.
- Its sanitize run, with any environment its vendor library needs set in its
  own `dune`.
