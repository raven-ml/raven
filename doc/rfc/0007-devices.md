# RFC 0007: Devices, runtimes and backends

- Status: discussion
- Date: 2026-09-25
- Packages: nx, tolk, rune, contrib/nx-oxcaml
- Amends: RFC 0005 (where devices come from, engines, placements, the tensor
  representation, routing, Laws 1, 2, 5 and 6), RFC 0006 (host cells)
- Depends on: RFC 0008, for the representation of a backend value

## Summary

nx owns the runtime of every device: its memory, queue and program loading,
shared by hand-written kernels and the programs tolk compiles. A device is the
hardware, one type shared by nx, tolk and every backend, opened by a call that
returns the same value each time, with no registry. A placement carries its
devices, its layout and one backend: by default `Nx.Backend.host`, nx.c's
kernels, which off the host move operands to the host and back; nx.cuda's
kernels on CUDA; or nx-oxcaml over storage it owns. An operation runs on its
operands' placement, with the same API everywhere. tolk's compiler and graph
replay run over nx's runtime and never see arrays; tolk's graph backend, nx's
operations as lazy graph nodes, is the one lowering from nx to tolk, and
rune's jit runs a function over it and compiles the graph.

Four nouns keep one meaning each. A **device** (`Nx.Device.t`, defined in
`nx.device`) is opened hardware of one kind: memory, a queue, program loading.
A **backend** (`Nx.Backend.t`) is a value that implements nx's operations.
A **placement** (`Nx.Placement.t`) is devices, a layout and one backend. A
**compiled device** (`Tolk.Compiled.t`) is a device with tolk's renderers and
queue encoders.

## Motivation

nx names devices it cannot use. `Nx.Device.t` and `Nx.place` are nx's
(RFC 0005), but a device other than the host comes only from `Rune.device`,
because the engine behind it lives in rune's jit (`jit.ml`, section "Devices,
opened by name"). A program that wants a GPU tensor links rune.

The runtime cannot run without the compiler. A tolk device carries its
renderers and compilers (`device.ml:137`), and every GPU queue is a host
program compiled by codegen and clang (`tolk_cuda.ml:370`,
`tolk_metal.ml:529`). Placing a tensor on CUDA loads NVRTC to ask whether the
device holds its dtype (`holds` in `jit.ml`).

Dtypes exist three times (`Nx_core.Dtype`, `Nx_buffer`, `tolk.uop`), and
buffers, device identity (bridged by rune's `by_name`), allocators and budgets
twice each. tolk and rune both register device openers, and initialisation
order decides which wins (`device.ml:328-329`).

A CUDA backend for nx with hand-written kernels would today bring a second
CUDA runtime next to tolk's: two memory pools blind to each other, which break
RFC 0005's budget; events exchanged across the runtimes on every jit call; and
kernels that reach compiled graphs only through host memory. Doing nothing
keeps every new device capability in rune.

## Guide

```ocaml
let gpu = Nx_cuda_device.v 0                    (* the device: CUDA:0 *)
let cu = Nx.Placement.device ~backend:Nx_cuda.backend gpu
let x = Nx.place cu (Nx.rand Nx.float32 [| 4096; 4096 |])
let y = Nx.tanh (Nx.matmul x x)                 (* runs on CUDA:0, nx.cuda's GEMM kernel *)
let s = Nx.item [] (Nx.sum y)                   (* copies 4 bytes to the host *)
let x' = Nx.place (Nx.Placement.device gpu) x   (* the host backend over the same GPU: a view *)
```

This program links `nx` and `nx.cuda` only, and the same code runs on every
device. A placement's backend defaults to `Nx.Backend.host`, whose eager
operations run on the host; Metal, AMD and NV have no hand-written backend, so
`Nx.Placement.device (Rune.device "METAL")` computes eagerly on the host, as
today, and its compiled functions run on the GPU.

An operation's operands share one device list and one backend; a host operand
joins any placement whose backend uses device buffers, as RFC 0005 says, and
any other mix raises, naming both placements. Moving between two placements
that differ only in backend makes a view when both backends use the device's
buffers: a compiled call that consumes `x` or `x'` consumes the storage both
reach (RFC 0006 Law 6), and `Nx.copy` gives independent storage.

```ocaml
let step = Rune.jit Nx.Ptree.(tensor @-> consumes params @@ returns (pair tensor params)) train_step
let loss, params = step x params                (* x placed by nx.cuda: bound with no copy *)
```

tolk compiles it for `x`'s devices, whatever backend placed it, over the
buffers eager code uses, and its results take `x`'s placement; leaves and
captures follow the operands' rule. rune's API is unchanged, `Rune.device`
included, and a jit over host inputs runs on the default device, as today.

## Reference

### Libraries

| Library | Contents | Depends on |
|---|---|---|
| `nx.dtype` | typed `('a, 'b) Dtype.t`, untyped `Scalar.t`, bf16, fp8 and int4 encodings (C header installed) | none |
| `nx.device` | `Nx_device`: devices, `Kind`, buffers, programs, events, the allocator and budget, `Stats`, the host devices; `Nx_buffer` | `nx.dtype` |
| `nx.core` | shape, view, the operation vocabulary `Ops.S` | `nx.dtype`, `nx.device` |
| `nx.effect` | the tensor representation, `Placement`, `Backend.S` and `Backend.t`, `E_op` | `nx.core`, `nx.device` |
| `nx.c` | nx.c's kernels | `nx.core`, `nx.effect` |
| `nx` | `Nx` (re-exporting `Nx.Device`, `Nx.Backend`, `Nx.Placement`), the numpy surface, the entry functions, `run` and dispatch, `Nx.Backend.host`, `Ptree` | the above |
| `nx.cuda.device` | the CUDA runtime over libcuda, moved from tolk | `nx.device` |
| `nx.cuda` | CUDA kernels as committed cubins; `Nx_cuda.backend` | `nx.core`, `nx.effect`, `nx.cuda.device` |
| `nx.metal.device`, `nx.amd.device`, `nx.nv.device` | the other runtimes, moved in tranche 2 when needed | `nx.device` |
| `tolk` (opam `tolk`) | compiler, graph replay, `Compiled.t` | `nx.dtype`, `nx.device` |
| `tolk.<vendor>` | each vendor's compiled device: renderers, compiler, queue encoders | `tolk`, the vendor's runtime |
| `tolk.frontend` | `Tolk_frontend`, tinygrad's `Tensor` port; the one match from a kind to its device and compiled device | `tolk`, every `tolk.<vendor>` |
| `tolk.nx_graph` | the graph backend: nx's vocabulary as lazy tolk nodes, compiled and run on read | `nx.core`, `nx.effect`, `tolk.frontend` |
| `rune` | transformations; `Quant.lower`; `Rune.device` | `nx`, `tolk`, `tolk.nx_graph` |

A backend library depends on `nx.core`, `nx.effect` and its runtime or tolk,
never on `nx`, and `nx` depends on no backend but nx.c, which
`Nx.Backend.host` uses. `nx.buffer` and the virtual `nx.backend` library go at
J2; until then nx.effect depends on the virtual library, which nx.c
implements, so nx.c implements `Backend.S` and dispatch moves to `nx` at J2.
`Nx_buffer` stays as the host's typed buffer. A vendor runtime not yet moved
stays in tolk behind `nx.device`'s interface. Building nx needs no SDK:
libcuda is opened with `dlopen`, AMD and NV use `ioctl` through vendored
headers, Metal links frameworks behind `enabled_if macosx`, and nx.cuda's
kernels are committed.

### The runtime (`nx.device`)

```ocaml
(* Nx_device, abridged; re-exported as Nx.Device *)
module Kind : sig type t = Cpu | Cuda | Nv | Metal | Amd  val of_string : string -> (t, string) result end
type t                                          (* a device *)
val host : t                                    (* "CPU" *)
val kind : t -> Kind.t
val name : t -> string                          (* "CUDA:1", tinygrad's spelling *)
val arch : t -> string                          (* "sm_89", "gfx1100", "arm64" *)
val equal : t -> t -> bool
exception Out_of_memory of t * int              (* the one definition *)
module Buffer : sig
  type device := t
  type t
  val create : device -> Nx_dtype.Scalar.t -> int -> t
  val view : t -> offset:int -> Nx_dtype.Scalar.t -> int -> t      (* offset in bytes *)
  val of_host : ('a, 'b) Nx_buffer.t -> t       (* on the host device; no copy *)
  val wrap : device -> ('a, 'b) Nx_buffer.t -> t  (* borrowed; no copy *)
  val copy : dst:t -> src:t -> unit             (* DMA or peer, else a host bounce *)
end
module Program : sig
  type device := t
  type t
  type arg = Buffer of Nx_dtype.Scalar.t | Scalar of Nx_dtype.Scalar.t
  type image = { binary : string; entry : string; args : arg array }
  val load : device -> image -> t               (* cubin, hsaco, metallib, host ELF *)
  val launch : t -> Buffer.t array -> int64 array -> global:int array -> local:int array -> unit
end
module Stats : sig type t val bytes_in : t -> int val bytes_out : t -> int val diff : t -> t -> t end
val stats : t -> Stats.t
(** {1:low Low-level} Each entry is tagged {b Implementers.} with its users. *)
module Event : sig type device := t type t val record : device -> t val wait : t -> unit end
val addr : Buffer.t -> nativeint
val launch_fenced : Program.t -> Buffer.t array -> int64 array -> global:int array -> local:int array -> unit
```

**The runtime is tolk's,** the port of tinygrad's `device.py` and `runtime/`,
with its capabilities and semantics: libcuda through `dlopen`, Metal, AMD's
KFD and AM, NV's NVK and GSP, and the same allocators, transfers, signals,
program loading, timed launches, synchronisation and names. Its one structural
divergence is that a device carries no compiler, where tinygrad's `Compiled`
holds its renderer and compiler; tolk pairs them in a compiled device. NVRTC,
comgr and Metal's compiler stay in tolk for its generated kernels, as in
tinygrad. No raven library calls cuBLAS; M's benchmark calls it only as a
yardstick.

**Opening.** A vendor runtime library exports `count : unit -> int`, `get :
int -> (Nx_device.t, string) result` and `v`, which raises `Invalid_argument`
with `get`'s message: a program that probes gets a value and catches nothing,
and one that requires the device fails loudly. `get` opens the device at its
first success and returns the same value after; its memo mirrors the driver's
one context per device. A device is what an opener returns: `CUDA:0` and
`NV:0` over one physical GPU are two devices with separate memory and budgets,
like two GPUs, and a move between them copies, as in tinygrad. Kinds are a
closed variant, so every match over them is checked for exhaustiveness.

**The interface.** The public part holds what callers observe: identity,
buffers, programs and statistics. What only vendor libraries, backends and
tolk use (raw addresses, a launch whose caller encodes its fences, events,
each vendor's `Native`) stays in the flat interface under a "Low-level"
section with a stable anchor, each entry's doc naming who it is for. An
`Event.t` is the completion of work submitted to a device's queue, used by
tolk's replay, backends, the allocator's deferred release and RFC 0005's calls
that return without waiting; users never see events (RFC 0005 Law 7).
Statistics are per-device snapshots that callers subtract with `Stats.diff`,
so no global counter is reset under another library.

**Submission.** The runtime orders every submission to a device after the
earlier ones: direct launches and copies through its own calls, compiled
batches through the handoff `Native` exports (today's `tolk_cuda_hcq_begin`).

**The runtime links no compiler.** Every GPU runtime already loads, launches
and synchronises without one, and copies by DMA where it can; the host bounce
(`Realize.copy_via_host`) moves into the runtime, and queue encoding,
`bufferize` and compiler setup leave it for tolk.

**Memory.** Each device has one caching allocator, one budget and one
`Out_of_memory`. A device buffer is owned, allocated by the allocator and
counted against the budget, or borrowed, wrapping host memory or a mapped file
that the device addresses, which is neither counted nor written nor lent (RFC
0005 Laws 8 and 9). nx attaches the finaliser of every cell over device
buffers, and lending moves owned storage to the new cell. A finaliser queues
buffers: an owned one returns to the cache once the event of its last
submission completes, and a borrowed one releases its source then; freeing to
the system synchronises first.

### Values and dispatch

The tensor type becomes `Placed of placed | Traced of traced`, with host
values placed on `Nx.Placement.host`. A cell is a view over a runtime buffer,
with its placement: stage 2's M2 already stores the placement in the cell
(`Nx_effect.cell ~placement`) and reaches the engine through the placement's
devices, and here the runtime takes the engine's place. Its storage is closed:
device buffer views, one per device of the placement; storage its backend owns
(nx-oxcaml); or a held one-element value. Its other state is RFC
0006's `Consumed`. A creation context becomes a placement.

**Backends.** A backend is a value, `Nx.Backend.t`, that travels in the
placement. Each backend library exports one (`Nx_cuda.backend`,
`Nx_oxcaml.backend`), and nx defines the default, `Nx.Backend.host`.
Transformations intercept operations through one effect carrying an operation
as a function of its interpreter, over the closed vocabulary `Ops.S`, so every
transformation is checked by signature. A backend value is `(module
Backend.S)` (RFC 0008): it says whether it runs on a device (`runs_on`), and
runs or refuses each operation.

**Placements.** `Nx.Placement.t` is abstract (RFC 0005, stage 2). The
constructors `device`, `replicated` and `sharded ~axis` take `?backend`,
defaulting to the one default backend, `Nx.Backend.host`: nx.c's kernels, run
in place on the host device and on `CPU:k` devices, whose memory is
host-addressable, allocating results on the same device; on any other device
it is host compute, which reads each operand's window to the host once, runs
nx.c and places the result back (RFC 0005 stage 1's routing). `host` is a
value; `host`, `device Nx.Device.host` and `device ~backend:Nx.Backend.host
Nx.Device.host` are one placement, equal and never mixed, and another backend
on the host is `device ~backend Nx.Device.host`. A constructor raises
`Invalid_argument`, naming the backend and the device, when the backend's
`runs_on` is false for that device.
The eliminators are `devices`, `window : t -> int array -> Device.t -> (int *
int) array` (the part of a value of that shape that device `d` holds, as
`(start, stop)` per axis, the ranges `Nx.shrink` takes; `Nx.tile` means
repetition), `backend` and `equal`, which
compares what each device holds and the backend. A placement's devices share
one backend and one kind by construction. An eager result over split or
replicated operands takes the placement of tolk's multi rule on every backend:
elementwise keeps the split, resharding to the last split axis among its
operands, and a reduction over the split axis is replicated (stage 2's plan,
its Q1, applied in M2).

**Dispatch.** An operation takes the device list and backend its placed
operands share, and the result's placement from the multi rule above, and has
the backend run it, which refuses in the operation's function (Backends). Host
values still join a placement whose backend uses device buffers, as RFC 0005's
Law 2 says: a host operand is copied there, or read in place on the host. They
do not join a placement whose backend owns its storage, where that would be a
silent copy. Every other mix raises, a different backend included: `Nx.add:
operands on CUDA:0 with nx.cuda and CUDA:0 with host; place one of them (no
copy)`. A compiled call applies the rule to its placed leaves and captures,
naming the leaf's path, and gives its results their placement.

**Moves.** `Nx.place` between two placements with the same devices and layout
and different backends copies nothing when both backends use device buffers:
the result is a view, which RFC 0005's and RFC 0006's view and consumption
rules cover, and in a trace it only sets the placement. It copies when either
backend owns its storage: a move onto such a placement is run by the target
backend, and nx makes the view itself only between
backends over device buffers. `Nx.place`'s documentation gains both cases.

Dispatch through a backend value adds an indirect call and a cell per result
on nx's hottest path, a cost the host gate (Order of work) decides.

### Backends

A backend implements each operation or refuses it by capability in the
operation's function, before that function does any work, by raising
`Nx.Backend.Refused`; dispatch names the operation, dtype and devices in its
message. The conformance suite calls every operation at every dtype, prints
each backend's refusal table from the calls that raised `Refused`, and checks
the others against nx.c up to RFC 0005's Law 1. Holding bytes is the runtime's
and computing is the backend's, so the runtime has no `supports`; hardware
facts such as Metal's missing float64 are inputs to the backends' functions. A
copying `Nx.place` creates its destination with the target backend's
`buffer`, which refuses before the copy, with no compiler loaded; a move that
makes a view copies nothing and refuses at its first operation.

- **`Nx.Backend.host`** (Placements) states where it computes. It serves
  Metal, AMD, NV and CUDA without nx.cuda, and compiled functions over its
  values run on their devices.
- **The graph backend** (`Tolk_nx_graph.backend`) runs operations as lazy tolk
  graph nodes and realises a value when it is read, the shape of tinygrad's
  `Tensor`. Its operations are rune's jit module (RFC 0008), retyped to its
  storage, and tolk's `quant_matmul` and `block_matmul` over its nodes; `Quant.lower`, which
  chooses among them in nx operations over `Nx_quant` (RFC 0004), stays in
  rune, which keeps handling `E_quant`. It is internal at first:
  `Tolk_nx_graph.backend` sits in its interface's Low-level section, tagged
  for rune and nx's tests, and placing user data with it is a future
  possibility. It refuses what jit refuses today (FFTs, `svd`, eigensolvers,
  complex, int4, float64 on Metal).

  ```ocaml
  val backend : Nx_effect.Backend.t             (* lazy nodes, realised on read *)
  val node : ('a, 'b) Nx_effect.t -> Tolk_frontend.Tensor.t  (* for staged scan *)
  val quant_matmul : ...                        (* tolk's kernels, for rune's Quant.lower *)
  val block_matmul : ...
  module Program : sig
    type t
    val compile : Nx_effect.Placement.t -> inputs:Nx_effect.packed list ->
      outputs:Nx_effect.packed list -> t
    val run : t -> Nx_device.Buffer.t array -> lend:Nx_device.Buffer.t option array ->
      Nx_device.Buffer.t array                  (* lend: None allocates *)
  end
  ```

- **nx.cuda** compiles and loads nothing at run time beyond its own cubins: no
  NVRTC, no PTX JIT, no cuBLAS. Its kernels, GEMM and attention included, are
  hand-written CUDA C compiled ahead of time by nvcc into cubins and loaded
  with `Program.load`. A cubin runs on later minor versions of its major
  architecture, so there is one per major family, plus tuned variants where
  they matter, such as tensor-core GEMM for sm_80 and sm_90; it refuses a
  device whose architecture has no cubin. Variants are generated offline, as
  nx.c's are, and fusion stays jit's: an eager kernel serves one operation,
  shapes and strides as arguments. Cubins are committed with their provenance
  (nvcc version, flags, source hash), regenerated by a committed script and
  hash-checked in CI. Its primary target is libcuda's `nx.cuda.device`; the
  same cubins also load on the driver-less `nx.nv.device`. At first it
  refuses the FFTs, `svd`, the eigensolvers, the complex and int4 dtypes, and
  split and replicated placements.
- **nx-oxcaml** is `Nx_oxcaml.backend`, on the host device, over storage it
  owns: OCaml unboxed arrays, which are its purpose and have no stable address.
  Moving between nx.c's and nx-oxcaml's host placements copies, and rune's jit
  refuses backend-owned storage before running, naming the leaf. Programs that
  linked nx-oxcaml place their inputs with its backend.

### tolk

A compiled device pairs a device with its renderers and queue encoders,
`{ device : Nx_device.t; renderers : Renderer_set.t; queue : queue option }`,
read where the compiler reads `Device.t` today; each `tolk.<vendor>` builds
one per device (`Tolk_cuda.compiled`), and `tolk.frontend` holds the one match
from a kind to its device and compiled device, used by its `DEV` selection,
the graph backend and `Rune.device`. Graph replay keeps its host submissions,
launched with `launch_fenced`, and Metal's indirect command buffers, which C1
re-measures. tolk's IR names devices by canonical string, resolved against the
devices a call binds, and the registry of openers goes. tolk's dtype stays,
converting to `nx.dtype`.

**tolk on its own.** tolk builds, compiles and runs graphs through
`Tolk_frontend`, its port of tinygrad's `Tensor`, on devices it opens, with no
rune and no nx arrays. Its 90 parity cases, which render only, are untouched,
and its end-to-end suites (compile, run, replay, several devices) stay. nx's
conformance suite runs against the graph backend with nx.c as reference, and
one nx program is benchmarked through nx.c and through tolk-compiled graphs,
with no rune. Runtime-only tests move with the runtime. Nothing that runs
today stops running.

Every port of a tinygrad runtime change splits along one line: whatever builds
or reads tolk's IR is tolk's. Allocators, interfaces, device initialisation,
signals and program loading go to the runtime; `pm_encode`, `pm_lower`,
`pm_batch`, `pm_bufferize` and the queue classes go to tolk. A port that needs
a runtime resource `Native` lacks adds it and says so in its commit body. If
more than one in three of the first 20 ports after G1 changes a `Native`
signature, the line is reviewed before the next vendor moves.

### rune

rune's jit runs the function with its operations dispatched to the graph
backend, its handler applying each call to a module over it, then compiles the
graph with `compile` and replays it with `run`. It keeps tracing, keys,
consumption, lending (RFC 0006) and staged scan, which builds its loop from
`node`. It compiles for the devices of its inputs' placement, whatever their
backend, and gives its results that placement; when every input is on the host
and it runs on the default device, its results take that device with
`Nx.Backend.host`. Its key holds each leaf's
devices and layout, so placements that differ only in backend share programs.

`Rune.device`, `devices` and `default_device` stay and return devices: a
name parses with `Kind.of_string`, and `tolk.frontend`'s match calls the
vendor's `v`, or gives `Nx_device.host` for `"CPU"`; `devices` enumerates
with `count`, and `default_device` reads `DEV` once, else tries `get` for
METAL, AMD, NV and CUDA in turn, else the host, catching nothing. They stay
because rune already links every runtime, a closed match is no registry, and
moving 178 caller lines in 39 files would buy nothing a law needs. The engine
record, storage accounting and budget move to `nx.device`; `Rune.jit_stats`
keeps `reused_bytes`, and transfers are `Nx_device.stats` snapshots.

### Fixed kernels inside compiled graphs

Every kernel runs on its device's runtime, so a compiled graph can hold
kernels tolk did not generate, sharing the device's memory pool and queue.
Pinned templates in tolk's IR, like RFC 0004's `quant_matmul`, run today. A
prebuilt binary, such as an nx.cuda cubin, enters as `CALL(PROGRAM ...
BINARY)` once `custom_kernel` accepts a program body as tinygrad's does, with
no runtime move. The graph backend chooses among these kinds.

### Amendments

RFC 0005:
- **Where devices come from.** From a vendor library in nx, and by name from
  `Rune.device`, which stays. "nx opens no device" reads "nx opens no device
  by name".
- **Engines.** This RFC's Reference replaces "Engines, and what belongs
  where", its paragraphs on device engines, `create_context` and "Nothing
  moves from tolk into nx", and the Summary's matching sentence. The engine
  record becomes the runtime and the backend, its host routing
  `Nx.Backend.host`.
  Stage 3's eager compute on devices waits for the graph backend as a
  placement's backend (Future possibilities); its other items stay.
- **Representation and routing.** `Host` and `Placed` merge, with the
  paragraph on why they differed; `context` becomes a placement; a cell is a
  view over a runtime buffer, with its placement, and holds device buffer
  views, backend-owned storage or a held value. `Placement.tile` is
  `Placement.window`. Routing
  dispatches to the placement's backend. In Moving, "the same engine" reads
  "the same kind", and a move that changes only the backend is a view; in
  Lifetime, "an engine records per cell the last submission" reads "the
  runtime records each buffer's last submission's event". The cache key
  holds devices and layouts, and `Out_of_memory` is `Nx_device`'s.
- **Laws.** Law 1 reads "its backend's rounding". Law 2 reads "A result lives
  on its placed operands' placement. Host operands join placements whose
  backend uses device buffers; operands on two device lists or with two
  backends raise." Law 5's "one engine and one backend" reads "one backend,
  over devices of one kind". Law 6 is replaced by this RFC's Law 3. Laws 3, 4
  and 7 hold as written.
- **Rationale.** Its rejections of moving tolk's runtimes into nx and of a
  second virtual `nx.backend` assumed a compiler in the runtime and the
  link-time seam, which go. Its Amended-by line gains "RFC 0007".

RFC 0006: its host cell becomes the host device's cell. The host gate (Order
of work) takes RFC 0006's criteria: RFC 0006 measures host cells as it
states, and J2 measures the merged representation against the same criteria.
If host cells failed RFC 0006's measurement, J2 keeps the host's direct path
from the start, and RFC 0006's fallback for its Law 6 applies.

### Order of work

The moves start after RFC 0005's stage 2 lands, so they carry its multi-device
code once, and after tolk's TODO milestone 2, which rewrites the files C1 and
G split, so those files move once. RFC 0008's phase 1 may land at a quiet
point of stage 2, and its phase 2 after stage 2 and before J1, so J1 writes
the backend value once, in RFC 0008's representation. A may land
now, B at stage 2's quiet point (after its M2, before its M3). Efforts are
agent-days, the sizing report's unit, include stage 2 (its plan's +3,050 −880
lines, about +3,210 and 42 engineer-days in the mesh proposal's revision) and
are re-derived before C1; RFC 0008 is sized in its own text.

**Tranche 1** (28 to 45) gives nx.cuda a runtime it shares with tolk, in three
phases whose pull requests each leave every package they touch green:
1. **Split in place** (11 to 18). A: rune's opam file declares `tolk` (0.25).
   B: `nx.dtype`, with one bf16 and fp8 codec rounding once, to nearest even,
   from its source's width, checked against tolk's over every fp8 code with
   f32 and f64 sweeps; a remaining difference is recorded in
   `DIVERGENCES.md` (3 to 4). E: tolk on `nx.dtype` (2 to 3). C1: inside tolk,
   a runtime library that links no compiler, compiled devices over it and
   device values in place of the registry; host, CUDA and Metal split, and
   AMD and NV get a compiled device over their unsplit runtime (6 to 11).
2. **The runtime moves into nx** (12 to 19). F: `nx.device` with the runtime
   interface, `Kind`, the host devices and the allocator; `on_device` and the
   registry go, and `Rune.device` opens through the vendors' `get` and `v` (7
   to 11). G1: the host's program loader and the CUDA runtime into nx (1 to 2
   each). H: one allocator, budget and `Stats` per device, with buffers owned
   or borrowed and only owned ones counted (3 to 4).
3. **Placements carry backends** (5 to 8). J1: backend values over the
   shared cell, `?backend` on the placement constructors and the `backend`
   eliminator, RFC 0005's engine becoming `Nx.Backend.host`, the host keeping
   its direct path, and rune binding cells of any backend.

Tranche 1 moves about 4,500 lines of runtime into nx. Two streams follow in
parallel. **M**, nx.cuda's kernels and the conformance suite, is sized before
it starts, against hand-written GEMM and attention and the architecture set
(which major families get cubins, which kernels get tuned variants), and stops
at 1.3 times that estimate; the earlier 30 to 60 assumed cuBLAS. **D and I**
(12 to 19): D makes the lowering the graph backend, retyping RFC 0008's jit
module to the graph backend's storage and moving it, the compile path and the
program cache from `jit.ml` into `tolk.nx_graph`, and switches rune's jit to
it (10 to 15, re-derived from that module); I deletes rune's
engine record, transfers and storage accounting, which F, H and J1 replaced (2
to 4).

**Tranche 2** (15 to 33 if every step lands) holds steps with triggers: F2,
`Nx_buffer` into `nx.device` and `nx.buffer` gone, after F (2 to 3); G2, Metal
(1 to 2), HCQ with AMD (2 to 4) and NV (2 to 4) with their splits, each when a
backend outside tolk needs it; J2, `Host` and `Placed` merged, the virtual
library gone, nx.c implementing `Backend.S` with dispatch moved to `nx`, and
the host gate, after RFC 0006's host-cell measurement (3 to 6); L,
`Nx_oxcaml.backend` over its own storage, after J2 (3 to 10); N,
prebuilt binaries in compiled graphs, which may land first (2 to 4). G2 waits
because the four runtimes add 27.8 MB to nx's 16 MB and 8 s to a cold build.

With M at its earlier estimate, and excluding its re-sizing, the plan costs 85
to 157, RFC 0008 excluded. If tranche 1 exceeds
its high estimate by half (about 68), this RFC returns to discussion, the split
stays inside tolk as C1 left it (Laws 2 and 3 hold), and M waits.

**Hardware.** A GPU CI job, a GitHub-hosted T4 runner billed per minute, runs
the CUDA runtime tests, tolk's NV kernel-driver tier and a rune jit, and
passes before C1's CUDA split, G1's CUDA move and M; its first run checks
whether the NV tier runs on a T4, with a rented machine as the fallback. A
skipped suite is not evidence. Each vendor's move reruns its suite where it
passed before C1 (CUDA on the runner, Metal locally, AMD's KFD path on its own
machine); NV moves after a first pass; G1's peer transfers need two GPUs.

Acceptance:
- tolk's device-opening suites stay green through C1, F and G, which rerun
  stage 2's probe on `CPU:1`..`CPU:4`;
- C1 times 10,000 launches of one kernel, direct and through
  `Realize.run_linear`, on the CPU and Metal, and keeps ICB replay of a chain
  of unfusable Metal kernels within 5% of before;
- after G1, J1 and D, gpt-oss-20b keeps its 228 checks, decode median within
  2% and peak within 0.2 GB of a same-machine baseline from the commit before;
  after J1 `Nx.Backend.host`, and after D the graph backend, pass the
  conformance suite on `CPU:1` and Metal with their generated refusal tables,
  and a test backend that refuses one operation shows it in its table;
- **the host gate**, after J2, on an idle M1 Max in alternating paired runs:
  RFC 0006's criteria (overhead rows within 5% or 10 ns, a rune eager training
  row within 2%, alloc rows growing by exactly the words of the record and
  cell a fresh host result adds, counted at J2), every other host row within
  3%, and RFC 0008's step 0 rows within 10% or 20 ns of RFC 0005 stage 1's
  commit, the sum of RFC 0006's host cells, RFC 0008 and J2;
- M: the conformance suite on CUDA; GPT-2's eager forward pass equal to the
  host's; an nx.cuda tensor entering `Rune.jit` with a zero `Stats.diff` of
  bytes in; one budget and `Out_of_memory` for nx.cuda and tolk; f16 and f32
  GEMM benched against cuBLAS called directly, as a yardstick;
- L: jit over backend-owned storage raises before running.

Stops: if J2 fails the host gate, the host keeps its direct path. A vendor
whose move exceeds its budget by half keeps its runtime in tolk.

## Laws

1. **One runtime per device,** where a device is what an opener returns:
   everything on it shares its memory, queue, program loading and submission
   order, and two devices over one GPU are two devices. *Prevents:* memory
   pools blind to each other, events exchanged across runtimes, and a direct
   launch racing a compiled batch.
2. **The runtime links no compiler.** *Prevents:* placing a tensor loading a
   compiler, and tools that cannot link a runtime without codegen.
3. **Opening is an explicit call that returns the same value each time:** a
   vendor library's `get` or `v`, one value per GPU and kind. Nothing
   registers an opener, and `Nx.Placement.host` carries `Nx.Backend.host`.
   Replaces RFC 0005's Law 6. *Prevents:* the opener race, devices that exist
   by link order, and two contexts for one device.
4. **Storage belongs to the device.** A value over device buffers is a view of
   them, and a move between backends that use them copies nothing. A backend
   that owns its storage is the one exception, and moves to and from it copy.
   *Prevents:* copies between backends on one GPU, and per-backend tensor
   types.
5. **A backend answers every operation where it says it runs, and user code
   names a backend only in a placement.** It runs an operation or refuses it
   by capability, in the operation's function, before any work runs;
   `Nx.Backend.host` computes on the host, as its name states, and every other
   backend on its devices. The conformance suite checks results and generates
   refusals. *Prevents:* silent host fallbacks, undocumented differences by
   device, and device-specific APIs.
6. **tolk's compiler never sees arrays, and `nx` needs nothing else in the
   project.** `tolk` depends on `nx.dtype` and `nx.device`, each
   `tolk.<vendor>` on its runtime, and nx's vocabulary reaches the compiler
   only through the graph backend; nx's libraries depend on no other package.
   *Prevents:* the compiler learning nx's vocabulary, and an opam cycle.
7. **One lowering.** Every path from nx's vocabulary to tolk goes through the
   graph backend, including rune's `Quant.lower`, which composes nx operations.
   *Prevents:* two lowerings that drift. Fusion still rounds compiled results
   differently from eager ones, which RFC 0005's Law 1 allows.

## Drawbacks

- **tolk diverges from tinygrad for good in one structural way:** the device
  has no compiler, where tinygrad's `Compiled` owns its renderer and
  compiler. The runtime therefore keeps a direct launch, every runtime port
  splits, and four `DIVERGENCES.md` entries lose their "when direct dispatch
  is removed".
- **The `nx` package grows by each runtime it takes** (1.1 MB for the host and
  CUDA), and its release depends on driver stubs building on opam-repository's
  platforms. tolk's OCaml floor rises to 5.5; `nx` gains `threads` with Metal.
- **Eager operations on Metal, AMD and NV run on the host,** copying operand
  windows, as RFC 0005's stage 1 does; compiled code runs on the device.
- **The payoff arrives with nx.cuda,** a large kernel-writing job with no
  local hardware, whose hand-written GEMM and attention must reach competitive
  throughput (measured here without a gate) from cubins built ahead of time.
- **The moves land in the most-edited code** (`jit.ml` and tolk's runtime
  took 160 commits between 2026-09-10 and 2026-09-25).

## Rationale and alternatives

**A device as a runtime paired with a backend** (per-backend openers) gives
"device" two meanings, built four ways; with the backend in the placement,
"device" means the hardware and mixed backends are mixed placements.

**An eager device backend over tolk's generated kernels now** runs each eager
operation on Metal, AMD or NV as a cached one-operation program, at a compiled
call each (about 0.19 ms on Metal); its lazy form, which would fuse eager
work, is designed better once the graph backend exists. That backend's name,
`tolk.nx_graph`, avoids tolk's senses of "backend" (a device kind) and "graph"
(replay).

**A bridge package** (`nx-tolk`) holding the lowering, engine and opening was
three blind designs' answer; it names no concept. **The runtime as its own
package** adds a release unit that no user installs alone.

**Two runtimes** (tolk unchanged, nx backends with their own) has every cost
Motivation lists, and each vendor gains a second allocator, transfer path and
launch (for CUDA, about 390 OCaml and 767 C lines).

**Engines registered in nx** bring initialisation-order state into nx, and
**backends that compose on one device** are partial; fixed kernels in compiled
graphs give the same result. **Kernels compiled at run time and vendor
libraries** (CuPy's NVRTC and cuBLAS) save hand-writing GEMM and attention,
and bring a compiler and dependencies beyond the driver interface into nx;
eager kernels that take shapes as arguments need no run-time specialisation.

**A separate module for implementer functions** splits one interface for a
handful of entries; a tagged "Low-level" section says the same in one.

**Fixed kernels as new nx operations,** such as attention, grow the
vocabulary and need rules in rune; choosing among kernels at lowering needs
neither. PyTorch's compiler launches Triton kernels on the stream and allocator
of its eager kernels, the arrangement Law 1 gives raven; PJRT and IREE keep
their runtimes apart from their compilers.

## Non-goals

- Ahead-of-time deployment of compiled programs without tolk.
- Several processes or nodes: a later multi-node RFC chooses a controller
  model by measurement and may extend device identity.

## Unresolved questions

None. RFC 0008 settled the representation of a backend value and the
libraries: `nx.core` keeps its name, and there is no `nx.frontend`.

## Future possibilities

- The graph backend as a placement's backend: lazy Metal, AMD and NV compute,
  realised on read.
- Backends that compose on one device, operation by operation.
- Hand-written backends for Metal and AMD.
- Vendor library calls (cuBLAS, cuDNN), excluded while nx.cuda depends only
  on the driver interface; they need a call node and a submission rule.

Nothing listed here is a reason to accept this or a later RFC.
