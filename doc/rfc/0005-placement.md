# RFC 0005: Placement

- Status: discussion
- Date: 2026-09-24
- Amended by: RFC 0006 (§Compiled functions, the cell state `Donated`,
  host storage, the cache key); stage 2's amendment, 2026-09-25 (an abstract
  `Placement.t` over a grid, cuts inside one tile, eager results over split
  operands, local reads, engines per backend, the withdrawal of §Fully sharded
  training, the stage 2 budget)
- Packages: nx (`Nx.Device`, `Nx.Placement`, `place`, `placement`, the tensor
  representation, routing and reads), rune (devices, the device engine, `jit`
  placement), tolk (ports), kaun (example importers)

## Summary

Where a tensor lives is a value nx can name. A device is a run-time value that
rune opens (`Rune.device "METAL"`) and that carries the engine holding memory
on it; the host is a device too, `Nx.Device.host`. A placement is an abstract
value built as one device, a list of devices holding full copies, or a list of
devices holding equal slices along one axis, and taken apart as the window
each device holds. `Nx.place p x` moves a value and leaves its source
where it was; `Nx.placement x` says where a value lives. The result of an
operation lives where its operands live, so eager code over values on a GPU
keeps its results there; a read copies the elements it reads and moves
nothing. A compiled function runs where its placed inputs and captures live,
and a function over split inputs is how raven runs data, tensor and expert
parallelism, so the global-view `pmap` goes. A compiled step reads one
argument and consumes another (`Rune.jit_step`), so many programs read one
set of placed weights while a cache is written in place, and `?donate` goes.
nx's link-time engine stays the host's engine; devices are values beside it,
and nothing moves from tolk into nx. The first stage ships all of this on one
device, then calls that return without waiting, and costs gpt-oss nothing;
several devices in one process follow the layer loop and a measured probe.

## Motivation

RFC 0001 made placement a run-time attribute of a value and reserved its
design. RFC 0003 shipped `Rune.to_device` for one device and listed what it
left out: sharded placement, a read that does not evict, a placement query,
moves between devices without the host, storage reuse per shard. The
maintainer asked the rest: if nx gets a CUDA or Metal backend, how do tensors
move and stay on the device, how does that compose with rune, tolk and kaun,
what belongs to a backend and what to nx proper, and does anything in tolk
belong in nx?

Today:

- **nx cannot hold a tensor on a device.** A resident value is a `Deferred`: a
  host tensor whose bytes have not arrived (`nx_effect.ml:21-45`). Its only
  concrete form is a host one, so every nx operation outside a compiled
  function reads it to the host, views included, and the read releases the
  device buffer. `Nx.item` on one element of a placed 10 GB tensor copies
  10 GB.
- **Moving evicts.** Placing a value resident on one device onto another reads
  it through the host, which releases the source (`jit.ml:2423,2645-2647`),
  and so does passing it to a compiled function on another device.
- **Sharded values have no constructor.** Only `pmap` outputs are sharded, and
  a resident capture of a `pmap` is read to the host and replicated: Llama 3.1
  70B at bfloat16, 141 GB, replicated on each of eight 80 GB GPUs does not fit.
- **Placement state is hidden.** It lives in rune's process-global table keyed
  by deferred id (`jit.ml:226`). Devices have two spellings, tolk's `METAL`
  and rune's `METAL:0`, and rune takes its default device from
  `Tolk_frontend.Run`, which can open the CPU device with aligned vector types
  before rune's opener is installed, against RFC 0003's Law 4.
- **Eager transformations read placed weights back.** Eager `grad` over a
  model placed on Metal copies every weight to the host (fixed on Metal in
  stage 1, on discrete GPUs in stage 3).
- **Donation is all or nothing.** `?donate:true` consumes every resident
  input leaf that no program binds (`jit.ml:3454-3458`), so a compiled block
  cannot read its weights and consume its cache. The layer-loop prototype
  spared weights by how they were made, placed with `to_device`, a rule that
  this RFC's removal of `to_device` takes away.

The forcing cases are gpt-oss-20b on one Metal device, which runs today, and
the goal posts that do not fit one device: Llama 3.1 70B tensor-parallel on
eight GPUs and fully sharded across two nodes, and DeepSeek V4, whose
reference splits experts, heads and vocabulary over one flat world of ranks
(`research/deepseek-prep/ref/v4_model.py:15-16,155-177,609-642`).

## Guide

### Devices and placements

```ocaml
let metal = Rune.device "METAL"            (* raises if absent *)
let gpus = Rune.devices "CUDA"             (* CUDA, CUDA:1, ..., CUDA:7 *)

let p1 = Nx.Placement.device metal
let tp = Nx.Placement.sharded ~axis:1 gpus  (* equal slices of axis 1, in order *)
let rp = Nx.Placement.replicated gpus       (* a full copy on each *)
```

A device has one name, tinygrad's: `"METAL"`, `"CUDA:3"`, never `":0"`.
`Nx.Device.host` is the host, named `"CPU"`; `Rune.device "CPU"` returns it,
and `Nx.Placement.host` is `Nx.Placement.device Nx.Device.host`. A placement
over one device is that device. A placement is abstract: `Nx.Placement.devices
p` lists its devices, `Nx.Placement.window p shape d` is the window, as
`(start, stop)` per axis, of a value of that shape that device `d` holds, and
two placements are equal when every device holds the same window.

### Placing and reading

```ocaml
let w = Nx.place p1 w_host                   (* w_host is unchanged *)
let y = Nx.tanh (Nx.matmul x w)              (* w is placed: y lives on Metal;
                                                x is uploaded for the operation *)
let s = Nx.item [] (Nx.sum y)                (* copies 4 bytes; y stays *)
let h = Nx.place Nx.Placement.host y         (* an explicit host copy *)
let _ = Nx.add y (Nx.place (Nx.Placement.device (Rune.device "CUDA")) y)
(* Invalid_argument: operands on METAL and CUDA; place one of them *)
```

The result of an operation lives where its placed operands live. A host
operand joins them. Operands on two different device sets raise. An
operation or dtype the device cannot take (float64 on Metal, an
eigendecomposition tolk cannot lower) raises at the operation and names the
device. A read (`item`, `to_array`, `pp`, a save) copies the elements of the
viewed window and leaves the value where it is. Storage is released when no
value reaches it, or when a compiled call it was donated to completes.

### Compiled functions

RFC 0006 replaces `jit_step` with one `jit` whose signature marks consumed
arguments with `consumes`; the examples keep this RFC's original spelling
(Reference, §Compiled functions).

```ocaml
(* gpt-oss-20b: the weights are placed as the importer builds them *)
let params = Gpt_oss.of_hf ~placement:(fun _ ~axis:_ -> p1) cfg Nx.bfloat16 ckpt
(* reads { token; index }; consumes { caches; next } and returns it, with the
   caches written in place and the sampled ids in [next] *)
let step = Rune.jit_step (module Input) (module State) (step cfg params)
let s = step { token; index } s
let ids = Nx.to_array s.next                   (* copies the ids; nothing moves *)
```

A compiled function runs where its placed inputs and captures live, or on
`?devices` if none is placed, or on the default device. Captures on its
devices are bound without copying (RFC 0003), split ones included. An input or
capture on other devices raises, naming the leaf and both placements.

`jit_step` reads its first argument and consumes its second. Weights, an
index and tokens are read and stay usable; the caches are donated, and each
output leaf takes the storage of the input leaf at its position. So several
programs (prefill buckets, decode, a block per layer kind, encoders, a
sampler) read one set of placed weights, captured or passed, without a copy,
and a block program of the layer loop reads its layer's weights as inputs:

```ocaml
(* one program per layer kind; the host loops over the layers *)
let compile kind = Rune.jit_step (module Block_in) (module Block_state) (Gpt_oss.block cfg kind)
let sliding = compile Gpt_oss.Sliding and full = compile Full
let { Block_state.cache; x } = sliding { b; index } { cache; x }
```

### More than one device

A function over split inputs is parallel, and nothing else names a device:

```ocaml
(* data parallelism: the batch split; the parameters start on the host, and
   the loss is a field of the state *)
let step = Rune.jit_step (module Batch) (module State) train_step
let state = step (Nx.place (Nx.Placement.sharded ~axis:0 gpus) batch) state

(* Llama 3.1 70B, tensor-parallel: the importer and the cache builder pass
   the axis each cut runs along in the tensor at hand *)
let placement role ~axis = match role with
  | Llama.Whole -> Nx.Placement.replicated gpus
  | Column | Row | Kv_heads -> Nx.Placement.sharded ~axis gpus
let params = Llama.of_hf ~placement cfg Nx.bfloat16 ckpt
let cache = Llama.cache ~placement cfg ~slots Nx.bfloat16   (* pools split on kv heads *)
```

The function sees global shapes, and a reduction over a split axis becomes an
allreduce, which tolk already emits. Replicated parameters come from the first
call's upload of host values and feed back placed, and the second call runs
the program the first compiled.

Expert parallelism maps over an expert axis split one slice per device:
`vmap` over that axis runs each lane on its own device, and the sum over it is
the allreduce.

```ocaml
(* DeepSeek V4: the importer stacks the 256 experts as
   [devices; per_device; ...] and splits axis 0 over the devices *)
let moe experts x ids gate =
  let per_device = 256 / List.length gpus in
  let lanes =
    Rune.vmap (module Args) (fun { experts; first } ->
        let local = Nx.sub ids first in
        let mine = Nx.logical_and (Nx.greater_equal_s local 0l)
            (Nx.less_s local (Int32.of_int per_device)) in
        (* Moe.apply calls Nx_quant.apply ~ids; an id outside [0, e) selects
           no expert: its product is zero and reads nothing (RFC 0004) *)
        Moe.apply experts x (Nx.where mine local (Nx.scalar_like local (-1l))) gate)
      { experts; first = Nx.arange Nx.int32 0 256 per_device }
  in
  Nx.sum ~axes:[ 0 ] lanes                    (* the reference's all_reduce *)
```

vmap puts the lane axis at the front of every part and of the ids, so each
lane gathers only its own experts with its own ids (RFC 0004's pairing), and a
route to another device's expert contributes zero and reads nothing. Each
device reads only the experts its tokens chose on it and multiplies only its
own routes, in decode and in prefill: a block of RFC 0004's block kernel,
`Op.block_matmul`, whose id is −1 runs no multiply-adds. The gate weights the
combine unmasked. Passing `local` as it is would select the same experts,
since an id outside `[0, e)` addresses nothing on every path (RFC 0004, as RFC
0002's `-1`); the `where` makes the intent visible.

## Reference

### nx: devices and placements

```ocaml
module Device : sig
  type t
  val host : t                                (* "CPU" *)
  val name : t -> string
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
  exception Out_of_memory of t * int          (* device, bytes requested *)
end

module Placement : sig
  type t                                      (* abstract *)
  val host : t                  (* device Device.host, the default backend *)
  val device : ?backend:Backend.t -> Device.t -> t
  val replicated : ?backend:Backend.t -> Device.t list -> t
  val sharded : ?backend:Backend.t -> axis:int -> Device.t list -> t
  val devices : t -> Device.t list
  val backend : t -> Backend.t
  val window : t -> int array -> Device.t -> (int * int) array
  (* [window p shape d]: the window of a value of [shape] that [d] holds, as
     [(start, stop)] per axis, stop exclusive, as [shrink] takes it; raises
     [Invalid_argument] if [d] is not in [devices p] *)
  val equal : t -> t -> bool   (* the same backend and window on every device *)
  val pp : Format.formatter -> t -> unit
end

val place : Placement.t -> ('a, 'b) t -> ('a, 'b) t
val placement : ('a, 'b) t -> Placement.t
```

- **Abstract, over a grid.** Callers build a placement with its four
  constructors and take it apart with `devices`, `backend`, `window` and
  `equal`; nothing outside `nx.effect` matches on its representation. Inside
  `nx.effect` a placement is one device or a grid: distinct devices in
  row-major order over the grid's extents, and for each cut tensor axis the
  grid axes that cut it, a grid axis that cuts nothing holding copies. Only
  one-dimensional grids have constructors: `replicated ds` is the grid over
  `ds` with no cut, and `sharded ~axis ds` the same grid cutting `axis`. A
  mesh constructor would add grids of more dimensions and change no caller,
  since no caller matches. Per-shard code in nx and rune takes a device's
  window from `window`, never from `shape.(axis) / n`.
- **A placement carries one backend.** `device`, `replicated` and `sharded`
  take `?backend`, which defaults to host compute, the host's kernels, which
  is stage 1's routing for devices with no backend of their own; another
  backend on the host is `device ~backend Nx.Device.host`. RFC 0007 defines
  devices as hardware and backends as values; RFC 0008 defines
  `Nx.Backend.t`'s shape. Operands on two different device sets, or with two
  different backends, raise under the mixed-placements rule (Routing).
- **Normal forms by construction.** `replicated [d]` and `sharded ~axis
  [d]` are `device d`, and a grid drops extents of 1 and merges adjacent grid
  axes when no cut names either or one cut names both in order, so a grid
  used flatly is the flat grid. An empty list, a repeated device, devices of
  two engines or a negative axis raise `Invalid_argument`. `place` raises when
  a cut does not divide its axis evenly. So `placement (place p x)` equals
  `p`.
- **Equality is the device-to-window map.** `equal p q` holds when `p` and
  `q` have the same backend and every device holds the same window under
  both, for every shape; it takes
  time proportional to the devices times the cut axes. A node × GPU grid
  cutting axis 0 over both of its axes, which no constructor builds yet,
  would equal `sharded ~axis:0` over the same 16 devices in the same order,
  and code written against the flat split would serve it unchanged.
- **Identity.** `Device.equal` is identity of the value, and rune returns one
  value per canonical name. A placement's devices are an ordered list
  (`devices p`), which decides which window lands where. All devices of a
  placement share one engine.
- **`placement` performs an effect,** as `view` does, so `vmap` answers for a
  batched value with the split axis shifted into the unbatched view. It
  raises when the mapped axis is the split axis: a lane of a map over devices
  has no single placement. Inside a compiled program on one device list,
  a traced value's placement is the program's; over several devices it is
  where nx's rules put the value as the function traces, which the program
  keeps (a result that tolk's rewrite places elsewhere raises at compile
  time).
- **`place` performs `E_place { placement; t_in }`,** which replaces
  `E_to_device` (whose `context` field no handler reads). Under `grad` and
  `jvp` it is linear, and a cotangent is placed back at its primal's
  placement; under `vmap` a split axis shifts past the batch axis; under `jit`
  it is the identity when its target is the program's placement, lowers to
  tolk's copy and shard nodes from stage 2, and raises `Jit_error` otherwise
  until then.
- **nx opens no device and keeps no registry.** Devices come from the library
  that owns runtimes: `Rune.device`, `Rune.devices`, `Rune.default_device`.
  `Rune.devices "CPU"` is `[Nx.Device.host]`; `"CPU:1"`, `"CPU:2"` are
  devices with storage of their own, opened by name, for tests of placement.
  rune gains `val device : string -> Nx.Device.t`, `val devices : string ->
  Nx.Device.t list` and `val default_device : unit -> Nx.Device.t`, and
  `?device : string` becomes `?devices : Nx.Device.t list` on `jit`, `jit2`
  and `jit'`, which lose `?donate`; `Rune.jit_step` (Compiled functions) is
  the call that donates.

### nx: the tensor representation

The engine-facing side, in `nx.effect`, replaces `Deferred` and `Symbolic`:

```ocaml
type ('a, 'b) t =
  | Host : ('a, 'b) Nx_backend.t -> ('a, 'b) t        (* the link-time engine's *)
  | Placed : ('a, 'b) resident -> ('a, 'b) t           (* a device engine's *)
  | Traced : ('a, 'b) traced -> ('a, 'b) t            (* a node of a trace *)
and ('a, 'b) resident = {
  r_id : int;                     (* fresh per value; identity tables key by it *)
  r_placement : Placement.t;      (* never the host *)
  r_dtype : ('a, 'b) Dtype.t;
  r_view : View.t;                (* per shard, the same on every shard *)
  r_cell : cell;                  (* one per storage, shared by all its views *)
}
and cell = {
  engine : engine;
  length : int;                   (* elements per shard *)
  mutable state : state;
  mutable bound : int;            (* reachable programs that bind the storage *)
}
and state = Live of storage | Donated
and ('a, 'b) traced = {
  t_id : int; t_context : context; t_dtype : ('a, 'b) Dtype.t; t_view : View.t;
  t_node : node;
}
and engine = {
  read : 'a 'b. ('a, 'b) resident -> ('a, 'b) Nx_buffer.t;  (* the view's elements *)
  place : 'a 'b. Placement.t -> ('a, 'b) t -> ('a, 'b) t;
}
and storage = ..                  (* rune adds tolk buffers *)
and node = ..                     (* rune adds its trace's id and the tolk tensor *)
```

- **`Host` and `Placed` differ by how their engine is bound.** The link-time
  engine is known statically, so its tensor type is concrete and the C backend
  takes it directly; device engines are values, so their storage is opaque to
  nx. Merging them would either make every host operation an indirect call
  over untyped storage, or put device memory inside `Nx_backend.t`. The
  constructor is `Placed` because it holds values on any device list but the
  host's, split and replicated ones included; `Device` names the module of
  devices and the placement on one device.
- **`Traced` carries its trace's payload.** jit's placeholders today are
  `Symbolic` values found through side tables keyed by physical identity
  (`st.table`, `st.traced` in `jit.ml`), with a fresh `s_id` only to keep
  their structural hashes apart. The payload replaces the placeholder
  tables (`st.traced`, `input_index`, and the placeholder entries of
  `st.table`); captures keep their table. The trace id in it keeps an inner
  jit from consuming an outer trace's value, and a traced value used after
  its trace raises; inside a trace the error names the trace, and in eager
  code it does not.
- **Views share a cell.** A view of a placed value is view arithmetic and
  allocates nothing (RFC 0001 Law 5). Donation, binding and the finaliser
  belong to the cell, so every view of a donated storage raises on read and on
  use, a bound storage is never donated, and storage is retired only when no
  view reaches it. Only a value whose view covers its whole storage can be
  donated; donating any other raises, naming the leaf. A cell is bound while
  a program that binds it is reachable: binding counts, and a compiled
  function's finaliser releases its count, so a weight captured by a program
  since dropped can be donated again.
- **Split values.** A split value's `r_view` is its per-shard view, and `view`
  answers it with the split axis's extent multiplied by the number of devices,
  so `Nx.shape` is global (Law 1); the strides and offset are each shard's, and
  `contiguous` and donation test the per-shard view against each shard's
  storage. Movement keeps the split axis when it survives whole: `permute`
  moves it, `expand` keeps it at its index, `reshape` keeps it when every
  shard boundary survives (tolk's `reshape_multi` rule), and `shrink`, `pad`,
  `flip` and `sliding_window` apply per shard when they leave it whole. A cut
  that stays inside one tile is a view of that tile's storage, placed on the
  devices that hold the tile and sharing the cell: on a one-dimensional split,
  the one device holding that shard. It copies nothing, so `Nx.item` or
  `Nx.get` on a split value reads one element; it does not cover its storage,
  so consuming it raises by RFC 0006's rule 4, naming the leaf and `Nx.copy`.
  Any other cut of the split axis (a `shrink` spanning tiles, a `pad`, `flip`
  or `sliding_window` along it, `cat` along it, a reshape that moves elements
  between shards) raises `Invalid_argument` naming the operation, the split
  axis and the value's shape: place the value replicated or on one device
  first. These are the rules of tolk's multi-device rewrite
  (`schedule/multi.ml:242-318`, over its `Unshard` node), except for cuts
  within one shard. There, tolk copies a cut of exactly one whole shard to
  every device of the list (`schedule/multi.ml:201-205`) and raises on a cut
  strictly inside one shard (`:206`), `Nx.item`'s included. A program over
  several devices cannot hold a value on one of them, so the cut inside one
  shard is where eager and compiled placements differ. From stage 2, rune checks split movements at trace
  time: a cut that nx refuses, or one strictly inside one shard, raises nx's
  message there, naming the operation, the split axis and the shape (for the
  second, the remedy is to place the value on one device first), and a
  whole-shard cut keeps tolk's copy.
- **Contexts.** `type context = Host of Nx_backend.context | On of Device.t
  list`. The frontend builds constants in its operand's context, at 81 sites,
  so a value created in `On ds` (a constant, a buffer, a host array, an index)
  is replicated over `ds`. A scalar created in a device context is a placed
  value whose storage is the scalar itself: it allocates nothing, and the
  engine passes it to a program as a run-time argument. A one-element result
  is held the same way, by nx, which is what lets `item` on it move 4 bytes.
  No operand value enters a program or its cache key.
- **Identity tables.** rune's `Tensor_map` and jit's constant and capture
  tables key a placed value by its `id` and never force it;
  `Tensor_map.stable` is deleted. That is the one change the transformations
  need.

### Routing

Every fallback in `nx_effect` routes by its operands, and creation effects by
their context: all on the host, the link-time engine; placed operands on one
device set, that set's engine; anything else raises. Routing compares device
sets, so one list in two orders routes as one. Whole-shard views of one value
join as copies on the value's full device list, as compiled code places them:
the sum of two whole shards of a value split over four devices lands as copies
on all four, as `pmap` does. A roll by one shard is no such combination, since
one of its pieces spans shards, and it raises, eagerly and compiled. Because
every transformation re-performs its operations into that fallback, eager
`grad`, `jvp` and `vmap` over placed values keep their results placed, with no
change to their rules.

Over split operands, an eager result takes the placement tolk's multi-device
rewrite gives the same operation in a program (`schedule/multi.ml:128-171`),
so eager and compiled placements agree on every operation but a cut inside
one shard (Split values), and stage 3 changes none of them. An
elementwise operation keeps its operands' split; a replicated or host operand
takes that split; operands split differently raise `Invalid_argument` ("place
them alike first"), as compiled code does from M3. A reduction over
the split axis gives a result replicated over the list, and a reduction over
other axes keeps the split. Movement follows Split values. Operands whose
placements have different device sets raise, as above. A take reads its
table along its axis: a table split there raises, and the result takes its
positions' split.

Until devices compute (stage 3), the engine runs an operation on the host: it
reads its placed operands' windows, runs the host engine, and places the
result. Each operand window is copied once, on Metal too: the engine has one
`read` for user reads and for routing, and a borrowed read would alias memory
that a later donation overwrites. On Metal that one copy reads the shared
buffer directly, through a port of tinygrad's `_as_buffer`
(`runtime/ops_metal.py:189`); going through `copyout` would copy twice (a
256 MiB placed read takes 14.6-19.5 ms through `_as_buffer` and 29.4-34.1 ms
through `copyout`, alternating runs of two otherwise identical executables).
Stage 3 replaces this path. The operation raises
first when the placement cannot hold the result's dtype or the device's
compiler refuses the operation, so stage 3 changes speed and nothing else.

### Reads

The frontend's element readers (`to_array`, `to_buffer`, `item`, `pp`,
`map_item`, `iter_item`, `fold_item`, boolean masks, `nonzero`, `argwhere`,
the scalar-reading linear-algebra helpers, saves) first take `contiguous`,
which returns a placed value itself only when its view covers its whole
storage (C order, offset 0, `numel` elements) and otherwise copies the window
into fresh storage on the same placement. They then read that storage with the
backend's `to_host`, and `item` and `pp` index it from 0. One element of a
placed 10 GB tensor copies one element. Reads are local and moves may
communicate: a read copies from devices this process holds and runs no
collective, and `place` is the call that moves data between devices. A read of
a split value gathers the shards in global order; a replicated value reads one
replica held by this process, and a tile held by several devices is read from
one of them. Replicas are meant to be equal bit for bit: tolk's allreduce
either reduces each chunk on one device and copies it to the others, or folds
the shards in the same order on every device (`schedule/allreduce.ml`), and
the second agrees only where every device runs the same compiled fold. Stage
2's probe found replicas equal under every strategy tolk picks by default, on
two to eight devices; the hierarchical strategy folded in a different order on
different devices at three or more boxes, and now folds in one order
everywhere (a tolk divergence). The probe ran float32 on `CPU:k` devices; the
half-precision path (`ALLREDUCE_CAST`) and the rented node's GPUs are not yet
measured. A compiled call's output covers its storage, so reading an engine's
sampled ids copies the ids and nothing else. Nothing is memoised. `Nx.data` of
a placed value raises `Invalid_argument` ("a placed value has no host storage;
read it with to_buffer, or place it on the host"), because its contract is the
storage that `offset` and `strides` index; `Nx.data` checks first, so readers
inside the frontend call `B.to_host`, not `data`. Host-only consumers take
their input to the host once at their boundary: talon's column constructors,
hugin's data preparation, kaun's `Metric`.

### Moving

| Source | `place p x` |
|---|---|
| already at `p` | `x` |
| the host, or a mapped file | RFC 0003's chunked upload; under a split placement each device reads only its own slice of the file |
| a device of the same engine | the view's window, then tolk's `transfer` for each shard that changes device, or a host bounce where the allocator has none (the CPU device); a strided window bounces through the host in rune until stage 3 compiles a contiguous copy on its source device |
| another engine | a host bounce, in chunks |
| the host, from a device | a read of the whole value, gathering shards |

The source is never released. An empty value is placed and allocates nothing.
Placing a host value on the host returns it; on `CPU:1` it gets
storage of its own. Storage that wraps memory the engine did not allocate (a
mapped file, host memory) is never written: donating such a value consumes it
but never lends its storage to an output. A value an engine uploads once per
step and many programs read, such as a layer loop's index, is placed once per
step; its buffer comes from the allocator cache like any other.

### Lifetime and donation

Storage is released when its cell is unreachable or donated, and only after
every submission that reads or writes it has completed: an engine records per
cell the last submission that uses it, and a submission keeps reachable every
host buffer it wraps until it completes. Donation state lives on the cell, and
`Rune.jit_step` is the only call that donates (Compiled functions). Bound
captures keep RFC 0003's Laws 8 and 9, and its Law 10 reads: an upload from a
mapped file bypasses the allocator cache, so a dropped model does not stay
allocated in it; every other allocation goes through the cache, which tolk
empties before an allocation fails (`device.ml:48-56`). An allocation that
still fails after the engine drains its queue, collects and retries once
raises `Nx.Device.Out_of_memory`; a call that raises has consumed no donated
input. The collection budget counts every device allocation since the last
major collection, eager results and per-step uploads included, 4 GiB by
default; placing Llama 3.1 70B then runs about 35 collections.

### Compiled functions

RFC 0006's amendment replaces `jit_step`, its positional pairing and
`?donate` with one `jit` over a signature, and where this section differs,
RFC 0006's rules govern. Where this section says a leaf of `'s` or donation,
read a leaf of a `consumes` argument. Three of its sentences change:

- a cell reached from two leaves, or from a leaf and a capture, makes the
  call raise (its rule 4), where this section reads it;
- a consumed host leaf is dead like any other (its Law 6, once host cells
  land), where this section keeps it usable;
- a result keeps the placement of the consumed input it derives from (its
  rule 5), where this section pairs by position.

Lending follows its rule 5 per shard (equal dtype, byte size and placement),
results are fresh (its rule 2), and the key is its rule 1's with this
section's placement and view clauses, a host leaf counting as replicated over
the program's devices.

```ocaml
val Rune.jit_step :
  ?devices:Nx.Device.t list ->
  (module Nx.Ptree.S with type t = 'r) ->     (* read, never consumed *)
  (module Nx.Ptree.S with type t = 's) ->     (* donated and returned *)
  ('r -> 's -> 's) -> 'r -> 's -> 's
```

A step compiled with `jit_step (module R) (module S) f` reads its first
argument and consumes its second.
Leaves of `'r` are never consumed: weights, an index or tokens read by many
programs stay usable. Leaves of `'s` are donated, RFC 0001's Law 3 counted by
cell: after the call each raises on read and on use, and the output leaf at
the same position takes its storage when its dtype and size match and no other
leaf of the call reaches its cell. Pairing by position needs no alias search.
A cell reached from both arguments, or bound by a program, is read: it lends
nothing and stays usable. A host leaf of `'s` is uploaded and stays usable,
since donation acts on cells. Outputs that are not state, such as sampled ids
or a loss, are fields of `'s`, as RFC 0002's step threads its tokens; a step
that reads nothing passes `(module Nx.Ptree)` and `Nx.Ptree.list []`. `jit`,
`jit2` and `jit'` never donate, and `?donate` is removed.

A program runs on the one device list its placed inputs and captures share
(axes may differ); captures are found at the first trace, which runs again
once on their devices when they decide. With none placed it runs on
`?devices : Nx.Device.t list`, else on the default device. A host leaf joins
the program's devices: it is uploaded for the call and replicated over the
list, and a host capture is uploaded once per compiled function (RFC 0003). A
`?devices`, or a leaf or capture placed on other devices, raises, naming the
leaf. Only a split value decides the order of the program's devices, which
decides where each slice lands: the first split leaf's order, else a split
capture's. Values that are copies match the devices as a set. A `?devices`
list fixes the order, and a split value in another order raises.

An output leaf of `'s` has the placement of the input leaf at its position.
When the compiler's placement differs, the program reshards at its end and
`RUNE_JIT_DEBUG` reports it, so donated state keeps its placement from call to
call and its storage can be reused. A state leaf built on the host enters
replicated over the program's devices and returns there, which for a split
cache is a full copy on each device: a model places its cache before the first
call (kaun). Other outputs take the placement the compiler gives them.

The cache key holds each leaf's placement in the program, a host leaf counting
as replicated over the program's devices (on one device, that device), and a
placed leaf's view (its strides; it binds the range the view reaches) only
when that view is not C order over its whole storage. A host value and the placed
value a call returns for it therefore share a program, and a loop whose state
starts on the host compiles once. A placed leaf or capture binds its storage
from the view's offset, as a tolk buffer view, and its strides are applied in
the program as movement, with no copy; two views that differ only in offset
share a program. If tolk cannot bind a buffer view at an offset (stage 1's
stop condition), the offset joins the key instead; that is the case for a
byte offset not aligned to 16, which tolk refuses and CUDA's vector loads
fault on.

Split leaves lower to a buffer on the device list under tolk's `Unshard`
node, which records each cut axis and its range over the devices; rune seeds
those ranges from the placement's grid and reads each output's placement back
from its ranges, never from `Uop.axis`, which raises on more than one cut
(`uop.ml:2420-2422`). The lowering applies, per shard, what today's
single-device lowering applies: storage reuse under donation, indexed scatter
and window writes, and staged scans. Today's `pmap` has none
of them: it reuses no storage (`jit.ml:2922`), writes by a one-hot scatter
over the whole destination (`jit.ml:1165-1181,1210-1212`), and unrolls scans
(`jit.ml:1136-1152`); a tensor-parallel decode step lowered that way holds two
copies of its cache and rewrites every row of every pool each step. Inside
`jit` over split inputs, `Nx.Rng.fold_in_axis` folds nothing: a global-view
program has one lane, so its value does not depend on how many devices hold
it. The per-device-index tests under `pmap` move to `vmap` over a split
axis, where `fold_in_axis` folds the lane index. `E_place` under `jit`
is the identity at the value's own placement, lowers to tolk's copy and shard
nodes for another placement over the program's devices, and raises
`Jit_error` for any other target.

A call waits for its work before it returns until the second half of stage 1,
as today. From then it returns once its work is submitted: a read waits for the
work that produces it (Law 7), a write into a buffer waits for that buffer's
last submission and never for the whole device, and donated storage is
released in queue order (Law 4). The layer loop loses 10 ms of a 132 ms
gpt-oss decode step to the wait at 26 calls, measured in its design pass.
Metal's upload synchronises the device before it copies
(`tolk_metal.ml:233-235`, as tinygrad's does), so once calls return without
waiting, each per-step upload still drains the queue once, and the host
cannot prepare the next step while the device runs this one. How much of the
10 ms that drain keeps is measured first. If it matters, the upload becomes a
copy ordered on the queue: tolk's Metal allocator records each buffer's last
command buffer and waits on that alone, a divergence with its own
`DIVERGENCES.md` entry. tolk's AMD and NV allocators returned a buffer to the
driver without waiting for the work that uses it, where tinygrad's
synchronises every device that maps the buffer first
(`runtime/support/hcq.py:567-568`); stage 1 restored that wait
(`tolk_amd.ml:2170-2173`, `tolk_nv.ml:2394-2397`).

### `pmap` and expert parallelism

`pmap` is `jit` over split inputs and is removed, with `in_axes`, in stage
2's M3, which brings `jit` over device lists: every caller migrates to `jit`
over placed values in the same change, with no bridge. Until then it keeps its
meaning. A function that must
act per device maps over an axis split one slice per device, as the Guide's
expert example does: `vmap` over it runs each lane on its own device,
`Nx.arange` over it gives each lane's index, `Nx.sum` over it is an
allreduce. Stacked `[devices; per_device; ...]`, experts are gathered
along the local axis, inside each shard, and a route marked with an id outside
`[0, e)` contributes zero and reads nothing (RFC 0004). Over experts split on
their leading axis, RFC 0004's product runs once per shard, built from shard
shapes, with no gather across shards, and its result is split on the lane
axis; any other layout falls back to decode-then-matmul, which
`RUNE_JIT_DEBUG` reports and which saves nothing on routes that select no
expert. RFC 0004 designs that per-shard product for the stage that brings
`jit` over device lists; stage 2's probe measured it through tolk (below),
and that stage measures it through rune. On its kernel and
grouped paths a foreign route reads nothing and multiplies nothing. When the
product groups routes by expert, a device reads each distinct local expert its
tokens chose once per block of its routes. It multiplies its own routes plus
at most `per_device · B` rows of padding, and the blocks that no local route
fills run no multiply-adds (RFC 0004's `Op.block_matmul`). Its own routes are
an eighth of the step's on average at eight devices. The lane's static bound
still counts every route of the step, so the ranking, the gathers into and out
of blocks, and the zeros of empty blocks follow the step. At a 4096-token
prefill on DeepSeek V4-Flash at eight devices (hidden width 4096 and expert
width 2048, which give its 13.4 MB per expert), a device fills about 64 of its
416 blocks of 64 rows. It multiplies about 4,100 rows per layer for its 3,072
routes, where it would multiply 26,600 with tolk's matmul over the gathered
matrices, which is 8.8 TFLOP of expert arithmetic per prefill against 57.8.
The ranking, the row gathers and the empty blocks' zeros move about 0.65 GB
per layer (derived). On DeepSeek V4-Flash at eight devices (13.4 MB per
expert, 43 layers, derived from its config), a one-token decode step reads at
most 3.5 GB on a device and 0.43 GB on average. At `T` tokens a device reads
the distinct local experts among `6T` routes, about `32 (1 - (255/256)^(6T))`
of its 32 per layer: 14 GB per step at 64 tokens, rising toward the 18.4 GB of
all 32 as the batch grows. A gather along the split axis itself would reduce
gathered weights across devices, which the stacking avoids. Whether the
example compiles to the per-shard product is measured in stage 2; if tolk's
rewrite gathers across shards, a per-device map with an explicit lane comes
back as its own RFC. Stage 2's probe measured it through tolk on
`CPU:1`..`CPU:4`, with a block kernel built from shard shapes: the
lane-stacked gather, the block product and the whole grouped form move no
expert bytes between devices, and only the output's lane sum crosses them, so
the per-device map is not needed. `Op.block_matmul` takes its shapes from the
global tensor and raises at graph build on split operands today; stage 2
builds it, and `Op.quant_matmul`, from each shard's shape.

### Fully sharded training

Parameters, gradients and optimiser state split on axis 0, and each weight
gathered where it is used with `Nx.place (Nx.Placement.replicated gpus) w`
inside the step, which tolk lowers today as an allreduce of zero-padded
shards (`schedule/multi.ml:105-113`), twice an all-gather's bytes. Without
that gather, tolk's
rewrite reshards to the last split axis among an operation's operands
(`schedule/multi.ml:139-143`): for a split batch times a weight split on its
input axis, it gathers the batch. The memory saving needs three things of the
execution: each layer's gathered weights freed after its use, gradients
reduce-scattered rather than allreduced then sliced, and the next layer's
gather overlapped with the current layer's compute. The first comes from
program boundaries: compiling one program per layer (the layer-loop design)
frees a layer's gathers when its program ends. The second is a tolk rewrite,
and the third needs asynchronous calls across programs. The probe runs a
fully sharded step in the layer-loop form, one program per layer, on
`CPU:1`..`CPU:4` before stage 2, so the layer loop comes first. If its
per-device peak exceeds (P + G + O)/n plus twice the largest layer,
placement with program boundaries does not express fully sharded training:
this RFC's amendment is withdrawn, and the roadmap's position that it is a
rune transformation stands.

**Withdrawn** (stage 2's probe, 2026-09-25). The probe ran eight layers of
`relu (h @ W)`, with 4 MiB weights, parameters, gradients and Adam's moments
split over `CPU:1`..`CPU:4`, and one tolk realize per layer. The worst
device's peak over its share of the state was 2.66 layers at best, above the
bound of 2.14 (two layers plus 0.14 of activations). It was 3.66 under tolk's
default lowering, 8.16 under the naive one, and 2.89 with both collectives
written out by hand. Three causes: a gather to a device list lowers to an
allreduce of zero-padded shards, which moves twice an all-gather's bytes;
gradients are allreduced and then sliced, since there is no reduce-scatter;
and every copy source is staged in its own buffer. An all-gather into slices,
a reduce-scatter rewrite, copies from views and the backward in two programs
per layer would each address one cause. None is built, and they were not
measured together. They are tolk collectives work, planned as its own stream
(all-gather, reduce-scatter, copies from views, hierarchical variants) for
the two-node training gate. Fully sharded training returns through placement
and `jit` if this probe, re-run after that stream and stage 2's M4, holds the
bound; until then this section stays withdrawn. Its throughput gate (90% of
torch) further needs gathers overlapped with compute, without which it
reaches about 88%.

### Engines, and what belongs where

| Layer | Owns |
|---|---|
| nx proper | `Device`, `Placement`, `place`, `placement`, the tensor representation, routing, reads, donation state |
| host engine (`nx.backend`: nx.c, nx-oxcaml) | host memory and kernels; one per executable, unchanged |
| device engine (a value) | storage, uploads, reads, transfers, and from stage 3 one-operation programs |
| rune | `device`, `devices`, `default_device`, opener registration; the engine over tolk; `jit` and `jit_step`; binding and budgets |
| tolk | runtimes, buffers, allocators, transfers, graphs, `Unshard` and its rewrite, allreduce; `Tolk_frontend` and `Tolk_nn` |

The engine record is in the representation above: `read` and `place`. The
engine attaches finalisers to the cells it creates. rune makes one engine
value per tolk backend, shared by that backend's devices, so a placement over
devices of two backends (`[METAL; CPU:1]`) raises by the rule that a
placement's devices share one engine, with no check of its own; Metal, with
one device, never enters a list of several. The engine moves split and
replicated values shard by shard: chunked uploads from the host, a mapped file
giving each device only its window, tolk's transfer between devices of its
backend (a host bounce on `CPU:k`), a chunked host bounce between backends
written in tolk, and `Out_of_memory` naming the device that failed. It gains
no field for this; `place` carries every move. Transfer code lives in tolk,
with one exception: a strided device-to-device copy bounces through the host
in rune until stage 3 compiles a contiguous copy on the source device. Stage 3
adds `run : 'r. 'r Effect.t -> 'r`, with one function over nx's effect
vocabulary that lists an effect's tensor operands; an engine raises
`Invalid_argument` naming any effect it does not implement, and nx's
conformance tests run against every engine. `Device.make` lives in
`nx.effect`.

**An nx CUDA or Metal backend is a device engine,** not another implementation
of the virtual `nx.backend`. The link-time seam allows one engine per
executable, and a process on a GPU still needs host tensors for its tokenizer,
its sampler and its data. From stage 3 the device engine executes an operation
as a cached one-operation compiled program: rune's lowering from nx effects to
tolk, keyed by the effect, its static arguments and its operands' dtypes,
shapes, strides and placements, compiled with tolk's heuristics only (`BEAM`
never applies). A vendor-library engine implements the same record in its own
library.

**The `create_context` TODO is deleted.** The host engine needs no parameters,
and a device index is part of a device's name. The docstrings that fence
devices out of `nx.backend` become "devices are engines carried by values;
this seam selects the host engine".

**Nothing moves from tolk into nx.** Runtimes, buffers, allocators, transfers,
graphs, `Unshard` and allreduce are ports of tinygrad's `device.py`, `runtime/`
and `schedule/`; nx would link drivers and a compiler to hold memory it has no
kernels for. `Tolk_frontend` is rune's lowering vocabulary, and `Tolk_nn` a
port used by tolk's tests. What moves goes the other way: rune takes its
device policy (`DEV`, probing, opener registration at initialisation) out of
`Tolk_frontend.Run`, and residency moves from rune's table into nx's values.
tolk's own work is porting its copy paths from tinygrad (transfers that name
both devices, CUDA peer access with a cross-context event, Metal's
`_as_buffer`, multi-device graphs), plus chunked host bounces as one
divergence entry; tolk's TODO tracks them.

### kaun

Example importers take `?placement : role -> axis:int -> Nx.Placement.t`,
where `role` is each example's closed variant of cuts (for example `Column |
Row | Whole`, plus `Experts` where a model has them), and the importer passes
the axis that cut runs along in the tensor at hand; one device is
`fun _ ~axis:_ -> p1`. A split leaf uploads only each device's window of the
file. An example's cache builder takes the same function, with a `Kv_heads`
role for the axis a tensor-parallel cache splits, and places each pool it
builds, so a split cache is on its devices before a step first reads it; the
engine that calls the step knows neither the split nor the format.
`Checkpoint` stays device-free. A packed quantised weight (RFC 0004) is
placed with `Nx_quant.place p`, which places every part with `p` and checks
that a split falls on the format's blocks. `Nx_quant.map` stays a traversal
that checks shapes only, since jit builds its placeholders with it and a
traced value has no placement. A weight stored as many entries, such as
DeepSeek's per-expert tensors, is stacked on the host one layer at a time and
then placed: a transient of one layer's experts, 3.4 GB for V4-Flash.

### Order of work

1. **One device.**
   - nx: `Device` with `host`, `Placement` (all constructors, one device used),
     `place`, `placement`, the tensor representation, routing with the host
     engine and `_as_buffer` on Metal for its one copy, reads, views and cells, `E_place` with
     its single-device rules.
   - rune: devices and their policy at initialisation, one spelling, the
     engine's `read` and `place`, `jit`'s rule and key for one device, binding
     placed views, `?device` becoming `?devices`, `jit_step` with `?donate`
     removed from `jit`, `jit2` and `jit'`, `Rune.to_device` removed, each
     with every caller; the host-only consumers' boundaries.
   - kaun: the importers and cache builders.
   - Acceptance on the M1 Max, against a baseline recorded on the same build:
     gpt-oss-20b keeps its 228 checks, its decode median within 2% and its
     peak within 0.2 GB; `Nx.item` on resident logits moves at most 4 bytes and
     leaves them resident; a move from Metal to the host keeps its source;
     mixed placements raise; a loop whose state starts on the host compiles
     once; a `jit_step` block program called for each of gpt-oss's 24 layers
     reads its weights and index and reports storage reuse for every cache
     leaf; no host row of nx's benchmark slows by more than 3%. Stop at 35
     engineer-days or 3,000 changed source lines (re-derived before the stage
     starts, since the `Traced` representation and `jit_step` were not
     priced), or if a contiguous window cannot seed a call without a copy;
     then ship the fixes that keep residency in rune (moves that keep their
     source, one spelling, the opener fix) and return this RFC to discussion.
   - Then, still on one device, calls that return without waiting: completion
     per cell, donated storage released in queue order, and tolk's AMD and NV
     frees waiting as tinygrad's do (Compiled functions). The upload's drain
     is measured, and Metal uploads are ordered on the queue only if that
     measurement shows it matters. Acceptance: the gpt-oss layer loop's decode
     median within 2% of the pre-stage baseline build, with identical ids.
2. **Several devices in one process,** after the layer loop and the probe on
   `CPU:1`..`CPU:4`: a tensor-parallel MLP (column then row, captured split
   weights) equals one device, uploads only its batch per call (once per
   device, `ndev` times the batch's bytes), and moves at most 1.1 times the
   allreduce volume between devices; the expert example, equal to one device
   on `CPU:1`..`CPU:4`, where each device reads only the local experts its
   tokens chose, and on the rented node, where each device multiplies only its
   own routes and the grouping's padding: its expert product at DeepSeek
   V4-Flash's shapes with random weights within 1.5 times the same product
   over its own routes' positions (RFC 0004's one-lane row); and a
   tensor-parallel decode step over an RFC 0002 cache split on its kv-heads
   axis, which reports storage reuse for every pool on every device and writes
   bytes proportional to the call's tokens. Then: the abstract placement over
   a grid, replicated and split placements, split `place` from the host, eager
   results over split operands (Routing), bound split captures, `jit` over
   device lists with storage reuse, indexed writes and staged scans per shard,
   and the removal of `pmap` with `jit` over device lists (M3), with every
   caller migrated to `jit` over placed values and no bridge. Acceptance: the
   probe, including replicas of an allreduced value equal bit for bit on
   every device, then Llama 3.1 70B tensor-parallel decode with its paged
   cache on a rented node of eight 80 GB GPUs (tolk.cuda). Budget, re-derived before the stage started: about 42
   engineer-days and about 3,210 inserted source lines over seven milestones
   (M0 probes, M1 movement of split values, M2 placing on device lists, M3
   `jit` over device lists, M4 per-shard lowering, M5 CPU acceptance, M6 the
   rented node), the grid representation's 1.5 days and 160 lines included,
   and the rented node's bring-up time-boxed at 6 of those days. Stop at 48 engineer-days or
   3,700 inserted source lines; then ship what landed through the CPU
   acceptance and open the hardware gate as a tolk item.
3. **Devices compute:** the engine's `run`, eager operations that do not wait,
   tolk's transfer ports; DeepSeek V4-Flash on one node.

## Laws

1. **A move changes no value, and a result depends on placement only through
   its engine's rounding.** `place p x` equals `x` in type, shape, dtype and
   elements, bit for bit. An operation on placed operands returns the host
   result up to the device's floating-point accuracy and the order of its
   sums, or raises before it runs. Placement is queryable and never changes a
   result's type, shape or dtype. Amends RFC 0003's Law 7. Prevents
   device-typed tensors, and code whose shapes depend on where it runs.
2. **A result lives where its placed operands live.** Host operands join; two
   device sets raise. Where the arithmetic runs is the engine's affair.
   Prevents silent copies between device sets and a guessed device; within
   one set, moving data between shards follows Routing, and `RUNE_JIT_DEBUG`
   reports it in a compiled program.
3. **A read moves nothing.** It copies the elements of its window from
   devices this process holds and runs no collective; reads are local, and
   moves may communicate. Prevents a print or a save evicting a model, and a
   read of one element costing the whole storage.
4. **Storage is released only by unreachability or donation, and only after
   the work that uses it completes.** Moving preserves its source. Prevents a
   move freeing weights on another device, use after release through views and
   bound captures, and a buffer reused under a running kernel.
5. **A placement has one normal form:** one engine and one backend; distinct
   devices in row-major order over a grid whose extents are at least 2 and
   whose adjacent axes merge when used flatly; each cut tensor axis divided
   evenly by the grid axes that cut it; a grid of one device being that
   device. Two placements are equal exactly when their backends are equal and
   every device holds the same window under both, and movement keeps a value
   in this form or raises. Prevents placements tolk cannot lower, two
   spellings of one placement keyed apart, and one device keyed twice.
6. **The link-time engine is the host's; devices are values, and nx keeps no
   registry.** Prevents a link-time choice excluding host and device values
   from one process.
7. **Completion is unobservable.** Reads wait for the work that produces them;
   capability errors raise at the operation; only execution failures may
   surface at the next use of an affected value. Prevents an API that changes
   when dispatch becomes asynchronous.

RFC 0001's Law 2 stands. Its Law 3 reads "resident" for "unforced", counted
by cell, and applies to the leaves of `jit_step`'s second argument, which
replaces its `~donate:true`; item 4 of its reuse mechanism changes: a read or
an eager operation no longer ends residency, so a read input is still seeded
resident and can be donated. From RFC 0003's Reference, a bound value's read
no longer keeps a host copy, and a capture on other devices raises instead of
being read back or replicated.

## Drawbacks

- A breaking sweep: `Rune.to_device` and `?donate` go in stage 1, and
  `?device` becomes `?devices`; every caller moves with them. `pmap` goes in
  stage 2's M3, its callers migrated to `jit` over placed values with no
  bridge. `?donate` alone has 89
  lines in 20 files (tests, examples, benches, docs, `vega.mli` and
  `attention.mli`).
- A state-to-state loop names a module for what it reads beside its state,
  and a loss or sampled ids become fields of the state.
- Until devices compute, an eager operation on placed operands runs on the
  host: each operand window is copied once and each result is copied into a
  new device buffer, on Metal as on a discrete GPU. Eager `grad` over a model placed on CUDA is
  slower than today until stage 3. Compiled code is unaffected.
- From stage 3, an eager device operation costs a compiled call, about
  0.19 ms on Metal before any work, and its first use compiles.
- A read of a placed value keeps it placed: a stray full read of a model holds
  it twice, where today it moved it.
- Expert-parallel prefill sizes each device's grouped blocks for the whole
  step. A device multiplies only its own routes, while its ranking, its row
  gathers and the zeros its empty blocks store follow every route of the
  step: about 0.65 GB per layer at DeepSeek V4-Flash's 4096-token prefill on
  eight devices, 8 ms of a prefill at 3.35 TB/s (derived). On `CPU:n`
  devices, where RFC 0004's `Op.block_matmul` keeps a constant loop bound, a
  device multiplies every block.
- Some host code raises on a device: float64 on Metal, operations tolk does
  not lower. talon, hugin and kaun's metrics read at their boundary.

## Rationale and alternatives

**Residency stays in rune.** The strongest alternative. nx gains nothing;
`Rune.to_device` gains a plural `to_devices ~axis`; moves stop evicting; a
non-evicting read is `to_device ~device:"CPU"`; eager operations on a placed
value read it to the host, as today; a mismatch moves the value silently. It
is the smallest change, and it ships sharding and non-evicting moves. It loses
on three counts. nx still cannot name a device, which is the maintainer's
question. Eager `grad` and `vmap` still read placed weights back for good,
because the fallback knows only the host. And placement and donation state
stay in a global table, with mismatches resolved by an invisible copy of a
model's size. It is also this RFC's kill path.

**A device engine as a second implementation of `nx.backend`.** The literal
reading of "a CUDA backend in nx". The seam is chosen at link time, one per
executable, so a Metal build could hold no host tensor and mapped checkpoints
would be uploaded at load; Metal lacks float64 against nx's total dtype set;
every compiled path bypasses it; and its kernels would duplicate tolk's. Its
idea survives: an engine per device, as a value.

**A per-device `pmap` with `psum` and `axis_index`** (JAX's `shard_map`). It
is `vmap` over an axis split one slice per device, so it would add a name, a
lane, a `psum` rule for `vmap` and a new lowering for what the composition
expresses. Stage 2's probe showed tolk keeps the lane-stacked gather local,
so it comes back as its own RFC only when a model needs all-to-all
dispatch.

**All-to-all dispatch for expert-parallel prefill.** Sending each route's row
to the device that holds its expert moves six rows per token each way per
layer. Under static shapes a lossless capacity is every route of the step, so
each device's buffer and grouped blocks keep the bound they have with
replicated rows, and the arithmetic divides only through RFC 0004's
`Op.block_matmul`, which serves replicated rows as well. All-to-all changes
what moves between devices, and it stays with the per-device map in its own
RFC.

**Keeping the global-view `pmap`.** It is `jit` over split inputs, so one
meaning would carry two names, and its positional `in_axes` needs 723 entries
for Llama 3.1 70B's parameters alone.

**Donation per argument.** A compiled block that takes its weights as inputs
reads them and consumes its cache, so donation is decided by where a value sits
in the call: `Rune.jit_step` reads its first argument and consumes its second,
and replaces `?donate`. The layer-loop prototype's rule, that a value placed
with `to_device` is never consumed, depends on how a value was made and goes
with `to_device`. A marker on the value would be hidden state, and a mask over
leaves would need 723 entries for Llama 3.1 70B. JAX's `donate_argnums` makes
the same split by argument and warns at run time when a donation is unused;
here the split is in the types, and a donated leaf's output is the leaf at its
position. A third module for outputs that are not state was weighed: every
caller in view (the gpt-oss step, RFC 0002's step, a training step, a layer
block) threads its outputs as state, so two modules cover them.

**A scoped handler, `Rune.on ~device f`.** Under an enclosing `grad`, the
re-performed backward operations escape past the inner scope; routing in the
fallback composes in any order.

**Lazy evaluation as API** (a realise call, or a graph the user manages). Law 7
already lets an engine defer and fuse work until a read, a `place` or a
compiled call; the API fixes only where errors surface. **Named meshes and
partition specs now** (JAX): every 1.0 gate fits a flat list cut on one axis:
the single-node deployments need no more, and state sharded 16 ways over two
nodes of eight is the flat 16-way split whatever grid spells it (that gate's
90% throughput needs gathers overlapped with compute, without which it reaches
about 88% of torch, and a mesh does not change that); the grid representation
makes a mesh a later constructor that no caller matches; and axis names as
identity would keep grids that differ only in names from joining. **A public
variant with a mesh case added later:** every caller that matches would break,
and stage 2's per-shard code would be written against one axis and derived
again; the abstract type costs stage 2 about 1.5 days and 160 lines.
**Evicting reads:** they made a save during training evict the state and a
move release its source. **Moving tolk's runtimes into nx** ends the port and
puts a compiler under the array library. **An explicit `free`** is a
use-after-release mode that donation and unreachability cover.

RFC 0003 shipped `to_device` as extensible by optional arguments. This RFC
replaces it instead, because a placement is a value nx must name. The roadmap's
position that fully sharded training is a rune transformation stands for
now: under tolk's current collectives, placement with program boundaries does
not express it. It returns through placement and `jit` if the probe holds
its bound after tolk's collectives stream and stage 2's M4 (§Fully sharded
training). Its other positions stand: collectives are tolk graph
operations ported from tinygrad, and sharded checkpoints and data live in
kaun.

## Non-goals

- Multi-node execution: the controller model, remote devices, the launcher and
  the transport. A later multi-node RFC chooses the controller model by
  measurement and may extend device identity accordingly; `Nx.Device.t` is
  abstract, so that extension breaks no caller. Replicas are meant to be
  equal bit for bit under any model (Reads). Stage 2 keeps any model open:
  devices compare with `Device.equal` and are never looked up by name, shard
  storage is indexed by grid position, collectives stay tolk graph nodes of
  device-to-device copies, and compile and cache keys name no process.
- Meshes: placements over grids of more than one dimension. The
  representation is a grid already, so a mesh constructor adds grids that no
  caller matches, with tolk's collectives over one grid axis, when a goal post
  needs one.
- A per-device map with explicit collectives, and all-to-all dispatch.
- Implicit moves between device lists, and an explicit release of memory.
- Sharded checkpoint files.

## Unresolved questions

During implementation:
- The cost of a one-operation program on Metal and CUDA, first compile
  included, which decides when stage 3's engine replaces host-speed
  operations.
- Program arenas. An engine keeps many programs alive (prefill buckets, a
  block per layer kind, encoders, a sampler), each holding its own arena today;
  two gpt-oss prefill programs held 19.4 GB where one held 16.0 GB in the
  layer-loop design pass. Whether rune shares one arena per device among
  programs its queue runs in turn is decided with the layer loop.
- How much memory a device has free after its weights are placed, which an
  engine needs to size its cache; the engine's RFC decides its form.

## Future possibilities

A no-copy Metal buffer over a page-aligned mapping, a tolk divergence, which
would bring placing gpt-oss-20b from 7.3 s to near zero on Apple machines; a
mesh constructor over the grid representation; an upload of a list of entries
into one buffer, if a load-time measurement shows the per-layer stack matters;
a per-shard checkpoint writer; moving vmap's and autodiff's identity tables
onto `Traced` payloads, as jit's are here. Nothing listed here is a reason to
accept this or a later RFC.
