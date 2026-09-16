# RFC 0001: Tensors as values

- Status: committed
- Date: 2026-09-16
- Packages: nx, rune, kaun, norn, hugin, talon
- Implementation: branch `rfc0001-values`

## Summary

An `Nx.t` is a value. Nothing writes into storage a tensor holds: the seven
in-place functions and their two operators are gone, one functional `set`
replaces them, the backend contract trades `assign` for a pure window write
`update`, and every view, overlapping or broadcast, is an ordinary tensor.
Reusing the storage of a carry threaded through `Rune.jit` (parameters,
optimizer state, a KV cache) is the compiler's decision: `~donate` releases
inputs, and an output that derives from a donated input at the same element
is computed straight into its storage, reported per leaf.

## Motivation

nx had two update models that disagreed. Eager nx mutated through views
(`set_slice`, `blit`, `set_item`); rune's `grad`, `jvp` and `vmap` raised on
any mutation; `jit` replayed a whole-buffer assign into an input leaf by
reading the result back to the host and copying it into the leaf every call,
and silently dropped a slice assign, because the shrink placeholder it wrote
into was not the parent. An in-place update meant three different things
depending on the handler above it, and the gap showed as bugs: the
nx-oxcaml backend's `assign` ignored the destination's strides and offset,
so its `set_slice` wrote the wrong region; a jitted decode step that updated
its cache with `set_slice` did nothing.

The trigger was a pull request restoring a zero-copy sliding window view,
which had to say what a write through an overlapping window means. Its
first version taught the C engine to recover aliasing from output strides on
every dispatch; its second tracked an injectivity bit on views. Both
answered a question that exists only because tensors can be written.

Every carry in the tree was already functional: kaun's KV cache is a ptree
threaded through the step, Vega returns new optimizer state, the examples
fold over batches. Outside nx, in-place writes appeared in hugin's raster
loop and norn's sampling loops, both element-at-a-time construction. Left
alone, every new backend would reimplement an unenforced write contract,
every transform would keep a refusal list, and the roadmap's in-place needs
would be served by a mechanism that copies the whole state through the host
per step.

## Guide

A tensor never changes. To get a tensor that differs from `x` at some
positions, ask for it:

```ocaml
let x = Nx.set [ I 0 ] row x                          (* row 0 *)
let x = Nx.set [ A; R (2, 5) ] v x                    (* columns 2..4 *)
let x = Nx.set [ M drop ] (Nx.scalar_like x 0.) x     (* rows where drop is true *)
let x = Nx.set [ L [ 0; 2 ] ] pair x                  (* rows 0 and 2 *)
```

`set specs v x` is `x` with `v`, broadcast to the selection, at the
positions `specs` select; the value comes before the tensor so the tensor
pipes. The specs are the `index` values `slice` takes, plus one form,
`D (start, len)`, a run of `len` positions whose start is a tensor:

```ocaml
let cache = Nx.set [ A; A; D (pos, seq) ] k cache   (* KV write at position pos *)
let frame = Nx.slice [ A; D (pos, 1024) ] signal    (* dynamic frame *)
```

`pos` is a rank-0 int32 tensor, so one compiled program serves every
position. Its value is clamped so the run always fits.

Building a tensor element by element is construction, not mutation. Fill a
buffer, then wrap it; nothing writes after the wrap:

```ocaml
let ba = Bigarray.Genarray.create Bigarray.int8_unsigned Bigarray.c_layout [| rows; cols; 3 |] in
fill_pixels ba;
let rgb = Nx.of_bigarray ba
let samples = Nx.stack (List.rev positions)
```

Views are free. `broadcast_to`, `transpose`, unit-step `slice` and
`sliding_window` share storage with their input, and nothing can tell:

```ocaml
let frames = Nx.sliding_window ~axis:(-1) ~window:1024 ~step:256 signal
let rms = Nx.sqrt (Nx.mean ~axes:[ -1 ] (Nx.square frames))
```

State threads through a jitted step as a value; the caller keeps it in a
`ref` or a fold, and donation lets the compiler reuse its storage
(`State` and `Params` are `Ptree.S` modules over the records):

```ocaml
let step =
  Rune.jit2 ~donate:true (module State) (module State) (fun s ->
      let grads = Rune.grad (module Params) (loss s.batch) s.params in
      let params, opt = Vega.adamw_step (module Params) ~lr s.opt ~params:s.params ~grads in
      { s with params; opt })

let state = ref (init ()) in
for _ = 1 to steps do state := step !state done
```

On a device, after a donated call the old `!state` is unreadable and the
new parameters and optimizer state occupy the buffers the old ones did. On
CPU outputs stay plain tensors and the GC does the work. `RUNE_JIT_DEBUG=1`
reports, per donated leaf, whether its storage was reused or copied, so an
in-place step is never a guess.

## Reference

### Semantics

A tensor is `(storage, View.t)`; every movement operation returns a view.
Exactly three functions in `nx.mli` cross between buffers and tensors, and
they carry the contract of `Bytes.unsafe_to_string`: `of_buffer` and
`of_bigarray` take a buffer the caller must not write afterwards, and `data`
returns one the caller must not write. `to_bigarray` copies. `copy` exists
to drop a view's parent storage (a small slice of a large tensor keeps the
large buffer alive until copied); `contiguous` yields a contiguous layout,
possibly the same tensor. Weight streaming wraps read-only mmap pages with
`of_bigarray`, legal because nothing writes; the device copy that reads
those pages directly belongs to a future RFC on devices.

Removed from `Nx`: the seven in-place functions `blit`, `set` (int-list
form), `set_slice`, `set_item`, `put`, `index_put`, `put_along_axis`, the
operators `.%{}<-` and `.${}<-`, `?mode` on `take`, and `empty` and
`empty_like` (a value has no uninitialized form to fill in; use `zeros`).
`fill` stays as the pure spelling of `full_like`. `nx.mli` contains no
function of type `... -> ('a, 'b) t -> unit`, the grep-able form of the
invariant.

### The update API

```ocaml
type index =
  | I of int | L of int list | R of int * int | Rs of int * int * int
  | A | N | M of (bool, bool_elt) t
  | D of (int32, int32_elt) t * int
      (** [D (start, len)] selects the run of [len] positions beginning at the
          run-time value of the rank-0 [start], clamped into [0, size - len].
          Keeps the axis, like [R]. [len] is static because traced shapes are. *)

val set : index list -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [set specs v t] is [t] with [v], broadcast to [shape (slice specs t)], at
    the positions [specs] select. [t] is unchanged. *)
```

`set` is total over `index`: `N` inserts a unit axis into the selection,
which `v` broadcasts across. Every selection is injective: `I`, `R`, `Rs`,
`D` and `M` by construction, and `L` must list distinct positions after
negative normalization or `set` raises `Invalid_argument`. `set` performs
one value-carrying operation on `t`, chosen from the spec syntax; the
operations that build its mask, starts or indices never read `t`:

| specs | lowering | eager cost | jit cost |
|---|---|---|---|
| `M mask` on one axis, every other spec `A` | `where` with the mask reshaped onto its axis | one pass over `t` | one pass |
| a window: every spec is `I`, `R`, `Rs` with step ±1, `D`, `A` or `N` | `update t ~starts v'` | copy of `t` plus the window | one pass for static starts; `len` passes for a traced start |
| anything else: an `L`, a stepped `Rs`, or `M` beside another spec | one `scatter` over a contiguous copy with a frontend-built flat index | copy of `t` plus the region | `region` passes |

On the window row, `v'` is `v` broadcast to the selection, with a unit axis
inserted at each `I` and each `N` axis squeezed so `rank v' = rank t`, and
flipped along each `Rs` axis with step −1, whose corner is then its far
end; `starts` stacks the per-axis corners, static ints and clamped `D`
tensors. On the last row, `M` reads its mask on the host, so a traced mask
raises under jit there.

Both operands differentiate through the rules of `where`, `update` and
`scatter`; a broadcast `v` receives the gathered cotangent summed over its
broadcast axes. `set` on a broadcast or non-contiguous `t` returns a fresh
contiguous tensor of `t`'s full shape.

`D` reads `start` on every call when it derives from an input leaf; a
captured `start` is a compile-time constant like every capture, frozen at
the first call. Under `vmap` a batched `start` batches the window: each
example gets its own copy of `t` eagerly and one fused pass under jit. On
the read side, `slice [ D (start, len) ]` is a gather with indices
`start + arange len`; `Rs` with step `k` gathers `start + k * arange n`, a
copy eagerly and one fused pass under jit; `M` on the read side reads the
mask on the host, so a traced mask raises under jit and a constant mask is a
static gather.

Index rule. Tensor-valued positions for `take`, `take_along_axis` and
`scatter` lie in `[0, size)`: eagerly an index outside raises
`Invalid_argument` (the C gather and scatter kernels no longer wrap
negatives, and report `Invalid_argument` rather than `Failure`); under jit
no check runs, an out-of-range position writes nothing and reads zero, and
no memory outside the tensor is touched. A `D` start is clamped by the
frontend, with a traced `clamp` when it is traced, so every backend sees a
window that fits; `len` larger than the axis raises `Invalid_argument`
eagerly, like a static `R` out of range. Static indices (`I`, `R`, `L`)
count negatives from the end. `wrap` is `mod_ (add_s i n) n` for `i` in
`[-n, n)`; `clip` is `clamp`.

### Buffer reuse under jit

Storage may be overwritten exactly when no handle can observe it. Without
ownership in the type, two sites qualify: intermediates, which the compiler
already reuses, and inputs the caller donated. The mechanism is the jit's,
and the user's only promise is `~donate` at the boundary.

1. Candidates. An output leaf `O` may take the resident buffer of a donated
   input leaf `I` when they have equal dtype, byte size and placement and
   every path from `I`'s node to `O`'s node in the traced graph passes only
   through elementwise operations (`where` included), casts of equal
   width, reshapes and contiguous markers. Any movement operation, a
   reduction, a matmul, a gather, or a scatter's index or update edge on a
   path disqualifies `O`. Equivalently, in the kernel that stores `O`,
   every load from `I`'s buffer is at the store's own index, whatever the
   scheduler fuses into it.
2. Ordering. In the linear schedule no kernel that mentions `I`'s buffer
   runs after the first kernel that mentions `O`'s, batched graph calls
   descended and staged loops treated as opaque, so the old value is never
   read through the new one.
3. Exclusions. A buffer that seeds more than one input leaf of the call, or
   that an output leaf other than `O` resolves to, is never reused. An
   input returned unchanged and donated moves its storage to the output
   with no copy and no kernel.
4. Mechanism. The path and ordering tests run once, at compile time.
   Replay adds what only it knows: the input seeded from a donated resident
   entry that nothing else claimed. It then binds `O`'s buffer node to that
   buffer instead of a fresh one and moves the resident entry to `O`'s
   handle; `I`'s handle becomes donated. No store enters the graph, so
   compiled programs and their cache keys are unchanged. Forcing a handle
   (any data read or eager operation) ends its residency, so a forced input
   is never seeded resident, never donated, and reports `not resident`.
   Otherwise `O` gets a fresh buffer and `I` is released after the call.
5. `RUNE_JIT_DEBUG=1` reports `reused`, `copied` or `not resident` per input
   leaf, numbered in traversal order; `jit_stats` counts the reused bytes.

The optimizer state, master weights and bf16 parameters of a training step
are elementwise chains from their own leaves and reuse storage. A KV cache
written with `set [ A; A; D (pos, seq) ]` is an `update` on its own leaf and
reuses storage. Programs under `pmap` keep two generations. The eager cost
is stated plainly: `set` copies `t`, and a decode step in eager nx is
O(cache) per token, as before.

### Backend contract

The backend operation `assign` is replaced by `update`. Its public callers
were the removed writers, its internal callers were `fill` and the jit
writeback, and one of two backends violated its stride clause. The window
write it served is a pure operation, so the op count of the contract is
unchanged:

```ocaml
val update :
  ('a, 'b) t -> starts:(int32, Dtype.int32_elt) t -> ('a, 'b) t -> ('a, 'b) t
(** [update t ~starts v] is [t] with [v] at the window whose corner is
    [starts] (rank 1, length [rank t], read at run time) and whose extent is
    [shape v]. {b Frontend guarantees:} [rank v = rank t]; [shape v] fits
    within [shape t]; [starts] is already clamped so the window fits.
    {b Backend must:} allocate and return a tensor that never shares
    storage with [t]. An output that reuses its input's storage under jit
    is a binding decision of the compiler above this contract. *)
```

The C backend implements it as `copy` followed by the strided write its
copy engine already performs through a shrunk view; nx-oxcaml as copy plus
a strided window store. The effect layer performs `E_update`; `grad`, `jvp`
and `vmap` have one rule each, built from `update`, `slice` and
`zeros_like`. The jit lowers a static start to `pad` (a view) and `where`,
and a traced start to a clamped gather per axis under a window mask.
Re-specified: `to_host` shares storage where the backend holds host memory
and copies otherwise, and the frontend never writes through it;
`from_host` takes ownership of its buffer; `scatter` states the index rule
above and never shares storage with its template. Nothing else in the
contract changes.

### Window primitives

`sliding_window` (a one-axis pure view: `stft` frames, rolling reductions),
`unfold` and `fold` (materializing kernels with padding and dilation:
`extract_patches`, `correlate`, `convolve`, `istft`'s overlap-add, the
im2col kaun's convolutions run on) all stay. Deriving either from the
other costs an extra materialization, so neither is redundant. The view is
public:

```ocaml
val sliding_window : axis:int -> window:int -> ?step:int -> ('a, 'b) t -> ('a, 'b) t
```

### Transforms

`E_assign`, its four handlers and the jit writeback are gone. `grad` and
`vmap` see `set` as `where`, `update` or `scatter`. Under `pmap` a window
write is elementwise over the shards, the mapped axis included: each device
writes the part of the window that falls in its shard, for a static or a
traced start. The remaining raise inside jit is reading a traced value.

## Laws

1. Values: no operation changes an observable of an existing tensor, so
   views, broadcasts, overlapping windows and replicas need no flag; the
   three buffer hatches carry the `unsafe_to_string` contract. Prevents
   identity-keyed transforms seeing a node change under them, and the
   silent loss of a slice write under jit.
2. Storage is written only before wrapping. Prevents stale device copies
   and writes through read-only mappings.
3. Reuse only through donation: an output takes storage only from a
   donated, unforced input whose buffer no other input or output of the
   call binds, and a donated handle raises on read on every device.
   Prevents use-after-donate and aliasing onto a live buffer.
4. Reuse never changes results: every load of the reused buffer in the
   storing kernel is at the store's own index, and no kernel reads the old
   value after the store. Prevents read-after-write inside a kernel and
   across kernels, including after fusion.
5. Cost is readable from the op: movement ops never allocate; `slice` with
   only `I`, `R`, `Rs`, `A` and `N` is a view and with `L`, `M` or `D` a
   gather; `pad`, `cat`, `unfold`, `fold`, `gather`, `scatter` and `update`
   allocate; `set` copies eagerly. Prevents hidden O(n) inside a view.
6. One index rule: `D` starts are clamped, tensor indices are in range,
   static indices count negatives from the end, and out of range raises
   `Invalid_argument` eagerly. Prevents index semantics drifting between
   functions and layers, as `take`'s modes had.
7. Every traced tensor's shape is known at trace time: a traced `M` mask
   raises under jit wherever it must be read. Prevents a data-dependent
   shape reaching the compiler.

## Drawbacks

- Eager `set` copies the whole tensor; a scalar write in a loop is O(n²).
  The idiom is construction, faster before this change too, but numpy users
  write the loop first.
- Storage reuse serves full fine-tuning and little else. Llama 3.1 70B full
  fine-tuning holds 987 GB of carry (bf16 params, fp32 master, Adam m and
  v) plus 141 GB of bf16 gradients: 70.5 GB per GPU across two 8×H100 nodes
  at one generation, 123 GB at two, and the goal post still needs the
  sharding transform and `remat`. LoRA (r = 16: 3.3 GB of state) and
  single-sequence decode (2.6 GB of KV at 8k tokens) fit at two
  generations, and a PagedAttention pool is not a jit carry.
- A KV write with a traced position costs `seq` fused passes per token, and
  a dynamic read a gather, until positions bind as compiler variables.
- Seven in-place functions, two creation functions, two operators and one
  option disappear; external numpy-style code breaks.
- Storage reuse is about two hundred lines of aliasing decisions in rune's
  jit; a wrong decision would corrupt results silently, which is why every
  rule is conservative and the refusals are tested.

## Rationale and alternatives

**User-visible whole-buffer assign replayed by jit (tinygrad's model, the
incumbent).** The strongest alternative: built, tested, and what rune's
documentation promised. It loses on evidence. The replay read the leaf back
to the host and copied it every call, so it peaked at two generations and
added PCIe traffic. `grad` and `vmap` refused it, so the training carry
needed donation anyway. Its slice form was silently dropped under jit, and
making it correct meant tracking the base of every movement placeholder, a
second identity notion. Two OCaml values over one buffer both observed the
write.

**An explicit state type (Flax NNX variables, Equinox state, an
`Nx.Cell`).** A noun whose only content is the donation promise; OCaml's
`ref` plus the jit carry spells it, and raven has one tensor type, owned by
nx, with no wrapper or alias anywhere in the stack.

**A writeability or injectivity bit (numpy's flag, the reworked pull
request).** Correct for a world with writers; without them it guards
nothing.

**Window writes as `scatter` on arange indices, no new backend op.**
Single-axis `scatter` cannot place a box offset on two axes, and under tolk
a scatter is one masked merge per position along the axis, so a static row
range or a prefill inside a jitted step would cost `extent` passes where
`update` costs one. `update` also keeps the eager fast path (copy plus a
strided write, no index tensor) and is the node a symbolic-window store
lowers from without pattern matching. An extended-tier effect instead of a
contract operation would give one operation two lowerings.

**Storage reuse emitted as a store into the input's buffer in the compiled
graph (tolk's `assign`).** It would leave the ordering to the compiler, but
it changes the compiled program and its cache key, and rune already holds
the linear schedule, so the ordering test reads that schedule and the graph
is untouched.

**A single donated arena for the whole carry.** Positional reuse needs no
path test, but it is a new noun coupling ptree traversal order to device
memory, cannot serve a carry whose leaf shapes change, and needs the same
ordering constraint per slot.

**`set_s`, `T of tensor`, keeping `?mode`.** `scalar_like` makes the scalar
case one call. Tensor-selected rows are `scatter ~indices` and `take`;
folding them into a `T` spec is a future possibility. `?mode` disagreed
across three functions and two layers; one rule and explicit `clamp` or
`mod_` is smaller and honest.

**Keeping `empty` as an alias of `zeros` (JAX's choice).** The name
promises to skip the fill and would not; a familiar name that lies about
cost is worse than a missing one.

Prior art is context, not motivation: JAX arrived at values partly because
of devices and made the copy visible in syntax; PyTorch's compiler
functionalizes mutation and replays it as the copy epilogue this design
deletes.

## Non-goals

- Device placement, explicit movement and sharding: a later RFC, on the
  ownership laws here; placement stays a runtime attribute of the resident
  entry, never a type parameter.
- Donation on CPU: outputs stay plain tensors and a CPU carry loop relies on
  the GC; the report prints `copied` there, so the difference is visible.
- Symbolic shapes and an O(window) KV store: a jit lowering once tolk's
  symbolic shrink is wired through rune.
- Copy-on-write or ownership inference in eager nx: construction for element
  loops, jit for tensor loops.

## Unresolved questions

None. Two were settled during implementation: `data` keeps its name, with
the read-only contract in its docstring; kaun's cached attention keeps its
slot mask until symbolic shapes, and its window is clamped rather than
dropped when a caller writes past the cache, since every run-time window
is.

## Future possibilities

Nothing here is a reason to accept this or a later RFC.

- Storage reuse under `pmap`, per shard.
- General same-call buffer reuse from a last-read/first-write analysis of
  the linear schedule.
- The O(window) store and view for `D` once positions bind as tolk
  variables.
- The C backend running `update` as a parallel row-run copy.
- A `T of (int32, int32_elt) t` spec unifying `L`, `take` and tensor-indexed
  writes in one syntax.
