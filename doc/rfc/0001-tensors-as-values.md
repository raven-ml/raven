# RFC 0001: Tensors as values

- Status: published
- Date: 2026-09-16
- Packages: nx, rune, kaun, norn, hugin, talon

Status lifecycle: ideation, discussion, published, committed, abandoned.
"Committed" means this document describes how the system works, not an
intention. Acceptance implies nothing about priority or who implements it.

## Summary

An `Nx.t` becomes a value. Nothing writes into storage a tensor holds: the
seven in-place functions and their two operators go, one functional `set`
replaces them, the backend contract trades `assign` for a pure window write
`update`, and every view, overlapping or broadcast, is an ordinary tensor.
Reusing the storage of a carry threaded through `Rune.jit` (parameters,
optimizer state, a KV cache) is the compiler's decision: the existing
`~donate` releases inputs and an elision pass writes outputs over them,
reported per leaf. PR 210's `sliding_window_view` lands as
`Nx.sliding_window` with no write guard.

## Motivation

nx today has two update models that disagree. Eager nx mutates through views
(`set_slice`, `blit`, `set_item`); rune's `grad`, `jvp` and `vmap` raise on
any mutation; `jit` replays a whole-buffer assign into an input leaf by
reading the result back to the host and copying it into the leaf every call,
and silently drops a slice assign, because the shrink placeholder it writes
into is not the parent. An in-place update means three different things
depending on the handler above it, and the gap shows as bugs: the nx-oxcaml
backend's `assign` ignores the destination's strides and offset, so its
`set_slice` writes the wrong region; a jitted decode step that updates its
cache with `set_slice` does nothing.

The trigger is PR 210, which restores a zero-copy sliding window view and
must say what a write through an overlapping window means. Its first
version taught the C engine to recover aliasing from output strides on
every dispatch; the reworked version tracks an injectivity bit on views.
Both answer a question that exists only because tensors can be written.

Every carry in the tree is already functional: kaun's KV cache is a ptree
threaded through the step and rebuilt with `where`, Vega returns new
optimizer state, the examples fold over batches. Outside nx, in-place
writes appear in hugin's raster loop and norn's sampling loops, both
element-at-a-time construction. If we do nothing, every new backend
reimplements an unenforced write contract, every transform keeps a refusal
list, and the roadmap's in-place needs are served by a mechanism that copies
the whole state through the host per step.

## Guide-level explanation

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
pipes. The specs are the `index` values `slice` takes, plus one new form,
`D (start, len)`, a run of `len` positions whose start is a tensor:

```ocaml
let cache = Nx.set [ A; A; D (pos, seq) ] k cache   (* KV write at position pos *)
let frame = Nx.slice [ A; D (pos, 1024) ] signal    (* dynamic frame *)
```

`pos` is a rank-0 int32 tensor, so one compiled program serves every
position. Its value is clamped so the run always fits.

Building a tensor element by element is construction, not mutation. Write a
buffer, then wrap it; nothing writes after the wrap:

```ocaml
let ba = Bigarray.Genarray.create Bigarray.int8_unsigned Bigarray.c_layout [| rows; cols; 3 |] in
fill_pixels ba;
let rgb = Nx.of_bigarray ba
let samples = Nx.stack (List.rev positions)
```

Views are free. `broadcast_to`, `transpose`, unit-step `slice` and the new
`sliding_window` share storage with their input, and nothing can tell:

```ocaml
let frames = Nx.sliding_window ~axis:(-1) ~window:1024 ~step:256 signal
let rms = Nx.sqrt (Nx.mean ~axes:[ -1 ] (Nx.square frames))
```

State threads through a jitted step as a value; the caller keeps it in a
`ref` or a fold, and donation lets the compiler reuse its storage:

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
new one owns the same buffers; on CPU outputs stay plain tensors and the
GC does the work. `RUNE_JIT_DEBUG=1` reports `reused` or `copied` per
donated leaf, so an in-place step is never a guess.

## Reference-level explanation

### Semantics

A tensor is `(storage, View.t)`; every movement operation returns a view.
Exactly three functions in `nx.mli` cross between buffers and tensors,
and they carry the contract of `Bytes.unsafe_to_string`: `of_buffer` and `of_bigarray`
take a buffer the caller must not write afterwards, and `data` returns one
the caller must not write. `to_bigarray` copies. `copy` exists to drop a
view's parent storage (a small slice of a large tensor keeps the large
buffer alive until copied); `contiguous` yields a contiguous layout,
possibly the same tensor. Weight streaming wraps read-only mmap pages with
`of_bigarray`, legal because nothing writes; the device copy that reads
those pages directly belongs to the device RFC.

Removed from `Nx`: the seven in-place functions `blit`, `set` (int-list
form), `set_slice`, `set_item`, `put`, `index_put`, `put_along_axis`, the
operators `.%{}<-` and `.${}<-`, `?mode` on `take`, and `empty` and
`empty_like` (a value nothing can write has no uninitialized form; the
frontend keeps `B.buffer` for its own fresh allocations). `fill` stays as the
pure spelling of `full_like`. `nx.mli` then contains no function of type
`... -> ('a, 'b) t -> unit`, the grep-able form of the invariant.

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
| `M mask` on axis `k`, every other spec `A` | `where m' v' t`, `m'` reshaped onto axis `k` (never numpy-broadcast) | one pass over `t` | one pass |
| every spec is `I`, `R`, `Rs` with step ±1, `D`, `A` or `N` (a window) | `update t ~starts v'`: `v'` is `v` broadcast to the selection, with a unit axis inserted at each `I` and each `N` axis squeezed so `rank v' = rank t`, and flipped along each `Rs` axis with step −1, whose corner is then its far end; `starts` stacks the per-axis starts, static ints and clamped `D` tensors | copy of `t` plus the window | one pass for static starts; `len` passes for a traced start until the symbolic-shrink lowering, then O(window) |
| any `L`, `Rs` with another step, or `M` beside another spec | flat `scatter ~axis:0 ~unique_indices:true` over `reshape [| numel t |] (contiguous t)` with a frontend-built flat index; `M` here reads the mask on the host, so a traced mask raises under jit | copy of `t` plus the region | `region` passes; rare |

Both operands differentiate through the rules of `where`, `update` and
`scatter`; a broadcast `v` receives the gathered cotangent summed over its
broadcast axes by the existing expand rule. `set` on a broadcast or
non-contiguous `t` returns a fresh contiguous tensor of `t`'s full shape.

`D` reads `start` on every call when it derives from an input leaf; a
captured `start` is a compile-time constant like every capture, frozen at
the first call. Under `vmap` a batched `start` batches the window: each
example gets its own copy of `t` eagerly and one fused pass under jit. On
the read side, `slice [D (start, len)]` is `take` with indices
`start + arange len` (a gather, `len` passes under jit until symbolic
shrink makes it a view); `Rs` with step `k` becomes a view eagerly (`shrink`, then
`sliding_window ~window:1 ~step:k`, then a squeeze) instead of today's
materializing gather, while under jit it still materializes through tolk's
`Movement.unfold` as `sliding_window` does; `M` on the read side reads the mask on the host, so
a traced mask raises under jit and a constant mask is a static gather.

Index rule. Tensor-valued positions for `take`, `take_along_axis` and
`scatter` lie in `[0, size)`: eagerly an index outside raises
`Invalid_argument` (the C gather and scatter kernels stop wrapping negatives
and their `Failure` becomes `Invalid_argument`); under jit no check runs, an
out-of-range position writes nothing and reads zero, and no memory outside
the tensor is touched. A `D` start is clamped by the frontend, with a
traced `clamp` when it is traced, so every backend sees a window that fits;
`len` larger than the axis raises `Invalid_argument` eagerly, like a static
`R` out of range. Static indices (`I`, `R`, `L`) count negatives from the
end as before. `wrap` is
`mod_ (add_s i n) n` for `i` in `[-n, n)`; `clip` is `clamp`.

### Buffer reuse under jit

Storage may be overwritten exactly when no handle can observe it. Without
ownership in the type, two sites qualify: intermediates, which the compiler
already reuses, and inputs the caller donated. The mechanism is the jit's,
and the user's only promise is `~donate` at the boundary.

1. Candidates. An output leaf `O` may take the resident buffer of a donated
   input leaf `I` when they have equal dtype, byte size and placement and
   every path from `I`'s node to `O`'s node in the traced graph passes only
   through elementwise operations (`where` included), casts of equal
   itemsize, reshapes of a contiguous operand, and the template edge of an
   `update` or `scatter`. Any other movement operation, a reduction, a
   matmul, a gather, or a scatter's index or update edge disqualifies `O`.
   Equivalently, in the kernel that stores `O`, every load from `I`'s
   buffer is at the store's own index, whatever tolk fuses into it.
2. Exclusions. A buffer that seeds more than one input node, that an output
   leaf other than `O` resolves to (an input returned unchanged), or that a
   capture binds is never aliased; a kernel inside a staged loop body never
   aliases. An input returned unchanged and donated moves its resident entry
   to the output with no copy and no kernel.
3. Mechanism. At trace end the jit emits tolk's `Op.assign` of `O`'s node
   into `I`'s buffer node (`tolk/lib/frontend/op.ml:26`, a store wrapped in
   an `after` at buffer identity). Every later read then depends on the
   store, and the implementation must confirm that tolk's scheduler orders
   reads of the prior value before it, as tinygrad's does; with that, rune
   carries no schedule analysis beyond the same-index test above. The
   resident entry moves from `I`'s handle to `O`'s; `I`'s handle becomes
   donated exactly as today. Forcing a handle (any data read or eager
   operation) ends its residency, as today, so a forced input is never
   seeded resident, never donated, and reports `copied`. Otherwise `O` gets
   a fresh buffer and `I` is released after the call, as today.
4. `RUNE_JIT_DEBUG=1` reports `reused` or `copied` per donated leaf,
   numbered in traversal order.

The optimizer state, master weights and bf16 parameters of a training step
are elementwise chains from their own leaves and reuse storage. A KV cache
written with `set [ A; A; D (pos, seq) ]` is an `update` on its own leaf and
reuses storage; a decode token costs `seq` passes over the cache with no
allocation, and a prefill of `seq` tokens or a dynamic read
`slice [ A; A; D (pos, len) ]` costs `seq` passes until the symbolic-shrink
lowering makes the write O(window) and the read a view. The eager cost is
stated plainly: `set` copies `t`; a decode step in eager nx is O(cache) per
token, as today.

### Backend contract

Replaced: the backend operation `assign` by `update` (tolk's `Op.assign`,
named below, is the compiler's store and unrelated). `assign`'s public
callers are gone, its
internal callers (`fill`, the jit writeback) are gone, and one of two
backends violates its stride clause. The window write it served becomes a
pure operation, so the op count of `S` is unchanged:

```ocaml
val update :
  ('a, 'b) t -> starts:(int32, Dtype.int32_elt) t -> ('a, 'b) t -> ('a, 'b) t
(** [update t ~starts v] is [t] with [v] at the window whose corner is
    [starts] (rank 1, length [rank t], read at run time) and whose extent is
    [shape v]. {b Frontend guarantees:} [rank v = rank t]; [shape v] fits
    within [shape t]; [starts] is already clamped so the window fits.
    {b Backend must:} allocate and return a tensor that never shares
    storage with [t]. (An output that reuses its input's storage under jit
    is a binding decision of the compiler above this contract.) *)
```

The C backend implements it as `copy` followed by the strided write its
`assign` engine already performs; nx-oxcaml as copy plus a strided window
store, replacing its wrong-region `assign`. The effect layer performs
`E_update`; `grad`, `jvp` and `vmap` gain one rule each, built from
`update`, `slice` and `zeros_like`. The jit lowers a static start to `pad`
(a view) and `where`, a traced start to the gather-and-mask form until
symbolic shrink is wired through rune, then to a store into the shrunk
buffer. Re-specified: `to_host` "shares storage where the backend holds
host memory, otherwise copies; the frontend never writes through it";
`from_host` "takes ownership of `buf`"; `scatter` gains the index rule above
and "the result never shares storage with the template". Nothing else
changes.

### Window primitives

`sliding_window` (one-axis pure view: `stft` frames, rolling reductions,
PR 210), `unfold` and `fold` (materializing kernels with padding and
dilation: `extract_patches`, `correlate`, `convolve`, `istft`'s
overlap-add, the im2col kaun's convolutions run on) all stay. Deriving
either from the other costs an extra materialization, so neither is
redundant.

```ocaml
val sliding_window : axis:int -> window:int -> ?step:int -> ('a, 'b) t -> ('a, 'b) t
```

### Transforms

`E_assign`, its four handlers and the jit writeback are deleted. `grad` and
`vmap` see `set` as `where`, `update` or `scatter`. `pmap` treats `set` as
shard-local; a `D` window on the mapped axis raises `Jit_error`. The
remaining raise inside jit is reading a traced value, as today.

### Migration

One sweep, then two follow-ups. The sweep: nx (`D`, `set`, `update` in `S`
and both backends, the deletions, `nonzero`, `nonzero_indices_only`,
`argwhere` and `map_item` rewritten over host buffers, contract text, tests,
`CHANGES.md`); rune (delete `E_assign`, its handlers, the writeback and
their tests; add `set` under jit at two positions, `grad` of `set` for both
operands, `slice [D]`; rune.mli's Jit_error and in-place paragraphs, README
and the two docs); consumers (kaun `apply_cached` to `set [ A; A; D ]` with
`pos` reshaped to rank 0 and `~donate:true` in the gpt2 loop; norn
accumulates then `stack` or `create`; hugin fills a uint8 bigarray then
`of_bigarray`; talon's `empty` to `zeros`; kaun's bench to
`astype float32 (one_hot ...)`; the nx tsne example and numpy-comparison
doc). Then elision as its own PR with a memory test shaped like a carry.
PR 210 rebases onto the sweep as `Nx.sliding_window` alone.

## Laws

1. Values: no operation changes an observable of an existing tensor, so
   views, broadcasts, overlapping windows and replicas need no flag; the
   three buffer hatches carry the `unsafe_to_string` contract. Prevents
   identity-keyed transforms seeing a node change, and the silent slice-set
   loss under jit.
2. Storage is written only before wrapping. Prevents stale device copies
   and writes through read-only mappings.
3. Reuse only through donation: an output aliases only a donated, unforced
   input whose buffer no other input, output or capture of the call binds,
   and a donated handle raises on read on every device. Prevents
   use-after-donate and aliasing onto a live buffer.
4. Elision never changes results: every load of the reused buffer in the
   storing kernel is at the store's own index, and the store is ordered
   after every other read of the prior value. Prevents read-after-write
   inside a kernel and across kernels, including after fusion.
5. Cost is readable from the op: movement ops never allocate; `slice` with
   only `I`, `R`, `Rs`, `A`, `N` is a view eagerly (under jit a stepped
   `Rs` materializes) and with `L`, `M` or `D` a gather; `pad`, `cat`, `unfold`, `fold`, `gather`, `scatter`, `update`
   allocate; `set` copies eagerly. Prevents hidden O(n) inside a view.
6. One index rule: `D` starts are clamped, tensor indices are in range,
   static indices count negatives from the end, out of range raises
   `Invalid_argument` eagerly. Prevents the three-way `mode` drift the audit
   found.
7. Every traced tensor's shape is known at trace time; `M` is the one spec
   whose selection depends on data, so a traced mask raises under jit
   wherever the mask must be read (a read, or a write beside another
   spec); `M` alone on the write side is a `where` and traces. Prevents a
   data-dependent shape reaching the compiler.

## Drawbacks

- Eager `set` copies the whole tensor; a scalar write in a loop is O(n²).
  The idiom is construction, faster today too, but numpy users write the
  loop first.
- Until elision lands, a jitted carry on device peaks at two generations
  of state, as the writeback did (it read every assigned leaf back to the
  host each call). Llama 3.1 70B full fine-tuning holds 987 GB of carry
  (bf16 params, fp32 master, Adam m and v) plus 141 GB of bf16 gradients:
  70.5 GB per GPU across two 8×H100 nodes at one generation, 123 GB at two.
  Elision removes the factor of two; the goal post still needs the sharding
  transform and `remat`. LoRA (r = 16: 3.3 GB of state) and single-sequence
  decode (2.6 GB of KV at 8k tokens) fit either way, and a PagedAttention
  pool is not a jit carry.
- Until the symbolic-shrink lowering, a KV write costs `seq` fused passes
  per token instead of an O(`seq`) store, and a dynamic read a gather.
- Seven in-place functions, two creation functions, two operators and one
  option disappear; external numpy-style code breaks.
- Elision is compiler machinery rune does not have; as a lowering onto
  tolk's `assign` plus the same-index test it is one to two hundred lines,
  and it is the piece the full fine-tuning goal post depends on.

## Rationale and alternatives

**User-visible whole-buffer assign replayed by jit (tinygrad's model, the
incumbent).** The strongest alternative: built, tested, and what rune.mli
promises. It loses on evidence. The replay reads the leaf back to the host
and copies it every call, so it peaks at two generations and adds PCIe
traffic. `grad` and `vmap` refuse it, so the training carry needs donation
anyway. Its slice form is silently dropped under jit, and making it correct
means tracking the base of every movement placeholder, a second identity
notion. Two OCaml values over one buffer both observe the write.

**An explicit state type (Flax NNX variables, Equinox state, an
`Nx.Cell`).** A noun whose only content is the donation promise; OCaml's
`ref` plus the jit carry spells it, and constraint C1 forbids a second
tensor-like type.

**A writeability or injectivity bit (numpy's flag, PR 210 as reworked).**
Correct for a world with writers; without them it guards nothing.

**Window writes as `scatter` on arange indices, no new backend op.** Single-
axis `scatter` cannot place a box offset on two axes, and under tolk a
scatter is one masked merge per position along the axis, so a static row
range or a prefill inside a jitted step would cost `extent` passes where
`update` costs one. `update` also keeps the eager fast path (copy plus a
strided write, no index tensor) and is the node the symbolic-shrink store
lowers from without pattern matching. An extended-tier effect instead of an
`S` op would give one operation two lowerings.

**Elision decided by a trace pattern plus a schedule last-reader count.**
Unsound once tolk fuses a movement read of the input into the storing
kernel, and unnecessary once tolk's `Op.assign` orders the store after the
prior reads: only the same-index test remains, which is a path property of
the trace.

**A single donated arena for the whole carry.** Positional aliasing needs no
alignment test, but it is a new noun coupling ptree traversal order to
device memory, cannot serve a carry whose leaf shapes change, and needs the
same ordering constraint per slot.

**`set_s`, `T of tensor`, keeping `?mode`.** `scalar_like` makes the scalar
case one call. Tensor-selected rows are `scatter ~indices` and `take`;
folding them into a `T` spec is a future possibility. `?mode` disagreed
across three functions and two layers; one rule and explicit `clamp` or
`mod_` is smaller and honest.

Prior art is context, not motivation: JAX arrived at values partly because
of devices and made the copy visible in syntax; PyTorch's compiler
functionalizes mutation and replays it as the copy epilogue this design
deletes.

## Non-goals

- Device placement, explicit movement and sharding: the next RFC, on the
  ownership laws here; placement stays a runtime attribute of the resident
  entry, never a type parameter (C1).
- Donation on CPU: outputs stay plain tensors and a CPU carry loop relies on
  the GC; the report prints `copied` there, so the difference is visible.
- Symbolic shapes and the O(window) KV store: a jit lowering once tolk's
  symbolic shrink is wired through rune.
- Copy-on-write or ownership inference in eager nx: construction for element
  loops, jit for tensor loops.

## Unresolved questions

Resolved during implementation:
- That tolk's scheduler orders reads of a buffer's prior value before the
  `Op.assign` store the elision emits; if it does not, the jit adds a
  last-reader-in-schedule check on the linear schedule before seeding.
- Whether `data` should be spelled `unsafe_data`, since a bigarray has no
  read-only view and the name is the only signal available.
- Whether kaun's `apply_cached` keeps its `len`-slot causal mask (it does
  until symbolic shapes) once the write is an `update`.

Explicitly out of scope:
- How the host-to-device copy reads mmap pages without a `Bytes` stage
  (today `copyin_tensor` stages through one); device RFC.

## Future possibilities

Nothing here is a reason to accept this or a later RFC.

- General same-call buffer reuse from a last-read/first-write analysis of
  the linear schedule (donate phase 3 in `TODO.md`).
- The O(window) store and view for `D` through `symbolic_shrink` once
  positions bind as tolk variables.
- The C backend running `update` as a parallel row-run copy.
- A `T of (int32, int32_elt) t` spec unifying `L`, `take` and tensor-indexed
  writes in one syntax.
