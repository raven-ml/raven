# RFC 0002: The decode contract

- Status: committed
- Date: 2026-09-18
- Packages: kaun, nx (one documentation line), rune and tolk (the indexed
  store)
- Implementation: commits `Make attention total` through `Group attention
  keys without copying them` on main
- Revision: 2. Replaces the first revision in place; Rationale says what it
  chose and why it lost.

## Summary

A decoder model in kaun is one parameter record and one function, `cached`,
that maps a cache, a cache index and tokens to the residual stream and the new
cache. `hidden`, the residual stream of whole sequences, is `cached` over the
index of whole sequences, by definition and by implementation, so a model has
one forward pass. A cache is any record of pools: tensors whose axis 0 is a
slot axis, with no batch axis. A cache index is an opaque value built from two
int32 tensors, each token's position and a table from a sequence's positions
to slots. Contiguous decoding, paging, shared prefixes and beam forks are
values of the table, so kaun has one cache type and one cached attention.
Attention is total: a query that sees no key yields zero. Kaun ships no model:
the contract is kaun's types, a convention and the laws that kaun's decoder
tests check, so training, a generate loop and a future inference engine run
one definition that lives in user code.

## Motivation

GPT-2 was the only decoder in the tree and its forward pass existed twice,
sharing no body. `Attention.apply_cached` assumed one position for the whole
batch, and the cache's batch extent was fixed at construction, so an engine
that admits or retires a request would change a compiled program's signature
and retrace. Llama-class models with validated import, mixed-precision
training and an inference engine all need the same models. Against that
surface each would have written them again, which is how a serving stack comes
to own a separate model zoo.

## Guide

### A model

```ocaml
(* a model, in user code: llama.mli; 'x is (float, 'b) Nx.t *)
module Params : Nx.Ptree.Uniform with type 'a t = 'a params
module Cache = Kaun.Attention.Cache.List       (* one cache per block *)

val make : config -> Nx.float32_t params
val cache : config -> slots:int -> (float, 'b) Nx.dtype -> 'x Cache.t

val cached : config -> 'x params -> 'x Cache.t -> Cache_index.t -> Nx.int32_t -> 'x * 'x Cache.t
(** The residual stream, [[batch; seq; dim]], of tokens that sit where the
    index says, and the cache with their keys and values written. *)

val hidden : config -> 'x params -> Nx.int32_t -> 'x
(** The residual stream of whole sequences. *)

val logits : config -> 'x params -> 'x -> 'x
(** Final norm and head, per position. Select positions before calling. *)
```

```ocaml
(* block cfg b c index x calls
   Attention.cached ~head_dim:cfg.head_dim ~rope:cfg.rope b.attn c index (norm x) *)
let cached cfg p caches index ids =
  let x, rev =
    List.fold_left2
      (fun (x, acc) b c ->
        let x, c = block cfg b c index x in
        (x, c :: acc))
      (Embedding.apply p.tok ids, [])
      p.blocks caches
  in
  (x, List.rev rev)

let hidden cfg p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let nothing = cache cfg ~slots:0 (Nx.dtype p.norm.Rms_norm.gamma) in
  fst (cached cfg p nothing (Cache_index.whole ~batch ~seq ()) ids)
```

The index is opaque to the model, which passes it to every layer. A slot holds
what one token stores, here its keys and values in every block. `hidden`
passes a cache of no slots because the fold threads one; over
`Cache_index.whole` no layer reads or writes it. Arguments run options,
parameters, carried state, inputs, which is `Batch_norm.apply`'s order;
results are output, carried state. A model may add its own optional arguments
(GPT-2's `?dropout`).

### Training

```ocaml
let loss p (ids, targets) =
  Loss.softmax_cross_entropy_sparse (Llama.logits cfg p (Llama.hidden cfg p ids)) targets
```

Over `Cache_index.whole` a layer touches no pool, so training costs what plain
attention costs. A right-padded training batch needs nothing from the index:
under a causal mask no real token sees the padding after it.
`Cache_index.whole ~lens` takes ids padded on the left, as `Cache_index.rows`
does, and a model that wants it gives `hidden` a `?lens`.

### Generating

```ocaml
let step ({ Step.tokens; index; key; temperature; k; p; cache } as s) =
  let h, cache = Llama.cached cfg params cache index tokens in
  let logits = Nx.cast Nx.float32 (Llama.logits cfg params (Nx.slice [ A; I (-1) ] h)) in
  let ks = Nx.Rng.split key in
  let next = Nx.Rng.categorical ks.(1) Nx.(Fn.top_p ~p (Fn.top_k ~k (div logits temperature))) in
  { s with tokens = Nx.reshape [| batch; 1 |] next; index = Cache_index.advance index; key = ks.(0); cache }

let step = Rune.jit2 ~donate:true (module Step) (module Step) step
let s = ref (step { tokens; key; temperature; k; p; index = Cache_index.rows ~context lens;
                    cache = Llama.cache cfg ~slots:(batch * context) dt })
```

`Step` is the caller's record of these seven fields; its three traversals use
`Cache_index.map`, `Cache.map` and the polymorphic function on the rest. A
lane is one entry of a call's batch. `Cache_index.rows ~context lens` gives
each sequence `context` slots of its own, hence the cache of `batch *
context`, and pads lanes on the left; the caller pads `tokens` the same way,
so the last column of `h` is every lane's last real token.
`Cache_index.advance` moves every lane to its next token. The sampling
parameters are fields of the step because a captured tensor is a constant of
the program. The first call has `seq = max lens` and the rest `seq = 1`, so
one source compiles to two programs.

### What an engine does per tick

```ocaml
(* one bucket of one tick: pos and tokens are new, the table was patched in place.
   pos : [batch; seq], table : [rows; context], row : [batch]; -1 is none *)
let index = Cache_index.make ~row ~pos ~table () in
let h, state = Model.cached cfg params state index tokens in
```

One call per `(batch, seq, rows, context)` bucket that has work, threading the
state through them: typically a prefill call at `[1; chunk]`, then a decode
call at `[batch; 1]`. Lanes shorter than `seq` are padded on the left with
`pos = -1`; an empty lane is `-1` throughout. The table is engine state kept
on the device and patched when allocation changes. Admitting or retiring a
request changes values inside fixed shapes.

An engine depends on neither kaun nor a model. It is parameterised by a step
function over nx tensors and an opaque state with a traversal. Because axis 0
of every state leaf is the slot axis, an engine moves state by slot (a swap or
a transfer between instances) with `Nx.take` and `Nx.scatter` over the
traversal, knowing nothing else about the model. Those calls follow nx's index
rule, so the engine moves allocated slots only and filters `-1` out of a table
row first. The laws below are the protocol between the two. The checks of the
battery under Reference are stated over a step's inputs and outputs, so an
engine's own tests can port them without model code.

## Reference

### The cache

A cache is an ordinary value of a type the model defines, whose leaves are
pools: tensors of shape `[slots + 1; ...]` of any width and dtype. There is no
batch axis; who owns a slot is the table's business. Kaun ships the record
most models use, `Attention.Cache.t = { keys; values }` with payloads `[slots
+ 1; kv_heads; head_dim]`, and `Cache.List`, the `Uniform` traversal of a list
of them. A layer with another payload (a latent and a rotary key, a quantised
entry beside its scales) defines its own record and uses the same cache index;
no layer in the tree does yet.

The last row of every pool is the scratch row. What addresses nothing is
written there, and what is unallocated is read from there and replaced by
zero, so its content is unspecified and never observed. It is last so that
slot `s` is row `s` of every pool. Slot numbers `0` to `slots - 1` are the
allocator's. An engine takes `slots` from whoever built the state; a leaf's
axis 0 is one longer, and slot `slots` addresses nothing, like any slot
outside the pool. `Cache.make ~slots` allocates the row and accepts `slots =
0`. A pool smaller than the slots a table names is no error: those columns
address nothing.

### The cache index

```ocaml
module Cache_index : sig
  type t
  val make : ?row:Nx.int32_t -> pos:Nx.int32_t -> table:Nx.int32_t -> unit -> t
  val rows : context:int -> int array -> t
  val whole : ?lens:int array -> batch:int -> seq:int -> unit -> t
  val advance : t -> t

  val batch : t -> int                      (* seq, context likewise: static sizes *)
  val positions : t -> Nx.int32_t           (* [batch; seq], in range *)

  val extend :
    ?window:int -> t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t
  val mask : ?window:int -> t -> Nx.bool_t  (* [batch; seq; context] *)

  val map : (Nx.int32_t -> Nx.int32_t) -> t -> t     (* map2, iter likewise *)
end
```

`pos : [batch; seq]` is each token's position in its sequence, `-1` for
padding. `table : [rows; context]` names the slot holding position `j` of
sequence `r`, `-1` when none is allocated. `row : [batch]` names the sequence
of each lane; without it lane `b` is sequence `b` and `rows` equals `batch`.
Several lanes may name one sequence, which is how a flattened batch of
one-token lanes is expressed.

**`-1` addresses nothing, everywhere:** as a position, a table entry or a
sequence, and so does any slot outside the pool. No slot number a cache index
computes is out of range, so an eager run and a compiled run agree on every
index. `Cache_index.positions` is the positions clamped into `[0, context)`,
for rotating and for indexing a table of position embeddings.

A token stores at the slot its table names at its own position. Padding and a
position past the last column store nothing, so a full context is never
overwritten. A token past the last column sees every column without itself and
its output is unspecified; an engine retires the sequence first.

`rows ~context lens` is the Guide's contiguous allocator: sequence `b` owns
slots `b * context ..`. `advance` moves every lane to one past its greatest
position, a lane of padding to `0`. It is per lane: where two lanes name one
sequence an engine sets positions itself. `whole` reads and keeps nothing.

`extend index values pool` is `(seen, pool')`. `pool'` is `pool` with `values :
[batch; seq; ...]` stored where the call's tokens sit, by one `Nx.scatter
~unique_indices:true` over the call's tokens, in place on a pool donated to
`Rune.jit`. `seen : [batch; context; ...]` is what those tokens attend over:
each lane's sequence read from `pool'`, with every column that is unallocated,
past the lane's positions or below every query's `window` as zero. It writes
and then reads the written pool, so a token below the last column sees itself
through the table, and RFC 0001's order for storage reuse holds by dataflow.
On a whole index `seen` is `values` and `pool'` is `pool`; that case is fixed
when the index is built, before any tensor has a value, so it holds under
`Rune.jit`. `mask` says which columns of `seen` each token sees: those at or
before its position, and with `window` only the last `window` of them. A
padded token sees nothing.

A cache index holds nothing resolved: the scratch row's number is the pool's
size, which an index does not know, and a value cached during a trace would
outlive it. Its functions are recomputed per leaf and the compiler's
hash-consing shares the work, as long as the slot numbers stay a lazy
expression; forcing them with `Nx.contiguous` costs a buffer per leaf per layer,
measured at 8% of a decode step.

### Total attention

`scaled_dot_product_attention ?mask q k v` is total: a query that sees no key
yields a zero output over finite values and a zero gradient, so padded lanes
and empty windows need no care from a mask builder. At half and quarter
precision the scores and the softmax run in the float32 island, as before.
`causal_mask` no longer keeps its diagonal for padded rows.

### Cached attention

```ocaml
val cached :
  head_dim:int -> ?rope:Rope.t -> ?window:int ->
  'x t -> 'x Cache.t -> Cache_index.t -> 'x -> 'x * 'x Cache.t
```

For `x : [batch; seq; embed]` the layer projects and rotates queries and keys
at `Cache_index.positions`, extends both pools with `Cache_index.extend`, and
attends once over what they return under `Cache_index.mask`. Queries reshape
to `[batch; kv_heads; groups; seq; head_dim]` against keys at `[batch;
kv_heads; 1; context; head_dim]`; both head counts are read from the
projection widths. No model in the tree uses `?window` yet.

`Attention.apply ~head_dim ?mask ?rope p x` stays for attention that is not
causal self-attention. `Span`, `route` and `Attention.route` are removed.

### One definition

`hidden` is `cached` over `Cache_index.whole` and a cache of no slots, as the
Guide writes it: that line is the definition and the implementation. `cached`
over a fresh cache and `Cache_index.rows` gives the same result up to
reassociation, which is the first check of the battery and what binds the two
cases of `extend`.

### The battery

Kaun runs the first eight checks on a two-layer decoder defined in
`kaun/test/test_decoder.ml`, each eagerly and under `Rune.jit2 ~donate:true`.
The window half of the poisoning check and the lane whose sequence is `-1` are
layer tests in `test_attention.ml`, as is the last row. The gradient check
runs eagerly and compiled, the reuse check compiled only. It is a test file
over that decoder, written against `cache`, `cached`, `hidden` and `logits`
alone: a model author ports the checks that apply, as
`examples/05-llama/validate.ml` does for the chunking and ragged-batch rows on
real weights.

| Check | Law |
| --- | --- |
| one call over `Cache_index.rows` equals `hidden` | 8 |
| chunks of 1, of 7 and the whole prompt agree, written slots included | 8 |
| a sequence fed as one-token lanes agrees | 8 |
| a random permutation of slots changes nothing | 1 |
| two sequences sharing the slots of an equal prefix | 1, 10 |
| a ragged batch: each lane equals the lane alone | 5 |
| every slot no table names, scratch row included, and every column below a window, filled with `nan` before each call: outputs finite and equal | 4, 5 |
| a lane of `-1` throughout, a lane whose sequence is `-1`, an all-padding call: finite outputs, pools unchanged | 3, 4 |
| the gradient of `hidden` with a fully padded lane: finite | 3 |
| every cache leaf reports storage reuse under donation | 6 |
| four decode steps at different positions share one trace | 2, 7 |

The poisoning checks were validated by mutation: with the zeroing select
removed they fail in both modes.

### Cost

Measured on an M1 Max with Metal, GPT-2 124M shape unless noted.

- The write costs the call's tokens. A two-layer decode step at a context of
  256 takes 3.6 ms over a pool of 4096 slots and 3.8 ms over 131072; the
  select over the pool it replaces took 3.8 ms and 24.6 ms. On the CPU
  device's zero-copy path no storage is reused, so a compiled CPU step still
  copies each written pool once per leaf; CPU reuse is tested with
  `RUNE_JIT_FORCE_COPY=1`.
- The decode step takes 8.0 ms and 8.9 ms at contexts of 256 and 1024 against
  9.1 ms and 10.3 ms before, and allocates 17% fewer host words per step. Of
  that, grouping the keys without copying them is 0.5 ms at 256 and 8% of
  the words.
- The compiled forward and gradient of `hidden` has 1380 kernels against 1397
  for the hand-written second pass it deletes, and returns the same gradient
  to nine digits. With rotary positions `extend` costs two kernels per layer
  in that graph, under 1% in time.
- Llama 3.2 1B reproduces the reference float32 logits to `1e-6` relative
  eagerly and `2.4e-6` compiled; fed in chunks of 1, 7 and the rest it equals
  its whole-sequence path exactly.
- A read costs the context, per lane: it lands in a context-sized buffer per
  leaf per layer before attention, float32 under the half-precision island.
  `T` one-token lanes of one sequence gather `T x context` rows where one lane
  of `T` tokens gathers `context`, so the flattened layout suits decode lanes
  and short chunks. `?window` bounds what is seen; a read bounded by the
  window is future work behind the same signature.

Compiled programs are keyed by `(batch, seq, rows, context)`, by whether the
index carries `row`, by the slot count and by the dtype. An engine buckets the
first four and builds every index of a bucket the same way.

### Layers and functions shipped with the contract

- `Rms_norm`: `{ gamma }`, with `Layer_norm`'s float32 island at half and
  quarter precision.
- `Rope`: the inverse frequencies of one head, host floats computed once in
  float64. `make ?theta ~head_dim ()` and `llama3 ~theta ~head_dim ~factor
  ~low_freq_factor ~high_freq_factor ~original_context` build one. `apply t
  ~pos x` rotates `[batch; heads; seq; head_dim]` at float32, feature `i`
  paired with `i + head_dim / 2`.
- `Fn.top_k ~k` and `Fn.top_p ~p` replace entries outside the kept set with
  negative infinity. `k` is an int32 tensor and `p` a float tensor, because a
  captured number is frozen at the first trace. Both threshold on `Nx.sort`'s
  values, take float32 logits and keep ties, and `top_p` always keeps the most
  probable token.
- `Loss.softmax_cross_entropy_sparse` computes its log-sum-exp at float32.
- SwiGLU is three `Linear`s and `Fn.silu`.

### Where models live

Models live outside kaun. A model is user code: records, `Params`, `make`,
`cache`, `cached`, `hidden`, `logits`, a config parse and an `of_hf`.
`Checkpoint` and `Kaun_hf` stay architecture-blind. The rejected models
library is under Rationale.

## Laws

Each names who owes it: K for kaun's layers, M for a model author, E for an
engine or any caller of `Cache_index.make`.

1. **Column is position (E).** `table.(r).(j)` holds position `j` of sequence
   `r`, and within a row a slot appears once. A slot shared by two sequences
   is shared at the same position after the same tokens, because a stored key
   is rotated at its position and computed from every token before it. Every
   column a token sees names a slot that this call or an earlier one stored; a
   column left `-1` inside that range reads as a zero key and takes the weight
   of a zero score. Prevents keys read at the wrong position or from another
   prefix, and attention diluted by holes. A sliding window keeps this law: a
   column is freed and becomes `-1` once it is below every window still to be
   fed.
2. **Positions, tables, keys and sampling parameters are inputs of the step
   (M).** A captured one compiles a program for one position or one
   temperature.
3. **Attention is total (K).** A query that sees no key yields zero and a zero
   gradient. Prevents `nan` from padded lanes and empty windows, in decoding
   and in training.
4. **`-1` and any slot outside the pool address nothing, eagerly and compiled,
   and the scratch row is never observed (K).** Prevents the two modes
   disagreeing on a bad index, and results that depend on what an allocator
   left in memory.
5. **A column that is unallocated, past every position of its lane, or below
   every window of its lane contributes exactly zero to that lane, whatever
   its slot holds (K).** Prevents one request's overflow becoming another's
   `nan`.
6. **Every cache leaf is one tensor of `slots + 1` rows whose axis 0 is the
   slot axis, leaves in a fixed order (M).** No two leaves hold one tensor: a
   donated tensor seeds one leaf. Prevents an engine needing model code to
   move state, lost storage reuse, and a compiled program keyed by another
   traversal.
7. **Shapes are static, values vary (M, E).** Prevents a retrace per
   admission.
8. **`cached` is invariant under chunking (K, M).** Given Law 1, tokens fed
   whole, in chunks, one by one, or as one-token lanes of one sequence give
   the same outputs and the same written slots up to reassociation, and a call
   over `Cache_index.whole` agrees with one over `Cache_index.rows`: at
   float32 to `1e-4 * max 1 |logit|`; at bf16 relative logit error under
   `5e-2` against the float32 reference. The constants are provisional until
   calibrated. Kaun owes it for its layers. A model author owes that
   everything outside kaun's layers acts on each token alone and takes
   positions from `Cache_index.positions`. Prevents a model that trains
   cleanly and decodes garbage.
9. **Conventions that import depends on are the layer's (K).** RoPE's pairing
   and the query-to-kv-head grouping are pinned by a gradient check and a
   reference-logit test at `groups > 1`.
10. **Write targets are distinct (E).** Within a call no two tokens sit at one
    position of one sequence, and no token sits at a slot that another
    sequence's table names: a shared slot is one an earlier call wrote. Two
    tokens aimed at one slot leave it holding, element by element, an
    unspecified one of their stores, and the eager and compiled runs may
    differ there; every other slot is exact. Prevents a fork rewriting its
    sibling's past and results that depend on store order.

## Drawbacks

- Addressing code grew: `attention.ml` went from 523 to 369 lines and
  `cache_index.ml` adds 255. The difference buys windows, the row indirection,
  the whole index and leaves of any rank and dtype.
- A cache index is one of two kinds behind one type, and `advance` raises on a
  whole index.
- An index is recomputed per leaf and relies on the compiler to share the
  work. An eager run pays it per layer, which only tests do.
- Each sampling mask is a bitonic sort of the vocabulary under jit until
  `Nx.top_k` exists.
- Laws 1 and 10 are obligations kaun cannot check inside a compiled step. An
  engine checks them in its own tests.
- The bf16 clause of Law 8 is checked by the Llama validator only; equal
  greedy tokens over many steps has no test.

## Rationale and alternatives

Three redesigns were written blind to each other: from model code, from an
engine's core, and from laws. All three chose an opaque address, a scratch
row, unordered duplicates, total attention, pools with slot axis 0, and
deferring a `decode` transformation. No production engine lets model code see
addressing.

**Attend over the past from the old pool, then over the call's own tokens,
merge by log-sum-exp, then write.** The strongest alternative: training needs
no branch, and draft tokens need not be columns. Built and measured, it costs
12% to 16% per decode step and 60 more kernels, and a sequence fed as
one-token lanes fails under it, which is how engines flatten a batch. Its
storage reuse held by a schedule order that nothing enforces. The log-sum-exp
and its merge ship when a layer splits a read; `extend`'s signature does not
change.

**The first revision of this RFC.** Implemented the day it was published and
never released. Its flat pool and its position-to-slot table survive. It lost
on four points. Its address was a public record, `Span.t = { pos; slots }`,
whose fields layers read, so the row indirection and the whole index would
each have changed every model and every layer. It ordered duplicate writes,
the later token winning, which cost an inverse map of `slots x tokens`
compares per call and one select over the pool per leaf, and on the indexed
store would have forced a serial range and a repeated write for padding, for a
case no allocator produces. Totality was an obligation on every mask builder,
and `causal_mask` kept its diagonal for that reason alone. `hidden` was
defined as `cached` over a fresh cache and implemented as a second pass,
because the definition cost up to three times as much to differentiate.

**A `decode` transformation in rune that derives the step from `hidden`.**
Outside mixing layers a decoder acts on each token alone, and such a function
is its own chunk form, so the transformation would derive nothing there. What
it adds is state threading, naming, and a static refusal of non-causal code.
It depends on this addressing either way and can take a cache index as its
argument later.

**A write target given by the caller.** All three redesigns chose it, for
speculation trees and for slots that hold a block of positions. With
write-then-read it is redundant with the table for every case built here, and
a token stored where its table does not name it never sees itself. It is left
out until a layout needs it.

**`read` and `write` as the index's verbs, with the layer branching on a whole
index.** A mode every layer author must match, and two halves that invite the
read order rejected above. `extend` performs both halves and matches the whole
index in one place.

**State declared as typed leaf lists.** Infers the state type and costs a list
of typed leaves at every call site plus a module that exists to hold it.
Ordinary records with `Ptree.S` do the same work.

**A constant slot 0 holding each pool's initial value.** Removes the redirect
and costs a select on every write plus a frame law, and an engine would index
pools with an offset.

**The engine supplies attention as a closure.** Addressing never crosses as
data, and the closure grows one verb per kind of state, or the engine learns
model arithmetic. An abstract index gives the same opacity with the verbs in
kaun.

**Pool rings for sliding windows.** They break prefix sharing and make a
column's position data. A windowed layer's table frees old columns and keeps
Law 1.

**A second, paged cache type; a mode on one function; a models library in
kaun; a `Decoder` module type.** Rejected: a block table is an affine function
of a slot index; a mode makes every caller match a case it knows cannot
happen, where `Cache_index.whole` is matched once inside `extend` and models
pass it through; a models library is what kaun removed to stay a small library
of layers; a signature cannot say "validated logits" and Laws 8 and 9 can.

## Non-goals

The engine: scheduling, allocation, request state, prefix caching, preemption.
Each of the following is left out and arrives behind the opaque index or as a
new module: slots that hold a block of positions; per-query reads at columns a
layer computed; per-sequence state for recurrent layers; speculation trees; a
window-bounded read. None changes the signature of a model written today; a
model whose layers allocate state separately will take one index per table.
Caches that evict or reorder, where a column's position is data. Quantised
state and tensor-parallel layout, which need compiler work first. Packed
training sequences with position reset. Mixture of experts. `Rune.remat` under
jit, an identity today.

## Unresolved questions

None blocks the contract.

- During implementation: re-record `kaun/bench/decode`'s baseline on a quiet
  host, and calibrate Law 8's constants.
- Before an engine exists: a rune test that alternates two compiled programs
  on one donated state and reports reuse for every leaf. Single-program reuse
  and pass-through are pinned today.

## Future possibilities

A surface for each capability under Non-goals, behind the index:
`Cache_index.make ?every` for slots that hold a block of positions, a
per-query read, a module for per-sequence state. A second layout constructor
for ragged batches once a variable-length kernel exists. The battery as a
function over a step. A checker that refuses token-mixing operations outside
kaun's layers. In nx and tolk: `Nx.top_k` with indices, a row-form store that
takes `[k]` indices and `[k; ...]` rows, and the fused attention kernel that
consumes the table. Nothing listed here is a reason to accept this or a later
RFC.
