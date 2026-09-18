# RFC 0002: The decode contract

- Status: committed
- Date: 2026-09-18
- Packages: kaun, rune (one pullback fix)
- Implementation: branch `decode-contract`

## Summary

A decoder model in kaun is one parameter record and one block body. It
exposes `hidden`, the residual stream of whole sequences, and `cached`, the
same for tokens that attend through a key-value cache, and `hidden` is
defined as `cached` over a fresh cache. A cache is a flat pool of slots with
no batch axis. A call says where its tokens sit with two int32 tensors: each
token's position, and a map from a sequence's positions to slots. Contiguous
decoding, paging, shared prefixes and beam forks are values of that map, so
kaun has one cache type, one cached attention, and no notion of a block.
Sampling is two logit masks beside what nx already has. Kaun ships no
model: the contract is kaun's types and a convention, so training, a
generate loop and a future inference engine run one definition that lives
in user code.

## Motivation

GPT-2 is the only decoder in the tree and it lives in an example. Its
forward pass exists twice: `logits` and `logits_cached` share no body, and
about twenty-two lines exist in both (`examples/04-gpt2/gpt2.ml:183-268`).
The copies already differ: one has dropout and returns every position, the
other has neither.

`Attention.apply_cached` assumes one position for the whole batch in four
places: a one-element guard, a rank-0 reshape, an `A` on the batch axis of
the write, and a mask whose leading extent is 1 (`attention.ml:180-212`).
Sequences at different positions cannot share a batch, and the cache's batch
extent is fixed at construction, so an engine that admits or retires a
request changes the compiled program's signature and retraces.

The next kaun milestone is Llama-class models with validated import, and
mixed-precision training runs on the same models. Against today's surface
they would be written twice, and an engine would write them a third time,
because a signature built around one contiguous past has nowhere to receive
an engine's addressing. That is how a serving stack comes to own a separate
model zoo. The contract has to exist before the models do.

## Guide

### A model

```ocaml
(* a model, in user code: llama.mli; 'x is (float, 'b) Nx.t *)
type 'a block = {
  attn_norm : 'a Rms_norm.t; attn : 'a Attention.t;
  ffn_norm : 'a Rms_norm.t;
  gate : 'a Linear.t; up : 'a Linear.t; down : 'a Linear.t;
}

type 'a params = {
  tok : 'a Embedding.t; blocks : 'a block list; norm : 'a Rms_norm.t;
  head : 'a Linear.t option;  (* None: tied to [tok] *)
}

module Params : Nx.Ptree.Uniform with type 'a t = 'a params
module Cache = Kaun.Attention.Cache.List       (* one cache per block *)

val make : config -> Nx.float32_t params
val cache : config -> slots:int -> (float, 'b) Nx.dtype -> 'x Cache.t

val cached : config -> 'x params -> 'x Cache.t -> Span.t -> Nx.int32_t -> 'x * 'x Cache.t
(** The residual stream, [[batch; seq; dim]], of tokens that sit where the
    span says and attend through the cache, and the cache with their keys
    and values written. *)

val hidden : config -> 'x params -> Nx.int32_t -> 'x
(** The residual stream of whole sequences: [fst (cached ...)] over a fresh
    cache. *)

val logits : config -> 'x params -> 'x -> 'x
(** Final norm and head, per position. Select positions before calling. *)
```

Arguments run options, parameters, carried state, inputs; results are output,
carried state. Parameters, state, inputs is `Batch_norm.apply`'s order, and
every function here follows it. A model may add its own optional arguments to
`hidden` (GPT-2's `?dropout`).

### Training

```ocaml
let loss p (ids, targets) =
  Loss.softmax_cross_entropy_sparse (Llama.logits cfg p (Llama.hidden cfg p ids)) targets
```

### Generating

```ocaml
let step ({ Step.tokens; span; key; temperature; k; p; cache } as s) =
  let h, cache = Llama.cached cfg params cache span tokens in
  let logits = Nx.cast Nx.float32 (Llama.logits cfg params (Nx.slice [ A; I (-1) ] h)) in
  let ks = Nx.Rng.split key in
  let next = Nx.Rng.categorical ks.(1) Nx.(Fn.top_p ~p (Fn.top_k ~k (div logits temperature))) in
  { s with tokens = Nx.reshape [| batch; 1 |] next; span = Span.advance span; key = ks.(0); cache }

let step = Rune.jit2 ~donate:true (module Step) (module Step) step
let s = ref (step { tokens; key; temperature; k; p; span = Span.rows ~context lens;
                    cache = Llama.cache cfg ~slots:(batch * context) dt })
```

`Span.rows ~context lens` pads rows on the left, and the caller pads
`tokens` the same way, so the last column of `h` is every row's last real
token. The sampling parameters are fields of the step because a captured
tensor is a constant of the program; the selected logits are cast up so the
masks and the draw run at float32. The first call has `seq = max lens` and the
rest `seq = 1`: two compiled programs from one source, as today. `Step` is
the caller's record of seven fields; its three traversals are about twenty
lines using `Span.map`, `Cache.map` and the polymorphic function on the rest.
Greedy decoding drops the key and calls `Nx.argmax`.

### What an engine does per tick

One call per `(batch, seq, context)` bucket that has work, threading the
cache through them: typically a prefill call at `[1; chunk]`, then a decode
call at `[batch; 1]`. Rows shorter than `seq` are padded on the left with
`pos = -1`; empty rows are `-1` throughout. One padded call mixing both is
legal and costs `batch x seq` lanes. `slots` is engine state kept on the
device and patched when allocation changes; only `pos` and the tokens are new
each tick. Admitting or retiring a request changes values, never a shape, and
the engine never touches the cache's contents.

An engine depends on neither kaun nor a model. It is parameterised by a step
function over nx tensors and an opaque state: tokens, positions, slots and
sampling parameters in, tokens and the state out. The caller writes that
step from `Span.make`, its model's `cached` and `logits` and the masks, and
compiles it. The addressing rules of this RFC are the protocol between the
two, and an engine can check any step against them behaviourally: a prompt
fed in chunks gives the tokens it gives whole, and a padded row does not
disturb its neighbours.

## Reference

### The span

```ocaml
module Span : sig
  type t = private { pos : Nx.int32_t; slots : Nx.int32_t }
  val make : pos:Nx.int32_t -> slots:Nx.int32_t -> t
  val rows : context:int -> int array -> t
  val advance : t -> t
  val positions : t -> Nx.int32_t          (* effective positions, in range *)
  val map : (Nx.int32_t -> Nx.int32_t) -> t -> t   (* map2, iter likewise *)
end
```

`pos : [batch; seq]` is each token's position in its sequence, or -1 for
padding. `slots : [batch; context]` names the slot holding position `j` of
row `b`'s sequence, or -1 when none is allocated. `make` raises unless both
have rank 2 and equal batch. `rows ~context lens` gives row `b` the slots
`b * context ..` and its `lens.(b)` tokens the positions `0 ..`, left-padded
to `max lens`; the matching cache has `batch * context` slots. `advance`
replaces `pos` with `1 + max_i pos.(b).(i)`, shape `[batch; 1]`; a row of
padding advances to 0.

One scalar did four jobs. This RFC separates one: where a key is stored.
The other three stay one tensor, `pos`: what RoPE rotates by, which column
a token occupies, how far it sees. It is a record so that addressing can
grow a field without changing a model's signature.

**An address outside its range addresses nothing.** A token whose `pos` is
outside `[0, context)` is padding: it writes nothing and its output is
unspecified. Its effective position is 0: with `p = where (0 <= pos <
context) pos 0`, rotation and visibility are stated on `p`, so padding
rotates as position 0 and sees column 0 only. A column whose slot is outside
`[0, slots)` is unallocated: nothing is written to it and it reads as zero.
`pos` and `slots` are addresses with a no-address value and are never passed
unclamped to `take`; this is why they do not follow nx's index rule, and why
eager and compiled runs agree. A model that indexes a table by position
(GPT-2's `wpe`) indexes it with `p`, which `Span.positions` returns.

### The cache

A cache is a pool of slots: `Cache.t = { keys; values }` with payloads
`[slots; kv_heads; head_dim]` at the parameters' dtype, zeros from
`Cache.make ~slots ~kv_heads ~head_dim dt`. There is no batch axis: who
owns a slot is the span's business. `Attention.Cache.List` is the `Uniform`
traversal of a list of caches, leaf paths `0.keys`, `0.values`, ...; a model
whose carried state is only its blocks' caches re-exports it. Order is fixed
because compiled programs are keyed by leaf order.

### Cached attention

```ocaml
type route
val route : slots:int -> Span.t -> route
val cached : head_dim:int -> ?rope:Rope.t -> 'x t -> 'x Cache.t -> route -> 'x -> 'x * 'x Cache.t
```

A model resolves the span once per call and hands every block the same
`route`: the token that writes each slot, the clamped read index, the mask
and the positions. It is built inside the step and is never an input leaf.
Per block the attention is `Attention.cached ~head_dim ~rope b.attn c r`.
For `x : [batch; seq; embed]` the layer projects, rotates queries and keys
at the positions, writes, reads, and attends.

- **Write.** Token `(b, i)` targets `slots.(b).(pos.(b).(i))` when its
  position is in range and nothing otherwise: `where (0 <= pos < context)
  (take_along_axis slots p) (-1)`. `route` inverts that into a `[slots]`
  int32 vector, the last token in row-major order that targets each slot or
  -1, by one compare-and-max over the call's tokens. The new leaf is `where
  (inv >= 0) (take new (max inv 0)) old`. Two tokens aimed at one slot: the
  later wins, `Nx.scatter`'s rule.
- **Read.** `take` along the slot axis with `slots` clamped, giving `[batch;
  context; kv_heads; head_dim]`, then zero wherever the column is unallocated
  or beyond every position of the row. Columns no query of the row may see
  therefore contribute exactly zero, not `0 * v`.
- **Mask.** Column `j` is visible to token `(b, i)` when `j <= p.(b).(i)`, so
  every query sees column 0.
- **Grouping.** Queries reshape to `[batch; kv_heads; groups; seq; head_dim]`
  against keys at `[batch; kv_heads; 1; context; head_dim]`. `head_dim` is
  the one integer shared by queries, keys, the cache and RoPE; both head
  counts are read from the projection widths. `Attention.make ?q_dim ?kv_dim
  ~embed_dim` sizes the projections.

`Attention.apply ~head_dim ?mask ?rope p x` is attention of tokens over
themselves. `?mask` replaces `?causal:bool` and broadcasts to `[batch; seq;
keys]`; the layer inserts the head axes. `causal_mask : seq:int ->
?valid:Nx.bool_t -> unit -> Nx.bool_t` builds the triangle, `[seq; seq]`, or
`[batch; seq; seq]` with `valid : [batch; seq]` marking real keys. It always
keeps the diagonal, so a padded row is never fully masked. `apply_cached` and
its scalar `pos` are removed.

### One definition

```
hidden cfg p ids = fst (cached cfg p (cache cfg ~slots:(batch * seq) dt)
                          (Span.rows ~context:seq (Array.make batch seq)) ids)
```

up to reassociation. That equation is the definition of `hidden`, so there
is one model. A model implements `hidden` as a second fold over the same
block body with `Attention.apply`, because under jit reverse mode through
the read is a scatter-add into the cache, which tolk lowers as a reduce over
the `batch x context` indices for every cache element
(`rune/lib/reverse.ml:676-681`, `tolk/lib/frontend/op.ml:453-462`), and the
write is a pass training discards. Measured on one attention layer on CPU,
the compiled gradient of the right-hand side returns the same values and
costs 1.2 to 1.5 times the second fold's at batch 1 and 3.1 times at batch
8, the ratio growing with the batch as the lowering predicts. The block
body takes the attention as an argument; the two folds are five lines each.
The second fold is deleted when that ratio reaches one, which takes a
scatter-add that visits the indices and not the destination. `cached` stays
differentiable and is tested for it, eagerly and compiled; it is not a
training path.

### Cost

The kernel shapes below were read from rendered kernels; the measurements
are under Unresolved questions.

A read costs the context: the gather collapses to one gated load per
element, through the clamp and the zeroing select. For one query token
against ungrouped keys the load fuses into the score kernel; for
`groups > 1` or `seq > 1` it lands in a context-sized buffer per leaf per
layer before attention, because tolk realizes a value before broadcasting
it, and under the half-precision island that buffer is float32.

A write costs one pass over the pool per leaf, because rune lowers every
functional write to a select over the destination; main's window write costs
the same against its cache length plus a gather per axis. The inverse map is
`slots x tokens` int32 compares once per call. `jit ~donate:true` writes the
new cache over the old on a device: the path between them is one `where`, and
the read that follows uses the written leaf. On CPU there is no storage
reuse, as today.

One piece of compiler work takes the pool out of the write: an indexed
store, a kernel ranged over the call's tokens that stores at a loaded
index. tolk's gated store would stop moving pool bytes but still visits
every pool element, and using it means stores into donated inputs enter the
compiled graph, which RFC 0001 declined. Until the indexed store exists no
throughput claim may be made from this RFC.

Compiled programs are keyed by `(batch, seq, context)`, the slot count and
the dtype. An engine buckets the first three.

### New layers and functions

- `Rms_norm`: `{ gamma }`, the four-part layer pattern, with `Layer_norm`'s
  float32 island for half and quarter precision.
- `Rope`: `type t`, the inverse frequencies of one head as host floats,
  computed once in float64. `make ?theta ~head_dim ()` and `llama3 ~theta
  ~head_dim ~factor ~low_freq_factor ~high_freq_factor ~original_context`
  (Llama 3.1: 500000., 8., 1., 4., 8192) build one; every published schedule
  is a frequency vector, so a new one is a constructor. `apply t ~pos x`
  rotates `x : [batch; heads; seq; head_dim]`, feature `i` paired with `i +
  head_dim / 2`, angles at float32. No table is threaded.
- SwiGLU is three `Linear`s and `Fn.silu`; no layer.
- `Fn.top_k ~k logits` and `Fn.top_p ~p logits` replace entries outside the
  kept set with negative infinity and keep shapes. `k` is an int32 tensor and
  `p` a float tensor, each broadcast over rows: nx's rule is that a sampler
  takes its parameters as tensors, and a captured number would be frozen at
  the first trace. Both compare against a threshold taken from `Nx.sort`'s
  values and never call `argsort`, whose compiled form is quadratic in the
  vocabulary. Both take float32 logits: the step casts the selected
  position's logits up before dividing, so the masks and the draw run at
  float32. Ties at the threshold are kept and `top_p` always keeps the most
  probable token. Temperature is `Nx.div`; the draw is `Nx.Rng.categorical`.
  Penalties are functions of what was generated, which is engine state; they
  are not shipped.
- `Loss.softmax_cross_entropy_sparse` gains a float32 island inside the loss:
  a bf16 log-sum-exp over 128k classes biases the gradient.

### Where models live

Outside kaun. Kaun had a model zoo and removed it to stay a small library of
layers; this contract does not bring it back. A model is user code: the
records, `Params`, `make`, `cache`, `hidden`, `cached`, `logits`, a config
parse and an `of_hf` built from `rename`/`transpose`/`split`. GPT-2 stays
the example it was, on this contract. `Checkpoint` and `Kaun_hf` stay
architecture-blind and the documentation's "there is no per-architecture
loader in the library" stays true. The contract is a convention, not a
module type: kaun has no layer abstraction and gains none. Kaun tests the
model-level law itself, on a small decoder defined in its test suite, and
its decode bench builds a GPT-2 shaped stack from layers.

## Laws

1. **Column is position.** `slots.(b).(j)` holds position `j` of row `b`,
   and a slot shared by two rows is shared at the same position, because
   keys are stored rotated. Prevents a causal mask that is wrong whenever
   an allocator hands out slots in any order but positional.
2. **Positions, slots, keys and sampling parameters are inputs of the
   step,** never captured or read on the host. Prevents a program compiled
   for one position or one temperature.
3. **Every query row of every mask kaun builds admits a key.** Prevents
   `nan` from a padded lane, in decoding and in training gradients.
4. **An address outside its range addresses nothing;** a repeated slot
   takes the later token; every index that reaches `take` is clamped first.
   Prevents eager and compiled runs disagreeing on a bad index, and a full
   context overwriting slot 0.
5. **A column no query of a row may see contributes exactly zero to that
   row's outputs.**
   Prevents one request's overflow becoming another request's `nan`.
6. **One tensor per cache leaf.** Sharing is two rows of `slots` naming
   the same slots, never two leaves holding one tensor. Prevents losing
   storage reuse, which needs each donated tensor to seed one leaf.
7. **Shapes are static, values vary.** Prevents a retrace per admission.
8. **`hidden` is `cached` over a fresh cache, and `cached` is invariant
   under chunking.** Each model tests both with chunks of 1, of 7 and of the
   whole prompt: at float32 to `1e-4 * max 1 |logit|`; at bf16 equal greedy
   tokens over 32 steps and relative logit error under `5e-2` (constants
   provisional until calibrated on GPT-2). Prevents a
   model that trains cleanly and decodes garbage.
9. **Conventions that import depends on are the layer's.** RoPE pairs `i`
   with `i + head_dim / 2`; a query head uses its group's kv head; each is
   pinned by a `check_grads` and a reference-logit test at `groups > 1`.
   Prevents plausible, wrong logits from a checkpoint.

## Drawbacks

- The write costs the pool until the indexed store exists. Llama 3.1 8B at
  bf16 holds 131 KB per slot across its 64 leaves: a 32768-slot pool is 4.3
  GB read and written every step beside 16 GB of weights, and a 40 GB pool is
  five times the weights. A single-sequence loop pays 4% more than the
  scalar position did at a cache of 1024. A two-call tick pays the pass twice.
- Read from the scheduler's rules, the read is a context-sized buffer per
  leaf per layer for every grouped model and every prefill, which main does
  not have.
- `hidden` has a second implementation per model, tied by a test and not a
  type, until the compiler makes it unnecessary.
- The simple loop mentions a span, one more noun than `pos`.
- Each sampling mask is a bitonic sort of the vocabulary under jit, about 150
  kernels at 128k by the lowering's count, until `Nx.top_k` exists. Float32
  logits for a training batch of 8 x 2048 over 128k classes are 8.4 GB.

## Rationale and alternatives

Two blind designs agreed on the flat pool, the position-to-slot map, no
length tensor, a hidden-state result with a separate head, grouping by
broadcast, no SwiGLU layer, no `Decoder` module type, and no `generate` in
the library. Both also proposed a `kaun.models` library; see below.

**Read inside the layer, write outside.** Attention returns its own tokens'
keys and values as a fresh cache, reads an optional past through the span,
and merging into the pool is a separate pure function. Training is then the
same function with no gather and no pool write, so there is one
implementation. It is the strongest alternative and the answer is not yet:
here attention reads the written pool, so RFC 0001's ordering rule for
storage reuse holds by dataflow; there, the layer's read of the old leaf
and the store of the new one are unordered, and a store scheduled first
refuses the reuse, which for an engine whose pool fills the device is an
allocation failure. It wins once rune orders a store into a donated leaf
after that leaf's other readers.

**The engine gathers contiguous keys and calls a contiguous layer.** The
layer would then return `k` and `v` for the engine to store, a fused kernel
that consumes `slots` has nowhere to go, and a `[batch; ...]` cache makes
admission a retrace.

**A second, paged cache type in kaun.** Same run-time cost as the map, since
a block table is an affine function of a slot index, plus a second cache, a
second cached attention, a block size, and a branch in every model.

**A mode on one function.** A flag or a `Cache.t option` makes every caller
match a case it knows cannot happen. A GADT-indexed context makes the result
type follow the mode; it is still a mode, matched in every layer, and kaun's
first public GADT would buy nothing two names do not.

**`Nx.set` with a `D` window as the write** cannot take a per-row start.
**`Nx.scatter`** lowers to one select per written row, chained
(`op.ml:427-451`): a 2048-token prefill is a 2048-deep expression.

**A flattened `[tokens; 1]` batch** mixes prefill and decode in one call but
makes each prompt token read its context separately. It needs a
variable-length attention kernel; the span admits it when one exists.

**A models library in kaun.** Proposed so that tests, a bench, import
validation and an engine could depend on a model. None of them needs it: the
layer's laws are tested on the layer and the model's on a decoder defined in
the test, the bench builds its stack from layers, import validation belongs
to whoever owns the adapter, and an engine takes a step function. What it
would cost is what made kaun drop its zoo: responsibility for architectures,
checkpoint formats and other people's weights, growing with the field. A
maintained model collection, when one is needed, is its own package.

**A `Decoder` module type** would let an engine be written once. What an
engine needs from a model is validated logits, which a signature cannot say
and Laws 8 and 9 can.

## Non-goals

The engine: scheduling, allocation, request state, prefix caching,
preemption. Any model where column differs from position: a sliding window
held in W columns, tree-structured speculation; both are expressible at a
cost, a sliding window as a mask over a full-length context and a tree as one
row per path sharing prefix slots and recomputing shared ancestors. Packed
training sequences with position reset. Mixture of experts. Quantized caches.
Tensor-parallel decode, which keeps two pool generations because storage
reuse is single-device. `Rune.remat` under jit, an identity today, and the
embedding gradient's scatter-add under jit; both sit under training at scale
and neither is changed here.

## Unresolved questions

None blocks the contract. What was measured while implementing it, on the
CPU renderer with pools of 4096 to 131072 slots and on a Metal device:

- Storage reuse holds with a read after the write in one program, and
  `rune/test/test_jit.ml` pins it.
- The write renders as one pass over the pool with a gated load of the new
  row and no loop over tokens, through the clamp, at 64 and at 2048 tokens.
  The inverse map is a separate `slots x tokens` int32 reduce.
- The read renders with no loop over slots. One query token against
  ungrouped keys fuses the gated load into the score kernel; `groups > 1` or
  `seq > 1` lands in one context-sized buffer first. At 32768 slots and
  above tolk splits the reduce before collapsing it, and the buffer is
  sixteen times the context; that is a tolk ordering fix, not a contract
  matter.
- The decode step of a GPT-2 124M shaped decoder costs 8.89 ms and 10.03 ms
  at cache lengths 256 and 1024, against 8.87 ms and 9.66 ms for the scalar
  position it replaces: the slot indirection costs 4% at the longer cache,
  inside the 5% budget of `kaun/bench/decode`.
- Llama 3.2 1B, written on the contract in `kaun/examples/05-llama`,
  reproduces the reference implementation's float32 logits to a few parts
  in a million, and its cached path fed in chunks equals its whole-sequence
  path exactly.

Open, and out of scope: who builds the indexed store and the scatter-add
that visits indices, the two pieces of compiler work that remove the pool
from the decode write and the second fold from models.

## Future possibilities

A `col` field on `Span.t` for models whose columns are not their positions;
a fused attention kernel that consumes `slots` inside its score loop; a
unified prefill and decode batch; `Nx.top_k` by partial selection. Nothing
listed here is a reason to accept this or a later RFC.
