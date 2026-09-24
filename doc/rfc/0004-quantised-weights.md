# RFC 0004: Quantised weights

- Status: published
- Date: 2026-09-24
- Packages: nx (the `nx.quant` library), rune (the product's rules and its
  lowering), tolk (two kernels), kaun (the gpt-oss example)

## Summary

A quantised weight is a value of one closed type, `Nx_quant.t`, in a library
of its own beside `nx.io`. It is built from the tensors a checkpoint stores,
codes and scales in the published byte layout, so RFC 0003's zero-copy loading
holds; its constructors check that the parts agree, and `Nx_quant.dequant` is
its meaning. Its one product, `Nx_quant.apply ?ids w x`, is `Nx.matmul x
(transpose w)` with `Nx.matmul`'s shapes, or with `ids` the product with each
position's selected expert, where an id outside the experts selects none.
`apply` and `dequant` perform one effect that `nx.quant` declares. Compiled, a
product takes one of three forms, chosen from static shapes by one rule: a
kernel that decodes in registers and reads each packed byte once per tile of
rows, for decode and batched decode; for experts that many routes share, the
routes grouped by expert first, so that each expert is read once per block and
the arithmetic follows the routes; and for long products, a decode followed by
a matmul on tensor cores. Eager, the library runs the reference in
bounded chunks. A quantised weight is never differentiated: a frozen base is
captured, and adapters are trained beside it. MXFP4 ships first, driven by
gpt-oss-20b; block-scaled FP8 is designed here and lands with DeepSeek, whose
V4 stores its routed experts in gpt-oss's MXFP4 and its other linears as FP8.
On the M1 Max the goal is an expert path of at most 0.35 ms per layer at batch
1, from about 3.2 ms today, which brings the decode step from 120 ms to
`F_1` + 8.5 ms, about 53 ms if `F_1` is the 44.5 ms the profile attributes to
the rest of the step. Batched decode at 8 and 32 sequences and a 512-token
prefill have targets of their own.

## Motivation

The gpt-oss-20b decode step takes 120 ms on Metal. In the profiled step
closest to that median, 75.7 ms go to four kernels per layer. One decodes the
four selected experts' `gate_up` to bfloat16 and writes 133 MB near the
machine's bandwidth; the next reads it back and multiplies at 120 GB/s; `down`
does the same, its product at 53 GB/s. The products are slow for the reason
the dense bfloat16 products are: a bfloat16 product accumulates at float32, so
its reduce source is a cast, and tolk's matrix-vector heuristic, which
matches tinygrad's line for line, declines. The form moves 450 MB per layer
where 53 MB of packed bytes are needed. The example materialises the decoded
rows on purpose (`packages/kaun/examples/06-gpt-oss/moe.ml`): with the decode
inside the product the heuristic declines too, and the step was four times
slower. Today's step is about eleven times its 11 ms bandwidth floor.

Neither of the example's forms scales past one sequence. Its Gather form
decodes one expert for every route, so a batch of 32 sequences decodes 128
experts per layer, four times the 32 it holds. Its Dense form, which a prompt
takes, multiplies every token by all 32 experts, eight times the arithmetic
the routes need; a DeepSeek-class model, with 256 experts and 6 per token,
would do 43 times.

Three other defects have the same root, which is that raven has no notion of a
quantised weight:

- **Eager decoding holds 19 GB.** The example's float32 decode over one
  layer's 32 experts keeps about nine temporaries of 2.1 GB alive, so
  validating the real model eagerly needs a workaround that decodes one expert
  at a time.
- **FP8 weights look trainable.** Float8 is a float dtype to rune and vega, so
  an FP8 weight inside a structure passed to `grad` is differentiated and
  updated at FP8 with its scales ignored.
- **Every model repeats the format.** MXFP4 lives in the gpt-oss example.
  DeepSeek V4 stores its routed experts in the same format, and its other
  linears as FP8 with one power-of-two scale per 128 by 128 block
  (its reference inference code stores FP4 experts as `[out; in/2]`
  with `float8_e8m0fnu` scales `[out; in/32]`, and FP8 linears with scales
  `[⌈out/128⌉; ⌈in/128⌉]`); DeepSeek V3 is the same
  with float32 scales.

## Guide

### Loading packed weights

An importer asks for each part by name, as RFC 0003's importers do, and
builds the weight with its format's constructor:

```ocaml
(* gpt-oss: codes [experts; outputs; inputs / 32; 16], one scale per 32 *)
let experts ~inputs ~outputs name =
  let bytes ~shape n = Checkpoint.to_tensor ~shape Nx.uint8 n ckpt in
  let groups = inputs / 32 in
  Nx_quant.mxfp4
    ~scales:(bytes ~shape:[| cfg.experts; outputs; groups |] (name ^ "_scales"))
    (Nx.reshape [| cfg.experts; outputs; inputs / 2 |]
       (bytes ~shape:[| cfg.experts; outputs; groups; 16 |] (name ^ "_blocks")))
  |> Nx_quant.place placement

(* DeepSeek V3: e4m3 values, one float32 scale per 128 x 128 block *)
let linear ~inputs ~outputs name =
  let blocks n = (n + 127) / 128 in
  Nx_quant.fp8 ~block:(128, 128)
    ~scales:(F32 (Checkpoint.to_tensor ~shape:[| blocks outputs; blocks inputs |]
                    Nx.float32 (name ^ ".weight_scale_inv") ckpt))
    (Checkpoint.to_tensor ~shape:[| outputs; inputs |] Nx.float8_e4m3
       (name ^ ".weight") ckpt)
  |> Nx_quant.place placement
```

DeepSeek V4's linears differ only in their scales, which are e8m0 bytes
passed as `E8m0`. A weight's logical shape is the file's `[...; outputs;
inputs]`, with blocks running along `inputs` inside each row. A constructor
raises `Invalid_argument`, naming the part, when the parts disagree.
`Nx_quant.place` places a weight part by part, and raises when a split would
cut a block. `Nx_quant.t` is a `Ptree.S`, so a weight can sit in any parameter
record.

### Using them

```ocaml
let q = Nx_quant.apply p.q_proj x      (* x [...; inputs] -> [...; outputs] *)

(* gpt-oss's experts: each token's four *)
let h = Nx_quant.apply ~ids p.gate_up (Nx.reshape [| tokens; 1; 1; width |] x)
(* [tokens; 4; 1; 2 * inter] *)
```

`apply` has `Nx.matmul`'s shapes. With `ids`, the weight's expert axis is
gathered by `ids` before the product; the gather is an address, and no packed
row is copied. An id outside `[0, e)`, such as `-1`, selects no expert: its
position of the result is zero, and nothing is read for it. `apply` returns
`x`'s dtype. The same call serves one token, a batch and a prompt: compiled,
it becomes a kernel that reads each packed byte once per tile of rows while
each expert meets few rows, groups the routes by expert when many routes
share experts, and decodes once and multiplies when an expert meets many rows.
Eager, it decodes one bounded chunk at a time, whatever the weight's size.

### Fine-tuning over a quantised base

A quantised weight has no gradient. The base is captured by the loss, and
only adapters are parameters:

```ocaml
let adapted base ad x =
  Nx.add (Nx_quant.apply base x) (Nx.mul_s (Nx.matmul (Nx.matmul x ad.a) ad.b) alpha)

let step =
  Rune.jit_step (module Batch) (module State) (fun batch { State.ad; opt; _ } ->
      let loss ad = xent (model base ad batch.x) batch.y in   (* base captured *)
      let loss, g = Rune.value_and_grad (module Adapters) loss ad in
      let ad, opt = Vega.adam_step (module Adapters) ~lr opt ~params:ad ~grads:g in
      { State.ad; opt; loss })
```

`Rune.jit_step` reads its first argument and donates its second (RFC 0005);
the captured base is bound once and never donated. A captured base never
reaches `grad` or an optimiser, so no gradient or moment of its size exists.
If a transformation tracks a part of a quantised weight, `apply` and `dequant`
raise: "a part of a quantised weight is differentiated; capture the weight, or
build it from `Rune.detach`ed tensors". Quantisation-aware training is user
code over float master weights, with the model's own rounding.

## Reference

### `Nx_quant`

`nx.quant` is a library of its own, with top-level module `Nx_quant`, as file
formats live in `nx.io`. It uses only nx's public API and declares its own
effect.

```ocaml
type scale = E8m0 of (int, Nx.uint8_elt) Nx.t | F32 of (float, Nx.float32_elt) Nx.t

type t = private
  | Mxfp4 of { codes : (int, Nx.uint8_elt) Nx.t;         (* [...; n; k/2] *)
               scales : (int, Nx.uint8_elt) Nx.t }       (* [...; n; k/32] *)
  | Fp8 of { values : (float, Nx.float8_e4m3_elt) Nx.t;  (* [...; n; k] *)
             scales : scale;                             (* [...; ⌈n/br⌉; ⌈k/bc⌉] *)
             block : int * int }

val mxfp4 : scales:(int, Nx.uint8_elt) Nx.t -> (int, Nx.uint8_elt) Nx.t -> t
val fp8 : block:int * int -> scales:scale -> (float, Nx.float8_e4m3_elt) Nx.t -> t

val shape : t -> int array                          (* logical [...; n; k] *)
val place : Nx.Placement.t -> t -> t
val dequant : (float, 'b) Nx.dtype -> t -> (float, 'b) Nx.t
val apply : ?ids:(int32, Nx.int32_elt) Nx.t -> t -> (float, 'b) Nx.t -> (float, 'b) Nx.t

include Nx.Ptree.S with type t := t
```

- **The type is private.** Users match on a weight's format and read its
  parts; only the constructors build one. They check shapes (for `Mxfp4`, `k` a
  multiple of 32; for `Fp8`, a scale grid of `[...; ⌈n/br⌉; ⌈k/bc⌉]`, partial
  blocks allowed) and read no bytes, so RFC 0003's Law 3 holds. `map` and
  `map2` check their results' shapes the same way and read no placement, so
  they run under every transformation, `jit`'s placeholders included; `map2`
  raises when two weights differ in format or geometry. Traversals visit codes
  before scales and preserve every part's dtype.
- **Formats.** Codes of 4-bit formats are two per byte, the low nibble first.

  | Format | A value | Checkpoints |
  |---|---|---|
  | `Mxfp4` | e2m1 code × 2^(scale − 127); scale 255 is NaN | gpt-oss experts; DeepSeek V4 routed experts |
  | `Fp8` | e4m3 value × its block's scale, a multiply although V3 names it `weight_scale_inv` | DeepSeek V3 (float32 scales) and V4 (e8m0 scales) linears |

  Fp8 blocks may be partial at the edges: V3's 576-row projection has five
  row blocks.
- **`place p w`** places every part with `p` (RFC 0005). Under a split
  placement it raises `Invalid_argument`, naming the part and the axis, when a
  shard boundary falls off the format's blocks: a leading axis splits
  anywhere, `n` at the scale block's rows, `k` at the group. The parts have
  equal rank, so one placement and one axis serve all of them.
- **`apply ?ids w x`** is `Nx.matmul x (Nx.matrix_transpose (dequant Float32
  w))`, computed as §Precision says and returned at `x`'s dtype, with
  `Nx.matmul`'s shapes: `w` is `[...; n; k]`, `x` is `[...; m; k]` or `[k]`,
  and batch axes broadcast. With `ids`, `w` is `[b…; e; n; k]` and `ids`
  is `[b…; s…]`, their leading `b` axes broadcasting: the product is over `w'`
  of shape `[b…; s…; n; k]`, whose entry at `(b, s)` is `w`'s at
  `(b, ids.(b, s))`, as `Nx.take_along_axis` pairs an index with its operand.
  A weight with no leading axes, as in every single-device call, is gathered
  by `ids` whole.
- **An id outside `[0, e)`, `-1` included, selects no expert,** eagerly and
  compiled, on every form: its position of the result is exactly zero,
  whatever `x` holds there, and nothing is read for it. This is RFC 0002's
  rule that `-1` and any slot outside the pool address nothing, and it holds
  for the same reason: eager and compiled runs agree on every id. Routing ids
  are computed in the graph, and a lane of expert parallelism writes `-1` for
  the experts it does not hold. Duplicate ids are allowed. An empty `ids` or
  `x` gives an empty result. The name keeps `Nx.matmul`'s one meaning.
- **`dequant dt w`** is the values of `w` at `dt`.

### The effect

`apply` and `dequant` perform one effect, declared in `Nx_quant.Effect`, the
vocabulary rune's handlers match on, as rune's `Custom` declares its own:

```ocaml
type (_, _) op =
  | Apply : { ids : (int32, Nx.int32_elt) Nx.t option; x : (float, 'b) Nx.t;
              transpose : bool } -> (float, 'b) op
  | Dequant : (float, 'b) Nx.dtype -> (float, 'b) op
type _ Effect.t += E_quant : { w : t; op : ('a, 'b) op } -> ('a, 'b) Nx.t Effect.t
val perform : t -> ('a, 'b) op -> ('a, 'b) Nx.t
```

`perform w op` performs the effect and, where no handler takes it, runs `op`
eagerly; `apply` and `dequant` are `perform`. It checks an `Apply`'s shapes
before performing, so no handler sees shapes that disagree. Only the reverse
rule sets `transpose`: `apply` leaves it false, and the rule reaches the
transposed product, and its eager loop, through `perform`.
`nx_effect.ml` and `backend_intf.ml` do not change, and no engine implements
anything.

**Eager, when nothing handles the effect,** `apply` runs a chunk loop: each
chunk decodes at most `C` values of the rows it needs, computes its block of
the output, and dies before the next; the blocks are concatenated once. With
`ids`, the loop runs once per distinct expert over the positions that select
it, so it is grouped by construction; a position that selects no expert is in
no chunk, and one gather puts the blocks in position order, reading a zero row
for those positions. `C` is an implementation constant, chosen so that a
chunk's dispatch costs under one percent of its decode on the C backend (2^22
values to start). `dequant` decodes chunk by chunk into a host buffer that it
wraps once full, which RFC 0001's Law 2 allows, so it holds its result and one
chunk, and places the result where its parts live (RFC 0005, Law 2); on a
device that does not share host memory, the result exists on both sides for
the time of that upload. The decode uses byte arithmetic until its last step
and a 256-entry table for power-of-two scales, since nx's `exp2` is not exact
at integer arguments.

**A handler without a rule of its own** (debug) re-performs the operation with
`Nx_quant.Effect.perform` from its clause, as debug does every operation. An
enclosing handler then sees one effect, and an unhandled one runs the chunk
loop.

### Lowering under `Rune.jit`

rune's jit chooses a form from static shapes, per lane (an index of `w'`'s
leading `b` axes). Let `m` be `x`'s rows per matrix instance and `i` the
instances; with `ids`, `i` counts the positions of `ids`, `R = i · m` the
routes and `d` is `w`'s expert extent `e`; without, each matrix of `w` meets
`r` rows, `m` times the instances that share it. Let `p` and `q` be one
matrix's packed and decoded bytes. The kernel's options give its row tile
`M`, the rows of `x` it multiplies per load of a group, and its row bound
`ρ`: the largest `r` with `⌈r/M⌉ · p ≤ p + 2q`, lowered to where the
kernel's arithmetic, which runs without tensor cores, starts to cost more
than its reads. The options are a total function of the device and the
shape, measured; a shape with no valid options, a symbolic dimension, and
every device not yet measured have `ρ = 0`. `ρ` is set per device and `x`
dtype, measured at gpt-oss's `gate_up` expert against decode-then-matmul: 32
on Metal at bfloat16 and float16 and 16 at float32, where the two forms tie;
64 on the CPU at bfloat16 and 2 at float32.

1. **Without `ids`,** a matrix that meets `r ≤ ρ` rows takes the kernel, which
   reads it `⌈r/M⌉` times. Otherwise it takes decode-then-matmul. A stack of
   matrices broadcast against a larger batch of `x` takes decode-then-matmul
   whatever its rows.
2. **With `ids` and one row per instance** (`m = 1`, the form a token's experts
   take), the routes are grouped when `R' > d` and `R' · (R' − 1) / 2d > τ`,
   where `R' = R · d / ē` estimates a lane's own routes, with `ē` the experts of
   every lane (rule 3), so `R' = R` on one device. The left side estimates,
   under uniform routing, the reads of an expert that a route shares with an
   earlier route, which grouping saves; `τ` is the grouping's fixed cost counted
   in reads of one matrix, measured per device and shape. With `R' ≤ d` the
   kernel's blocks are one row (rule 3), and a block of one row shares no read.
   On the M1 Max at gpt-oss-20b's decode shapes `τ = 16`: grouping lost at 32
   routes over 32 experts, where every block is one row (111 against 101 ms per
   step), and won from 64 (135 against 150 ms; 213 against 249 ms at 128), so
   `τ` lies between 0 and 63 there; nothing between 33 and 63 routes was
   measured. It is infinite on the CPU, where grouping lost at every size
   measured (16 to 512 routes of gpt-oss's MoE block; at 512, 10.2 against 7.1
   s) with the block kernel unoptimised, and on every device not measured.
   Ungrouped, and with `m > 1`, each instance takes rule 1 with `r = m`.
3. **Grouped,** the blocks take the kernel while the rows an expert meets on
   average, `R / ē`, are at most `ρ`, each block as many rows as that average
   rounded up to a power of two, at most `M`: a row the kernel carries costs a
   share of a matrix read, and gpt-oss-20b's decode step at 128 routes took 281
   ms in blocks of `M` = 8 against 213 ms in blocks of 4. Here `ē` counts the
   experts of every lane: `d` times the lanes of a weight split on its leading
   axis, and `d` itself on one device, so `R / ē = R / d` until split parts
   (Stage 2). A weight mapped over lanes on one device (vmap) counts `d` times
   its lanes as experts, and `R` counts every lane's routes, since one ranking
   serves them all. Under expert parallelism a lane's `R` counts every route of
   the step, most of them −1, so `R / d` would overstate its rows per expert by
   the number of lanes. Otherwise the blocks take decode-then-matmul, in blocks
   of `B` rows, the largest of the matmul's row tiles at most `R / ē` and a
   multiple of the product's row tile per work group: on Metal 64, 32, 16 or 8
   rows, a tensor-core tile of 8 upcast by up to 8. The padding is then at most
   about the routes' own rows, and the arithmetic scales with the `k` experts a
   token selects, whatever `e` is. If the grouped prefill has stopped (§Target),
   a call that would take this branch takes the dense form instead: every row of
   `x` multiplied by every expert of its lane, each position keeping its own
   expert's product. Per product that is `e` times the routes' arithmetic where
   `x` is per route, as gpt-oss's `down` is, and `e / k` times where it is per
   token, as `gate_up` is: for gpt-oss twice the arithmetic of the example's
   Dense form, which ran each expert's whole block on every token.

No form multiplies a row by an expert that did not select it, outside the
grouped prefill's stop outcome (§Target) and, until split parts land, a program
over several devices (below).

**The kernel** is tolk's `Op.quant_matmul`, beside `Op.scatter_indexed`, built
through the ported `Tensor.custom_kernel` with its optimisation options
pinned, so neither the heuristic nor a search picks them. For each output row,
32-value group and tile of `M` rows of `x`, it loads the group's code bytes
and its scale byte once, decodes in registers, takes `M` float32 dot products,
and multiplies each partial sum by the group's scale 2^(s − 127): the float
whose bits are `s << 23` for 1 ≤ s ≤ 254, 2^−127 (bits `0x00400000`) for s =
0, and NaN for s = 255. Row addresses with `ids` are computed at tolk's index
dtype. A position outside the gate is written as zero. On a GPU its loop over
`k` is an outer loop, whose bound is zero for a work group whose positions all
fall outside the gate, around the constant loop its options split, as in the
block kernel (below), so no load of `w` or `x` runs for it; on the CPU the
bound is constant and the loads are gated. In a bounded loop of at most one
iteration every use of the index folds to 0, and the reduce over the loop,
left unparented, is rewritten to its body times the loop's size (tinygrad's
`reduce_unparented`, ported), which evaluates the loads of a position outside
the gate. So the outer loop keeps two iterations or more, and a row of a
single group (`k = 32`) is gated at its loads on a GPU too. Its
pinned options keep the position axis off local and upcast dimensions, and its
builder raises if they do not. It takes each part as whole storage, contiguous
along its last axis; loaded and placed parts are, and a part that is not is
copied on every call, which `RUNE_JIT_DEBUG` reports. The options are
measured: on Metal, the group amount is the largest divisor of `k / 32` at
most 15 (15 for gpt-oss's 90 groups: 45 to 49 µs per expert against 58 µs at
6, tinygrad's matrix-vector amount), with 4 columns per work group and up to 4
per thread, and `x`'s rows are upcast by `M` up to 8, columns per thread times
rows at most 4, beyond which registers spill. The kernel is an additive
`DIVERGENCES.md` entry, since
tinygrad's only fused quantised products are AMD kernels in `extra/`, and
tolk's tests hold it to the reference on every device.

**The grouped form** is a composition in rune's lowering, with no kernel of
its own. Each route is a row of `x` and an id. A one-hot of the ids against
`[0, d)` and its running sum over the routes give each route's rank among its
expert's routes. The running sum is split in chunks of 256, as tinygrad's
`cumsum` splits it, so it costs about `256 · R · d` integer operations: 1.6e9
for DeepSeek's 4096-token prompt, where tolk's unsplit scan costs `R² · d`,
1.5e11. A running sum of each expert's block count `⌈c / B⌉` gives its first
block. A route's slot is its expert's first block times `B` plus its rank. The
rows are gathered into `[nb; B; k]`, where `nb = ⌈R / B⌉ + d` is a static
bound that counts every position of `ids`. A slot that no route fills reads
nothing, and a block that no route fills has id −1. Filled blocks come first,
in expert order. The blocks are multiplied as an `ids` product with `m = B`,
and each route's result is gathered back from its slot. A filled block reads
its expert once and multiplies its `B` rows. An empty block reads no weights
and, on a GPU, runs no multiply-adds, so it costs a launch, one read of its id
and a store of zeros (the block kernel, below). The arithmetic is the routes'
plus at most `d · B` rows of padding. The CPU does not group (`τ` is infinite
there) until the block kernel has options measured on it. A route with no
expert has no slot, so under expert parallelism, where a lane's ids name its own experts and
−1 the others, a device reads and multiplies only for its own routes. The
ranking, the gathers of rows and the empty blocks' zeros follow the bound,
which counts the step's routes on every lane.

**Decode-then-matmul** decodes `min(i, d)` matrices. With `ids` and `i < d`,
it gathers the selected experts' packed rows first, as the example does today;
otherwise it decodes every matrix once, and tolk's block kernel,
`Op.block_matmul`, multiplies each instance or block by the decoded matrix
its id addresses, reading it in place. A copy of the gathered matrices, which
tolk makes for a gather that feeds its ordinary matmul, as tinygrad does,
would write `nb · n · k` values per product: 2.1 GB for gpt-oss's `gate_up` at
a 512-token prefill. It decodes at `x`'s dtype when that dtype has float32's
exponent range (bfloat16, float32), otherwise at float32. Without `ids`, and
for the gathered rows when `i < d`, it multiplies with tolk's ordinary matmul,
which keeps tensor cores where they exist; there a position that selects no
expert is set to zero by a select after the product, since its gathered row
is zero and `0 · x` is NaN where `x` is not finite.

**The block kernel** is built through `Tensor.custom_kernel`, as the kernel
is, with tensor-core options pinned per device and shape class. Its loop over
`k` is written as two:

- an outer loop over tiles of the tensor core's depth, whose bound is zero
  when the block's id is outside `[0, e)`;
- an inner loop of that depth, which the tensor-core option splits.

A block that selects no expert reads its id, runs no multiply-adds and stores
zeros: the reduction's identity, with no select after it. The bound is one
value per work group, so barriers and tensor-core instructions stay
convergent. The pinned options never split or vectorise the block axis, and
the kernel's builder raises if they do: an upcast block axis turns the bound
into a vector that does not compile. On the CPU, tolk runs work groups as a
loop, and a loop bound that reads that loop's index miscompiles, the
reduction's accumulator reset moving inside that loop, so there the bound is
constant and a select on the id zeroes the store. A contraction of one input
takes that form on every device: a bounded loop of at most one iteration drops
out of its reduce, as the kernel's does (above), so the bounded loop keeps two
trips or more and the depth of 8 applies only past 8 inputs. At gpt-oss's `down`
shape, with 46 of 65 blocks of 64 rows filled, it measures 16.9 ms on the M1
Max at float32, where tolk's matmul over the gathered matrices takes 23.9 ms
plus 10.4 ms to copy them, 34.3 ms in all; the rejected gated rule (Rationale)
brings that matmul to 16.8 ms and keeps the copy. Stage 1's build, at bfloat16
with its options pinned on Metal (tensor cores, rows upcast by up to 8 tiles,
columns by 3 and split 4 ways across a work group), measures 6.6 to 7.6 TFLOPS
on 46 filled blocks of 64 rows at gpt-oss's shapes, where `G` is 8.2; 64 empty
blocks take 120 µs, the store of their zeros. Its results are exact at
float32, float16 and bfloat16. The kernel is an additive `DIVERGENCES.md`
entry, and tolk's tests hold it to the reference on every device.

**Over split parts** (RFC 0005), the kernel runs once per shard, built from
the shard's shapes. The lowering takes it when every part is split on the
same axis at a block boundary and `x` and `ids` are split or replicated to
match: a split leading axis, the lane of expert parallelism, and a split `n`
give a result split on the matching axis, and a split `k` gives partial sums,
which the lowering reduces with tolk's allreduce. Grouping and the block
kernel run per lane. Any other layout takes decode-then-matmul, and
`RUNE_JIT_DEBUG` reports it. Until split parts land, a program over several
devices groups nothing, since the ranking would run along a sharded axis, and
multiplies as many positions as experts or more in rule 3's dense form, whose
products hold `e` values per output where a copy of one decoded matrix per
position would hold `n · k`.

### Transformations

- **reverse:** the cotangent of `x` is the same effect applied to the
  cotangent with `transpose` flipped, summed over the batch axes along which `x`
  was broadcast against `w`, or against `w'` with `ids`, as `Nx.matmul`'s rule
  does; a position that selects no expert has a zero cotangent. Eagerly it
  runs the forward's chunk loop; compiled, it follows the forward's rule
  without the kernel, so it is grouped where the forward is and otherwise
  decode-then-matmul. The tape holds `w` and `ids`, and nothing decoded.
- **reverse and forward** raise when a part of `w` is tracked or has a
  tangent, for `apply` and `dequant` alike. Under `grad`, integer parts passed
  inside differentiated parameters are carried like any integer leaf; `jvp`
  takes a tangent for every leaf it is given, so under `jvp` the weight must
  be captured.
- **forward, vmap:** the forward rule is `apply ~ids w ẋ`. vmap puts a lane
  axis at the front of `x`, of every part and of `ids`, inserting a unit axis
  where one is unbatched, so the lane is a leading `b` axis: each lane of a
  weight mapped over a leading axis (experts split one slice per device,
  RFC 0005) gathers its own experts with its own ids.
- **debug** logs the call and performs it again in the enclosing context: a
  jit around it lowers it, and eagerly it runs the chunk loop.

### Precision

- **Decode** is exact barring overflow (MXFP4 codes of magnitude 4 or more at
  scale byte 253 and 2 or more at 254, and FP8 values at scale bytes 247 and
  above). Every other MXFP4 value is a float32 and a bfloat16. An e8m0-scaled
  FP8 value is a float32, and a bfloat16 from scale byte 3 up. V3's
  float32-scaled FP8 rounds once, at float32, and once more when
  decode-then-matmul decodes it at bfloat16.
- **Accumulation** is at float32 and the result is rounded once to `x`'s dtype.
  Decode-then-matmul may round each decoded weight to `x`'s dtype, and a path
  without tensor cores may round each product to `x`'s dtype before the
  float32 sum, as tolk's matmul does.
- **Scales** multiply group partial sums. For power-of-two scales this equals
  the per-value product bit for bit when both sums run in the same order,
  barring overflow and underflow.
- **Batch shape.** The forms differ within Law 2's bound, and the form is
  chosen from the call's shapes, so a row's result can change with the rows
  and routes it shares a call with: `apply` is not batch-invariant.
- **Subnormals.** Metal flushes subnormal float32: scale byte 0 decodes to
  zero there, and a byte-1 group loses its ±0.5 codes. gpt-oss-20b's scale
  bytes span 115 to 136 (all 597,196,800 measured), and DeepSeek's quantiser
  never writes byte 0. Per term such a device loses a byte-0 group's value,
  at most `3 · 2^−126 · |x|` since the flushed scale zeroes values up to
  `6 · 2^−127`, a product below `2^−126` and a running sum below `2^−126`,
  hence Law 2's `k · 2^−126 · (3 · max |x| + 2)`. A subnormal `x` on such a
  device is outside Law 2.
- **Activation quantisation** is not part of the product. DeepSeek rounds
  activations to e4m3 per 128 values before each product, a relative error of
  about 2.7% RMS, sixteen times bfloat16's, and V4 was trained that way. raven
  computes at bfloat16 or float32; a model that wants the reference's bits
  writes the rounding in its own code.

### Target and kill criterion

Every target is measured on the real gpt-oss-20b, on Metal, on the M1 Max, at
bfloat16. The expert path at batch `N`, `E_N`, is a decode step's time for `N`
sequences minus the time of the same step with each expert product replaced
by zeros of its shape (`F_N`, the floor). In the profiled batch-1 step closest
to the 120 ms median, the expert kernels take 75.7 ms, about 3.2 ms per layer,
and the rest 44.5 ms, which estimates `F_1`.

**Stage 0, measure and commit nothing (at most two days).** Measure `F_1`,
`F_8` and `F_32`; today's `E_8`, `E_32` and expert path of a 512-token
prefill; `P`, the fused composition (arithmetic decode, no `contiguous`) with
the matrix-vector options pinned onto its kernel in a scratch harness, as
tinygrad's own linearizer tests pin options, which is an early reading of the
kernel at one row; and `G`, tolk's bfloat16 matmul throughput at `[512; 2880]`
by `[2880; 5760]`.

| Form | Measured | Target | Ship, restating the target | Stop |
|---|---|---|---|---|
| Kernel, one row | `E_1` | at most 0.35 ms per layer: 150 GB/s over the selected experts' packed bytes, 8.5 ms per step | 0.35 to 0.66 ms | above 0.66 ms (80 GB/s) |
| Kernel, row tile | one `gate_up` expert without `ids`, at 8 and 32 rows against 1 row | 8 rows within 3.5 times, 32 rows within 12.9 times (restated at Stage 1's measurement; first 1.25 and 2.5 times) | 8 rows within 4 times, the 32-row figure restated | 8 rows above 4 times |
| Grouped, decode | `E_8`, `E_32` | 150 GB/s over the packed bytes of the distinct experts the step selects, counted from its ids (restated at Stage 1's measurement: 114 GB/s at batch 8, 61 GB/s at batch 32) | 80 to 150 GB/s | below 80 GB/s |
| Grouped, prefill | expert path of a 512-token prefill | the routes' arithmetic at half of `G` or better (restated at Stage 1's measurement: 0.41 of `G`) | a quarter to a half of `G` | below a quarter of `G` |
| Grouped, one lane of eight | expert path of a 512-token prefill whose ids for experts 4 to 31 are −1, as lane 0 of eight (`ē` = 32: `B` = 64, 36 blocks) | within 1.25 times the same product called with only the positions of experts 0 to 3 | 1.25 to 1.5 times | above 1.5 times, or a Law 2 failure on a GPU |

Under uniform routing the grouped decode target is about 1.9 ms per layer at
batch 8, where the step selects about 21 of 32 experts, and 2.8 ms at batch 32,
where it selects about 31 (derived). The 512-token prefill's routes do 2.45
TFLOP of expert arithmetic per step, an eighth of today's Dense form. That
lane fills about 6 of its 36 blocks under uniform routing, and multiplying
the gathered matrices with tolk's matmul it would multiply all 36 (derived).

Stage 1 measured the kernel's first two rows. At one row the expert kernels
take 0.26 ms per layer, in the target, and the real 20b decodes a token in 48
to 54 ms. The row tile takes 3.5 times one row at 8 rows and 12.9 times at 32,
in the ship band: each further row's cost is attributed to reloading `x`, a
thread holding a group's 32 inputs per row.

Stage 1 measured the grouped rows on the real gpt-oss-20b, bfloat16, the M1
Max, each step against its floor, with the kernel in place. The 512-token
prefill's expert path takes 0.73 s, 0.41 of `G`, in the ship band; the prefill
takes 1.09 s against 3.14 s ungrouped. The one-lane row is 1.02 to 1.08 times,
in the target. At batch 8, rule 2 leaves 32 routes over 32 experts ungrouped,
and the kernel takes each route: `E_8` is 45 ms per step, 114 GB/s, in the
ship band. At batch 32 the routes are grouped in blocks of 4 rows on the
kernel: `E_32` is 135 ms per step, 61 GB/s, in the stop band, bound by the
kernel's cost per row. It ships, as it beats today's form (701 ms per layer at
Stage 0) and the ungrouped kernel (249 against 209 ms per step); the kernel's
row tile is what moves it. The prefill's remaining cost is the blocks' padding,
about 1.9 times the routes at gpt-oss's routing, and decoding every expert,
about 10 ms per layer.

Each form is time-boxed: the kernel at five days from its first compile, its
row tile at three more, and the grouped form, with the block kernel, at five.
At the end of its time box a form in its target is done, and one in its ship
band ships with its target restated at the measurement. One in its stop band
stops work; it ships anyway, with its target restated, if it beats today's
form at its measurement point, and otherwise the rule excludes it: at one row,
the compiled product is today's form, the selected experts' rows decoded at
bfloat16 into a buffer that dies with the product and then multiplied; without
the row tile, `M` is 1; without grouped decode, `τ` is infinite; without
grouped prefill, a prompt takes the per-product dense form of rule 3; and
without the block kernel, the blocks multiply the gathered matrices with tolk's
matmul, empty blocks included, and RFC 0005 states that expert-parallel prefill
divides reads only. The surface of `Nx_quant` is the same in every outcome.

If Stage 0 measures `F_1` at 45 ms or more, this RFC states no step figure
and keeps `E_1` alone.

### What placement provides

A quantised weight is placed and sharded as its parts, so RFC 0005 needs no
notion of a format. `Nx_quant.place p` places it and checks that a split falls
on the format's blocks; `map` reads no placement, so it runs inside compiled
programs, where RFC 0005 answers no placement query. Expert parallelism stacks
a layer's experts as `[devices; per_device; n; k]`, places them split on
axis 0, and maps over that axis; each lane's ids name its own experts and −1
the others. A weight stored as many entries, such as DeepSeek's per-expert
tensors (`model.layers.3.mlp.experts.0.down_proj.weight` in V3), is stacked on
the host one layer at a time and then placed, a transient of one layer's
experts; that is a placement copy, not a repacking.

### Stages

0. The measurements of §Target.
1. `nx.quant` with `Mxfp4` only: constructor, `dequant`, `apply`, the `Ptree.S`
   traversal, the effect and its eager loop, and `place` once RFC 0005's first
   stage gives nx its placements; rune's reverse, forward, vmap and debug rules;
   the three compiled forms on Metal, time-boxed as §Target says: the kernel
   with its row tile, the grouped form and decode-then-matmul; before the
   grouped form, tolk's `cumsum` split in chunks of 256 as tinygrad's
   `_split_cumalu` does (`mixin/op.py:754-765`), a parity fix; tolk's block
   kernel, with its `DIVERGENCES.md` entry, its pinned options on Metal, a
   builder that refuses options on the block axis, a codegen test for each
   renderer that its loop bound, and the kernel's, reads the id on a GPU and is
   constant on the CPU, and its kernels in tolk's opt-correctness fuzzer under
   options that leave the block axis alone; gpt-oss migrated. Gates:
   `validate_stream`'s 228 checks at their tolerances; Law 2's battery for every
   form on each device that takes it, Metal and the CPU; the eager validator's
   dense block without `--experts-by-one` at most 0.5 GB above the weights,
   against 18.97 GB today; §Target's five rows; at batch 1, peak footprint at
   most 16.5 GB and first call at most 48 s. Stage 1's budget in engineer-days
   is derived before it starts: the forms' time boxes, thirteen days, bound only
   the compiled work, and the surface, the eager loop, the rules, the scan, the
   block kernel's options and the Law 2 battery come on top. At that budget the
   stage stops and ships the forms that have reached their ship bands.
2. Split parts: the kernel over shards, grouping and the block kernel per
   lane, and the block kernel's options on each device of the node, with RFC
   0005's stage for several devices. Gate: split weights on `CPU:1` to `CPU:4`
   equal one device under Law 2, for every form.

`Fp8` is designed here and lands with DeepSeek, after NV hardware is
validated, with its Law 2 rows.

A tolk parity bug found here was fixed on its own (8b26ea10a): tolk's `Range`
rule folded any range whose bounds are equal (`uop/symbolic.ml:1496`), where
tinygrad folds only a range of constant size (`uop/symbolic.py:250`).

## Laws

1. **A quantised weight's parts agree by construction, and no traversal
   changes a part's dtype.** Prevents a wrong scale grid decoding silently,
   and a traversal value-casting codes as numbers.
2. **Every path of `apply` equals `cast dt (matmul (cast f32 x) (transpose
   (dequant f32 w')))`, where `w'` is `w` gathered by `ids` as §Reference
   defines it, or `w` itself without `ids`, and a position that selects no
   expert is exactly zero, within the error of a float32 sum of `k` terms in
   unspecified order whose decoded weights and products may each be rounded
   to `x`'s dtype, plus `k · 2^−126 · (3 · max |x| + 2)` on a device that
   flushes subnormals, for an `x` with no subnormal values, and is NaN exactly
   where that reference is, barring a decoded value that overflows**, in each
   device's default math mode. Held by a property test per format, form, device
   and `x` dtype over random codes and scale bytes whose decoded values are
   finite (at most 252 for MXFP4 and 246 for e8m0-scaled FP8), byte 255
   included, and over ids with duplicates and ids outside `[0, e)`; on a device
   CI lacks, it runs locally before landing. Prevents a fast path computing
   another function, or substituting a value for a NaN group.
3. **`apply` and `dequant` never differentiate a quantised weight: a tracked
   part, or a part with a tangent, raises.** Prevents FP8 weights trained
   without their scales.
4. **No eager path of `apply` or `dequant`, forward or backward, holds more
   than one chunk of decoded weight beyond its result; a compiled product
   decodes a weight only into a buffer that dies with its product; the tape
   holds nothing decoded.** Prevents the 19 GB trap, and a tape keeping every
   decoded weight until the backward pass (about 280 GB for a 70B model
   eagerly).
5. **A layout `Nx_quant` names loads as it is: nothing repacks a part's bytes
   at load.** Prevents losing zero-copy to a kernel's preferred layout
   (Marlin, ggml's interleaved blocks).
6. **On a GPU, the kernel and the block kernel run no multiply-adds for a work
   group whose positions or block select no expert,** except the kernel's rows
   of a single group (`k = 32`) and the block kernel's contractions of one
   input, whose loads are gated instead. The cause is a bounded loop of at most
   one iteration: every use of its index folds to 0, so it drops out of its
   reduce, and the reduce, left unparented, is rewritten to its body times the
   loop's size (`reduce_unparented`, as in tinygrad), which runs the body
   whatever the bound. It is the issue of a reduce over a possibly empty range,
   where the exceptions' removal belongs. Held by tolk's codegen test for each
   renderer, by the kernels' builders refusing options on the position and block
   axes, and by §Target's one-lane row. Prevents expert parallelism dividing a
   device's expert reads while leaving it the whole step's arithmetic.

## Drawbacks

- One effect with rules in five rune handlers, one kernel with a decode per
  format, and options per device and shape class to measure and maintain: the
  row tile `M`, the row bound `ρ` and the grouping cost `τ`. The compiled
  forms must stay equal to the reference (Law 2).
- The grouped form costs every call it serves a ranking of its routes, about
  `256 · R · d` integer operations, two gathers of rows, up to `d · B` rows of
  padding, and a launch and a store of zeros for each block that no route
  fills. Its bound counts every position of `ids`, so a lane of expert
  parallelism pays these for the step's routes while it multiplies only its
  own.
- tolk carries a second custom kernel, the block kernel, with its own
  `DIVERGENCES.md` entry and options per device and shape class, measured and
  maintained as the kernel's are. On the CPU its empty blocks run their loops.
- The format set is closed: a new format is a constructor and a decode in
  `nx.quant`, a decode in tolk's kernel, and a lowering case in rune, about 100
  lines plus its rows in the Law 2 battery.
- Two layout conventions: `Nx_quant` is `[outputs; inputs]` as files are,
  while kaun's `Linear` is `[inputs; outputs]`.
- On Metal, scale byte 0 decodes to zero and byte 1 loses its ±0.5 codes,
  where the CPU keeps both.
- Eager `dequant` of a weight placed on a device that does not share host
  memory holds its result twice while it uploads it.
- `Fp8` is designed but unbuilt until DeepSeek runs, since no model in the
  format fits the development machine.

## Rationale and alternatives

**No primitive: a composition, and a divergence widening tolk's matrix-vector
heuristic.** The strongest alternative. A quantised weight stays a record of
tensors in each model, its decode stays an nx composition, and `detect_matvec`
learns to see through a decode chain. It adds nothing to nx or rune, and it
would also fix the dense bfloat16 products. It loses on three counts. Eagerly
the decoded weight exists before `matmul`, and backward keeps it on the tape.
FP8 values stay trainable floats. And whether the kernel is fused depends on
how an expression is written: a `contiguous` or a cast between the decode and
the product silently removes it, which is exactly what the example does today.
Its heuristic also serves one row: the row tile and the grouped blocks need
options over `x`'s rows, which only a pinned kernel has. Stage 0's `P` is its
one-row reading. For context, llama.cpp, MLX and vLLM all name the quantised
product; tinygrad's loader keeps the decode lazy by default, with a note that
materialising it is faster (`tinygrad/llm/model.py:413-416`).

**A decode primitive, with rune fusing a matmul over its output.** `Nx.matmul`
would stay the only product. The fusion then depends on a provenance table in
the jit and on how the expression is written, eagerly the decoded weight exists
whole, and the tape holds it until backward.

**The product in rune** (`Rune.quant_apply`). nx code without rune would have
no bounded product, and rune's public values would gain their first array
operation beside its transformations. **In kaun,** kaun would own an array
operation that rune must lower, above rune.

**`dequant` as an ordinary composition,** differentiable in its float parts.
Quantisation-aware training could then differentiate through it, but an FP8
weight in a structure passed to `grad` would be trained at FP8 with its
scales, which Law 3 prevents; training rounds float master weights in the
model's own code instead.

**The kernel built in rune** from `Tensor.custom_kernel`, tolk unchanged.
tolk's one custom kernel, `Op.scatter_indexed`, lives in its frontend with
parity cases and a `DIVERGENCES.md` entry; a kernel body in rune would sit
outside tolk's kernel tests.

**Block counts read on the host once per layer, in buckets.** This is exact
and needs no tolk change. It costs a wait per expert layer, which RFC 0005's
calls that return without waiting exist to remove: about 0.4 ms on Metal, from
the layer loop's 10 ms over 26 waits. It also costs a compile per bucket and
block program, and up to twice the filled blocks' arithmetic at power-of-two
buckets. Under expert parallelism every device takes the largest count. It is
the fallback if the block kernel stops. **Prefill on the kernel.** Its
arithmetic follows the routes, but it runs without tensor cores: an H100's
dense bfloat16 tensor-core rate is about fifteen times its float32 rate. Rule
3 already sends it past `ρ`.

**A gate recovered from the kernel's shape.** A lowering rule could bound a
reduce's loop by a select that discards its value, and serve the grouped form
without a kernel of its own. It misses silently when the select lands in
another kernel, when the product has a second use, or when tolk's heuristic
puts the block axis on a local dimension. Unguarded, it fails to compile
gpt-oss's decode shapes, where the default options vectorise the instance
axis. It also leaves the copy of the gathered matrices, which the block kernel
removes, and at gpt-oss's `down` shape it is no faster (16.8 ms against 16.9).
**A predicate carried on the reduce** would survive fusion, but every option
and lowering step would have to carry it on each tinygrad pin move. A gated
custom kernel takes tensor cores once its gated loop sits around a constant
loop the tensor-core option splits, which is how the block kernel is written.

**Every route multiplied by every expert** (the example's Dense form). It
needs no ranking of routes, and while a step is bound by its reads it costs
little more than decoding every expert. Run on a whole expert block, as the
example did, its arithmetic is `e / k` times the routes', 8 for gpt-oss and 43
for DeepSeek. Run per product, as a lowering must, it is `e` times the routes'
where `x` is per route: 16 times over gpt-oss's two projections, twice the
example's form. A prefill pays it in full; it stays only as the grouped
prefill's stop outcome.

**Grouping in user code**, sorting tokens into `[experts; capacity; k]` for a
product without `ids`. A fixed capacity drops the routes past it, which
changes the model's function, and a capacity that drops nothing is every
token, the Dense form's arithmetic. Lossless grouping under static shapes needs
blocks addressed by id and a lowering that reads each expert once per block,
which is this RFC's grouped form.

**A mask beside `ids`, or a compacted list of routes, for positions that
select no expert.** A mask carries the id's absence in a second tensor that
must match `ids`'s shape. A compacted list has a static length that is its
worst case, every route, and so needs a value for an empty slot, which is an
id that selects nothing again. RFC 0002 already says `-1` addresses nothing.
Raising eagerly on an id at or above `e`, as `Nx.take` does, would make eager
and compiled runs disagree on it, which RFC 0002's Law 4 rules out for the
cache index.

**A backend operation, `qmatmul`, with the code table as data.** It gives the
eager path a real kernel: the C GEMM's packing step already converts storage
to its compute type. It obliges every engine, and the effect gives the
compiled path its kernel and the eager path a bounded loop without that. A
dequantising pack can be proposed later with a CPU measurement.

**Dense as one of the formats.** One record could then serve a bfloat16
fine-tune and an int4 deployment. A dense weight is trainable and a packed one
is not, so one name would carry two meanings; a model that needs both writes
`Dense of … | Quant of …`.

**Formats as data: a code table, a block and a scale kind.** FP8 values are a
native cast on the hardware that has them, e8m0 and float scales decode
differently, and a 256-entry table is slower than a cast. A descriptor of mode
strings, as in MLX, is the same idea checked only at run time.

**New nx dtypes for fp4 and e8m0.** A dtype is one element kind, while a
weight is codes plus scales with a block geometry. nx refuses sub-byte views,
and neither tolk nor tinygrad has either dtype.

**A quantised tensor that claims a float dtype** (torchao's tensor
subclasses). Every nx operation and engine would learn packed storage or
silently materialise it, and one name would carry two meanings.

**Reproducing DeepSeek's activation quantisation inside the product.** It
trades sixteen times bfloat16's noise per product for agreement with one
vendor's kernels (see §Precision).

**A kaun `QuantizedLinear`.** The expert product, this RFC's main case, lies
outside `Linear`, and `Linear` is laid out `[inputs; outputs]` against every
block format's `[outputs; inputs]`.

## Non-goals

- Int4 group formats and NF4: each lands as a constructor with the first
  checkpoint raven runs in it, its layout read from that file.
- Quantising float weights: `quantise` becomes API with the first
  post-training quantisation user and the format it needs. Calibrated methods
  (GPTQ, AWQ) are out of scope; raven loads their results, converted offline.
- Quantised activations, and FP8 by FP8 tensor-core products.
- Gradients with respect to codes or scales.
- Batch invariance: a row's result may change with the rows and routes it
  shares a call with (§Precision). Whether the engine offers an invariant
  mode is the engine RFC's decision.
- The dense bfloat16 matrix-vector products, about a sixth of today's step.
- Carried leaves in `grad` and optimiser state (`tape.ml:39`, `vega.ml:945`),
  which allocate a full-size zero and two moments for any integer leaf. That
  is a rune and vega defect with its own forcing case; a captured base never
  reaches it.

## Unresolved questions

During implementation:
- DeepSeek V4's on-disk entry names and dtypes, and whether its experts are
  stored one entry each. If its FP4 codes are stored as I8 rather than U8 or
  F4 (which `to_tensor` already hands out as bytes), a byte accessor on
  `Checkpoint` is proposed then, checking the stored shape in bytes.
- Whether scale bytes 1 and 255 occur in DeepSeek V4's files.
- A block-scaled FP8 checkpoint small enough for the development machine,
  which would let `Fp8` land before DeepSeek.
- How far a bfloat16 decode step without activation quantisation agrees with
  DeepSeek's reference: top-1 agreement over a validation set.
- Whether tolk widens hand-written index arithmetic past 2^31 for expert
  tensors that large.
- rune's jit keys a call by its leaves, so two `Fp8` weights whose parts agree
  in shape and dtype and differ only in `block` would replay one trace.
  Whether the jit checks a structure's fields beyond its leaves, or `fp8`
  excludes the ambiguity, is decided when `Fp8` lands.
- Whether `τ`'s uniform-routing estimate `R' (R' − 1) / 2d` under-groups
  skewed routers, and whether a measured duplicate rate replaces it. Stage 1
  did not test the estimate: at gpt-oss's shapes the `R' > d` guard decides
  before `τ` does. A model whose decode step has more routes than experts at
  small `R' (R' − 1) / 2d`, or a second router, decides it.
- Whether tolk's CPU linearizer accepts a loop bound that reads an enclosing
  loop's index, which would let the CPU skip empty blocks too. Today the
  reduction's accumulator reset moves inside that loop.
- The block kernel's options on CUDA and AMD, whose tensor-core tiles are 16
  rows where Metal's are 8, and whether rule 3's `B` follows them.

## Future possibilities

A transposed kernel for the backward pass; a dequantising pack in the C GEMM,
as a backend operation proposed with a measurement; an FP8 activation operand
for H100 and H200 tensor cores; a device layout produced by placement, never
by the loader, on devices whose upload is a copy anyway, proposed with a
kernel measurement; a kaun MoE API once a second model exists;
`Nx_quant.quantise` and new formats with their forcing cases. Nothing listed
here is a reason to accept this or a later RFC.
