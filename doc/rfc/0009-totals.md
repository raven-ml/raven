# RFC 0009: Totals and lanes

- Status: discussion
- Date: 2026-09-29
- Packages: rune (`Rune.Total`, `Rune.axis`, `Rune.lanes`, `?axis` on `vmap`
  and `vmap'`; no-tensor custom calls under forward and reverse; `custom_jvp`
  under vmap; the `E_psum` effect goes), nx
  (`E_psum` and `op_psum` go). Motivating consumer: sofo, outside this
  repository.

## Summary

A total is a write-only sum that code anywhere inside a function adds to and
that the caller reads when the function returns: `Rune.Total.collect t ~zero
f` is `f ()` with `zero` plus everything `f` added to `t`. The scope that
collects a total owns it and threads it through scans and remats itself, so a
staged loop stays one loop, a replay recomputes the total and a restarted
trace discards its additions. `Rune.lanes a x` stacks `x` across the lanes of
the map named `a`, as data. With these and two custom-call rules (a
transformation runs a custom call with no tensor result; vmap passes a
`custom_jvp` on), a marked little loss reports its curvature from anywhere in a model: sofo's mark is a `custom_jvp` with a unit
result whose rule gathers the direction lanes and adds the `k×k` Gauss-Newton
block to a total that the sketch driver collects.

## Motivation

SOFO, the second-order forward-mode optimizer sofo implements, measures a
model along `k` directions Θ: the loss, the gradient sketch `C = Θᵀ∇c` and the
sketched Gauss-Newton matrix `Σ Yᵀ H Y`, where `Y` holds the `k` tangents of a
prediction `y` and `H` is the curvature of the little loss `ℓ(y)` it feeds.
The directions are pushed with `vmap` over the tangents around `jvp`, so the
primal runs once. The loss and `C` are the objective's value and tangent. The
Gauss-Newton block is bilinear in the tangents, so no single lane can compute
it, and it has to be formed where the prediction is, deep inside the model,
often at every step of a recurrence.

Two designs have been tried.

- A tangent query (PR 232, `Rune.tangent x`) let a collector ask forward mode
  for a tangent it keeps privately. Under `vmap` around `jvp` the only way to
  see all `k` lanes was to carry a batched tensor out of the map through an
  effect, which exposes vmap's physical layout and mixes axes once another map
  encloses the sketch. The collector accumulated by side effect while the body
  ran, so it saw one step of a staged loop, and every scan had to unroll
  (`Rune.Scan_claim`). The function could also read its own derivative. It was
  rejected.
- Sketches as values (sofo today) push each step by hand with the state's
  lanes in a scan's carry (`Sofo.scan`). It is correct and constant in the
  horizon, but it covers one top-level recurrence, and users restructure their
  model around it. They report it as too specialized.

Nothing in rune lets code anywhere contribute to a value the transformations
thread. An effect performed inside a transformed function passes every
handler without a case for it (`forward.ml:849`, `reverse.ml:1176`,
`vmap.ml:734`, `jit.ml:2714`): out of vmap it carries a physically batched
payload, out of jit a traced one, and at replay it is not performed. Handlers
also run user code in their own context (scan folds at `forward.ml:133-136`,
`vmap.ml:663-665`; `stage_scan` at `jit.ml:2738-2745`; a remat at
`jit.ml:2353`), so a scope inside a transformation misses what they relocate.

Two needs of the consumer also meet defects in custom calls. One model must
train under `grad` and be sketched unchanged, but reverse raises on any
`custom_jvp` with a tracked parameter (`reverse.ml:1062-1073`). A little loss
may sit inside the user's own map over examples, but vmap runs a custom call's
function and drops its rule (`vmap.ml:580-602`).

## Guide

### Marking a little loss

A user writes the loss they would minimise with any optimizer and calls a
sofo loss wherever a prediction meets a target:

```ocaml
let loss params (xs, targets) =
  let _, ls =
    Rune.scan state Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor ~init:h0
      ~f:(fun h (x, target) ->
        let h = cell params h x in
        (h, Sofo.mse ~target (readout params h)))
      (xs, targets)
  in
  Nx.sum ls

let l, g = Rune.value_and_grad params_ptree (fun p -> loss p batch) params
let sk = Sofo.sketch params_ptree (fun p -> loss p batch) params dirs
```

`Sofo.mse ~target y` returns the mean squared error, which the model sums like
any other value, and marks it: under a sketch its curvature joins the
Gauss-Newton matrix, under Adam the mark does nothing. The loss sums its little
losses; a mean over steps or trials goes inside each little loss (law 6). A
mark may sit in nested functions, in several scans, inside `Rune.remat` or
inside the user's own `vmap`, and a sketch compiled with `Rune.jit` keeps every
scan a loop.

### Totals

`Rune.Total.make ()` is a fresh total; `Rune.Total.add t v` adds `v` to the
innermost open `collect` of `t`, and does nothing if none is open;
`Rune.Total.collect t ~zero f` runs `f` and returns its result with the sum.

```ocaml
let saturated = Rune.Total.make ()

let cell params h x =
  let h = Nx.tanh (Nx.add (Nx.matmul h params.w) (Nx.matmul x params.u)) in
  Rune.Total.add saturated (Nx.mean (Nx.cast Nx.float32 (Nx.greater (Nx.abs h) threshold)));
  h

let train_step =
  Rune.jit sig_ (fun params batch ->
    let (l, g), sat =
      Rune.Total.collect saturated ~zero:(Nx.zeros Nx.float32 [||]) (fun () ->
        Rune.value_and_grad params_ptree (fun p -> loss p batch) params)
    in
    (update params g, l, sat))
```

Nothing reads a total before its `collect` returns, so an addition never
changes a value the function computes. To compile, open the scope inside the
function `jit` compiles and return the total: a `jit` inside a scope runs its
function eagerly, as it does inside `grad` and `vmap`.

### Axes and lanes

`Rune.axis ()` is a fresh name for a map, and `vmap ~axis:a` gives it. Inside
the map named `a`, `Rune.lanes a x` is every lane's `x` stacked on a new
leading axis, the same value in every lane. Sofo's mark gathers the direction
lanes in a forward rule:

```ocaml
let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let mark curv y =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit ~f:ignore y
    ~jvp:(fun _ dy ->
      let ys = Rune.lanes directions dy in                  (* [k; shape y] *)
      let rows t = Nx.reshape [| Nx.dim 0 t; -1 |] t in
      let b = Nx.cast Nx.float64 (Nx.matmul (rows ys) (Nx.transpose (rows (Curv.apply curv ys)))) in
      Rune.Total.add curvature (Nx.mul_s (Nx.add b (Nx.transpose b)) 0.5);
      ((), ()))

let mse ~target y =
  mark (Curv.scale (2.0 /. float (Nx.numel y))) y;
  Nx.mean (Nx.square (Nx.sub y target))

let sketch p loss params dirs =
  let k = Nx.dim 0 (first_leaf p dirs) in
  let (l, c), ggn =
    Rune.vmap ~axis:directions Nx.Ptree.(p @-> returns (pair (pair tensor tensor) tensor))
      (fun dir ->
        Rune.Total.collect curvature ~zero:(Nx.zeros Nx.float64 [| k; k |])
          (fun () -> Rune.jvp p Nx.Ptree.tensor loss params dir))
      dirs
  in
  { loss = Nx.slice [ I 0 ] l; c; ggn = Nx.slice [ I 0 ] ggn }
```

The loss and `C` are `jvp`'s primal and tangent of the returned loss. The
total is collected inside the direction map, so each lane collects the whole
block once, and the driver reads lane 0.

## Reference

### `Rune.Total`

```ocaml
module Total : sig
  type ('a, 'b) t
  val make : unit -> ('a, 'b) t
  val add : ('a, 'b) t -> ('a, 'b) Nx.t -> unit
  val collect : ('a, 'b) t -> zero:('a, 'b) Nx.t -> (unit -> 'r) -> 'r * ('a, 'b) Nx.t
end
```

- `make ()` is generative.
- `add t v` performs an effect; unhandled, it does nothing.
- `collect t ~zero f` handles `t`'s additions and runs `f` as a
  transformation, so a `jit` inside it runs its function eagerly
  (`gate.ml:46-52`, `jit.ml:5511`). The scope checks each addition's shape
  against `zero`'s when it receives it, in its own context, where every map
  inside the scope has already summed its lanes; a mismatch raises
  `Invalid_argument` at the `add`. If `f` raises, `collect` raises the same
  exception; an addition made before an exception that `f` itself catches
  counts.

The scope discharges its total itself, by one rule: code that a handler runs
away from its call site runs under a nested scope started at `Nx.zeros_like
zero`, its sum leaves as a value, and the scope adds it.

- **`E_scan`.** The scope answers the probe by passing it on and claims the
  scan with `Scan.pass_on` (`scan.ml:76-78`). When a stager lies beyond and
  stages it, the scope passes the scan on with one more carry leaf,
  `Nx.zeros_like zero`, and a step that runs the received step under a nested
  scope started at that leaf and returns `c' @ [total']`; the scope adds the
  final leaf. An attempt another claimer aborts (a forward or vmap `Grow`
  restart, jit's placement restarts) discards its additions with its carry.
  Otherwise it declines with `Scan.Not_staged`, and the performer folds where
  it performed the scan, inside the scope, so each step's additions reach the
  scope as they are made and no handler between them (`Nx.Rng.with_key`,
  `with_debug`) is skipped. Every other exception raised while
  the scan runs is delivered to its performer (law 7). `stage_scan` sees an
  ordinary carry.
- **`E_remat`.** The scope passes on the remat of `f'`, which runs `f` under a
  nested scope started at `Nx.zeros_like zero` and returns its sum as an extra
  result, forward's pattern (`forward.ml:740-790`); the scope adds that sum. A
  recompute of `f'` discards the extra result. An exception `f` raises is
  delivered to the remat's performer (law 7).
- **`cond`, `while_loop`** branch in OCaml (`rune.ml:374-380`); an addition
  inside them is straight-line code.

An addition that crosses a transformation:

- **jvp** passes it on; the addition has no tangent.
- **vmap** sums a batched `v` over its lanes and multiplies an unbatched `v`
  by the lane count, then passes it on (law 3).
- **grad** passes it on when reverse first runs the function, and drops it
  whenever reverse runs the function again (law 2). The rerun state belongs to
  the tape: the tapes of the staged backward step (`reverse.ml:1212`) and of
  `recompute` (`reverse.ml:1289`) are rerun tapes, and a tape created under one
  (`Tape.create ~parent`) is one too, which covers the scratch tapes of a scan
  or remat inside rerun code and the eager fold that reuses its tape. A handler
  over a rerun tape drops an addition before its `no_grad` check, since an
  addition is never taped, and runs the functions it runs in its own context
  (a custom call's `fwd`, a unit-result `f`) under a handler that drops
  additions. Under `no_grad` it keeps its claim on the code another handler
  would run away from it, untaped: it passes a scan on with its step under
  that handler or declines it, so that it folds where it was performed; it
  passes a remat on with its function under that handler; and it runs a
  custom call's `f` or `fwd` under that handler itself.
- **jit** runs its function eagerly inside a scope; with no scope anywhere,
  the addition passes out and is dropped.

A scope inside a transformation is ordinary arithmetic to it: differentiated
under jvp, taped under grad, per lane under vmap (the map returns the totals
stacked), and a traced value a compiled program returns.

### Axes and `Rune.lanes`

```ocaml
type axis
val axis : unit -> axis
val vmap : ?axis:axis -> ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val vmap' : ?axis:axis -> (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('c, 'd) Nx.t
val lanes : axis -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
```

- `axis ()` is generative. A map given `~axis:a` is named `a`; other maps are
  anonymous.
- **The map named `a`** answers `lanes a x` for a batched `x` with a fresh
  alias of its physical tensor (a reshape to its own shape,
  `structure.ml:23`), left unmarked, so it is a constant of the map;
  enclosing transformations see the reshape. An unbatched `x` is broadcast to
  `n :: shape x`: every lane holds the same `x`. A compiled sketch relies on
  the broadcast, because vmap's first attempt carries a tangent slot
  unbatched until a step batches it.
- **Every other map** passes `lanes a` on. For an operand it batches, it
  re-performs `lanes a` on the physical operand, swaps the two leading axes of
  the answer and marks it batched; an operand it does not batch is passed on
  and the answer returned unmarked.
- **A named map answers no other collective**: it passes `E_axis_index` on,
  so `Nx.Rng.fold_in_axis` (`nx.ml:43`) in a sketched model addresses the
  user's maps.
- **jvp**: `lanes` is linear, `lanes a dx`. **grad**: it raises when its
  operand is tracked; no consumer needs the transpose.
- **When no map named `a` lies around the call**, so the effect reaches jit
  or no handler, `lanes a x` is one lane, `expand_dims [0] x`, as
  `axis_index` is one lane outside a map (`nx_effect.ml:2011`).

`E_psum` goes (`packages/nx/lib/effect/nx_effect.ml:1008, 1715-1717`, its
cases at `vmap.ml:563`, `forward.ml:556-558`, `reverse.ml:833-835`,
`jit.ml:2709-2713`, and its mentions in `backend_intf.ml:75`,
`packages/rune/README.md:119` and `packages/rune/doc/02-transformations.md:421`,
`03-how-it-works.md:108`, `04-jax-comparison.md:346`). It has no public caller
and no batched rule, and `Nx.sum ~axes:[0] (lanes a x)` is the same sum.

### Custom calls

- **A transformation runs a custom call whose result holds no tensor in place
  of a rule it cannot apply**: there is nothing to differentiate. Reverse
  runs such a `custom_jvp`'s `f`, and forward's mirror case, a `custom_vjp`
  with no tensor result (`forward.ml:721-734`), follows the same line. A
  custom call with a tensor result still raises in the mode its rule does not
  cover.
- **vmap passes a `custom_jvp` on.** Every `custom_jvp` is passed on as the
  custom call of the batched `f` and the batched rule, remat's pattern
  (`vmap.ml:610-639`), in place of running `f` and dropping the rule
  (`vmap.ml:592-602`). vmap claims one whose parameters it does not batch too,
  as it claims every remat and scan: `f` and the rule may read a tensor the
  map batches, which a claimer outside the map would read as a constant of
  shape `n :: s`. Each marks the physical arguments at the batched positions
  and runs under this handler's state, and the batched `jvp` returns primal
  and tangent physically batched at every position where either is batched,
  as forward's shape check requires (`forward.ml:704-712`).

`custom.ml`'s header (`custom.ml:16-18`) and the `custom_jvp` entry in
`rune.mli` change with these two rules.

### Memory

Compiled, a sketched recurrence carries its state, the state's `k` lanes and
one `k×k` total: constant in the horizon, the step traced three times (the
existing `Grow` restarts) and compiled once. Eagerly the total is constant,
but forward's tangent store and vmap's batched set are strong tables
(`tensor_map.ml:17, 41`) shared by every step of an eager fold
(`forward.ml:133-136`), so an eager sketch of a recurrence keeps every step's
intermediates until it returns: 583 MB at 64 steps and 1606 MB at 512 in
sofo's Lorenz model, against 58 MB and 62 MB for the per-step push.
Step-scoped stores in forward and vmap, a separate change, are the
prerequisite for eager constant memory, and until they land a consumer that
needs it eagerly keeps an explicit per-step push. That change must keep a
tensor that reaches a later step through an OCaml capture tracked. Differentiating a
sketch over a staged scan stacks the carried total per step, O(T·k²)
residuals.

### Order of work

Along the consumer's path:

1. Axes and `lanes`, removing `E_psum`.
2. `Rune.Total`, with reverse's rerun rule.
3. The no-tensor custom call rule. With 1-3 a mark works anywhere except
   inside the user's own map, under `jit`, with constant compiled memory, in
   the same model trained with `grad`.
4. vmap passing a `custom_jvp` on, for a mark inside the user's own map.

## Laws

1. **A total is write-only inside its scope.** No function reads it before
   `collect` returns. Prevents a function from depending on its own
   derivative's sketch, and makes dropping an addition with no scope
   unobservable.
2. **An addition counts once per execution of the code that makes it.** The
   scope takes the sums of scans and remats out as values, and reverse drops
   what it reruns, so a restarted trace, a replay and a reverse-mode rerun
   neither lose nor repeat an addition. Prevents the double counts and lost
   steps of the tangent query's collector.
3. **An addition crossing a map is the sum of its lanes' additions.** vmap
   equals the loop it replaces. Prevents a batched payload from leaving a map,
   the tangent query's leak.
4. **`lanes a` addresses the map named `a`, and a named map answers no other
   collective.** Prevents a user's map inside a model from capturing the
   gather meant for the directions, and the direction map from capturing the
   model's own collectives.
5. **A custom call's functions run where the transformation that handles it
   runs, and their additions reach the scopes around that transformation.**
   This holds for the rule (`forward.ml:700`), for `fwd` under reverse
   (`reverse.ml:1034`) and for `f` when a transformation runs it in its own
   context (`forward.ml:695`). A scope between the call and that
   transformation misses them, so a rule never adds a tangent-derived value to
   a scope inside the differentiated function.
6. **A mark enters the Gauss-Newton matrix with weight one.** The sketch is
   the Gauss-Newton matrix of the returned loss exactly when that loss is the
   plain sum of its marked little losses plus unmarked terms. The weight a
   later mean or scale gives a marked loss is `∂L/∂ℓ`, a reverse-mode quantity
   forward mode cannot see.
7. **A handler answers the operation that asked.** Every case of every
   handler computes an answer and resumes the performer with it, a value
   through `continue` or an exception through `discontinue` (`Gate.deliver`,
   `gate.ml:57-70`); a rule never raises past its performer. A performer that
   falls back when unhandled matches only `Effect.Unhandled` of its own
   operation. Prevents an error in a rule, or in the code a handler runs for a
   call (a scan's step, a remat's function, a custom call's functions), from
   leaving through the `match_with` that installed the handler: it would skip
   the performer's handlers and finalisers, and an error the function catches
   would abort the whole transformation.

## Drawbacks

- Three public values (`Total`, `axis`, `lanes`), the type `axis`, and
  `?axis` on `vmap` and `vmap'`.
- The scope claims every scan it would stage: a body with no additions still
  carries the total, `k²` floats per scan here.
- Law 6 is a contract no check can enforce when the loss has unmarked terms.
  Averaging marked losses over trials after a `vmap`, the natural style of a
  per-trial model, scales the loss and `C` by `1/M` and leaves the matrix
  summed.
- A mark belongs to the innermost differentiation around it: inside a `grad`
  the model computes itself it is inert, and inside a `jvp` the model computes
  itself its rule adds a block that is no term of the objective.
- An addition inside a custom call's function skips a scope opened between
  the call and a transformation around it (law 5); under `vmap` this departs
  from the loop of law 3. No consumer adds there.
- An eager remat inside a scope runs in the scope's context, so a `with_key`
  between them is missed, as it is under any transformation today.
- Passing a `custom_jvp` on through vmap moves its claim outside the map:
  `jvp` of a map over a `custom_jvp` applies the rule where it differentiated
  `f`, and `grad` of a map over a `custom_jvp` with a tensor result raises, as
  it does without the map, where it now differentiates `f`.

## Rationale and alternatives

- **Forward over forward on a linearised loss.** The Gauss-Newton matrix along
  Θ is the Hessian along Θ of the loss with each marked prediction linearised:
  a mark `custom_jvp ~f:Fun.id ~jvp:(fun y dy -> (y, detach dy))` under two
  nested jvps over pairs of lanes yields loss, `C` and the matrix from one
  scalar, with no total and no collective. Every operation upstream of a mark
  then carries `k²` second-order tangents and a recurrence carries
  `k²·|state|`, a factor of `k` over this design at SOFO's `k` of tens to
  hundreds. It remains the test oracle for small `k`.
- **Reverse over forward.** A rule returning the little loss with tangent
  `∇ℓ·dy + q − detach q`, `q = ½ dyᵀ H dy`, makes the gradient of the summed
  tangents give the matrix with no new concept. Reverse stacks every step's
  carry and lanes, O(T·k·|state|) compiled, and the extra reverse pass costs
  about three times the forward one. It fails constant memory, which is what
  buys `Total` and `lanes`.
- **Accumulating the marked losses in a total too.** A second total collected
  inside `jvp` would make the objective the marked total plus what the model
  returns: each term's weight would be written once, at its mark, and the
  loss, `C` and the matrix would agree by construction. The model would then
  return only its unmarked terms and need a wrapper to train with any other
  optimizer, and marked losses would stop being ordinary values the user
  combines. Taking the loss and `C` from the returned objective keeps the
  plain loss, at the price of law 6.
- **Innermost addressing** (`lanes x` gathers over the innermost map). A map
  batching a custom rule must then be transparent, a second kind of map, and
  the direction map still captures the model's own collectives:
  `Nx.Rng.fold_in_axis` in a model sketched inside a map over trials folds in
  the direction index, so each direction draws its own noise and the primal no
  longer runs once. A name is one type, one value and an optional argument
  on the two maps, and no existing call site changes.
- **`psum (one_hot (axis_index) ⊗ dy)`.** It needs `k` inside the map, which
  vmap hides behind virtual shapes; `lanes` reads `k` off its answer.
- **Readable state with discharge** (JAX's refs, Dex's `State`). Reads need an
  order across steps, vmap must refuse a batched write into an unbatched
  state, and reverse must transpose reads into sums. The consumer needs only
  sums, and a write-only sum commutes, which is what makes vmap's and scan's
  rules one line each.
- **Transformations discharging the total**, every scan rule carrying every
  open total. Only the scope knows the zero and the key; the scope claiming
  `E_scan` keeps the transformations to one local rule per addition.
- **Reverse differentiating `f` for every `custom_jvp`.** One function would
  get two derivatives. The unit-result case needs only "nothing to
  differentiate".
- **A dedicated tap primitive.** It would need the same two custom-call rules
  and save nothing; a `custom_jvp` with a unit result already is one.
- **A scope outside the direction map.** The block is the same in every lane
  and would be added `k` times.

JAX's state discharge, Dex's `Accum` effect and Flax's `sow` solve adjacent
problems. `Total` keeps what they share (the owner threads the state, a
write-only commutative sum, inert without a collector) and drops what does not
fit rune's handlers (an IR pass, readable state, per-transform boilerplate
naming collections).

## Non-goals

- vmap passing a `custom_vjp` on: its batched `bwd` must sum the cotangent of
  a parameter the map does not batch, and no consumer needs it yet. Until
  that separate change, `grad` of a map over a `custom_vjp` differentiates
  `fwd`, as today.
- Totals of structures, or of monoids other than `+` on one tensor.
- Discharging an addition into a program output when the scope lies outside
  `jit`: a `jit` inside a scope runs eagerly instead.
- `Nx.Rng.with_key`'s missing `E_scan` and `E_remat` cases, which make draws
  in a scan or remat body miss a key scope that a handler relocates the body
  past. Both effects are rune's (`scan.ml:56-58`, `remat.ml:37-39`), so the
  fix is rune's too, as a separate change.
- A native `k`-tangent forward mode.

## Unresolved questions

- Before merge: none.
- During implementation: the messages of the shape check and of `lanes` under
  `grad`.

## Future possibilities

Nothing here is a reason to accept this RFC or a later one.

- Totals of structures, when a consumer adds several tensors at once.
- Named `axis_index`, when a consumer must address a map other than the
  innermost anonymous one.
