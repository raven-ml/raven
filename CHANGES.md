# Changelog

All notable changes to this project will be documented in this file.

- Only document user-facing changes (features, bug fixes, performance improvements, API changes, etc.)
- Add new entries at the top of the appropriate section (most recent first)

## [1.0.0~beta1] - Unreleased

### General

- `nx-oxcaml` moves to its own repository,
  <https://github.com/raven-ml/nx-oxcaml>, and raven no longer ships it. Raven
  now uses OCaml 5.5 syntax and the newest OxCaml compiler is based on OCaml
  5.4, so the backend cannot build in this tree. It builds against the `nx`
  1.0.0~alpha3 release until OxCaml supports OCaml 5.5.
- **Breaking:** a structure of tensors is a module with `type 'a t` and one
  `walk` over an `Nx.Ptree.Walk` cursor (RFC 0006), and transformations,
  optimizers and checkpoints take it as an `'s Nx.Ptree.t` value. Each deleted
  value and its replacement:
  - `Nx.Ptree.S`'s `map`, `map2` and `iter`, `Uniform`, `Traverse`, `Make`,
    `leaf` and `Tree` are one `walk` with `Nx.Ptree.instantiate (module M)`,
    `nest`, `pair`, `list`, `option`, `iso`, `Nx.Ptree.map`, `map2`, `fold`,
    `cast` and `Payload`. The type `Nx.Ptree.tensor` and `Nx_io.P` are
    `Nx.packed`, and `Nx.Ptree.unpack` is `Nx.unpack`.
  - `Rune.grad (module P) f` is `Rune.grad p f`, and likewise for every
    transformation; `vjp2`, `jvp2` and `vmap2` are `vjp p q`, `jvp p q` and
    `vmap` on a signature; `?in_axes` and `?out_axis` are a capture and
    `Nx.moveaxis`; `Rune.Ptree` is `Nx.Ptree`.
  - `jit (module P) f` is `jit Nx.Ptree.(p @-> returns tensor) f`,
    `jit2 p q f` is `jit Nx.Ptree.(p @-> returns q) f`, `jit_step r s f` is
    `jit Nx.Ptree.(r @-> consumes s @@ returns s) f`, `?donate` is
    `consumes`, `pmap2` is `pmap` on a signature, and `?device` is
    `?devices`.
  - `Vega.Sgd_state`, `Adam_state` and `Lbfgs_state` applied to a module are
    `Vega.sgd_ptree p`, `adam_ptree p` and `lbfgs_ptree p`.
  - `Kaun.ptree (module M)` is `Nx.Ptree.instantiate (module M)`,
    `Attention.Cache.List` is
    `Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))`,
    `Checkpoint.of_params (module P)` and `of_packed` are `of_value p`,
    `to_params` and `to_packed` are `to_value p`, and `Vega.Loss_scale`'s
    traversals are `Vega.Loss_scale.ptree`.
  - `[@@deriving ptree]`'s `map`, `map2`, `iter`, `fold`, `fold2`, `names`
    and `Uniform` are one derived `walk`.
- `fehu` and `sowilo` move to `contrib/`. Each is its own
  dune project with its own version, builds against `main`, and sits outside
  the 1.0 API commitment. Changes to `fehu` and `sowilo` are now
  recorded in `contrib/<package>/CHANGES.md`. The `raven` package no longer
  installs `fehu` and `sowilo`; install them by name.
- Add compatibility with OCaml 5.5.

### Norn

- Add `Norn.Smc`, waste-free tempered sequential Monte Carlo, which returns
  an `Evidence.t`. Particles move by Hamiltonian transitions (`Hmc`, the
  default) whose step size and length are tuned between temperatures, or by
  slice steps (`Slice`) for a likelihood without a derivative.
- Add `Norn.Nested`, nested sampling by slice moves on the prior, which
  returns an `Evidence.t`; a spent budget returns the estimate with the nats
  the live points could still add.
- Add `Norn.Ensemble`, ensemble sampling by the affine-invariant stretch
  move, without derivatives: walkers in independent ensembles move in two
  halves, one density evaluation per moving walker per transition.
- Add `Norn.Evidence`, an estimate of the evidence `ln Z` with its error,
  information and weighted posterior sample, and why its run stopped
  (`Evidence.stop`): a spent budget is a result, not an error.
- Add `Norn.Weighted`, draws with log weights, with Kish's effective sample
  size `ess` and systematic `resample`.
- Add `Norn.Hmc`, Hamiltonian Monte Carlo whose chains share one step size,
  trajectory length and geometry, so every chain takes the same number of
  leapfrog steps; `Hmc.warmup` tunes the length by ChEES.
- **Breaking:** norn is a core package again, rewritten around log densities
  over the caller's structures batched over a chain axis: `Norn.Dist`,
  `Norn.Bij`, `Norn.Gaussian`, `Norn.Nuts`, `Norn.Draws`, `Norn.Diag`,
  `Norn.Summary` and the model language `norn.model` replace the flat-vector
  `Norn.hmc` and `nuts`.

### Hugin

- `Nx.bit` values are not real, as `Nx.bool` ones are not: `Stats.histogram`
  refuses them.
- **Breaking:** hugin is redesigned: a figure (`Hugin.t`) is marks that bind
  channels (`num`, `cat`, `dim`) to roles over named `Scale.t`s, composed with
  `layer`, `grid` and facets. `imshow`, `hstack`, `Cmap` and decorations go.
- `hugin` no longer depends on `bytesrw`: the PDF writers deflate their
  streams with `Compress_deflate.Zlib.compress`.
- Hugin no longer depends on cairo or SDL2. Figures render through
  `hugin.vg` with the bundled Inter font, so `opam install hugin` needs no
  system libraries and a figure produces the same bytes on every machine.
  PDF output keeps text as text with the font embedded, and SVG output embeds
  the font too.
- Remove `Hugin.show`. There is no interactive window any more; view figures
  in Quill or save them to a file.

### Vega

- **Breaking:** `Vega.global_dot` and `global_norm` are removed for
  `Nx.Ptree.dot` and `Nx.Ptree.norm`, which return scalar tensors; read one
  for logging with `Nx.item []`.
- `clip_by_global_norm` no longer scales a tensor that is not a float, which
  rounded an integer counter to zero, and its norm no longer counts such
  tensors, so a structure with a counter clips at the norm of its floats.
- `Loss_scale.step` replaces `Loss_scale.unscale`, `grads_finite` and
  `adjust`: it wraps an optimizer step, divides the gradients by the scale,
  skips the step on overflow and returns the next scale. A skipped step keeps
  the whole value it is given, so the optimizer state can no longer be left out.
- A compiled `Schedule.polynomial_decay` with a fractional `power` was NaN
  from `decay_steps` on, so training kept a NaN learning rate after the
  decay: `1 - s / steps` compiled to a value just below zero. It now takes
  the remaining fraction `(steps - s) / steps`, exactly zero there.
- `Vega.global_dot` takes each leaf's inner product with `Nx.vdot`, so a
  `bfloat16` or `float16` leaf's products are summed at `float32` and rounded
  once, as every other product of vectors is. It rounded each product first.
- `adafactor_init` gives a leaf that is not a float, such as an RNG key or a
  counter, zeros of its own shape in every part of the state, as the other
  states do. A vector leaf held a scalar zero, a value its slot's type does not
  admit.
- Steps compute at float32, or float64 for a float64 leaf, and store each leaf
  at its own dtype: `adam_step` no longer returns NaN on float16 or bfloat16
  leaves, and RAdam switches at the exact step with its rectification accurate
  in float32.
- **Breaking** (relative to earlier unreleased revisions): the per-tensor tier
  is removed. Its state was opaque, carried its own chain and covered one
  tensor, so it could neither be a compiled step's argument nor be saved by
  path. `Vega.adam lr` with `init`/`step` is `adam_init p`/`adam_step p ~lr`,
  and likewise for every alias; a chain is function application, transforming
  the gradients (`clip_by_global_norm`, `clip_by_value`) before the step, and
  a state is saved through its structure (`adam_ptree p`, ...) in place of
  `state_to_tensors`. Against the old aliases, `lars_step` scales the
  decayed gradient by its trust ratio before accumulating momentum (the
  paper's order; the rate still scales the velocity from outside, as in
  `sgd_step`), and `adafactor_step` takes its rate as `~lr` in place of a
  built-in `1e-3 / sqrt t`, with `?decay_rate` for `?b2_decay` (a constant β2
  is gone) and `?factored` moved to `adafactor_init`.
- `sgd_step`, `adam_step` and `adamw_step` raise `Invalid_argument` on a
  momentum or `b1`/`b2` outside `[0, 1)`, a non-positive `eps` or a negative
  `weight_decay`, where `b1 = 1.` used to divide by zero and return NaNs.
- Add `lion_step`, `radam_step`, `lamb_step`, `lars_step`, `adafactor_step`,
  `adan_step`, `rmsprop_step` and `adagrad_step`, each with its `*_init` and a
  state whose structure lets it ride a `Rune.jit`-compiled step as arguments.
- **Breaking:** structural functions take `'p Nx.Ptree.t` in place of a module;
  `sgd_ptree`, `adam_ptree`, `lbfgs_ptree` and `Loss_scale.ptree` replace the
  state functors. A step raises naming the first path where its values differ.
- Add L-BFGS at a fixed rate to the structural tier: `lbfgs_step p ~lr grads
  st` takes the gradient at the state's parameters, as every step does, and
  preconditions it by the inverse Hessian the last `history` pairs of
  `lbfgs_init` define; it traces under `Rune.jit`. A deterministic objective
  minimised to a tolerance, with its derivative, is `Jera.Minimize`'s.
- Add `Vega.global_dot p dt a b`, the inner product of two parameter
  trees over all their float leaves as a scalar tensor accumulated at `dt`,
  in tensor arithmetic so it traces under `Rune.jit`.
- Optimizer state now compiles: every scalar that changes across steps is a
  tensor leaf of the state, so the state threads across compiled calls as
  ordinary leaves.
- **Breaking:** `sgd_state` is `{ velocity; step }` — every structural state
  now carries its step counter, a scalar `int32` tensor, so a schedule
  applies to `st.step` whichever optimizer is stepping; without it an SGD
  loop under `jit` had to thread a counter of its own.
- **Breaking:** `adam_state` is `{ mu; nu; step }` — `step` is a scalar
  `int32` tensor, the number of completed steps. A host `int` counter burned
  the Adam bias corrections into the trace at compile time and replayed them
  stale on every later call; the tensor counter tracks correctly under `jit`,
  and the corrections `1 - b^t` are derived from it inside each step
  (`float64` parameters keep their exact analytic corrections). The state
  carries nothing the counter does not determine, so checkpoints hold the
  moments plus one scalar.
- **Breaking:** the structural step functions take the learning rate as a
  scalar tensor: `~lr:(float, 'b) Nx.t`, cast to each leaf's dtype. `Vega.lr
  v` is the constant-rate helper (`Nx.scalar Nx.float32 v`; any float dtype
  is accepted, so `float64` loops can pass a `float64` rate); a scheduled
  rate is the schedule applied to the state's `step` leaf. The per-element
  arithmetic is otherwise unchanged.
- **Breaking:** `Schedule.t` is now a function from a scalar `int32` step
  tensor to a scalar `float32` rate tensor — pure tensor arithmetic, so one
  schedule family serves eager loops and compiled steps alike, every
  schedule included (`exponential_decay`, `polynomial_decay` and
  `cosine_decay_restarts` too). `Schedule.eval` reads a schedule at a host
  `int` for logging and eager loops that keep their own count.
  `cosine_decay_restarts` now validates `t_mul >= 1` and `m_mul > 0`, and
  `exponential_decay` validates `decay_rate > 0`.
- `clip_by_global_norm` computes its scale factor in `float32` tensor
  arithmetic and selects it with `Nx.where` — no host read — so it traces
  under `jit` on any device and can sit between a jitted backward pass and a
  jitted optimizer step. `global_norm` remains the `float64` host read for
  reporting.

### Rune

- A compiled `Rune.iterate` or `Rune.scan` whose step calls `Nx.take` with a
  constant index tensor compiles. It raised `Invalid_argument "a
  dtypes.weakint cannot be stored"`.
- On the host, a compiled function runs its kernels, scans and iterates as
  one host program per run of them: no OCaml runs per step or per kernel. A
  staged scan of 256 steps runs in 291 us instead of 462 us on an M1 Max, an
  iterate of 256 in 268 us instead of 495 us, and a float64 solve of 32
  equations in 50 us instead of 89 us. The call holds the OCaml runtime
  released until the program returns.

- A compiled `Nx.qr`, LU factorization and `Nx.solve_triangular`, and so
  `Nx.solve`, `Nx.inv` and `Nx.det`, run one kernel fewer per step: the loop
  passes each step its index. A float64 solve of 32 equations runs in 92 us
  instead of 101 us on an M1 Max.
- `Rune.jit` computes the incomplete beta family as nx does eagerly, and the
  derivatives of `Nx.betainc`, `Nx.betaincc` and their logarithms in each
  argument are within the bounds `rune.mli` states, eagerly and compiled,
  with their one-sided limits at the ends of `x`. A float64
  compile takes about 3 s, and 7 to 11 s for a derivative.
- A compiled `Nx.solve_triangular` and LU factorization, and so `Nx.solve`,
  `Nx.inv` and `Nx.det`, hold one step in a loop: compiling no longer grows
  with the number of rows. A float64 solve of 32 equations compiles in 1.1 s
  instead of 16 s, and one of 128 runs in 3.5 ms instead of 17 ms.
- A compiled `Nx.cholesky` stores its factor once: a product or reduction that
  read the factor recomputed the factorization for every term.
- `Rune.jit` computes `Nx.gammainc`, `Nx.gammaincc`, their logarithms and
  their inverses as nx does eagerly, and their derivatives are within the
  bounds `rune.mli` states, eagerly and compiled; an inverse's derivative is
  the implicit one.
- A compiled `Nx.cholesky`, `solve_triangular`, `solve`, `inv`, `svd` and
  `eigh` gives a failing matrix NaN in every element of its results, as
  eagerly, where it gave NaN or infinities from the failing column or row on,
  or finite factors for a one-element matrix holding NaN or an infinity. A
  compiled `svd` or `eigh` whose rotations do not converge is NaN too.
- `Rune.jit` compiles a `Rune.scan` over the rows of a slice that starts off a
  16-byte boundary, such as `Nx.reshape [| 4; 2 |] (Nx.shrink [| (1, 9) |] x)`
  of `float64`: compiling failed "UOp verification failed" on the loop's call.
- `Rune.jit` compiles a `Rune.scan` or `Rune.iterate` whose step reads a copy,
  or the bits, of a slice that starts off a 16-byte boundary, such as
  `Nx.copy (Nx.shrink [| (1, n) |] x)` of `float64`: compiling failed "UOp
  verification failed" on the loop's call.
- `Rune.jit` compiles a function whose results are views of storage that a
  loop gathering by integer positions writes, such as a NUTS warmup and
  sampling over many chains: compiling raised `Invalid_argument "option is
  None"` from tolk's division rules, or on a reshape of the gather's index.
- Step `i` of a `Rune.scan` or `Rune.iterate` draws from a key scope rooted at
  `Nx.Rng.fold_in k i`, `k` one key the loop takes from the scope at its call
  and computes at its first draw, eagerly, compiled, batched and transformed
  alike, whatever trips the loop or each lane takes. Every loop shifts the
  draws after it by one key. A compiled loop whose step draws now stages
  instead of being written out step by step, or refused for `iterate`.

- A reverse derivative runs a `Rune.remat`'s function, and the step of a loop
  it stages or batches, once, at the call, and keeps a record of its
  operations, which the backward pass replays: the code no longer runs again
  after the call returned. Effects such code performs no longer raise
  `Effect.Unhandled` under `jit (grad f)` or `vmap (grad f)`, a function that
  would behave otherwise on a second run gives the first run's gradient, and an
  eager `remat` keeps none of its function's intermediates.
- The functions rune's constructs carry run at their call, inside the
  handlers, `Rune.Total.collect` scopes and transformations around it, under
  every transformation: a staged or batched loop's step, a nested `jit`, a
  `custom_jvp` or `custom_vjp` rule and its tangent map, a `remat`'s function
  and a `root`'s `solve`. An effect such a function performs reaches the
  handlers around the call instead of raising `Effect.Unhandled` under `jit`,
  `grad` or `vmap`, and a rule's additions reach the scopes around its call.
- Under `jit`, a value a loop's step computes that reaches the program other
  than through the carry, the outputs or a `Rune.Total` the step adds to, as
  through a handler around the loop that computes from it, raises
  `Invalid_argument`; it raised `Not_found`.
- In a `vmap`ped `Rune.iterate` whose lanes stop apart, a stopped lane's
  additions inside a compiled call in the step are dropped, as the step's own
  are; they were counted.
- A compiled call reads a one-element capture and places a capture that lies
  elsewhere through its storage, so an `Nx.Op.intercept` around the call no
  longer meets operations on the call that traces only.
- Compiled `x * recip y` on the host now equals eager bit for bit; it was
  divided, rounding once where eager rounds twice.
- A compiled check returns its data at the failing index, so a compiled
  `Nx.check` raises the exception an eager call raises, built from computed
  values; a staged scan's check reads its first failing trip.
- `Rune.jit` computes `Nx.i0e` and `Nx.i1e` as nx does eagerly, and their
  derivatives are within the bounds `rune.mli` states, eagerly and compiled.
- `Rune.jit` compiles `Nx.erfinv` at `float64` 2.4 times as fast and its
  derivative 5.7 times as fast. The derivative is within 64 ulps.
- `Rune.jit` computes `Nx.erfc`, `Nx.ndtr`, `Nx.log_ndtr`, `Nx.ndtri`,
  `Nx.lgamma`, `Nx.digamma` and `Nx.lbeta` as nx does eagerly, and their
  derivatives, in every argument and at every order, are within the bounds
  `rune.mli` states, eagerly and compiled.
- `Rune.jit` compiles `bit`, `int4` and `uint4` values, which it refused. A
  compiled function reads and writes their packed bytes, views at sub-byte
  offsets included, and computes `int4` and `uint4` as integers modulo 16.
- A compiled scan whose carry holds a tensor the function moved to the host is
  written out, so the tensor comes back on the host; it came back on the loop's
  device.
- A compiled scan of one row stages as a loop, its step traced once, where it
  was written out; a scan inside a compiled loop's step is a loop nested in it.
- A compiled comparison of a value padded by a float the tensor's type rounds
  to 0, as a float32 pad of `5e-324`, compares the rounded value: the compiler
  bounded it unrounded and folded `Nx.equal` with 0 to false.
- A compiled selection between a widened load and another value, as a pad of
  a converted tensor read through a slice, keeps that value: a float32 pad of
  `0x1p-149` around `Nx.cast Nx.float32` of a float16 tensor read 0, since the
  pad was read at float16. A value the load's type cannot hold, a fraction or
  `-0.` through an integer load, or past float16's range, no longer moves into
  the load.
- Compiled `Nx.log` is within one unit in the last place of the correctly
  rounded result, from up to four in float32 on the host. It is computed from
  its argument's bits and a polynomial instead of a target's `log2` times
  `ln 2`, whose rounding adds to the target's error on every device.
- Add `Rune.root`, a value stated to solve an equation. Its derivative is the
  implicit function theorem's at the value, under every transformation, and
  the solve that found it is never differentiated, so it may iterate and stop
  early. `?linear_solve` replaces the default dense solve.
- Add `Rune.iterate` and `Rune.iterate'`, a loop that applies a step until a
  condition holds and raises after `max` steps. Under `vmap` each lane stops on
  its own; derivatives cover the steps each lane took. Under `jit` it compiles
  to one loop that tests its condition before each step, nested inside a
  compiled scan's or iterate's step too.
- `grad` of a `scan` whose carry and outputs depend on no tracked value keeps
  no carry per step under `jit`.
- A compiled integer remainder by zero is its dividend, as `Nx.mod_`'s.
- `RUNE_JIT_DEBUG` is a tolk setting: it holds an integer, nonzero to report,
  and any other value raises when the program starts.
- A `jit`-compiled function compiles again when any tolk setting that shapes
  compilation changes around a call (`Tolk.Setting.shaping`). It keyed its
  programs on `NOOPT` and the search width alone, so a change of
  `SPLIT_REDUCEOP`, `RING`, `MAX_KERNEL_BUFFERS` and the like ran the old
  program.
- `jit` of `Nx.eigh`, `Nx.svd` and `Nx.qr` compiles in one to four seconds at
  any size: the program holds one Jacobi round or Householder step, which a
  loop repeats. It held every step, so an 8 x 8 `eigh` took 35 s to compile, a
  16 x 16 one 280 s, and a 32 x 32 one ten minutes. A round of `svd` is now
  O(n^2), where it multiplied matrices: 64 matrices of 16 x 16 take 2.9 ms,
  where they took 41 ms.
- `jit` compiles `Nx.eigh` and `Nx.eigvalsh` of real matrices, by two-sided
  Jacobi rotations for a sweep count fixed by the size and dtype, so a step
  that solves with a symmetric eigendecomposition runs as one program. Before,
  `jit` refused them. Their eigenvalues are `float64`, which Metal refuses.
- `jit` reads a host scalar operand of a value on a device, such as `add_s`'s,
  as a constant where it lies. It used to copy it to the device and back, and
  each copy waited for the work queued there, so a trace stalled behind the
  previous call's kernels.
- **Breaking:** `grad`, `jvp` and `vmap` of a `jit` function, and a
  `Total.collect` around one, compile: each runs programs for the derived
  function, kept in the compiled function. Before, `jit` ran its function as
  plain code under a transformation. Under `grad` and `vjp`, a forward program
  returns the values the backward program reads, so the function's forward
  work runs once per call; a transpose that cannot be traced raises `Jit_error`
  at the forward call. Under every transformation, a compiled function that
  reads a tracked value through its closure raises (pass it as an argument),
  and so does one that reads a traced value another compiled function leaked.
- `jit` writes `Nx.scatter` into a consumed value in place with one indexed
  store, in every mode, unless it has more repeated updates than rows: a
  key-value cache's rows from a page table take one kernel, with no kernel
  for their offsets and no limit of 16 rows.
- A bitcast between a complex dtype and the float of its components, such as
  `Nx.bitcast Nx.float32` of a `complex64` tensor, is differentiable. Its
  tangent was dropped, so a gradient through it was zero.

- **Breaking:** where the operands of `Nx.maximum` or `Nx.minimum` tie, each
  takes half of the derivative, as tied elements share it along the axes of
  `Nx.max` and `Nx.min`; the second operand took all of it. `Nx.relu` and
  `Nx.abs` keep derivative 0 at 0, and the `Rune` interface states all three.

- `jit` compiles `Nx.take`, `Nx.take_along_axis` and the sorts as loads at
  the indices, which fuse into the operations that read them, where a one-hot
  sum cost a reduction over the axis per element and a kernel of its own.
- A compiled function answers `Nx.check` when its call returns, raising the
  first failed check's message, and differentiates, maps and compiles
  `Nx.fma`, `Nx.log1p` and `Nx.expm1`.
- `Rune.jit` and the compiled operations compile a program again for each set
  of counters and each trace request a profile asks for
  (`Nx_device.Profile.start ~counters ~trace`). A
  program compiled under a profile that counted was kept and reused: once the
  profile stopped its next link was refused, and under other counters it
  counted into memory nothing read.
- **Breaking:** rune is rewritten. A transformation is an interpreter that owns
  its values, reverse mode transposes forward mode, and `jit` compiles through
  tolk on the devices its arguments are placed on. For callers:
  - `vjp p q f x` returns the result and its pullback; `vjp_fun` and
    `vjp_fun'` go.
  - `custom_jvp p q (fun x -> (y, tangent))` and
    `custom_vjp p q (fun x -> (y, pullback))` take one function. A rule that
    reads a value its own differentiation tracks raises; pass it as an
    argument.
  - `value_and_grad_aux p a f` takes the structure `a` of the auxiliary value.
  - `jit` takes no `~devices`: a call runs where its arguments and captures
    are placed. `device`, `devices`, `default_device`, `jit_stats` and
    `reset_jit_stats` go. A device is an `Nx.Device.t`
    (`Nx_metal.device 0`), and `Nx_device.stats` of its memory counts its
    transfers and allocations.
  - Gradients and tangents are fresh, contiguous values.
- `Rune.jit` refuses `Nx.scatter ~mode:`Max` and `` `Min `` (and so
  `Nx.reduce_segments` by them), and `Rune.grad` and `Rune.jvp` refuse to
  differentiate through them; `Rune.vmap` maps them.
- A compiled call that consumes an argument another domain is reading computes
  from a copy instead of writing over it, for host values as for placed ones;
  a host read could see the call's write before.
- A consumed argument of `Rune.jit` that views part of its storage, or
  broadcasts it, is computed from a copy and its storage is consumed, where the
  call raised "does not cover its whole storage". One whose buffer is a window
  of larger memory, such as a weight loaded from a SafeTensors file, is copied
  and stays live. A compiled function's captures, host ones included, are never
  written over by another call that consumes them.
- `Rune.jit` and `Rune.vmap` refusals of a read name the function that read,
  such as `Jit_error "Nx.item: …"`, instead of a fixed "item, to_host, or a
  data-dependent branch". `Rune.jit` refuses a `bitcast` between widths.
- A `Rune.scan` compiled by `Rune.jit` reads rows computed from constants
  alone, such as `Nx.arange`'s running sum, as eager computes them. Such rows,
  when their size was not a multiple of 16 bytes, were padded through a store
  that tolk scheduled wrongly: the loop read wrong values from 4-byte rows, and
  compiling failed with `Failure "nth"` on 20-byte rows.
- **Breaking:** `Rune.cond`, `Rune.while_loop`, `Rune.no_grad`,
  `Rune.hessian'`, `Rune.hvp`, `Rune.hvp'`, `Rune.jvp_aux` and
  `Rune.with_debug` are removed. Each is a line of ordinary code:
  - `cond` and `while_loop` staged nothing: write `if Nx.item [] p` and a
    recursion. Both differentiate the path taken.
  - `no_grad f` around code no differentiation sees is `f ()`. Inside a
    differentiated function, `Rune.detach` holds a value constant.
  - `hessian' f` is `Rune.jacfwd' (Rune.grad' f)`, and `hvp p f params v` is
    `snd (Rune.jvp p p (Rune.grad p f) params v)`.
  - `jvp_aux`'s auxiliary value is part of `jvp`'s result structure.
  - `with_debug` is an interpreter installed with `Nx.Op.intercept` that
    prints each operation and evaluates it.
- The derivatives of `Nx.cummax`, `Nx.cummin` and `Nx.cumprod` are right.
  `cummax` and `cummin` gave a running extremum's derivative only where it
  changed, so `grad (sum ∘ cummax)` at `[3; 1; 2]` was `[1; 0; 0]`; each
  running extremum now takes the derivative of its element, the first of
  equal ones, giving `[3; 0; 0]`. `cumprod`'s gradient divided by the input
  and lost every term past a zero (`[1; 2; 6]` at `[2; 3; 0]` instead of
  `[4; 2; 6]`), and its tangent was NaN at a zero. Both now solve the product
  rule's recurrence without division, exactly at zeros and at every order.
- `Rune.jvp` of `x ** 2.` at `x = 0` is `0` instead of NaN, as `Rune.grad`
  gives. Forward mode filled an operand without a tangent with zeros and
  multiplied them by its coefficient, which is infinite or NaN where the
  operand is constant (the exponent's `log x` at `0`, an infinite factor of a
  product). A binary operation now builds terms only for operands with a
  tangent.
- Compiled `Nx.maximum`, `Nx.minimum`, `Nx.cummax`, `Nx.cummin`, `Nx.argmax`,
  `Nx.argmin`, `Nx.sort` and `Nx.argsort` order `-0.` below `0.`, as eager
  now does: a compiled `maximum` of two zeros returned its second operand, and
  the compiled scans, arg-reductions and sorts kept the first of two zeros.
- **Breaking:** `Rune.vmap` hands the mapped function lanes, traced values of
  the unbatched shape that hold the batched tensor, instead of batched tensors
  whose shape queries answered unbatched. `Nx.placement` of a lane of a map
  over an axis split across devices is a copy on each device; it raised.
- New `Rune.lane_index ?axis ()`: the calling lane's index in the map named
  `axis`, or in the innermost anonymous map, as `Rune.lanes` addresses maps;
  `0` outside any. It replaces `Nx.Rng.fold_in_axis`.
- **Breaking:** on a complex tensor, `Rune.grad` returns the gradient
  `dL/dre + i*dL/dim`, the direction in which the objective grows fastest, so
  `z - lr * g` descends as returned. It returned the conjugate, which had to be
  conjugated again before a step; code that did so must stop. `vjp` and `vjp'`
  take and return cotangents in the same sense (the pullback is the adjoint of
  `jvp` under `Re (sum (conj u * v))`), and so does a `custom_vjp` rule's
  pullback. `jacrev'` still equals `jacfwd'` on complex-differentiable
  functions, and `check_grads` pairs a complex gradient with
  its direction through the conjugate. Real tensors are unaffected.
- Gradients through `Nx.qr` are right on complex matrices. The reverse rule
  used plain transposes where the unitary factor needs conjugate transposes.
- Gradients and tangents of `Nx.cholesky` are right on complex Hermitian
  matrices, which the rules treated as symmetric, and follow what the
  factorisation reads: the lower triangle and the real part of the diagonal.
  `Rune.jvp` read the upper triangle of the tangent too, so a tangent that
  was not symmetric gave a wrong result on real matrices as well.
- Gradients through `Nx.solve_triangular`, `Nx.solve` and `Nx.inv` on complex
  matrices are right. The reverse rule solved against the conjugate
  transpose where the plain transpose is needed, and with `~transpose:true`
  both engines differentiated `aᵀ` where the solve reads `aᴴ`.
- `Rune.grad`, `vjp` and `jvp` differentiate `Nx.sign` on complex tensors,
  where it is `z / |z|` and turns with `z`. It had a zero derivative, as on
  real tensors, so a gradient through a complex `sign` was silently zero and
  the second derivative of `Nx.abs` on complex was wrong.
- A compiled function's first call with a warm compile cache is about a
  quarter faster: 15.3 s to 11.0-12.3 s for GPT-2's training step on Metal.
  Rebinding a cached program's parameters looked each node up in the list of
  bindings, so it took time proportional to nodes times bindings.
- `Nx.place` of a value on the disk, such as an entry of
  `Nx_io.load_safetensors`, on Metal (Apple silicon) or a `CPU:k` device
  borrows the file's pages copy-on-write and keeps its view: gpt-oss-20b's
  weights take 1.3 GB of the process's memory instead of 14 GB and import in
  3.5 s instead of 9.5 s cold. Borrowed storage is never lent to a compiled
  call's output or written, and counts nothing in `resident_bytes` or the
  collection budget. On another device the value is read from its file, 64 MiB
  at a time, into device memory. As an argument or a capture of `Rune.jit` a
  value on the disk is read as a host value is.
- A scan staged under `Rune.jit` no longer copies a carry its body returns
  unchanged at every step: the loop reads its initial value and the scan
  returns it. A `Rune.Total` scope around a scan whose body adds nothing adds
  such a carry, so it now costs the loop nothing.

- `Rune.vmap` passes a `Rune.custom_vjp` on as the custom call of its batched
  `fwd` and batched `bwd`, where it ran `fwd` batched and dropped `bwd`: a
  `grad` around a map now applies the rule to the map's lanes, summing the
  cotangent of a parameter the map does not batch. A call whose parameters the
  map does not batch is passed on too, where a `grad` outside read a tensor
  the map batches as the stacked tensor and returned a wrong value and
  gradient. `Rune.jvp` of a map over a `custom_vjp` with a tensor result now
  raises, as it does without the map.

- An exception raised while a transformation handles an operation now raises
  at the operation, where a `try` around it catches it and a `Fun.protect`
  around it runs its finaliser. It left through the transformation instead,
  aborting `grad`, `jvp`, `vmap` or `jit` even when the function caught it:
  a `remat`, `custom_jvp` or `custom_vjp` whose code raised, an operation
  with no rule, a value read inside `vmap`, a custom rule's tangent of the
  wrong shape.

- **Breaking:** a compiled call consumes a host argument its signature marks
  with `consumes`, as it consumes a placed one: the value must be all of its
  buffer, every handle to that buffer then raises, and on the host an output
  derived from it is written over its memory. A host argument was uploaded and
  stayed usable, so a donating step held its state twice on the CPU.
- `Rune.vmap` passes a `Rune.custom_jvp` on as the custom call of its batched
  function and batched rule, where it ran the function and dropped the rule:
  a `jvp` around a map now applies the rule to the map's lanes, so a mark
  inside a model's own map over examples reaches the forward mode outside it.
  `Rune.grad` of a map over a `custom_jvp` with a tensor result now raises, as
  it does without the map, where it differentiated `f`. The map passes on a
  call whose parameters it does not batch too: its `f` or rule may read a
  tensor the map batches, which a differentiation outside the map read as the
  stacked tensor, so `grad` returned a wrong value and gradient and `jvp` a
  result with an extra axis.

- `Rune.grad` of a function that calls a `Rune.custom_jvp` with a result that
  holds no tensor (a unit result) runs its `f`, where it raised on a
  differentiated parameter: there is nothing to differentiate. Likewise,
  `Rune.jvp` runs the `fwd` of a `Rune.custom_vjp` with no tensor result. A
  unit-result `custom_jvp` observes the tangents of its parameters in forward
  mode and is inert under `grad`, so one model trains with `grad` and is
  measured with `jvp`. A call with a tensor result still raises in the mode
  its rule does not cover.

- `Rune.Total` is a write-only sum that code anywhere inside a function adds
  to (`Rune.Total.add t v`) and that `Rune.Total.collect t ~zero f` returns
  beside `f ()`. With no scope open an addition does nothing, so one model
  serves every driver. Additions count once per execution of the code that
  makes them: a map adds the sum of its lanes', reverse mode drops the ones
  it makes again while rerunning code, and a scan staged under `Rune.jit`
  carries the sum out of its loop, so it stays one loop and a replay computes
  the total again. A `jit` inside a scope compiles the function that also
  returns its additions.

- `Rune.lanes a x` gathers `x` across the lanes of a map, as data: inside the
  map named `a` (`Rune.vmap ~axis:a`, `Rune.vmap' ~axis:a`, with
  `a = Rune.axis ()`) it is every lane's `x` stacked on a new leading axis,
  the same value in every lane. A map in between keeps its own lanes, and with
  no map named `a` around the call there is one lane. A named map passes the
  lane index on, so `Nx.Rng.fold_in_axis` inside it addresses the anonymous
  map around it.

- `Rune.jit` compiles for the memories of its arguments and captures,
  whatever backends their devices carry, and places its results on the
  arguments' devices. Arguments on two devices over one memory raise, naming
  `Nx.place`.

- `Rune.jvp` and `Rune.vmap` of a gradient through a `Rune.scan`, and
  `Rune.grad` of one, compile as loops under `Rune.jit`, where they unrolled
  every step: the gradient's reversed loop is now a scan itself, which every
  transformation passes on. Hessian-vector products and per-example gradients
  over recurrent models trace the body a fixed number of times whatever the
  length. The reversed loop also stops computing cotangents nothing asks for,
  of rows and captured tensors that are not differentiated (a 32-step MLP cell
  differentiated in its weights only: 11 to 9 kernels per step, 576 to 428 KiB
  of device memory).

- A `Rune.scan` compiled by `Rune.jit` over the stacked outputs of another
  staged scan reads them where the first loop wrote them. Rows of a size that
  is not a multiple of 16 bytes were copied once per call, and a scan over such
  rows split across devices unrolled every step; it now compiles as a loop.

- `Rune.jit` replays launch a compiled kernel without rewriting graphs, reading
  the environment or selecting a renderer. Each launch re-derived the storage
  views of its arguments with a graph rewrite and rendered every compile
  setting into its runtime cache key: host time per kernel on the CPU falls
  from about 32 µs to 5 µs (a chain of 65 small kernels replays in 0.33 ms,
  was 2.1 ms).

- `Rune.jvp` and `Rune.vmap` of a `Rune.scan` compile as one loop under
  `Rune.jit`, where they unrolled every step. The loop's carry gains a tangent
  or a lane only where one reaches it, so `vmap` over tangents around `jvp`
  computes the primal state once for every lane (32 lanes around a 32-step MLP
  cell: 21 ms per call to 7 ms on the CPU). `Rune.grad` of such a `jvp` or
  `vmap` compiles its reversed loop too.

- A `Rune.scan` compiled by `Rune.jit` computes a value its body reads from
  before the scan once, before the loop. The loop recomputed such a value at
  every step: a draw from a key argument, or a whole earlier scan, ran once per
  step (a 32-step scan over the result of another staged scan launched 642
  kernels per call, now 84).

- `Rune.grad` and `Rune.jvp` through `Nx.contiguous` of a placed value that
  fills its storage no longer count its derivative twice: the value came back
  as its own result, which the tape took for a second node. A transformation
  now raises when an operation returns its operand, where the derivatives were
  silently wrong.

- Placed storage retires through the shared device-safe queue, preserving
  allocation lifetimes during native callbacks and reporting failed retirement
  at explicit safe points.

- **Breaking:** `Rune.jit` and `Rune.jit'` take `?parallel` instead of
  `?beam_parallel`, controlling both independent kernel compilation and beam
  candidates through the shared `PARALLEL` setting.

- Consuming a placed argument now requires exclusive access to its storage
  across compiled functions, reads and placements. Captures retain ownership
  during tracing, preventing concurrent consumption from changing their values.

- Preserve higher-order compiled `Rune.remat` gradients with strict buffer-state
  validation. Written residuals get a separate checkpoint after their cotangents
  are ready, retaining the activation-memory bound without ambiguous reads.

- Overlapping or reentrant calls to the same `jit` closure now raise
  `Invalid_argument` before accessing shared compilation or replay state.
  Sequential calls can move between domains and threads.

- `grad` and `jvp` now handle products containing zeros and share extrema
  derivatives among tied values. Half-precision extrema count ties in float32
  so large reductions do not overflow their gradient normalization.

- Independent `jit` callers can share read-only captures without losing their
  ownership counts or racing first-time cache initialization. Transformation
  scopes no longer leak across domains or system threads.

- Keep placed storage alive until a read of it (`Nx.to_array`, `Nx.pp`)
  finishes.
  Collected values now hand ownership to one deferred-release path; failed
  allocator cleanup retains its backing without losing other queued releases.

- Give each compiled function its own planned intermediate buffers, preventing
  interleaved JIT calls from corrupting one another. Dropping a compiled graph
  releases its arena without retaining the largest allocation process-wide.

- Prevent independent uploads of 64 MiB or more from overwriting each other's
  staging bytes.

- Support compiled `Nx.bitcast` to and from float8, preserving all byte
  encodings through direct outputs, transposes and slices on CPU and Metal.

- Fix `Rune.jit` outputs that directly bitcast an input, including a tuple
  returning both: each result now retains its own storage dtype.

- `jit ~beam:0` disables default autotuning under a nonzero `BEAM` context,
  and the persistent cache key uses that explicit choice.

- A compiled float sum of -0s, the negation of a zero sum, and `c ? t : 0 +
  c ? 0 : f` keep eager's sign of zero: a sum of -0s was -0 and `-(x + 3)` at
  x = -3 was +0.
- A compiled `Nx.maximum` or `Nx.minimum` propagates NaN from either operand
  and keeps eager's second operand on a tie: `max 3 nan` was 3 and
  `max (-0) (+0)` was -0.
- A compiled `>=` or `<=` with a NaN operand is false, as eager's is: it was
  true, so a mask like `where (x >= 0) x 0` kept NaNs that eager drops.
- `Rune.jit` over several devices retains its planned intermediate buffers
  across calls. Each compiled graph owns its storage independently on every
  device, avoiding repeated allocation without sharing writable arenas.
- `Rune.remat` inside `Rune.jit` over several devices checkpoints as on one
  device: the block's arguments are kept on each device and the block is
  recomputed in the backward pass. It kept every intermediate.
- `Rune.scan` inside `Rune.jit` over several devices stages as a loop, as on
  one device: its body is traced once, each device running it over its slices
  of the carry and the rows. It unrolled into the program, one copy of the
  body per step. A scan whose rows are split along the scanned axis, or whose
  row slices are not whole 16-byte units, still unrolls.
- `Nx_quant.apply` inside `Rune.jit` over several devices takes the kernels
  one device takes, on each device's slices. Over experts split across the
  devices, each device multiplies the routes to its own experts and the
  partial products sum, whatever device a route's token is on. It decoded
  every expert of every lane (the dense form).
- A consumed argument of `Rune.jit` over several devices lends its storage,
  its buffer on every device, to the result that continues it, as on one
  device: a carry or a pool written by index keeps one generation on each
  device instead of two, and the pool is no longer copied before the write.
- Inside `Rune.jit` over several devices, `Nx.scatter` into a split value, and
  a traced `Nx.set` window into a value split along its first axis, write each
  device's slice with the indexed kernel one device uses, which stores each
  update's elements once. They compared every row with every update: over 256
  rows, 8,192 estimated operations at 2 updates and 79,872 at 16.

- A compiled float `x / x`, `x * 0`, `(x * y) / y` and `-0 + 0` compute what
  they say: they folded to 1, 0, `x` and -0, wrong at 0, inf, NaN and
  overflow. `x / (1 + x)` was rewritten to `1 - 1 / (1 + x)`, which is 0 at
  x = 1e-8.
- A compiled integer constant folded from constants wraps at its dtype as
  eager's does: `uint8` `x < full 200 + 100` compared with 300, true for
  every x, where eager compares with 44 (likewise `255 * 255`, `int8`
  `127 + 1`, `uint16` `65535 + 1` and `int32` `max + 1`).
- A compiled `Nx.pow` of a tensor base compiles for any float exponent, a
  tensor or a constant that is not a whole or half number, where it raised
  `unhandled op POW`, and takes pow's special values: the sign of an odd
  power of -0 or a negative base, NaN for a negative base to a fractional
  power, `1 ** y = 1`, and exact zeros and infinities at -0 and -inf, and on
  the CPU at subnormal bases (`(-inf) ** 2.5` was NaN, `1e-40 ** -0.8` inf).
  A 16-bit float power is computed at `float32`.
- A compiled comparison of an integer sum or product that may wrap no longer
  folds as if it could not: `uint32` `x - 1 < 5` compiled to `x < 6`, true at
  x = 0, and `uint8` `x - 1 < 255` to true. Signed 32- and 64-bit overflow
  stays undefined in the C that CPU and Metal kernels compile to.
- A compiled 8- or 16-bit integer sum, difference, product, negation, left
  shift or quotient wraps at its width before it is read: C computed it in
  `int`, so `uint8` `x - 1 < x` was false at every x where eager is true
  from x = 1.
- A compiled float sum or product groups as the program does, constants
  included: `c + (a + b)` was computed as `(c + a) + b`, 0 instead of 1 for
  a = 1e8, b = -1e8, c = 1 in `float32`, and `(x + 1e8) - 1e8` as `x`.
  Unrolled reductions add their lanes' sum to the accumulator and run up to
  4x faster on the CPU; a tanh-form GELU runs 14% slower there.
- Compiled mxfp4 products weigh tolk's kernel against decoding by one row
  bound per device, the same for one matrix and for routes grouped by expert:
  8 rows on Metal and 64 on the CPU, at every dtype. It was 32 at bfloat16
  and 16 at float32 on Metal, 64 and 2 on the CPU. On Metal grouped products
  of 9 to 32 rows per expert now decode (1024 routes at bfloat16: 38 against
  52-55 ms); on the CPU a float32 matrix keeps the kernel up to 64 rows (8
  rows: 10-12 against 21 ms).
- Compiled mxfp4 products that decode on one device multiply with tolk's
  block kernel, without ids and with fewer positions than experts as with
  more, each position's rows padded to the kernel's tile: exact products at
  every dtype, and on the CPU 1.6 to 3.1 times faster at float32 and 7 to 11
  times at bfloat16 (one gpt-oss expert, 8 to 128 rows); on Metal as fast at
  multiples of 8 rows and up to 4.8 times faster otherwise (509 rows at
  float32: 56-57 against 264-268 ms).
- **Breaking:** `Rune.pmap` and its `in_axes` are removed: place the values and
  call `Rune.jit`. `Some a` is `Nx.place (Nx.Placement.sharded ~axis:a ds)`;
  `None` leaves a value on the host, where it enters as a copy on each device.
  A per-device key (`Nx.Rng.fold_in_axis`) comes from `Rune.vmap` over an axis
  split one slice per device; a program over several devices has one lane.
- A consumed carry of `Rune.jit` over several devices keeps its placement:
  a result derived from it that lands elsewhere is resharded at the end of the
  program, so a carry that starts on the host stays a copy on each device and
  the next call reuses the program.
- **Breaking:** inside `Rune.jit` over several devices, values live where
  nx's rules put them, and operands split differently (a row-split matrix
  times a column-split one), an operation along a split axis, or a movement
  across devices raise as the function traces, as they do eagerly, instead of
  the compiler moving data silently. `Nx.place` inside the function gathers or
  splits a value, and `Nx.placement` answers there.
- `Rune.jit` runs over several devices: where its split or replicated leaves
  and captures live, or on `?devices`, which now takes any list of one
  backend. Host leaves enter as a copy on each device, placed leaves and
  captures (split ones included) are read in place, results come back placed,
  and a call returns without waiting for its devices.
- Reading a value on Metal or `CPU:k` that nothing else holds (the operand
  of an eager operation, say) no longer reads freed memory: the collector
  could release its buffer during the copy, which crashed when the buffer was
  uploaded from a file.
- A compiled `Nx.matmul` of `bfloat16`, `float16` or float8 values multiplies
  and sums at `float32` and rounds once, as eager does. Rounding each product
  moved greedy gpt-oss-20b off its `float32` ids at the fifth token.
- A sum over a value split across three or more devices keeps a -0 result
  under the ring, all-to-all and hierarchical allreduces (ring is the default
  above 256000 elements): their replicas were laid back together by adding
  zero-padded chunks.
- A compiled `Nx.concatenate` of pieces of unequal extent returns their
  elements bit for bit. It summed zero-padded pieces, which could turn -0 into
  +0, quiet a signalling NaN and, on Metal, flush a subnormal to zero.
- A compiled gather inside a padded or concatenated value reads its indices
  under their guard. It read them unconditionally, out of bounds wherever the
  guard was false, which could crash the process.
- A compiled `float64` sort returns its input's elements, `-0.` and NaN bits
  included, and a compiled argsort of 64-bit keys sorts their two 32-bit halves
  instead of matching values n×n: 65536 `float64` entries take 6 ms, not 4 s.
- `Nx.place` puts a value on several of rune's devices, `replicated` or
  `sharded`: each device gets its window alone, uploaded from the host (from
  its file, for a value on the disk) or copied by tolk from the devices
  holding it.
- **Breaking:** devices of different backends (`METAL` and `CPU:1`) have
  different engines, so a placement over both raises.
- A compiled `Nx.scatter` drops an update whose index is outside the axis
  when there is one update or the axis has size 1, and the gradient of a
  compiled `Nx.take` over an axis of size 1 no longer adds the cotangent of an
  out-of-range read. Both used to land at the index modulo the axis.
- `Rune.remat` saves memory under `jit`: its recomputation shared the forward
  pass's nodes, which kept every block's intermediates live (16 MLP blocks,
  batch 2048, CPU: 169 MiB with or without `remat`, now 62 MiB).
- `Rune.remat` works under `jvp` and differentiates the tensors its function
  captures, such as the weights a layer closes over: their gradients and
  tangents were zero, or missed the captured share. It raised under `jvp`.
- A compiled `Rune.scan` no longer copies each stored carry, output and
  cotangent before writing it to its loop buffer (16 MLP blocks, batch 2048,
  CPU: 16 kernels instead of 21 and 70 MiB instead of 78 for the gradient).
- Compiled mxfp4 products on the CPU group routes by expert from 96 routes
  over 32 experts, at every dtype: gpt-oss-20b's MoE block runs 1.2 to 1.6
  times faster at float32 from 96 to 160 routes, 1.5 times at bfloat16 at 160,
  and 2.6-2.7 times at both at 512, while bfloat16 is up to 14% slower from 96
  to 127 routes. The CPU used to never group.
- A compiled function traced under `PROFILE=1` or `DEBUG>=2` no longer serves
  its persistent cache entry to later unprofiled runs, which ran with
  per-kernel queue timestamps (512-token gpt-oss-20b prefill 7-10% slower).
- A compiled function's persistent cache entry is no longer served under
  other scheduling or queue settings, such as AMD's `AMD_AQL`, `WAVES_PER_SH`,
  `AMD_DISABLE_SDMA` or interface, whose queues it would replay wrongly.
- A compiled `Nx.sort` of floats returns the input's elements at its
  indices: -0 stays -0 and a NaN keeps its sign and payload. A values-only
  sort costs up to 1.8x more (131k bfloat16, CPU).
- A compiled integer constant outside its dtype's range, such as
  `Nx.add_s x 256` on uint8, takes the wrapped value as eager does. Folding
  used it unwrapped: `x + 256 < 5` gave false and `x / 257` gave 0.
- **Breaking:** `Rune.jit` takes the signature of the function it compiles and
  returns a function of the same type, with arguments read (`@->`) or consumed
  (`consumes ... @@`) and a result of any structure. `jit2`, `jit_step`,
  `pmap2` and `?donate` go: `jit p f` is `jit Nx.Ptree.(p @-> returns tensor)
  f`, `jit2 p q f` is `jit Nx.Ptree.(p @-> returns q) f`, and `jit_step r s f`
  is `jit Nx.Ptree.(r @-> consumes s @@ returns s) f`.
- **Breaking:** a compiled call marks what it consumes before its first kernel,
  and a value over consumed storage raises on use, naming where it was consumed
  ("this value was consumed at 1.keys in a compiled call's arguments"): a
  leaf's path starts with its argument's position from 0.
  A consumed leaf that views part of its storage, or whose storage another leaf
  of the call or a capture of the function reaches, raises before the call and
  consumes nothing; such a storage was read instead. Consuming a storage a
  compiled function binds ends it for its values, and that function keeps
  replaying with its buffer.
- A result takes the storage of the consumed leaf it derives from, or of one no
  kernel reads after it is written, where it took only the state leaf at its
  own position: a state returned in another order reuses its storage.
- **Breaking:** a compiled function that returns one value at two leaves of
  its result returns two values, each with storage of its own; they were one
  value, which a call consuming one of them ended for the other.
- A compiled function's programs are keyed on each leaf's path and on what its
  argument's structure reports (a window, a list's length, an option's
  presence, a case), besides dtypes and shapes: an argument with another window
  compiles its own program, where it replayed the old one. `RUNE_JIT_DEBUG=1`
  reports each retrace with the first difference from the previous call.
- **Breaking:** transformations take structures as `'s Nx.Ptree.t` values in
  place of modules. `vjp`, `jvp`, `custom_vjp` and `custom_jvp` also
  take the result's structure, so `vjp2` and `jvp2` go; so
  do `Rune.Ptree` and `while_loop`'s module.
- **Breaking:** `Rune.vmap` and `Rune.remat` take the signature of the function
  they transform, such as `Nx.Ptree.(tensor @-> tensor @-> returns tensor)`,
  and `vmap` maps axis 0 of every argument. Capture a value to hold it fixed
  and move another axis with `Nx.moveaxis`; `vmap2`, `?in_axes`, `?out_axis`
  and `vmap'`'s `?in_axis` and `?out_axis` go.
- **Breaking:** a transformation replaces each argument tensor by a fresh alias
  before it differentiates or maps it, so a captured tensor is a constant even
  when it is also the argument: `Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) w`
  is `w`, where it was `2 w`.
- `Rune.scan` raises when its body returns a carry whose structure differs from
  the one it received, or outputs that differ from the first step's, naming the
  first path where they differ. It checks eagerly and under `Rune.jit`.
- Fix `Rune.custom_vjp` and `Rune.custom_jvp` when the rule returns one of its
  parameters: its cotangent was added to the parameter's twice, and its tangent
  replaced the parameter's for every later use. `Rune.remat` of a function
  returning an argument doubled that argument's gradient.
- Compiled `Nx.max`, `Nx.min`, `Nx.argmax` and `Nx.argmin` of floats return
  NaN (the first NaN's index) when any element is NaN, as eager does; a NaN was
  ignored unless it came first, and Metal flushed subnormals. Compiled `max`
  and `min` of -0 and +0 give +0 and -0 (IEEE).
- Fix staged `Rune.scan` compilation on Metal by expressing the loop as an
  effect on its output buffers. Nested scans and their gradients use the same
  call protocol as other compiled schedules.
- Route JIT output and scatter-prefill copies through Tolk’s shared executor
  so device transfer and host fallback follow the same storage protocol.

- Preserve donated storage for indexed writes and in-place `Rune.scan` carries
  after Tolk materializes their allocations. Queue batching retains kernel
  access order so donation cannot overwrite an input before its last reader.

- Compiled `Nx.cummax` and `Nx.cummin` order subnormals correctly on Metal,
  where they compared as zero; compile for 8-bit floats along an axis longer
  than 512, where the C and Metal compilers failed; and keep the first of equal
  zeros, as eager does. Where int64 is native they run one scan instead of two,
  faster on some shapes and slower on others.
- A compiled `Nx.prod` of `bfloat16` or `float16` values multiplies at
  `float32` and rounds once, as the eager one does. Rounding after every factor
  drifted: 256 bfloat16 values just under 1 multiplied to 0.3633 on the CPU
  against 0.3672 eager.
- A compiled `Nx.sum` or `Nx.mean` of `bfloat16` or `float16` values
  accumulates at `float32`, as the eager one does. It accumulated at the input's
  precision: 4096 bfloat16 ones summed to 1024 on the CPU and 16384 to 4096 on
  Metal.
- A compiled function on Metal takes, returns and captures values with no
  elements. It raised "Metal OOM while allocating buffer": an empty value was
  given device storage, and Metal has none of size zero.
- On Metal, a compiled `Nx_quant.apply ~ids` over many routes groups them by
  expert, reading each expert once per block of its routes. gpt-oss-20b's
  512-token prompt takes 1.09 s on an M1 Max, against 5.5 s before.
- Compiled `Nx.argsort` and the indices of `Nx.sort` no longer cost n² work for
  dtypes of up to 32 bits: a float32 argsort of 131072 entries on Metal drops
  from 61 ms to 12 ms. 64-bit dtypes keep the quadratic path.
- Compiled `Nx_quant.apply` runs tolk's kernel, which decodes MXFP4 weights in
  registers, when a matrix meets few rows (32 on Metal and 64 on the CPU, at
  bfloat16). On an M1 Max, gpt-oss-20b decodes a token in 48-54 ms against
  122-126 ms, 8 sequences in 100 ms against 446 ms, and a 512-token prompt in
  3.5 s against 8.8 s.
- `Nx_quant.apply` and `Nx_quant.dequant` compile under `Rune.jit`, and `grad`,
  `jvp`, `vmap` and `with_debug` take them: gradients flow to `x` only, and a
  quantised weight whose part is differentiated raises.
- Fix compiled `int64` and `uint64` constants beyond 2^62: `Rune.jit` read
  them through OCaml's 63-bit `int`, so `Int64.min_int` became 0 and
  `0x4000000000000000L` became `-2^62`. They now keep every bit.
- Fix reading a strided view of a placed value (`Nx.to_array`, `Nx.copy`, or
  an eager operation on `Nx.transpose` of it): the elements were copied
  through OCaml values, which quieted signalling NaNs. They are now copied as
  bits.
- Compiled `Nx.sort` and `Nx.argsort` put NaN after every number in either
  direction, as eager ones do. A NaN dropped out of the sorted values, with
  another value repeated in its place, and the positions around it were wrong.
- Compiled `Nx.cummax` and `Nx.cummin` are NaN from the first NaN on, as
  eager ones are. They kept the running maximum or minimum past a NaN.
- **Breaking:** Remove `Rune.to_device`. Place values with
  `Nx.place (Nx.Placement.on (Nx_metal.device 0)) x`; on the host,
  `Nx.place` returns its argument where `to_device` made it contiguous.
- A placed leaf or capture that views part of its storage (a slice, a
  transpose, a flip, a broadcast) is read in place by a compiled function on its
  device, where it was copied through the host on every call.
- A compiled function runs where its placed inputs and captures live, else on
  the host. A leaf or capture elsewhere, or a `pmap` output, raises
  instead of going through the host; so does `float64` on Metal.
- **Breaking:** Remove `RUNE_JIT_FORCE_COPY`; compile for `"CPU:1"` to run the
  device path without a GPU. `Rune.pmap` over `"CPU"` raises.
- Reading a compiled function's output no longer moves it to the host:
  `Nx.item` on resident logits copies one element and the logits stay
  resident, and eager operations and `grad` over resident values keep their
  results on the device. `Nx.place Nx.Placement.host` makes a host copy.
- `Rune.to_device` is no longer the identity inside transformations: under
  `grad` and `jvp` it is linear and a gradient returns to its input's
  placement, under `vmap` it places the batched value, and inside `jit` it
  raises `Jit_error` unless the device is the program's.
- A `Rune.pmap` output stays on its devices when read, and an nx operation on
  it reads it and returns a host value, as before.
- Placing a value on a device that cannot hold its dtype raises
  `Invalid_argument`, at `Rune.to_device` and at an eager operation whose
  result has that dtype: `float64` on Metal, and complex and 4-bit integers on
  every rune device, so `Nx.rfft` of a placed value raises; place it on the
  host first.
- `RUNE_JIT_RESIDENT_BUDGET` counts every device allocation since the last
  major collection, eager results and uploads included, and a device that
  still cannot allocate after a collection raises `Nx_device.Out_of_memory`.
- A compiled function whose output has no elements returns an empty tensor of
  that output's dtype and shape instead of raising "an output of the traced
  function was not scheduled to a buffer".
- `Nx.cumsum` under `Rune.jit` keeps int8 and int16 results in their dtype. The
  compiled scan left them in int32, so values came back wrong (int8
  `[100; 100; 100; 1]` gave `[100; 0; 0; 0]`) and the process could crash.
- Compiled functions on a device share the memory their intermediates need,
  sized to the largest, instead of each holding its own: gpt-oss-20b run one
  compiled layer kind at a time peaks at 17.4 GB instead of 19.4 GB.
- A `Rune.scan` staged under `Rune.jit` updates a carry in place when its step
  writes it with `Nx.set` or reads it only where it writes: a step writing one
  row of a stacked cache no longer copies the whole cache.
- `Rune.scan c x y` folds over structures and `scan'` over single tensors. Put
  per-step data such as stacked layer weights in `xs`: `jit` reads it in place.
- A `Rune.scan` staged under `Rune.jit` replays its body as batched device
  graphs instead of launching each kernel: a 64-step scan on Metal goes from
  18.5 ms to 6.4 ms.
- A `Rune.scan` staged under `Rune.jit` no longer waits for the device on
  every iteration: the body writes what the loop used to copy. A 64-step scan
  on Metal goes from 60 ms to 18 ms.
- A compiled call on a GPU returns without waiting for its kernels; reads wait.
  Twenty-six chained small calls on Metal take 2.9 ms instead of 10.8 ms.
- Tracing a function under `Rune.jit` no longer allocates a buffer for every
  traced value. Each placeholder was an uninitialised tensor of the result's
  full size; the pages were never touched, but OCaml counted the bytes and ran
  major collections throughout the trace. The first call of a gpt-oss-20b step
  goes from 47 s to 11 s. Placeholders are now traced values
  (`Nx.Repr.Traced`): dtype and shape, no bytes.
- A compiled function plans the memory of its intermediates: buffers whose
  lifetimes do not overlap share one arena per device instead of each owning an
  allocation for the life of the function. A single-token step of gpt-oss-20b
  on Metal held 11 GB of intermediates beside its 13.8 GB of weights and now
  holds 0.2 GB; no longer under memory pressure, it drops from 6.7 s to 0.42 s.
  `NO_MEMORY_PLANNER=1` turns it off.
- `RUNE_JIT_RESIDENT_BUDGET` counts the outputs of compiled calls only. Placed
  weights stay resident by design, and counting them ran a major collection
  before every output allocation once a model larger than the budget was
  placed. `jit_stats ().resident_bytes` still counts them.
- A compiled function that captures a value placed with `Rune.to_device` on
  its own device binds the value's buffer as its constant: nothing is uploaded,
  and every compiled function over the same weights shares one device copy,
  where each used to upload its own. A bound value keeps its buffer for as long
  as it is reachable: a host read copies it out and leaves the buffer in place.
- Add `Rune.to_device ?device x`, `x` with its bytes held by a device. The
  result has `x`'s type and value and is resident like an unread output of a
  compiled call: feeding it to a compiled function on that device moves no
  bytes, and a host read brings it back. Its buffer
  bypasses the allocator's cache. On the CPU device it is `Nx.contiguous x`,
  and inside `jit`, `grad`, `jvp` and `vmap` it is `x`.
- Copies between host and device move 64 MiB at a time. `Rune.jit` staged each
  upload and read-back in a host buffer the size of the tensor and kept one per
  distinct size for the life of the compiled function, 1.1 GB for a 1B
  parameter model, and made a strided tensor contiguous whole before staging
  it. A strided tensor is now copied piece by piece, and a long run of copies
  synchronizes the device every 256 MiB.
- `Rune.jit` on the CPU device compiles kernels that read host memory at any
  address. The kernels declared their vector types aligned to their size while
  tensors were read in place wherever their data sat, so on x86-64 a slice
  starting inside a buffer, a tensor over a mapped file, or a four-wide float64
  kernel over ordinary allocated memory could kill the process.
- Compiled programs on Metal replay as batched GPU submissions: the kernels of
  a `Rune.jit` trace are recorded once and each call submits them in a few
  command buffers instead of one per kernel. The per-kernel launch cost drops
  from about 27 to 3 microseconds, level with the CPU device: a trace of 256
  small kernels takes 0.9 ms per call where it took 7.7 ms, and a GPT-2 124M
  decode step 6.6 ms where it took 9.2 ms. `JIT=2` restores one submission
  per kernel.
- Reverse mode no longer copies every cotangent. A cotangent is held as the
  lazy view its pull produced (a transpose, a broadcast) and materialized where
  a reshape needs it and where a gradient leaves `Rune.grad`. Compiled
  gradients run far fewer kernels, 95 against 210 for a two-block decoder, and
  transposes that cancel are free in the backward pass as they were in the
  forward one.
- `Nx.set` with a run-time window start (`Nx.D`) under `Rune.jit` costs the
  window instead of the destination: it compiles to a store at the window's
  flat positions, in place on a consumed tensor. A one-row write into a
  1048576x64 cache takes 0.26 ms on Metal, the same as into 4096 rows, where
  it took 3.4 ms.
- Fix `Rune.grad` through `Nx.scatter` in `` `Set `` mode with repeated indices:
  every update aimed at a position received the cotangent, where only the
  last one reaches the output. Shadowed updates now get zero.
- A compiled call writes a scatter over the storage of a consumed destination,
  so `Nx.scatter` into a consumed tensor costs its updates alone: a 64-row
  write into a 131072x8x64 pool takes 0.35 ms on Metal, the same as into 4096
  rows. The program never copies a destination that is an input: the write
  lands in the output's buffer, which takes the consumed storage or, on a call
  that cannot lend it, is given the input's value by one device copy.
  `reused_bytes` counts the pool.
- `Nx.scatter` under `Rune.jit` costs the number of updates plus one copy of
  the destination, instead of destination size times update count. The
  gradient of `Nx.take` and `Nx.take_along_axis` is such a scatter: an
  embedding gradient for 1024 tokens over a 131072x256 table went from 2.1 s
  to 20 ms on CPU. `unique_indices` now reaches the compiled kernel, and an
  index outside the axis still writes nothing.
- `Rune.grad` through `Nx.matmul` now sums the cotangent of an operand over
  its batch axes of extent one that broadcast against the other operand. The
  gradient used to come back at the broadcast shape and a later pullback
  raised on the element count, so grouped-query attention (several query
  groups against one key head) could not be differentiated.
- `Rune.jit`'s compile cache now keys on every setting that changes a compiled
  trace: tolk's program configuration (`NOLOCALS`, `TC`, `IMAGE`,
  `TRANSCENDENTAL`, ...) and the scheduling variables (`SPLIT_REDUCEOP`,
  `REDUCEOP_SPLIT_THRESHOLD`, `PCONTIG`, `RING`, ...). Changing one after a
  first compile kept serving the entry built under the old value.
- `Rune.jit` no longer lets a fresh output share a buffer with an input the
  program does not read: the two buffer-slot counters (rune's and tolk's)
  could hand out the same slot, so a resident input fed to a later call had
  its bytes overwritten by that call's output. Rune now draws every slot from
  tolk's process-wide counter.
- A compiled call writes an output over the consumed input it derives from
  when every path between them stays at the same element and no later kernel
  reads the input, so a jitted training step or decode step holds one
  generation of state on the device instead of two. `jit_stats` counts the
  bytes reused in `reused_bytes`, and `RUNE_JIT_DEBUG=1` reports per consumed
  leaf whether its storage was reused or released.
- `Nx.set` with a `D` window follows every transform: `grad` differentiates
  both operands, a mapped start under `vmap` writes each example at its own
  clamped position, and a window on `pmap`'s mapped axis is written shard by
  shard.
- **Breaking**: with tensors as values (RFC 0001) there is no in-place
  update to replay. `Rune.jit` no longer writes an assigned input leaf back
  to the host on every call, and `grad`, `jvp` and `vmap` have nothing to
  refuse; carry state by returning it, as `Rune.jit`'s example shows.
- `jit_stats` retires the outputs that were dropped unread and collected
  before it reports, so `resident_bytes` counts only reachable handles. Their
  buffers used to wait for the next compiled call, which made the counter
  depend on when the GC ran.
- A parameter structure is a positional sequence of leaves: a tensor behind
  two leaves is two parameters. `jit` bound two such leaves to a single input
  at trace time, so a later call passing distinct tensors read one of them for
  both positions; every leaf visit is now its own input. `grad`, `vjp`, `jvp`
  and their variants gave both leaves the summed gradient and, for `jvp`, one
  leaf's tangent; each leaf now gets its own. Tie weights by structure, not by
  aliasing.

### Jera (new)

- Add `Minimize.levenberg_marquardt`, for a sum of squares given by its
  residual, with its damping falling on accepted steps, and
  `Minimize.nelder_mead`, for an objective with no useful gradient, which
  restarts from its best vertex before it converges (Kelley's test) and
  returns a detached answer.
- Add `Minimize.solve` and `Minimize.iterates` with the gradient methods
  `Minimize.bfgs`, `lbfgs ~memory` (both searched to the strong Wolfe
  conditions) and `newton` (steps solved by `~linear` on Hessian-vector
  products). The minimum is stated as the zero of rune's gradient, so its
  derivative is the implicit one through the Hessian.
- Add `System.solve`, a zero of a system on a structure: `System.newton`
  with a sufficient-decrease line search (Newton–Krylov with `Linear.cg` or
  `gmres`), `System.broyden` for a residual with no usable derivative, and
  `System.anderson ~memory` for fixed points. The answer is stated through
  `Rune.root`, so its derivative is the implicit one, solved by `~linear`.
- Add `Linear.banded ~width`, for systems whose matrix has entries near its
  diagonal only: `2 width + 1` products probe the band, factored by LU with
  partial pivoting in loops of fixed-size windows.
- Add `Linear.gmres`, restarted GMRES preconditioned on the right, for
  non-symmetric systems: each cycle of `restart` Arnoldi steps is a scan,
  and the solve stops between cycles on the residual.
- Add `Linear.cg`, preconditioned conjugate gradients for symmetric
  positive-definite systems, stopped at `‖a u − r‖ ≤ rel ‖r‖` or on a
  direction of non-positive curvature.
- Add `Linear.solve` and `Linear.dense`: the `u` with `a u = r` for a linear
  function `a` on a structure, materialised from its products and solved by
  `Nx.solve`, checked against the backward error of an LU factorisation, and
  differentiated through `Rune.root` with the same solver.
- ODE solves bound the step by the span of their times: after many short
  intervals the step could grow to infinity, and a rejection then never
  shrank it, so the next long interval spent the budget.
- Add `Ode.delay`, for `y' t = f t (y t) (y (t − τ))` with constant lags
  read from the accepted steps' continuous extensions and a `history`,
  stepping onto the breakpoints the lags create.
- Add `Ode.event`, a solve that ends at the first sign change of any
  component of an event tensor, returning the time, the state and the
  component's index, with the crossing time's derivative.
- Add `Ode.path`, the solution as a `Piecewise.t` of one piece per
  accepted step, each the method's continuous extension (order 4 for
  `tsit5` and `dopri5`, 3 for `bs3`).
- A failed solve's report gives its settings, the lane's inputs, the
  budget used in its own unit and what to change; `Solution.pp` prints it
  for the first failing lane. `Ode.sample`'s times out of order end their
  lane `Stalled` instead of raising, and `Ode.solve` returns its start at
  `t0 = t1`.
- Add `Ode.solve` and `Ode.sample`, adaptive solves with the embedded
  methods `bs3`, `tsit5` and `dopri5`: a proportional–integral controller
  chooses steps on detached values, and the answer takes the accepted steps
  again, so its derivative is theirs.
- Add `Piecewise.adapt`, a piecewise Chebyshev fit of a function to a
  tolerance on a partition of at most `budget` pieces, refined where the
  series' tail is largest.
- Add `Quad.qmc`, randomised quasi-Monte Carlo over boxes of up to 1111
  dimensions: a Sobol sequence under 16 random digital shifts drawn from a
  key, stopped on the standard error at powers of two.
- Add `Quad.cubature`, Genz and Malik's adaptive rule of degree 7 over boxes
  of 2 to 10 dimensions, one problem per lane of `Quad.Box.v lo hi`.
- Add `Quad.tanh_sinh`, double-exponential integration: tanh-sinh on a
  finite range, exp-sinh on `Range.from` and sinh-sinh on `Range.line`,
  for endpoint singularities and infinite ranges.
- Add `Quad.adaptive`, Gauss–Kronrod integration on a partition refined by
  bisecting the piece of largest error, elementwise, whose answer is the rule
  over the final partition.
- Add `Minimize.bracket`, Brent's method elementwise with a forced golden
  step, its minimum stated as a zero of rune's derivative of the function.
- Add solves, which return a `Solution.t` with a status per lane:
  `Solution.get` returns an answer that converged everywhere and raises
  `Failure` with the first failing lane's report, `best` and `ok` read every
  lane. Tolerances are `Tol.v`, `rel`, `abs` and `ulps`. `Root.bracket`
  (ITP with bisection, at most 2b + 2 evaluations) and `Root.newton` find
  zeros elementwise, stated through `Rune.root` so their derivative is the
  implicit one, and zero at a lane that did not converge.
- Add `Sde.march`, fixed-step marches of stochastic differential equations
  along a `Sde.Brownian` path, a virtual tree that returns increments and
  space–time Lévy areas as a pure function of a key: `euler_maruyama`,
  `milstein` and `sra1` (Itô) and `reversible_heun` (Stratonovich). A
  compiled march takes its path as an argument, of `Sde.Brownian.ptree`.
- Add `Piecewise`, piecewise Chebyshev series closed under `eval`,
  `derivative` and `integral`: interpolants `linear`, `cubic` (`Natural`,
  `Not_a_knot` and `Clamped` ends), `steffen` (monotone) and `hermite`,
  fits by `chebyshev`, `eval_at` and `extend`. `Grid` is their tensor
  product over several axes: `linear`, `cubic` and `chebyshev`, with
  partial `derivative` and `integral`.
- Add `Ode.march`, explicit Runge–Kutta marches through the times of `at` in
  equal steps: `Ode.euler`, `rk4`, `ssprk3`, `bs3`, `tsit5` and `dopri5`, and
  `Ode.tableau` for a caller's Butcher tableau. A method's tag says which
  drivers take it.
- Add `Quad.fixed` and `Quad.cumulative`, integrals of an elementwise
  integrand by a rule: `Quad.Rule.gauss n` and the Gauss–Kronrod rules
  `Quad.Rule.kronrod 7` and `10`, on finite ranges and, by a change of
  variable, on `Quad.Range.from a` and `Quad.Range.line c`.
- Add `jera`, numerical methods over nx tensors that rune differentiates,
  maps and compiles. `Split` composes exact kicks and drifts of a separable
  Hamiltonian: `leapfrog`, `mclachlan`, Yoshida's `yoshida4`, `yoshida6` and
  `yoshida8`, and `Split.v` for palindromic coefficients, with `Split.step`
  and `Split.march`.

### Ppx_ptree (new)

- A mistyped `[@ptree.walk f]` and a `ptree` of another type are reported at
  the part instead of at the declaration or a generated name. Every
  diagnostic names its part, and a fixed instance can nest a type derived
  earlier in the module (`state Kaun.Linear.t` with `ptree_state`).
- `[@ptree.int]`, `[@ptree.skip]` and `[@ptree.walk f]` after a constructor's
  single argument apply to it. They were ignored there, so
  `Frozen of Nx.float32_t [@ptree.skip]` still walked the tensor. On a
  constructor with no argument or several, and on a type declaration, they
  are errors.
- Add the `ppx_ptree` deriver: `[@@deriving ptree]` writes a type's `walk`,
  the one function of `Nx.Ptree.S`, and a type without a parameter's
  structure, `ptree`. `[@ptree.int]` marks an integer a compiled program
  depends on, `[@ptree.skip]` data no program reads, and `[@ptree.walk f]` a
  part walked with `f`.
- Add a runnable linear-regression example that trains a derived structure
  with `Rune.grad` under `Rune.jit`.

### Munin (new)

- Artifact digests are BLAKE2b-256 from the standard library instead of
  SHA-256, so `munin` no longer depends on the `sha` package. Blobs live
  under `blobs/blake2b` in the store.
- The system statistics stubs build on Windows. CPU times come from
  `GetSystemTimes`, memory from `GlobalMemoryStatusEx`, disk space and page
  size from the Win32 calls; the load average, which Windows does not keep,
  reads as zero.

Local experiment tracking for Raven. Evolves `kaun-board` into a full
experiment tracker — the Raven equivalent of W&B or MLFlow, without a server.
Declare a metric once with `Session.metric` and log samples through the handle
it returns, so a key is never spelled twice and a typo cannot silently create a
second channel. Save artifacts from your training script, monitor runs live in
the terminal with `munin watch`, then compare results with `munin compare`.
Data is plain JSON on disk, so `jq` and shell scripts work out of the box. Git
commit, command line, and system info are captured automatically. The
`munin.sys` sub-library adds opt-in CPU and memory monitoring in a background
thread.

- Add `x-kaun-mnist-jit`, an example tracking a `Rune.jit`-compiled CNN
  training run (forward, backward, and SGD update in one compiled program,
  Metal by default) with live `munin watch` monitoring.

### Tolk (new)

- `Tolk_engine.time` always times a schedule on its own profile and its
  devices' stamps, so a profile taken around a search no longer changes what
  it measures or the kernels a beam search picks. `Tolk_engine.clock` no
  longer depends on whether a profile is taken.
- The host runs a run of its calls, with the ranges and back edges around
  them, as one host program that calls each kernel through
  `Nx_device.Program.entry`: no OCaml runs per trip or per kernel.
  **Breaking:** `Hcq2.device`'s `queues` is `work`, `Queues`, `Programs` or
  `Calls`.
- A precompiled call inside a loop may pass a scalar parameter of its body a
  weak integer expression of the loop's ranges, such as `2r + 1`: each trip's
  kernels read it, and their programs compile once for any loop.
  `Realize.get_call_arg_uops` leaves out every scalar argument.
- tolk is no longer linked with `-linkall`: a program links only the modules
  it reaches. tolk's test of `Op` shrinks from 8.3 MB to 4.4 MB.
- **Breaking:** the rules `Shape.simplify` rewrites with are `Shape`'s:
  `symbolic`, `symbolic_simple`, `commutative`, `invalid_gate` and
  `pm_remove_invalid` from `Symbolic`, `div_and_mod_symbolic` and its
  `number`, `rule` and `Not_a_number` from `Divandmod`, `mop_cleanup` from
  `Movement`, and `pm_uncast_const` from `Uop_weak`. `Divandmod` and
  `Movement` are removed, and so is `Shape.Private.set_symbolic`: `Shape`
  sets its own rules, so `simplify` works in a program that does not link
  `Symbolic`.
- **Breaking:** `Ops` keeps the node language: arguments, construction,
  bounds, ranges, patterns and rewriting. What reads a shape needs
  simplification and moves to two new modules. `Shape` holds shapes and
  simplification, which call each other: `Shape.shape`, the movements
  (`Shape.reshape`, `Shape.mop`, ...), `Shape.simplify`, `Shape.resolve`,
  `Shape.Sint`, divisibility and `Shape.sym_infer`. `Call` holds what calls
  take and the calls: storage (`Call.param`, `Call.alloc`, ...), sharding,
  the bindings of variables, `Call.call_with_outputs` and
  `Call.program_info_of_sink`.
- **Breaking:** `SPEC=2` and `SPEC=3` no longer check each node as
  `Ops.v` builds it; any nonzero `SPEC` checks the graphs passed between
  stages. The check needed every module of tolk linked into every program.
  `Ops.Private.set_spec` is removed.
- A compiled gather through a pad, such as `Nx.concatenate` of a buffer and an
  `Nx.take`, no longer reads its indices before their buffer and faults: the
  index's simplification under the pad's gate dropped the gate of the loads
  inside it.
- Stored values of one shape that share a computation, and do not read each
  other, run as one kernel that writes them all: a compiled `Nx_wide.add`
  runs one kernel, from six, and computes its chain once.
- On the host, a kernel without a reduce computes 64 bytes of elements at
  once as Clang vectors, 16 float32 or 8 float64, and Clang compiles its body
  once: `Nx.erf`, `Nx.erfc`, `Nx.erfinv` and `Nx.ndtri` replay 1.2 to 1.6
  times as fast on a core as without the upcast, with the same bits. A kernel
  whose own operations already run 5 at a time, such as float64
  `Nx.log_betainc`, stays scalar and compiles in a fraction of the time.
- A host program's variables take slots past its blocks' bounds when its
  buffers skip a slot; a variable could take `block_hi`'s slot and read it.
- A cast of an unsigned mask, or of a right shift by less than the operand's
  width, to a wider unsigned type widens the operands first, so a 4-bit value
  unpacked from its byte decodes at its consumer's width: the host kernel of a
  4096 x 4096 MXFP4 product runs in 22.7 ms instead of 43.0 ms on one core.
- A schedule runs a loop of calls while a flag holds: `Ops.backedge` around a
  call, with a range bounding its trips and one boolean of storage the calls
  write. The engine reads the flag before each trip; on devices with command
  queues each trip is one batch. `Rune.iterate` compiles to it.
- On a device that stamps its own runs, `Search.beam_search` times a round's
  candidates in order while later ones compile (new `Worker.iter`); on the
  host's clock, which compiles would slow, its rounds still compile first. It
  takes `~clock` and reads it from its kernel's linked program: new
  `Search.clock` and `Tolk_engine.clock`. A search round of gpt-oss-20b's key
  and value projection on an M1 Max's Metal takes 352 ms, from 430 ms.
- `Tolk_engine.time` raises `Nx_device.Lost` when a device that runs the
  timed kernels is lost during the run, where it raised `Invalid_argument`
  "no span": a search on NV that a GPU co-tenant held past the hang timeout
  died naming a missing span instead of the lost device.
- `Search.beam_search` compiles each source once: candidates of distinct
  kernels that render one source each ran the compiler, a rejected one again
  for every domain waiting on it. A domain asking for a source another is
  compiling now waits for its binary or rejection. Candidates are compiled
  without the disk cache of their compiler's binaries.
- `Compiler_cpu.clang` runs Clang as a process of its own, waited for with
  the runtime released, instead of through the C library's `system`, which
  macOS runs one at a time: 16 compiles on 8 domains take 0.11 s, from
  0.66 s.
- Lowering a kernel allocates 22% less again: a graph walk pushes its work
  without allocating, and looking a node up builds a smaller probe. 8 domains
  lowering lorenz_simple's candidates collect 25% less often.
- A beam search keeps nothing of its candidates: they are compiled by
  `Codegen.linearize` then the new `Codegen.compile`, which keep nothing, and
  only the search's choice and the kernel compiled with it stay in the disk
  cache. Candidates grew the cache without bound, to 8.9 GB on one machine.
- **Breaking:** `Codegen.to_program` takes only a kernel's sink, and `Codegen.compile` only
  what `Codegen.linearize` returns; `Realize.lower_and_compile` no longer
  completes a program compiled in part.
- `ASSERT_COMPILE` no longer stops a beam search at its first candidate: a
  search that is not kept compiles its candidates, and the kernel compiled
  with its result raises at its own compilation.
- **Breaking:** `Ops.toposort`, `Ops.reaches`, `Ops.graph_rewrite`,
  `Ops.substitute` and `Spec.type_verify` take a required
  `~calls:(Enter | Skip)` (new `Ops.calls`) in place of `?enter_calls`, whose
  default entered call bodies in the first two and `type_verify` and skipped
  them in the others.
- **Breaking:** `Ops.backward_slice`, `Ops.backward_slice_with_self` and
  `Ops.op_in_backward_slice_with_self` take the required `~calls`, as their
  sibling walks do; they always left call bodies out.
- **Breaking:** `Ops.graph_rewrite` takes its rules as an `Ops.rules`
  (`After_sources`, `Before_sources` or `Around_sources`) and a required
  `~pass:(Fixed_point | Once)` in place of `?bottom_up`, `?bpm` and `?walk`,
  so a rewrite before the sources with a second matcher, which raised, cannot
  be written. `Ops.substitute` takes `~pass` in place of `?walk`.
- **Breaking:** `Search.get_kernel_actions` takes no `?include_0` and returns
  each candidate with the action that made it, `(Opt.t * Scheduler.t)`, in
  place of positions in the action table.
- `Symbolic.sym` folds the constants of nested integer maxima at the
  maximum's width: on uint8, `max(max(w, 1), -3)` became `max(w, 1)`, where
  the machine computes `max(w, 253)`.
- `Transcendental.xexp2`, `xlog2` and `xsin` are within one unit in the last
  place of the correctly rounded result in every float; float16 now computes
  at float32 and rounds once, and the functions refuse it. `xsin` was 205266
  units off in float32 near multiples of pi below 30, where its Cody-Waite
  reduction subtracted a wrong part of pi, and up to two units elsewhere;
  `xlog2` was up to five units off. `cody_waite_reduction` removes quarter
  turns.
- Lowering a kernel allocates 2.4 to 2.9 times less: a pattern that fails to
  match allocates nothing, and a rewrite walk no longer builds a table per
  node. A beam search's candidates collect less often across its domains: 8
  domains lowering lorenz_simple's candidates take 2.9 s, from 6.1 s.
- `BEAM_DEBUG`'s compile time of a candidate counts its rendering and its
  compiler, which it left out.
- On arm64 the host hides a float zero from clang only where a comparison
  reads it, which is all its select-to-`fminnm` lowering needs, so a select of
  a zero by another condition compiles to a mask again. A routed MXFP4 product
  whose rows are selected by their validity ran 1.17 times as long as without
  the select, and runs 1.05 times as long.
- A value one kernel stores keeps its bounds in the kernels that read it (new
  `Ops.stored_bounds`), and a bitcast between integer types of one width keeps
  them, so a gather whose stored indices have a known range keeps no per-load
  check. A routed MXFP4 product whose experts are sorted in kernels of their
  own masked 40 byte loads a step: gpt-oss-20b's gate and up product of 512
  tokens on Metal takes 36.4 ms, from 42.6 ms.
- `Compiler_amd.hip` compiles each source in a process of its own, from a
  small program the library carries, so compiles from several domains run at
  once: comgr serialises the compiles of a process. A source that crashes
  LLVM raises `Compile_error` instead of ending the program. A beam search of GNODE's step on an R9700
  takes 19 s at `PARALLEL=8`, from 43 s.
- On AMD, a profile's span of a kernel covers its run, so `Tolk_engine.time`
  and searches time long kernels right. The compute queue's timestamps waited
  for the pipe to drain, which on GFX12 included the next dispatch: a 0.6 s
  kernel timed at 11 us, and searches kept such candidates.
- `Search.beam_search` times its kernel before its first round and starts
  from it, and a round progresses only when every sample of its fastest
  candidate beats every sample of the beam's first kernel by
  `BEAM_MIN_PROGRESS`, so a search never answers a kernel slower than the one
  it was given. The least of noisy timings is biased low, and searches ran
  rounds on gains within the noise, as PR #235 found.
- **Breaking:** `Tolk_engine.measure` is replaced by `Tolk_engine.slots`,
  `Tolk_engine.link_program` and `Tolk_engine.time`, and `Search.beam_search`
  takes `~link` and `~time`. A search allocates its buffers once, and not at
  all when its result is cached, links each candidate once, runs each sample
  once and silently, compiles each kernel once, and compiles no candidate past
  `BEAM_UOPS_MAX` (new `Codegen.linearize`). A candidate's setup on Metal
  takes about 1.2 ms, from about 3.9 ms. PR #235 found these costs.
- Every environment variable is one setting of the new module `Tolk.Setting`,
  declared and documented there with its reach, so each can be bound with
  `Setting.context` (`Ops.rewrite_stack_limit` is `Setting.rewrite_stack_limit`).
  `Helpers.getenv*` and `variable*` are removed, so the caches key on every
  variable that changes compiled output (`Setting.shaping`). A setting reaches
  output when a change of it alone can change what a compilation returns:
  `BEAM` and `JITBEAM` do; `CC` and `CUDA_PATH`, which a compiler's cache key
  names, do not.
- `TC_OPT` is the level of hand-coded optimizations alone, default 0; the
  search's tensor-core candidates always take level 2, which admits every
  kernel. `Search.actions` reads `TC` and `BEAM_PADTO` as bound.
- A compiler's cache key names it whatever `CCACHE` holds, so programs and
  searches key on it; `CCACHE`, now read when compiling, decides only whether
  binaries and programs go to disk. With `CCACHE=0`, two compilers differing in
  their options alone shared in-memory programs.
- `Hcq2.compile_linear`, `Hcq2.sched_batches` and `Jit.jit_lower` take a
  required `~profile:Hcq2.profile` (`Stamped` or `Unstamped`) and no longer
  read `DEBUG`; `Tolk_engine.profile` gives the one the engine's reports need.
- Scheduling a large graph no longer grows with the square of its kernels:
  weighing whether a stage could be inlined walked every kernel upstream of
  it, and the where-closure rule searched branches its INDEX gate rejects.
  An unrolled training step of 2031 kernels schedules in 5.3 s, from 27 s.
  `Ops.reaches` stops at the node it looks for.
- A beam search is read back from the disk cache only for the candidates it
  chose among, so `TC_OPT=0` no longer reads back a search made with `TC_OPT`
  unset. The search's `BEAM_*` variables, `SUM_DTYPE`, `HCQ_NUM_SDMA` and
  `WAVES_PER_SH` are declared, and `NO_MEMORY_PLANNER` reaches results, so the
  caches key on them too.
- A function with two loops or reductions of one length, such as a scan body
  of `sum (mul x (matmul x c))`, schedules again: numbering its ranges for the
  schedule cache's key raised "a bottom-up rewrite cycles" when two ranges
  traded numbers, and merged them when one took the other's (#236).
- A compiled binary is read back from the disk cache only for the compiler,
  version and options that made it: `Renderer.Compiler.v` takes the table as a
  function, which Clang's, Metal's, NVRTC's and comgr's compilers derive from
  their toolchain. A changed flag or an upgraded toolchain was answered with a
  stale binary. Beam searches are kept under their settings and compiler too.
- A compiled call run back to back on a GFX12 AMD GPU no longer hangs it,
  which made the driver reset the GPU for every process on it: a batch's
  memory barrier only invalidates the GPU's caches, and the host flushes the
  host data path before each submission, as it already did. A routed MXFP4
  product hung a Radeon AI PRO R9700 within a few hundred calls.
- A matrix-vector product with few outputs, along a matrix laid out by
  columns, spreads them over more workgroups with fewer threads across each.
  gpt-oss-20b's key and value projections, 512 outputs, ran on 8 of an RTX
  5000 Ada's 100 multiprocessors: 7.8 us each instead of 9.05 us, and its
  router 3.7 us instead of 6.7 us.
- A value that every output of a reduction reads, such as the normalised
  vector of a matrix-vector product, is stored once instead of computed again
  for each output when it reads more than one buffer. gpt-oss-20b's decode
  step on an RTX 5000 Ada spends 9.7 ms in kernels, from 11.9 ms.
- On the host, the hand-coded optimisations fill a kernel's 32 lanes with an
  upcast by 2 when 3 or 4 overflow, and a kernel of several reduces is not
  unrolled past them. sofo-raven's lorenz_simple step without a beam search
  goes from 117 ms to 83 ms on 6 cores (80 ms with `BEAM=2`).
- A compiled program on CUDA no longer hangs when a kernel waits for a copy
  queued after it in the program: CUDA's streams take commands as the host
  issues them, so `Hcq2.sched_batches` now submits each streamed queue after
  the queues it waits for (`Hcq2.submission`), splitting the batch otherwise.
- A value that kernels read widened to a type holding each of its values, as
  `bfloat16` to `float32`, is stored at its own type and widened as it is
  read: half the bytes, and a `float32` product of such values runs on the
  tensor cores. gpt-oss-20b's attention output projection of a 512-token
  prefill on an RTX 5000 Ada takes 0.30 ms, from 2.34 ms.
- The tensor cores take a product whose rows share the other operand by
  blocks, as `Nx_quant.apply ~ids` on a prompt multiplies each block of
  positions by its expert's matrix, with either operand as the core's first,
  and a product of decoded weights whatever its number of reduce axes.
  gpt-oss-20b's 512-token prefill on an RTX 5000 Ada takes 0.22 s, from
  0.71 s.
- A value of no axes that its consumers broadcast, such as a reduction of a
  whole tensor, is computed once. Each consumer computed it again for each
  of its elements: `x - Nx.sum x` summed `x` once per element, and
  `Vega.Loss_scale.step` checked every gradient in five kernels.
- A kernel's independent reductions share the threads of its workgroup. A
  check of many values, such as `Vega.Loss_scale`'s, ran all but its first
  reduction in each thread: in GPT-2 124M's float16 step on an M1 Max's
  Metal it takes 0.77 ms, from 3.1 ms.
- A precompiled call whose argument has no element compiles: the argument
  passes as its constant. `Rune.jit` of a scan whose carry has no element
  and whose step reads its rows failed verification, as under `vmap (grad
  (jit f))` over no lane.
- The C-style renderers write an integer constant as the value its type
  holds, the integer part of a float wrapped to the type's width. A constant
  past 64 bits, such as a large float padding value converted to an integer,
  was written as a literal C refuses.
- A matrix-vector product's workgroup takes rows of its matrix, which share
  the vector's loads, before rows that each read a vector of their own.
  `Nx_quant.apply ~ids` on a prompt laid its workgroups over blocks of
  positions: gpt-oss-20b's 512-token gate and up product takes 162 ms on an
  M1 Max's Metal, from 224 ms, and 21.9 ms on an RTX 5000 Ada, from 56.6 ms,
  in blocks of 4.
- A decoded operand of a summed product that an output axis reads only
  through its quotient by a block is decoded once per block. `Nx_quant.apply
  ~ids` on a prompt decodes each expert's matrix once per block of 4
  positions: gpt-oss-20b's 512-token prefill on an RTX 5000 Ada takes 0.71 s,
  from 1.49 s.
- A host kernel of more than 2^31 operations runs on all the host's cores.
  Its count of operations wrapped in 32 bits, sometimes below one block's
  worth, and it ran as one block.
- Compiling a long chain of operations no longer takes time growing with the
  square of its length: a jitted chain of 100 `Nx.sin` compiles from a cold
  cache in 21 s on an M1 Max, from 698 s. The node table's shards grew to 22
  buckets and stopped, the where-closure rule built each node's set of
  booleans, and `Ops.op_in_backward_slice_with_self` each node's slice.
  `Ops.reaches` replaces `Ops.bool_slice`.
- On the host, the hand-coded optimisations stop upcasting at 32 lanes, past
  which a kernel's values spill out of registers: a lorenz_simple tangent
  kernel without a beam search takes 1.51 ms instead of 2.12 ms.
- A sum unrolled beside upcast axes adds its products as multiply-adds, as
  other sums of products do. The unrolled products reached the reduce through
  a permuted view, which the fusion rule did not look through.
- `Tolk_engine.link` resolves each host program's launch once: the buffers,
  offsets and variables a run passes it, including those that move with the
  trips of a range, are not looked up and simplified again on each run. A
  jitted scan of 256 small steps on the host went from 5.1 ms to 0.27 ms on an
  M1 Max. `Shape.sym_compile` computes a symbolic integer as `Shape.sym_infer` does,
  simplified once.
- GPU matrix-vector products take their layout from the matrix's and split
  the reduce, also one inside another reduce, until the GPU is busy: gpt-oss-20b
  decodes in 11.1 ms on an RTX 5000 Ada, from 45.5 ms. `MV_BLOCKSIZE`,
  `MV_THREADS_PER_ROW` and `MV_ROWS_PER_THREAD` are removed.
- `Spec.program` refuses an index of a vector value at a lane that is not a
  constant, which CUDA cannot compile: its vectors are structs. Such a kernel
  fails at lowering where it failed in NVRTC.
- A process reads back the schedules of a function with loops, such as a
  staged scan, that an earlier process kept on disk. The loop's range was
  numbered from a counter that depends on what the process did before, so a
  warm run missed the schedules made after its first miss or hit.
- A program cached under one host compiler (`CC`) is compiled again under
  another, in the compile cache and the program cache: the object the first
  compiler made was served.
- `Tolk_engine.link` and `Tolk_engine.run` raise `Invalid_argument` naming
  both requests when an AMD batch encoded for one profile request (counters,
  traces) links or runs under another, whose shared trace buffers it would
  overwrite.
- `DEBUG` 4 and 7 print the source and instructions of a program read back
  from the disk cache, as of one compiled; they printed nothing.
- The CUDA device's kernels compile to cubins, which the driver loads as they
  are: a process no longer waits for the driver to translate PTX (137 ms a
  kernel on an RTX 5000 Ada when its cache is cold).
- A program kept on disk or in memory is made again under another value of
  `DMC`, `ALLOW_HALF8` or, within a process, `TUPLE_ORDER`, and a schedule
  under another value of any setting that shapes it: a cached one made under
  the old value was returned.
- A float `Mulacc` that a graph states folds and evaluates rounded once, as
  every target computes it, where it rounded twice, and an Invalid gate around
  one of its operands moves out of it, where it reached verification.
- A compiled CPU kernel runs on every compute core: its launch splits the
  largest output loop that every store reads into blocks the host's threads
  run at once, when the kernel does enough work to repay waking them.
  `Tolk_engine.Program.blocks` is the number of blocks a run takes.
- A compiled matrix-vector product whose vector is converted, or whose matrix
  is decoded, such as `Nx_quant.apply`'s, spreads each row's reduction over a
  group of threads: it ran each row in one thread.
- **Breaking:** tolk is a new port of tinygrad's compiler, at the tinygrad
  commit its test generator pins, running on `nx.device`'s runtimes. It
  replaces the first port: `tolk.uop`, `tolk.frontend`, `tolk.nn`, `tolk.cpu`,
  `tolk.cuda`, `tolk.metal`, `tolk.amd`, `tolk.hcq`, `tolk.nv` and
  `tolk.nvrtc` go, and the libraries are `tolk` and `tolk.engine` (modules
  `Tolk` and `Tolk_engine`). Tensors, devices and buffers are nx's, and rune
  compiles through it.
- Tolk no longer depends on Zarith, and so needs no GMP on the system. Its
  exact integers, the values of integer constants, dtype limits and bounds,
  are `Bigint.t`, an arbitrary-precision integer in pure OCaml with the names
  and semantics of the Zarith functions tolk used.

- The NV device refuses a release of NVIDIA's kernel driver it does not
  describe (any but 570, 580 and 610), so the default device moves on to CUDA.
  Under 615 it used 610's layouts, and the driver wrote a channel group's
  parameters past their buffer: the process aborted with glibc's
  `munmap_chunk(): invalid pointer`.
- A buffer over an external pointer (`Device.Buffer.borrow`) takes no
  allocator owner: its memory is the caller's. Borrowing host memory, or
  copying from or to it, inside an operation of another device, such as from a
  kernel's runtime callback, raised `Invalid_argument "device operation: owner
  was not prepared"`.
- `Device.Buffer.copy_from` between two host-memory devices holds both
  devices' owners for the whole copy. A copy that one of them started from its
  synchronize while the first was running, as a replaying caller does, raised
  `Invalid_argument "device operation: owner was not prepared"`.

- Fix building Tolk's CUDA ABI tests on Windows by declaring the timing and
  system functions used by the runtime's C stubs.

- An NV queue submission whose command stream lies beyond the GPFIFO entry's
  40-bit address field fails with `HCQ command stream lies outside a GPFIFO
  entry's 40-bit address`, where its address spilled into the entry's length.

- CPU kernels hold bfloat16 as its bits (`unsigned short`) instead of C's
  `__bf16`, so a bfloat16 value keeps every bit on every host: on x86-64 a
  gated bfloat16 load failed to link (`__truncsfbf2`), or with AVX512-BF16
  quieted signalling NaNs and flushed subnormals. `Cstyle.clang` loses
  `?native_bf16`, `Cstyle.clang_no_abi` its unused architecture, and
  `Compiler_cpu.supports_bf16` is removed: no compiler support is needed.

- Fix host calls to C runtime functions such as `memcpy` on Windows: the CPU
  device looked them up in the executable alone, where POSIX searches every
  loaded library, and failed with `link_symbol: undefined symbol memcpy`.

- A CPU kernel that calls a symbol the process lacks fails naming it,
  `link_symbol: undefined symbol NAME`, where it said `link_symbol failed`.

- Failed NVIDIA GSP initialization now retires mapped context and channel
  allocations after a successful device stop. Failed stops retain their
  virtual addresses, page tables and backing storage.

- Fix concurrent raw-PCI AMD/NV allocations across devices corrupting their
  shared virtual-address allocator or failing during its first initialization.

- CPU and Metal scheduling no longer split kernels at 31 buffers. Their packed
  argument interfaces support wider inputs, avoiding unnecessary intermediate
  kernels in large gradient sums; explicit buffer limits still apply.

- Raw-PCI AMD virtual functions now use mailbox access leases, gated registers,
  and privileged translation-invalidation queues. PF-only boot operations and
  forced recovery are excluded; uncertain setup storage remains owned.
- Low-level AMD `Amdev.make` now requires byte-register callbacks, and `Iface`
  carries `can_recover`. `Pci_iface.register` receives the SDMA constructor and
  device count to finish VF queue setup before returning initialization access.

- Long-running graph construction reclaims dead hash-consing bucket capacity
  instead of repeatedly growing sparse tables, reducing retained memory while
  preserving the identity of live `Uop` nodes.

- `Uop.get_idx` and `Uop.get_valid` handle invalid lanes and generic values
  consistently. The duplicate `Indexing.get_idx` and `Indexing.get_valid`
  functions are removed; use the canonical `Uop` functions.

- PCI page-table teardown retains an empty child table if clearing its
  parent entry fails, preventing physical memory reuse while the device
  may still reach that table.

- Symbolic shapes survive staged-storage cleanup and multi-device view
  movement. Empty reductions retain their identities, and reused padded
  materializations preserve the source extent and offset.

- Concurrent HCQ clock calibration keeps each timestamp submission, host
  timing and readback together, preventing overwritten samples and waits
  blocked by another calibration.

- Scheduled calls preserve the extent of small views over large symbolic
  storage. Byte-size comparisons no longer overflow and expand a view to
  its entire backing buffer.

- Mixed weak and concrete operands in `Uop.usum`, `Uop.uprod` and floating
  power decomposition are promoted before arithmetic, preventing invalid
  integer operations in generated floating-point kernels.

- CUDA cross-device copies use shared host staging for device-only storage.
  Pinned imports retain their original allocation context across device
  replacement, avoiding name-based peer-context lookup.

- Concurrent native allocation, submission and teardown share device ownership,
  retaining original owners across device replacement. Pending waits and
  replay addresses are validated under the same ownership scope.

- **Breaking:** low-level buffer `get`, `addr` and `synchronize` select a
  retained allocator with `?target` instead of resolving a `?device` name.

- Hand-built partial `PROGRAM` graphs complete only their missing compilation
  stages, preserving supplied instructions, source and ABI metadata. Complete
  programs can be reused without invoking a compiler.

- Generated kernel declarations preserve explicit parameter names. Rendering
  and ABI validation now share the same naming rule and reject collisions
  between buffer or scalar parameters before native dispatch.

- Kernel splitting preserves independent store and value loops, including
  symbolic loops with equal maximum sizes. They no longer collapse into one
  loop and change which output elements are written.

- Independent kernel lowering and beam candidates share bounded compilation
  workers through `PARALLEL`. Workers inherit scoped settings and finish
  before errors propagate; device timing stays in the caller.

- Symbolic partial reshapes preserve their actual dimensions when mapping
  indices. Equal upper bounds no longer make distinct trailing dimensions
  interchangeable, avoiding incorrect coordinates for dynamic shapes.

- Fix image load/store lowering to carry two scalar coordinates throughout
  the pipeline, preserving four-lane pixel values and correctly rendering
  constant coordinates in OpenCL kernels.

- Preserve weak exponent constants and promote mixed integer/float operands
  when lowering `exp2` and `log2`, keeping exponent arithmetic consistent with
  ordinary graph operations.

- Apply `CCACHE` when creating a compiler and `CACHELEVEL` at each disk-cache
  access, including scoped worker contexts. `ASSERT_COMPILE` now rejects cache
  misses before invoking a compiler while allowing cached binaries.

- Keep shard shape and placement metadata through repeated unbuffered
  `zeros_like` and `ones_like` calls. Device-less partitions now materialize
  correctly on one device; `Uop.Multi` retains optional device names per lane.

- Preserve mixed-width symbolic offsets and GPU launch dimensions by promoting
  operands before combining them; apply the same promotion to image coordinates,
  WMMA accumulation and optimizer predicates.

- Buffer finalizers remain deferred until every overlapping operation on a
  domain has returned. One system thread could previously trigger teardown
  while another still had a device update in progress.

- Conditional GPU loops now finish shared-memory reads before the next iteration
  writes the same storage. Their backedge condition is preserved when inserting
  the required workgroup barrier.

- Range optimization preserves loop iteration counts after successive merges.
  Splits and gated shrinking also update enclosing loop binders when the body
  contains a nested sink, preventing duplicated or unclosed ranges.

- Integer decomposition now promotes introduced operands consistently, including
  signed division corrections and shift counts. Exact comparison thresholds
  stay weak until proved, so an out-of-range threshold cannot become zero.

- GPU launch lowering rejects symbolic dimensions that require physical
  splitting instead of launching their maximum size without a bounds gate.
  Dimension grouping and splitting also keep intermediate products exact.

- Failed KFD and NVK peer imports remove partially installed receiver mappings
  before releasing their source. Failed cleanup retains the source because
  the receiving GPU may still access it.

- Symbolic and range optimization now use tinygrad's operand promotion when
  constructing arithmetic. Constant-base powers with integer exponents no
  longer truncate the logarithm used in their decomposition.

- `CHECK_OOB` rejects affine index proofs that rely on unsigned multiplication
  or shifts not wrapping. Widening an already wrapped value could previously
  make an unsafe access appear valid.

- Driver-less NV startup retires boot allocations after initialization fails,
  stopping the device first if firmware has started. Failed resets now report
  an error and retain memory that the device may still access.

- Driver-less AMD startup preserves resident boot memory when software
  preparation fails. Root-page and PSP-fence clearing now starts only after
  the session is marked dirty.


- A Metal kernel on a unified-memory device reads and writes a buffer of the
  CPU device in place, through a no-copy Metal buffer over the pages that hold
  it: Metal's allocator maps host storage, where binding one raised.
  `Device.shares_host_memory` says whether a device's memory is the host's.

- `Device.Buffer.borrow` wraps host memory as a buffer of the host device that
  keeps the memory's owner reachable until it and every mapping of it are
  released, and `Device.Buffer.ownership` tells such borrowed storage from
  owned storage. Only owned storage counts in `mem_used`.

- Keep each compiled queue submission's address table, signal storage and
  Metal command arguments separate. Equal-sized batches could overwrite one
  another's linked addresses or arguments, corrupting replayed computations.

- Preserve buffer and scalar arguments used only by a conditional store's
  gate. Indexed writes beside unit axes could fail compilation after their
  bounds checks moved into control flow.

- Frontend `Jit.create` automatically realizes the tensors enumerated by
  `outputs` during warmup and capture, so lazy return values compute correctly
  on replay without explicit `Run.realize` calls.

- Cross-device `Op.assign` preserves contiguous destination slices and their
  aliases through bulk transfer lowering; noncontiguous destinations are rejected.

- Explicit `STAGE` graphs use their canonical storage capacity, preserving
  vector extents and avoiding duplicate materialization. Zero-coordinate stages
  reuse existing storage even when they carry stage options.

- `Jit.call` rejects changed input layouts and reads symbolic bindings carried
  by input views. Equivalent composed reshapes and slices share a capture;
  conflicting input and explicit bindings fail before execution.

- `Jit.call` refreshes symbolic output views with each replay binding instead
  of retaining capture-time extents. `Jit.create ~outputs` explicitly identifies
  tensors inside structured return values; inner fixed bindings remain fixed.

- Image stores convert non-stacked half vectors to float lane by lane,
  preserving valid image-write types through late lowering.

- Beam search follows the device's cache-invalidation capability, avoiding
  unnecessary eviction kernels during CPU timing. `Device.invalidate_caches`
  now returns the optional operation without invoking it.

- Concurrent scheduling now protects its shared cache, and independent buffer
  views retain accurate ownership counts through allocation and retirement.

- `Uop.shape` reports reserved maximum extents for symbolic `STAGE` nodes.
  Lowering preserves active dimensions such as `[n; 3]` instead of shrinking
  fixed axes to one when another axis is symbolic.

- `CHECK_OOB=1` now verifies leading-axis long cumulative sums and padded
  Metal tensor-core loads, including BF16 accumulation. Distributed indexes
  previously hid the bounded expressions named by their masks.

- Independent callers now receive distinct buffer identities and retain every
  live tensor handle during concurrent construction. Lost identities or registry
  entries could alias allocations or prevent graph rebinding.

- Render dynamic vector lane indexes as element access instead of vector
  addition. C renderers now require canonical flat memory indexes, rejecting
  malformed multidimensional indexes instead of guessing strides.

- Keep beam-search trial launches within the size budget when symbolic
  dimensions multiply beyond host integer limits; timing scales now preserve
  the full launch size.

- Preserve execution and allocation counts across concurrent callers.
  `Global_counters.snapshot ()`, `Storage.mem_used ?device ()` and
  `Realize.queue_submissions ()` replace externally mutable counters.

- Reject replay of a linked queue after one of its devices has been replaced,
  before its old native addresses reach the replacement runtime. `Uop.export`
  now rejects linked queues; serialize their unlinked templates instead.

- Retire cached queue templates and linked storage when any participating
  device is replaced. Concurrent first links now share one published result
  and one lazily initialized Metal/CUDA timeline.

- Reject beam-search candidates whose upcast or local lane products exceed
  their limits even when the product is larger than a host integer.

- Keep internal buffers distinct during concurrent scheduling and JIT cache
  imports; racing slot reservations could merge unrelated buffer identities.

- Prevent overlapping linked queue replays from overwriting address tables
  or reserving the same timeline value. Failed queue preparation leaves the
  retained address table unchanged.

- Keep optimizer upcast products exact and symbolic instead of overflowing
  host integers or treating unknown extents as one. `Postrange.upcast_size`
  now returns a `Uop.t`; heuristics require proven resource bounds.

- Prove gated gather, scatter, selection and scan accesses under `CHECK_OOB`
  using their component bounds. Reject unsafe narrowing casts and keep proof
  variables distinct from user parameters.

- Optimizer padding and shared-memory limits use exact arithmetic. Large
  extents no longer wrap negative, and symbolic local sizes must prove that
  their full storage requirement fits the device budget.

- Compiled program and runtime caches synchronize concurrent misses and retire
  with replaced device owners. Live CPU executables remain usable across GC;
  unreachable executable mappings now have automatic cleanup.

- Concurrent `Device.get` calls open each device once and wait for its
  initialization. Failed openers retry without publishing partial devices;
  recursive bootstrap can still resolve its own provisional device.

- Replace the global `Realize.capturing` registry with `with_capture` and
  `current_capture`. Concurrent callers keep separate capture scopes, and
  nested Rune compilation restores outer capture on return or exception.

- `TC_MIN_GLOBALS` preserves global work when tensor-core heuristics choose
  upcasts. Scoped changes to this policy select distinct compiled programs.

- Ordinary tensor slices retain their base storage in compiled call signatures;
  only explicitly materialized contiguous views become narrowed call inputs.

- `Creation.empty ~device` no longer opens the default backend while building
  storage for an explicitly selected device.

- Generated kernels avoid redundant 64-bit index arithmetic when bounds prove
  32-bit operations safe, and preserve shared floating-point expressions when
  distributing negation.

- Late scalar variables, including sharded kernels' device index, receive
  parameter slots after existing buffers so generated declarations and runtime
  argument packing agree.

- Kernel costs remain exact beyond host-integer limits, and symbolic memory
  traffic is capped by its buffer footprint. `Global_counters.global_ops` and
  `global_mem` now use `Tolk_uop.Bigint.t`; beam ranking converts costs only
  at comparison.

- Remove the obsolete AMD/NV `Compute_queue`, `Copy_queue` and raw program
  APIs. Queue execution uses `Encoded_queue`; static setup and cache packets
  remain private to the runtimes.

- Committed stack indexes remain explicit until lowering, preserving vector
  lane rendering. Bare constant stack indexes follow tuple indexing, and
  diagnostic expressions print committed constants by value.

- RDNA3 int8 tensor-core wrappers and call sites use tinygrad’s sanitized
  `signed_char` type name consistently.

- AMD compiled SDMA queues support typed writes with patched 32/64-bit replay
  values. Copy chunk limits now follow the full SDMA revision, including the
  1 GiB limit introduced at 4.4.2.

- Symbolic kernel estimates count repeated loop contributions correctly and
  simplify final FLOP expressions. Equal symbolic terms previously lost an
  addend, understating beam-search costs.

- Compiled signatures reject conflicting scalar declarations before dispatch,
  preventing rendered arguments and native packing from disagreeing. Distinct
  explicit scalar slots can still share a symbolic binding name.

- Large stacked lookup tables lower to balanced selections, bounding
  conditional depth while preserving the final-element fallback for invalid
  indices, as in tinygrad.

- Anonymous allocations preserve scalar and symbolic shapes across calls and
  kernel lowering. Local and register storage stays internal to the kernel
  instead of being counted as external runtime arguments.

- Dynamic scalar bindings in execution, JIT replay and beam search now carry
  the complete signed 64-bit range. Launch dimensions remain checked host
  integers, and scalar metadata preserves exact bounds.

- Compilation policy overrides are isolated across domains and systhreads;
  beam workers inherit an immutable snapshot, so concurrent or nested searches
  cannot change one another’s settings.

- AMD PCI boot marks the session unfinished before programming hardware and
  restores that marker after a reset, so a failed initialization cannot leave
  the previous session’s clean-shutdown stamp in place.

- `State.load_state_dict` transfers placed checkpoint values to a parameter’s
  device and shards single-device values along a multi-device parameter’s axis;
  loading weights previously discarded that placement.

- `Uop.placeholder` and `Tensor.numel` reject element counts that exceed the
  host integer range instead of wrapping; zero-sized shapes remain empty even
  when preceding dimensions have a large product.

- Constant folding preserves tensor shapes, including symbolic dimensions and
  conditions reused through views, so JIT `top_k` and comparison/permute graphs
  remain valid during shared view analysis.

- Beam compilation propagates interruption immediately and joins started workers
  before returning; cancellation was previously treated as a rejected candidate.

- Preserve raw compact-float bits through bitcasts, copies and value selection,
  including FP8 subnormals and NaN payloads; numeric arithmetic keeps its normal
  conversion and masked fallback semantics.

- Beam search now uses shared `PROGRAM` compilation, preserving the compiler
  callback's target, signature and profiling identity when timing candidates.

- Remove `Tiny_elf.pack`; command submission uses `Tiny_elf.layout` and shared
  typed queue patches, leaving one argument encoding path.

- Reject hand-built scalar parameters missing a name or bounds before building
  the runtime ABI; they previously disappeared from the signature while native
  code could still read their missing argument.

- Tensor-core selection now tries smaller valid tiles after a symbolic split
  rejects an earlier candidate, restoring the optimizer before retrying.

- Fix Metal compilation of four-lane `int8`, `uint8`, `uint16` and `uint64`
  accesses by using native vector type names.

- Preserve closed reduction dependencies when finalizing optimizer ranges;
  nested range extents no longer reintroduce an already closed axis.

- `Prepare.contiguous_view` replaces `Uop.contiguous_view` and proves aliases
  using shared indexing rewrites, including cancelling transposes and symbolic
  leading slices while preserving byte offsets and pending effects.

- Remove ignored `dtype` arguments from `Uop.load`, `reduce`, `wmma` and
  `noop`. Their types come from the source, accumulator or void-marker contract;
  callers should use an explicit cast when a conversion is required.

- PCI peer imports reject source addresses outside the receiving GPU’s virtual
  address range before editing page tables, allowing shared execution to stage
  unsupported transfers.

- AMD `Program.image` zero-pads relocated binaries to a four-byte boundary,
  matching the reference contract for program uploads.

- Late signed comparison rewrites keep proof arithmetic exact at integer
  boundaries, preventing an empty interval from wrapping into equality.

- Range analysis reuses ended-range results for shared `AFTER` and `BARRIER`
  dependencies, avoiding exponential traversal and allocation during compilation.

- `Creation.full_like`, `zeros_like` and `ones_like` preserve device placement
  and allocate local shard sizes, including symbolic nonsharded dimensions.
  Fill and random constructors now use the same placement rules.

- `Uop.semantic_key` ignores auxiliary fallback graphs consistently, so calls
  that differ only in fallback metadata share a semantic cache key.

- Keep compiled programs and runtime handles with their `Device.t` owner.
  Replacing a device with the same name no longer reuses incompatible renderer
  assumptions, loaders or linked queue templates.

- Remove unused `Buffer.uop_refcount`, `Buffer.add_ref` and
  `Multi_buffer.add_ref`; live views and owning UOps retain storage directly.

- Failed LRU cache cleanup retains uncertain backing and restores untouched
  entries, preventing a single retirement failure from losing ownership of
  every remaining cached buffer.

- `Rand.rand_like`, `randn_like` and dropout preserve source placement and
  sharding. RNG keys and counters now belong to their requested device;
  sharded draws advance independent streams while replicated draws share one.

- `Creation.clone` preserves sharding and allocates each shard separately.
  `Op.assign` accepts cross-device values through copy preparation and rejects
  incompatible shard axes; clones reject DISK destinations before allocation.

- Symbolic launch dimensions use exact arithmetic, preserving large intermediate
  products, signed shifts and scalar casts. Both launch APIs now agree with
  `Uop.sym_infer` instead of overflowing host integers independently.

- `Op.arange` preserves values and lengths at OCaml integer boundaries.
  Exact endpoint and offset arithmetic prevents sign changes and empty ranges
  caused by overflowing intermediate calculations.

- Failed AMD/NV PCI discovery releases constructor-owned BAR mappings.
  Address-space reservation now precedes PCI claiming, so a reservation failure
  cannot strand a device claim; uncertain hardware state keeps its claim held.

- AMD PCI boot uses updated firmware and register tables, including SMU
  13.0.15 protocols. Missing or outdated firmware is downloaded from the
  pinned source with `curl`, SHA-verified and cached without replacing local files.

- NV submissions share immutable program images and allocate descriptors and
  arguments together. Retained replays keep separate writable storage while
  avoiding repeated image allocation and relocation setup.

- AMD reset and firmware settling delays now sleep instead of consuming a CPU
  core in busy loops. Scripted hardware tests use the device's injected clock.

- KFD and NVK retain host backing when driver-memory retirement fails during
  cleanup, preventing failed allocation rollback from unmapping live storage.

- Buffers remain allocated when a failed import rollback may leave device
  mappings live. Explicit deallocation reports the original failure instead
  of freeing memory that the receiving device can still address.

- Buffer byte transfers now use synchronized host access or bounded owned
  staging through shared submissions. Remove allocator copy callbacks and
  unused eager AMD/NV staging pools, saving at least 64 MiB per device.

- Beam search waits for every started compilation worker before reporting a
  worker or spawn failure, preventing work from escaping its compilation context.

- Changing `TC_SELECT` or `TC_OPT` now selects a program compiled for that
  tensor-core policy instead of reusing one cached under a previous setting.

- Autotuning clears caches through a shared fill workload when no backend
  hook exists. Explicit `beam:0` now disables default search even when `BEAM`
  is set; code generation uses the kernel's resolved width.

- Mixed kernel/copy replays preserve snapshot-copy semantics for overlapping
  views. Kernel mappings are checked before execution, so an unsupported later
  mapping cannot leave earlier outputs partially changed.

- NV channel initialization and local-memory growth share kernel submission
  ordering and retirement. Failed local-memory setup preserves the previous
  capacity and retains storage whose completion is uncertain.

- Reuse Metal pipelines and CUDA functions across independently linked
  submissions, avoiding repeated native program creation. Retiring one replay
  keeps programs used by other replays alive until device shutdown.

- `Device.make` accepts an initialization callback for bootstrap submissions
  that need to resolve their own device. Failed initialization restores the
  previous registration so later opens can retry.

- AMD and NV kernels use compiled queue submissions exclusively, removing an
  unused 16 MiB argument allocation per device and the obsolete
  `Hcq.Kernargs` and backend direct-program APIs.

- Metal and CUDA kernels execute exclusively through compiled queues.
  Multi-device calls share this path, and unsupported mappings fail before
  fallback copies execute.
- Buffer views refresh after their base storage is reallocated. Remove the
  unused allocator disk-copy hook and redundant capability flags.

- Preserve exact dimensions in padding, repeats, pooling and bitcasts. Large
  lazy shapes no longer wrap to empty tensors, and invalid reshapes and slices
  are rejected before allocation.

- `Buffer.copy_from` now uses scheduled `STORE` submissions, sharing queue
  dependencies and transfers with normal execution. Provably overlapping
  copies preserve their contents through the existing bounded host fallback.

- Remove the obsolete `Realize.Compiled_runner` API. Compiled kernels and
  optimizer candidates execute through shared `CALL` submissions and
  `Realize.time_call` instead of a separate direct-dispatch adapter.

- AMD mode1 reset now waits for PCI configuration readiness before resuming
  MMIO, avoiding access to a device that has not returned from reset.
  Non-hive devices report a bounded readiness timeout.

- Execution and JIT replay now use owned `BUFFER` nodes and explicit `PARAM`
  arguments. Remove `Realize.Buffers` and the `Jit.call` buffer resolver; Rune
  inputs, arenas and nested loops use per-invocation parameter scopes.

- C-style rendering uses IR shapes for vector types and memory accesses,
  fixing size-changing bitcasts and vector casts. Rendered graphs can now be
  collected instead of remaining in a permanent expression-width cache.

- Metal queue completion now collects command buffers in submission order.
  Later pending profiling work cannot hide an earlier completion, and host
  waits observe command failures and timestamps before publishing progress.

- Beam search now times the same compiled queue submissions used for GPU
  execution, including device timestamps and supported timeout budgets.
  Temporary timing programs are released only after successful completion.

- Fix reading an unallocated buffer view whose base is already allocated.
  `Buffer.is_allocated` now describes the view itself, and tensor reads ensure
  its storage exists before accessing it.

- Use the correct SMU 13.0.10 command family and exclude auxiliary PSP v2.1
  images during AMD PCI firmware loading, preventing wrong power-management
  commands and firmware selection.

- Allocate AMD/NV compiled command streams uncached, so device command fetches
  observe patched commands during replay.

- Load NVRTC-produced cubins on the NV backend. The ELF loader previously
  rejected their executable format before reading the kernel.

- `Hcq.Timeline.submit` replaces `next_timeline`: AMD/NV direct submissions
  advance their completion counter only after publication succeeds. Failed
  submissions retain storage and timing slots instead of leaving phantom waits.

- AMD compiled PM4 dispatches give each compute die a disjoint scratch region,
  preventing simultaneous dies from overwriting each other's private storage.

- Failed PCI device claims release their lock and descriptors, allowing a
  retry after fixing permissions or device configuration.

- Failed AMD/NV PCI runtime setup stops queues before releasing their buffers
  and mappings. Partially programmed AMD copy queues are included in teardown;
  failed retirement retains storage and disables later shutdown callbacks.

- AMD PCI treats a fabric segment as a multi-die hive only when the hardware
  advertises peer regions, preserving single-device boot waits and avoiding
  fabric firmware setup on standalone devices.

- Metal compiled queues report GPU command failures when waiting or preparing
  another submission. Host reads wait for checked command completion, avoiding
  partial results from a failed command whose timeline event was signaled.

- Buffer finalizers defer teardown until device operations finish, preventing
  GC from waiting on unsubmitted work or re-entering native allocators. Failed
  teardown retains its storage instead of retrying a partially completed free.

- `Uop.axis` of an operation whose operand of lower rank is split counts the
  operand's axis from where it broadcasts, as the multi rewrite does. It kept
  the operand's own index, so a matrix product over a column-split weight
  reported its split on the wrong axis, and a custom kernel over such a value
  failed to build ("reshape moved items between shards").

- `Op.scatter_indexed`, `Op.block_matmul` and `Op.quant_matmul` take operands
  split across devices: each device runs the kernel over its own slices and
  writes its slice of the result, or its partial product when the matrices or
  inputs are split, which the result sums. Built from the whole value's
  extents, they failed to build over split operands (`block_matmul`,
  `quant_matmul`) or wrote out of their slice (`scatter_indexed`).

- **Breaking:** `Op.quant_row_bound` takes only the renderer: the bound is one
  measured value per device, at every dtype and shape.

- `Op.quant_matmul` with ids on the CPU reads matrix 0 for an id outside the
  matrices, and zeroes its result, instead of gating every load on the id,
  which made clang spill the kernel: 1.7 to 3.2 times faster at bfloat16 and
  1.2 to 1.3 times at float32 from two rows per position (gpt-oss's expert,
  5760 by 2880). One row per position at float32 is 10% slower.

- With `ALLREDUCE_NODE_NDEVS` set to a box size h, an all-gather or
  reduce-scatter of concrete shape over its own n devices crosses boxes only
  between devices at the same position in their boxes: each device moves
  (b-1)/n of the value between b boxes, where it moved (n-h)/n, at the cost
  of (b-1)/n of the partial in a reduce-scatter's peak. The reduce-scatter
  folds within each box first, so its blocks equal the hierarchical
  allreduce's rows bit for bit.

- A product of narrow floats widened to `float32` before the multiply takes a
  narrow-in, `float32`-out tensor core, which computes the same product. It
  took no narrow tensor core, and on CUDA none at all without `ALLOW_TF32`.

- `Uop.call_info.name` is a `Uop.call_name`: `Label` for a callable's name,
  or `Collective` for the collective a precompiled call implements
  (`Allreduce op`, `Allgather axes`, `Reducescatter (op, axis)`), which no
  longer passes as a string.

- Resharding an allreduced tensor to split rows (`Creation.shard ~axis` of a
  sum over a split axis) is a reduce-scatter when nothing else uses the sum:
  each device sends (n-1)/n of its partial and keeps only its rows, bit-equal
  to the naive allreduce's. It allreduced the whole value first. A fully
  sharded training step now stays within its share of state plus two layers
  and its activations.

- An allreduced tensor that a realization returns is written straight into its
  storage. It was reduced into an intermediate buffer and then copied, which
  held one more copy of the tensor on each device.

- A `bfloat16` constant rounds once from the double, so compiled constants
  equal nx's eager values.

- A `float16` constant between `2^-25` and `2^-24` in a compiled graph folds
  to the smallest subnormal, `2^-24`. It folded to zero, so `3e-8` became
  `0`.

- Realizing a slice of a split tensor whose part on each device is a
  contiguous window of that device's shard gives a view of the shards:
  nothing is allocated or copied. Each part was copied into a new buffer.

- Gathering a slice of a split tensor whose part on each device is contiguous
  reads each part in place. Each device first copied its part into a buffer
  of its own.

- A copy to another device from a contiguous slice of a realized value reads
  the slice in place. It was first copied into a buffer of its own on the
  source device.

- A precompiled call reading a view argument read the wrong elements: a
  non-contiguous view as the flat bytes from its offset, and a window of a
  value computed in the same realization from the start of that value, with
  its writes landing there too. A window of storage now reaches the call in
  place, another view that the call reads is copied in, and storing into
  one raises, naming the call.

- A gathered tensor that a realization returns is written straight into its
  storage. It was written to an intermediate buffer and then copied, which
  held two copies of the tensor on each device.

- Copying a split tensor to a list of devices moves each shard once into its
  place: every device receives (n-1)/n of the tensor. The copy was an
  allreduce of zero-padded shards, which moved 2(n-1)/n per device under ring
  and n-1 whole tensors under the naive strategy, and held a padded copy of
  each shard. Gathering on one device moves the same bytes as before.

- A precompiled call or custom kernel that stores into an argument which is a
  view, not storage, raises `Invalid_argument` naming the call. The view was
  scheduled as a copy of its values, so the writes were silently lost.

- A realized allreduce whose result is a symbolic slice of an inner axis
  returned zeros; it now returns the sum. The reduction wrote a copy of the
  result's view instead of its storage.

- `Op.block_matmul` on the CPU keeps a register tile of rows and columns and
  unrolls the contraction: 9 to 37 times faster at float32 and 8 to 18 times
  at bfloat16 at gpt-oss's shapes, where it ran with no options.

- A compiled program is no longer reused, in the process or from rune's disk
  cache, under other values of `TC_SELECT`, `TC_OPT`, `FLOAT16`, `MV*`,
  `OCCUPANCY_FLOOR`, `DMC`, `ALLOW_HALF8`, `EXPAND_SSA`, `ALIGNED` or the CPU
  compiler `CC`, which change the kernels emitted.

- `Creation.shard ~axis` accepts a tensor already replicated on the target
  devices and splits it without copying: each device keeps its own shard of its
  replica. Other multi-device sources still raise.

- `Run.realize` and `Run.realize_many` accept sharded and replicated tensors.
  They ran the schedule, then raised while reading back a buffer that a
  multi-device tensor does not have.

- Replicas of a hierarchical allreduce (`ALLREDUCE_NODE_NDEVS`) are now equal
  bit for bit on every device. With three or more boxes, devices summed the
  boxes' partials in different orders and their replicas differed in the last
  bits.

- Changing a setting that scheduling reads (the allreduce strategy settings,
  `SPLIT_REDUCEOP` and its thresholds, `FLOAT16`, `PCONTIG`,
  `MAX_KERNEL_BUFFERS`, `OPENPILOT_HACKS`) between realizations now takes
  effect for a graph scheduled before. The schedule cache ignored them and
  replayed the first schedule. `Schedule.config` lists them.

- Compiled programs on Metal no longer return wrong values from kernels of 16
  to 29 arguments, such as a concatenation of 16 tensors. An M1 Max
  miscomputes them from an indirect command buffer, so they now run as direct
  dispatches between queued batches.

- `Op.quant_matmul` and `Op.block_matmul` use the migrated optimizer and storage
  APIs. Quantized kernels count dynamic reduction axes when choosing splits,
  preserving gated products across CPU and GPU renderers.

- Broadcast dimensions in contiguous-view detection no longer mistake repeated
  input rows for adjacent storage. Quantized products with no inputs create
  their zero output directly on the selected device.

- AMD PCI initialization disables PCIe link power saving before reading MMIO,
  avoiding unstable register reads on links with retimers.

- AMD PCI boots reserve the firmware-reported trusted memory region at a stable
  address across sessions. The boot layout now uses the target protocol stamp;
  GC 9.5 keeps resident fabric state after an unclean matching session.

- AMD PCI initialization lowers clocks and halts compute and DMA engines
  before resetting a live device, avoiding resets while engines run at full speed.

- NVIDIA PCI command rings stay in device memory on small-BAR cards, matching
  the physical addresses used when GSP creates their channels.

- NVIDIA PCI compute submissions include the channel runlist in work-submit
  tokens, so GSP-created channels reach the correct hardware queue.

- Failed PCI host-page discovery and BAR mapping release CPU mappings and
  GPU memory reservations. If page-table rollback itself fails, backing
  storage stays reserved so the device cannot access reused memory.

- Failed NVIDIA NVK device construction unregisters channels before releasing
  queue storage, then unwinds UVM registrations, control mappings and device
  objects. Cleanup uses the installed driver's channel-unregister layout.

- Failed AMD KFD device construction retires queues before releasing their
  buffers, events and descriptors. Late scratch-buffer finalizers cannot
  touch abandoned timelines, and failed queue retirement retains its storage.

- Failed AMD KFD and NVIDIA NVK bootstrap releases acquired descriptors and
  per-device KFD events, so retries do not accumulate abandoned resources.
  KFD retains its shared event page if registration may have reached the kernel.

- `Device.profile` calibrates GPU timestamps to the host clock; `Profile.output`
  preserves timing between devices instead of starting each device at zero.
  Operation durations retain their original device-clock measurements.

- AMD PCI XGMI address conversion reads fabric topology from the compute hub,
  preventing device-local memory from being assigned the wrong peer aperture.

- AMD PCI initialization reads harvest information and selects live I/O dies,
  avoiding dead memory-hub waits and indirect doorbell routes that can stall
  the fabric. Harvested compute dies are fenced off from doorbells.

- AMD PCI multi-die setup programs each memory hub and compute die separately,
  including TLB acknowledgements, AQL descriptors, engine reset and clock
  gating. These paths previously repeated writes to instance zero.

- AMD PCI register bindings honor discovered IP instances. SDMA 4.4 initializes
  and tears down each selected engine and copy ring instead of repeatedly
  programming instance zero.

- Successful AMD PCI compute recovery releases retained submission errors while
  still reporting the failed work to its caller. Recovery targets only the
  failed device and remains blocked while SDMA work is outstanding.

- AMD/NV staging waits now retain device fault reports, so subsequent waits
  consistently report the original failure instead of timing out again.

- Interleaved compatible device groups share compiled batches when byte
  dependencies permit reordering. Runtime alias checks also cover writable
  accesses moved across another group.

- Compiled batches wait for earlier foreign accesses that their queue timelines
  do not cover, preventing races when separate device groups share host memory.
  Already ordered device accesses retain asynchronous submission.

- AMD compiled transfers use distinct SDMA rings and distribute all-to-all
  copies according to `ALL2ALL` and `HCQ_NUM_SDMA`, allowing independent
  transfers to overlap.

- AMD without SDMA keeps bulk copies in compiled submissions by using a
  byte-copy kernel, including staged imports. `Device.queue.copy` now selects
  the copy or compute queue for each transfer.

- Unsupported queue copy imports use two 64 MiB staging slots with dependencies
  before reuse, retaining asynchronous replay and rebinding. PCI host signals
  remain importable on small-BAR devices; device and allocation faults propagate.

- Compatible accelerator peers share compiled queue batches. Host-device
  synchronization waits for captured foreign memory accesses, and AMD/NV
  host and peer copies can use the shared queue protocol.

- Compiled queue replay binds cached storage mappings without synchronizing
  their owners. Direct runtime calls and native transfers still wait for foreign
  storage users, preserving safety while queue fences control asynchronous work.

- Sparse compiled calls no longer try to resolve unused argument slots during
  eager preparation, allowing programs with non-contiguous formal slots to run.

- `PROFILE=1` collects asynchronous queue timings at synchronization.
  `Device.profile` drains completed events, and `Profile.output` exports
  Chrome trace JSON with separate compute/copy lanes and per-device clocks.

- NV asynchronous submissions wait for FIFO capacity and command-storage
  retirement, preventing long batches or independent replays from overwriting
  commands still in use. Stalled waits preserve live storage and report failure.

- AMD KFD and NVIDIA NVK unwind failed memory setup without releasing borrowed
  host storage; NVK closes temporary mapping descriptors. PCI allocation retries
  reclaim reservations and partially written page tables after allocation,
  zeroing or mapping failures.

- Large fallback copies use 64 MiB host chunks on allocators with offset views,
  bounding temporary host and native upload memory while preserving overlapping
  view copies.

- CUDA queues accept host and peer copies, checking mappings before submission
  and falling back to ordinary execution when imports are unsupported. Peer
  timeline work now uses each device's own encoder and context.

- `Realize.compile_linear ~profile:true` records per-call queue timestamps;
  waited replay sums them instead of host submission overhead. `DEBUG=2`
  enables this by default, including CUDA callbacks and Metal GPU timings.

- Reject queue replay bindings that introduce writable overlaps between
  unordered calls, including overlapping external pointers. Ordered buffer
  donation, read-only aliases and disjoint views remain valid.

- Compile scheduled kernels for their argument devices, fixing CPU-sharded
  custom kernels when Metal is the default. `Realize` and `Jit` compiler
  callbacks now receive the execution device.

- Eager execution reuses compiled queue templates across fresh inputs while
  preserving shared-base views. Small schedules reuse linked command storage;
  `HCQ_CACHE_THRESH` keeps larger schedules' inputs bound at link time.
- CUDA initializes its driver once across domains and reports missing symbols
  consistently. Device creation unwinds failures, and shutdown releases streams,
  events and contexts even after a synchronization error.
- Compiled host submissions preserve address-load ordering in verified IR,
  allowing Metal queue replay with `SPEC=1` without relaxing validation.
- Gated reductions and indexed loads collapse again with typed constants,
  avoiding unnecessary loops. Divisibility folding now handles symbolic
  variables before and after conversion to kernel parameters.
- Fix a compilation rewrite cycle in BF16 arithmetic, including half-width
  random draws and Rune gradients, by preserving literal widths during late
  float emulation.
- IR verification rejects invalid COPY operands and non-storage STORE targets,
  while accepting opaque calls and cast-sized buffer slices at the proper
  stages. Integer range splitting preserves explicitly typed widths.
- `Uop.wmma_info` no longer carries a device name; the selected renderer
  determines the target, avoiding conflicting tensor-core metadata.

- Launch dimensions and program signatures recognize symbolic `BUFFER`
  variables as scalar arguments, fixing missing values and unsupported-op
  errors after the variable representation change.

- `Creation.shard` partitions a tensor along an axis or replicates it across
  devices. Sharded results can be gathered with `Creation.clone` on one device.

- Allreduce preserves symbolic output extents and supports hierarchical
  reduction through `ALLREDUCE_NODE_NDEVS`. Cross-device symbolic copies
  transfer padded storage instead of repeating values to fill the allocation.

- Sharded parameters now allocate only their local storage. `Uop.unshard`
  carries sorted axes and explicit ranges, preserving symbolic sizes and
  supporting per-thread fragment indexing and multi-axis device gathering.

- AMD uses AQL queues by default on multi-XCC devices, matching tinygrad. Direct
  launches and compiled submissions share HSA dispatch packets, queue-owned
  scratch configuration and the mapped producer counter.
- AMD PCI queue initialization omits the kernel-driver control stack; passing
  that unsupported parameter previously caused queue creation to fail.

- NV compute and DMA queues use compiled host submissions with retained kernel
  images, typed arguments and chained Ada/Blackwell launch descriptors. Timed-out
  replays preserve in-flight storage; failed local-memory growth preserves the old store.

- Timed-out AMD queue replays preserve the timeline, command buffers and
  arguments still owned by in-flight work, and reject further submissions.

- Compiled queue linking applies casts and bitcasts around constant expressions,
  so split device addresses and packed descriptor fields are initialized before
  submission instead of remaining unapplied patches.

- NV initializes the instruction-prefetch guard after each loaded kernel and
  rejects invalid relocations before allocating device memory.

- AMD direct launches and retained queues share owned scratch storage with
  tinygrad's minimum sizing. Resizing preserves backing captured by older
  replays; allocation failure rejects larger launches without discarding it.

- Direct AMD launches support kernels requiring an HSA dispatch-packet pointer.
  AMD waits and AMD/NV DMA completion packets preserve timeline epochs by
  encoding only the low word in hardware dword fields.

- AMD uses compiled host submission for PM4 kernels, SDMA copies and retained
  replay. Polling and ring backpressure are bounded; a stalled submission
  preserves unread commands and reports its failure at synchronization.

- AMD and NV submissions share mapped queue positions and timeline counters.
  Timeline rollover retains linked signal addresses and replay fence values;
  NV completion signals no longer overwrite the submitted counter with timestamps.

- Compiled queue submissions preserve argument-buffer patches when linking
  their addresses and rewrite native callee dependencies consistently, avoiding
  stale replay arguments and spurious control-flow cycles.

- AMD and NV bind host and peer buffers through shared storage mappings,
  preserving view offsets and freeing imports before their source allocation.
  NV PCI allocations retain their actual owner and backing pages.
- CPU storage uses zeroed pages suitable for GPU registration. Host access
  waits for devices that mapped the buffer, so pending GPU writes are visible.

- PCI allocations release system-memory virtual ranges and the actual CPU
  mapping, preventing leaks when BAR addresses differ from GPU addresses.
  Large-BAR GPUs keep uncached CPU-visible allocations in device memory.

- Remove `Device.Graph` and `Jit.batch_graphs`; queue submission is compiled
  by `Realize.compile_linear` for both eager execution and JIT replay.
  `Realize.queue_submissions` replaces the old graph launch counter.

- Replace CUDA graph replay with host-compiled compute and copy queues.
  Linked kernels and argument storage survive replay; event handoffs order
  ordinary dispatches and uploads with queued work.

- Preserve queue dependencies between differently shaped views of the same
  input parameter slot, so a copy completes before its consumer kernel.

- CUDA buffers expose pinned host storage and mapped views to the shared
  linker. Cross-device copies enable peer access when supported and otherwise
  use the executor’s host fallback.

- Fix BF16 scatter compilation looping during float emulation. Committed
  literals retain their width, and weak literals are rounded to their emulated
  peer's dtype before arithmetic is promoted to float32.

- Compile Metal batches into host submission programs shared by eager execution
  and JIT replay. Command storage is linked once and fenced before runtime
  address, scalar and launch-size updates; the old Metal graph API is removed.

- Track graph dependencies over byte views instead of their entire backing
  allocations, allowing disjoint views to remain independent while preserving
  ordering for overlapping reads and writes.

- Separate `Realize.compile_linear` (formerly `pm_compile`) from
  `Realize.link_linear`. Linking initializes command storage once and retains
  allocations referenced by native addresses across replay.
- `Uop.getaddr` accepts a target device. Generated host calls now lower
  function-pointer loads correctly; `Tolk_cpu.link_symbol` resolves native
  symbols for linked host programs.

- Execute bulk transfers as `CALL(STORE)` and apply disk views after copying
  to the destination, avoiding temporary disk allocations. `Uop.copy` now
  rejects weak dtypes and requires explicit stores for disk destinations.

- `Storage.get ~device` maps an allocation into another device and shares
  that mapping across byte views. Mappings synchronize and unmap before
  their source storage is freed.
- CPU uses `Storage.Host_allocator` and can run directly over CPU-accessible
  device storage, including Metal buffers and offset views, without copying.

- Runtime programs and graph replay now take `Device.Buffer.t` arguments.
  Allocator type identities protect dispatch and transfers; Metal keeps its
  native handles and byte offsets directly, without a token registry.
- Transfer hooks receive both device identities and can decline unsupported
  pairs for host fallback. Replay detects reallocated buffers even when the
  allocator reuses an address.

- `Uop.max_shard_numel` computes per-device allocation sizes with exact
  multiplication. View and replay sizing now use the same checked shape
  properties instead of multiplying host integers independently.

- Reject cyclic buffer dependencies during scheduling instead of silently
  dropping mutually dependent kernels or returning a partial schedule.

- Remove `Uop.replace ~dtype`: result types follow source edges or typed
  metadata. Cast and storage rewrites must update their payload explicitly.

- Execution now requires graph-owned storage or an explicit buffer binding;
  it no longer allocates unplaced placeholders on a default device.
  `Realize.Buffers.create ()` therefore takes no device argument.

- `Uop.contiguous` now creates bare `STAGE` materializations, replacing
  `CONTIGUOUS`. Preparation lowers them to call-local stores and shares
  assignment-hazard boundaries with copies and existing store effects.

- Storage views now use `SHRINK`/`BITCAST` instead of `SLICE` and executable
  view calls. `Uop.contiguous_view` reports one byte offset for scheduling,
  replay and frontend aliases, including symbolic leading dimensions.

- `Uop.call_with_outputs` passes tensor results through explicit storage
  arguments, replacing `FUNCTION`/`TUPLE`/`GETTUPLE`. Inline and precompiled
  calls share allocation ownership and symbolic output shapes.
- `Bufferize.run` separates persistent tensor storage and post-realization
  identities from `Callify.transform_to_call`, which now returns only a call.

- Tensor preparation helpers move from `Rangeify` to `Prepare`:
  `movement_ops` and `detect_expanded` share the preparation pass used by scheduling.
- `Uop.alloc` declares temporary storage. Cached schedules bind it separately
  for each invocation while preserving buffers that already own storage.
- Fix sharded reductions to a scalar emitting an unindexed output pointer;
  scalar storage parameters now acquire a flat size-one kernel view.
- `Uop.variable` now creates a scalar storage identity; `Uop.bind` sequences
  its value through a store and rejects violations of its declared divisor.
  Nested calls resolve scalar parameters within their own scope.
- `Uop.const` represents concrete numeric literals as typed casts over weak
  values. Weak arithmetic commits widths at its consumers before code emission.
- Preserve half-precision rounding in `arange` and ordinary arithmetic when
  promoting back to float32; image-store conversion no longer removes it.

- `Uop` rewrites derive result dtypes from their new inputs. Integer `FDIV`
  produces a float, and `Uop.stack` promotes mixed element dtypes.
- `Uop.param` and `Uop.buffer` describe flat maximum storage sizes, with
  symbolic and multidimensional shapes represented as views.

- `Uop.from_buffer` retains storage directly in the graph, including external
  buffers and views. Realization no longer needs a frontend storage registry.
- `Uop.export` preserves buffer contents and shared views across import while
  leaving unallocated buffers lazy; native pointers are never serialized.

- Reject negative or overflowing buffer sizes and view ranges before native
  memory access. Large offsets can no longer bypass MMIO bounds checks.

- Honor declared workgroup sizes, including symbolic extents during replay,
  without extra trial executions on first launch.
- `Uop.program_info` stores launch expressions for global and local dimensions.
  It no longer duplicates the kernel name; `Uop.program_function_name` now
  reads that name from the program itself.

- Keep a single cast around conditional expressions during symbolic
  simplification instead of duplicating it across both branches.

- Use tinygrad's lexical argument ordering for weak-index simplification
  and kernel instruction scheduling.

- Remove `Uop.kernel_info.axis_types`; kernel axes are determined by their
  range nodes. Older exported graphs and compiled caches are invalidated.

- Use FNUZ FP8 formats on AMD CDNA3 for casts and matrix multiplication,
  including its 240-value E4M3 saturation limit. CDNA4 keeps OCP formats.

- `Tc.create` derives tensor-core tile geometry from operand fragment layouts
  instead of separate scheduling and swizzle tables. Tensor-core padding now
  preserves contraction masks, and contracted output axes are rejected.

- Remove unused `Postrange.shape_str`, `shape_str_to_axis` and `output_shape`
  queries; optimizer axes are identified directly by their ranges and owners.

- Keep tensor-core warp dimensions separate when folding logical local axes
  into GPU launch dimensions, so hardware lane indices retain their meaning.

- Replace `Opt.Upcast`, `Unroll`, `Local`, `Group` and `Grouptop` with
  `Opt.Split` using absolute range indices. Match tinygrad's split validation
  and beam choices; remove `Nolocals` and `kernel_info.dont_use_locals`.
- `Opt.Swap` preserves unrelated node tags when exchanging axes. Existing
  compiled caches and exported graphs are invalidated by the optimizer update.

- Fix `Opt.Padto` reductions: padded lanes now contribute the reduction identity,
  including products and negative maxima, and padded loads remain valid inputs
  to memory coalescing. Reject padding warp axes and multiples below two.

- Workgroup reductions use `Local` and `Warp` axes, with shared-memory staging
  derived from the contracted ranges. Remove `Axis_type.Group_reduce` and
  invalidate older compiled-program exports and caches.

- `Device.compile_program` groups buffer formals before scalar formals when
  compiling hand-built kernels, so interleaved declarations agree with the
  binary signature and cannot shift GPU argument values.

- CUDA dispatch and graph replay pack arguments using the binary signature,
  preserving mixed-width alignment, compact slot order and 64-bit scalar
  values when launching kernels or rebinding replay arguments.

- CPU dispatch honors compact slots in compiled binary signatures, including
  reordered buffer and mixed-width scalar bindings. Ordinary generated
  signatures retain the allocation-free argument path.

- AMD and NV dispatch bind arguments by their compiled slots and scalar widths,
  preserving mixed-width alignment and 64-bit values instead of packing every
  scalar into 32 bits. Invalid argument layouts leave mapped memory unchanged.

- Recompiling a kernel produces the same structural name regardless of earlier
  compilations, preserving source-cache reuse and stable program identities.

- Metal kernels use one typed argument structure for dispatch and replay,
  preserving 64-bit scalar values and buffer view addresses and allowing more
  than 31 buffer arguments. Older compiled-program exports are invalidated.
- Hand-built linear programs preserve buffer and scalar declaration order in
  `Program_spec`, preventing dispatch from swapping arguments when slot or
  variable-name order differs from the rendered function signature.
- `Device.runtime` now loads `Tiny_elf.t` binaries with their compilation
  target and argument signature. Dispatch checks argument counts, binds only
  the buffers used by each kernel, and supplies scalars during local-size tuning.
- Compiled programs and beam-search caches distinguish the selected renderer
  and target architecture. Switching targets can no longer reuse a program
  compiled for an earlier architecture on the same device.

- CPU JIT kernels on x86-64 can call external functions more than 2 GiB away.
  The ELF loader emits an absolute-address trampoline instead of truncating
  the call displacement and jumping to the wrong address.

- CPU kernels now finish on the calling domain, matching tinygrad's threadless
  execution model. SIMD remains enabled; `THREADS`, `NUM_CPU_THREADS`, and
  implicit `core_id` arguments no longer control dispatch.

- CPU kernel timings on macOS use the raw monotonic clock so fast synchronous
  kernels remain distinguishable during beam search.

- Conditional loops pass through kernel optimization without querying numeric
  bounds for their void scopes. New split axes avoid IDs held by loop scopes,
  device axes, and size-one ranges.

- AMD kernels honor loads marked `nontemporal` by emitting the cache-bypassing
  builtin. Scalar and vector loads retain the pointer type of their access.

- Elementwise computations over storage slices no longer fail with a false
  indexing-cycle error. Call normalization retains the sliced argument and
  its offset through JIT capture and replay.

- `Uop.param`, `Uop.buffer`, and global `Uop.placeholder` accept `volatile`.
  Scheduling and rendering retain the qualifier, and coalescing keeps volatile
  accesses separate and no longer combines accesses with different flags.

- Failed buffer-view allocation no longer retains a nonexistent live view.
  Allocation can be retried and the base buffer freed after the view is released.

- Empty host arrays have a registered storage identity and can participate in
  JIT capture and replay. Zero-byte buffers, copies, and views require no native
  allocation; empty views may start at the end of a buffer.

- `Search.beam_search` requires explicit `var_vals` for candidate filtering and
  timing, and rejects missing or out-of-bounds values. Codegen supplies
  floor-rounded interval midpoints by default, including negative intervals.

- Host defaults use the configured OCaml target, so CPU target selection and
  JIT initialization do not require `uname`. macOS caches use `~/Library/Caches`
  even before that directory exists; other systems consistently use `~/.cache`.

- Beam search releases temporary program handles after timing completes,
  including failures and interruptions, instead of retaining every candidate
  for the process lifetime or reusing handles across device instances.

- `END` validation requires a void effect and bounded integer ranges.
  Simplifying folded loop ranges preserves the enclosed effect and removes
  redundant loop boundaries.

- Scheduling uses `Uop.shape_opt` for shape inference, preserving symbolic
  extents and bitcast sizes. Incompatible broadcast dimensions raise an error
  instead of silently selecting an operand’s dimension.
- Empty slices and reductions preserve their output shape during scheduling,
  so padding and ring allreduce can consume zero-sized chunks.

- `Uop.call_info.dtype` declares a call’s scalar return type. C renderers
  support indirect host calls with typed arguments and preserve calls inside
  their enclosing loops.

- `Uop.loop` and `Uop.backedge` represent conditional loops explicitly,
  preserving enclosing range dependencies through validation, scheduling,
  and C rendering.

- 64-bit integer emulation shares fully lowered word splits within each pass,
  avoiding repeated arithmetic expansion and redundant casts.
  `Decomp_dtype.pm_long_decomp ()` creates a matcher for one pass.

- Integer floor division by powers of two lowers directly to arithmetic
  shifts, including negative inputs. `MAX` keeps its bounds until late
  lowering so subsequent compiler passes can use them.

- Constant integer division computes multiplier bounds exactly, avoiding host
  integer overflow. Metal uses the same proven multiply-shift rewrites; modulo
  stays native when no supported replacement exists.

- Software `sin` uses integer shifts during large-angle reduction, avoiding
  floating-point powers and conversions when selecting bits from the
  reduction table.

- Emulated `int64` and `uint64` comparisons split both operands before
  constructing word arithmetic, correctly distinguishing values with equal
  low words and different high words.

- Compact-float emulation constructs loads, vector slices, and bitcasts from
  their converted integer storage, keeping access dtypes consistent throughout
  lowering and preserving masked reads and writes.

- FP8 emulation uses `float32` arithmetic even on backends with native
  `float16`, preserving small normal `fp8e5m2fnuz` values that previously
  underflowed to zero.

- Emulated `int64` and `uint64` arithmetic accepts weak integer literals and
  preserves their high words through explicit casts, including masked-load
  fallback values.

- Emulated `int64` and `uint64` buffer accesses use both 32-bit storage words
  consistently. Masked loads and stores preserve their bounds and fallback
  values instead of leaving unsupported 64-bit accesses in the kernel.

- Backends that emulate 64-bit integers reject unsupported scalar runtime
  variables by name instead of silently narrowing their bindings to 32 bits.

- Add `Op.block_matmul`: each block of rows times the matrix of a stack its id
  addresses, read in place. A block whose id is out of range is exactly zero,
  and on a GPU it runs no multiply-adds. Products and sums are float32 on every
  device, so products of narrower floats are exact.

- Add `Op.quant_matmul`, a product with MXFP4 weights that decodes them in
  registers and reads each packed byte once per tile of rows, and
  `Op.quant_row_bound`, the most rows a matrix should meet before decoding it
  costs less. One gpt-oss expert (5760 by 2880) takes 50 us at one row on an
  M1 Max, 180 GB/s over its packed bytes.
- `Op.scatter_indexed ~unique:true`, and so `Nx.scatter ~unique_indices:true`
  under `Rune.jit`, gets launch sizes from the optimizer instead of one thread
  per workgroup: 64 rows of 32768 on Metal take 0.17 ms instead of 2.9 ms. A
  scatter without the promise keeps its duplicates in index order.
- A store whose length is a variable of at most 1 writes nothing when the
  length is 0. It wrote one element, which in a decode step overwrote row 0.

- A loop whose size is a variable of at most 1 stays a loop. The symbolic rules
  replaced it with its single index, so it ran once when the size was 0.

- Add `Decomp_dtype.is_dtype_supported`: whether a renderer's programs can use
  a dtype, natively or by emulation. `float64` on Metal is neither.

- Add `Device.Buffer.as_buffer`, a buffer's bytes as host memory without a
  copy on devices the host addresses (Metal, the CPU device), as tinygrad's
  `_as_buffer`.

- `Op.cumsum`, `Op.cumprod` and `Op.cummax` values scan axes over 512 in chunks
  of 256 (262,144 elements on Metal: 209 ms to 0.44 ms; `cummax` indices stay
  quadratic). An empty int8 `Op.cumsum` returns int32, like a non-empty one.

- `Device.Lru_allocator` no longer leaks a buffer that the GC frees while an
  allocation searches the cache. The allocation stored the cache it had read,
  dropping the new entry, so that buffer was never reused or freed.

- Dropping a Metal graph no longer risks a crash at a later synchronize. Its
  finaliser pruned the in-flight command list and could run while another
  graph was pruning it, leaving a released command buffer to be awaited.

- Long Metal runs no longer exit silently with status 2. Releasing a buffer
  view from a GC finaliser could relock the buffer table inside an allocation
  that already held it, which killed decode loops after a few dozen steps.

- Emulated `int64` and `uint64` casts to `float64` preserve double precision
  and the correct high word, fixing rounded results and zero output for `2^63`.

- Split ranges retain their complete axis identities through expansion,
  scheduling and tensor-core contraction. `Uop.axis_id` exposes that identity;
  `wmma_info.tc_upcast_axes` now carries full IDs instead of root axis numbers.

- `Device.compile_program` preserves the selected renderer's alignment and
  each call's device and optimization metadata. It uses the source-based
  compiler cache, avoiding unsafe reuse of aligned kernels for unaligned inputs.

- A device's target selects lazy renderer factories and exact compiler
  architectures, including CPU tuning and feature flags. Program caches
  distinguish targets;
  CUDA compilation uses the exact GPU architecture instead of a rendering tier.

- `Device.Renderer_set.make` accepts named target-aware factories. Superseded
  renderer controls and GPU architecture environment readers are removed.

- `Op.getitem` preserves symbolic dimensions outside advanced-index axes and
  accepts symbolically sized index tensors. Both combined and separate index
  axes retain their logical shapes through gathering and masking.

- `Op.getitem` supports symbolic axis lengths and negative slice bounds.
  `Movement.parsed` and `parse_view_index` now carry symbolic sizes and bounds;
  symbolic slices require a unit step and a provably non-negative length.

- Advanced integer-tensor indexing rejects indices on a different device,
  matching tinygrad instead of constructing an invalid mixed-device graph.

- `Run.data` rejects symbolic logical shapes, matching tinygrad and the typed
  host readers. It no longer exposes the allocation's maximum-size bytes as
  though they were the tensor's concrete shape.

- `Creation.clone` supports symbolic shapes, allocating at dimension bounds
  while preserving the logical view. Assignment can now initialize symbolic
  pending tensors, and clones retain independent storage through later writes.

- `Op.assign` propagates writes through bitcast aliases and initializes pending
  values without evaluating the computation being overwritten. Partial writes
  into pending contiguous storage preserve initialization and update effects.

- `Rand.rand` supports concrete float widths beyond float32, including
  float16, bfloat16 and float64 on capable devices. Packed draws and counter
  advancement match tinygrad, including odd-sized and empty half-width draws.

- Kernels whose writes simplify away accept the resulting empty effect group,
  as tinygrad does. Identity assignments and reading the random counter after
  an empty draw no longer fail during linearization.

- `Dtype_ops.bitcast` supports different element widths by rescaling the last
  axis. Packing preserves byte order through non-contiguous views and
  unaligned subword slices.

- `Rand.rand` draws and dropout masks own fresh storage, matching tinygrad.
  Random draws can receive indexed updates before their first realization.

- Reading an assigned tensor or view now executes its pending write before
  returning bytes. `Run.data` no longer returns stale values through the
  contiguous-view shortcut, and repeated reads do not repeat the assignment.

- `Op.assign` accepts weak scalar values that promote to the destination dtype
  and commits weak destinations to concrete storage. Assignment and
  `Op.scatter_indexed` reject weak-float updates to integer storage.

- Conditional simplification folds known branch conditions before merging
  guards, avoiding redundant predicates. Constant guards stay outside index
  validity rewrites instead of repeatedly adding masks to the same access.

- Staged weak arithmetic preserves wide integer values when it needs an
  intermediate buffer. `Run.of_bytes` rejects weak dtypes, which have no storage
  representation.

- `Creation.clone`, `Creation.full`, and scalar reads select a concrete storage
  dtype without narrowing large weak integers. Scan, scatter, sort, and pooling
  padding use extrema of that concrete dtype; `Creation.empty` rejects weak dtypes.

- `Creation.full ~buffer:false` leaves an inferred numeric dtype weak, so its
  consumers choose the precision. `Op.scatter_indexed` commits weak update values
  to the destination dtype, matching assignment.

- `Reduce.sum`, `Reduce.prod`, `Reduce.max`, and `Op.mean` preserve weak integer
  inputs beyond `int32` by choosing a storage dtype from their bounds. Exact values
  beyond all integer storage types are rejected before accumulation.

- `Uop.max_numel` rejects overflowing element counts and handles zero-sized
  shapes with large dimensions exactly. Large shape products no longer cause
  gated buffer indices to narrow incorrectly to `int32`.

- Weak integer conversions preserve truncation through nested casts and
  arithmetic. A narrow result cast no longer narrows wide division operands
  before the computation.

- Preserve floating-point rounding when simplifying comparisons with added
  constants. The rewrite `(c0 + x) < c1` to `x < c1 - c0` now applies only
  to integers.

- Fix symbolic simplification changing integer bit masks into multiplication.
  Validity-predicate rewrites now apply only to booleans, preserving expressions
  such as `(x & 12) & x`.

- `Uop.vmin`, `Uop.vmax` and parameter bounds retain exact integers and
  floating-point limits. Weak scalars can commit to `uint64`; values beyond
  all supported integer widths are rejected instead of silently narrowing.
- Remove the obsolete `Dtype.Uint128` and `Dtype.Uint256` storage helpers,
  matching the target scalar dtype set. Existing serialized caches are invalidated.
- Missing runtime variables report their name and program or replay call,
  including variables needed to compute launch dimensions.
- `CHECK_OOB` rejects unproved scalar accesses containing bitcasts or stacks
  and accesses without a known buffer extent. Out-of-range signed casts no
  longer produce spurious empty bounds.
- Beam search uses the 0.01µs progress threshold and reconsiders candidates
  rejected by a previous round's compute filter while reusing compiled code.
- `State.safe_load` decodes escaped Unicode names and validates metadata,
  shapes and byte offsets before uploading tensors. Empty host inputs no
  longer attempt zero-byte native allocations.
- Fix `asinh` for large negative inputs, log-space operations at infinity,
  activation saturation, and integer `Op.var`.
- Extrema, scans, sort and padded integer `Op.max_pool2d` preserve full-width
  integer identities. `Op.arange` accumulates small floats in float32 before
  narrowing, and `Op.pad_to` supports a nonzero fill value.
- Release unreachable tensor buffers and cached graph properties. Live views
  and JIT captures retain their storage; serialized program cache keys no
  longer depend on process-local node identities.
- `Uop.exec_alu` preserves full-width integer values and exact weak-integer
  intermediates. Literal matching avoids float rounding; weak promotion
  preserves padded values. `Const.Int` now carries a `Tolk_uop.Bigint.t`
  mathematical value.
- `Dtype.min` and `Dtype.max` return exact `Tolk_uop.Bigint.t` integer bounds,
  including weak integers, and finite limits for FP8 formats without
  infinities.
- AMD and NV devices automatically fall back to PCI when kernel-driver
  interface initialization fails. Explicit `AMD_IFACE`/`NV_IFACE` selections
  continue to restrict the device to the requested interface.
- Metal kernels cast between `bfloat16` and `float32` by bit manipulation, as
  the reference does to avoid a Metal compiler bug with
  `as_type<half>((bfloat)(const))`. tolk rendered native casts. Values are
  unchanged: both round half to even.
- `DEBUG=2` prints one line for every executed kernel, view, copy and batched
  graph: device, call count, name, memory in use, time, and GFLOPS and GB/s
  from the kernel's estimates. `Helpers.Global_counters` holds the running
  totals. Batching hides kernels inside a graph call, so `JIT=2` gives the
  per-kernel profile of a compiled function.
- A compiled function that is dropped now releases the device memory of its
  batched graphs. Recorded graphs were kept in a table that was never emptied,
  and each one holds the buffers of its intermediates: a process that compiled
  many functions, or one function at many shapes, grew without bound on Metal
  and CUDA (108 MB per dropped function in a three-matmul probe at 3072x3072).
  `Tolk.Realize.graph_runners` reports how many recorded graphs are live.
- `Cstyle.clang` and `Tolk_cpu.create` take `?aligned`. `~aligned:false`
  declares vector types aligned to one byte, for a device that binds memory it
  did not allocate. Absent, the `ALIGNED` environment variable decides, as
  before.
- The Metal device carries a `Device.Graph` capability: a batched call sequence
  is encoded once into an indirect command buffer and replayed as a single
  command buffer, with rebound buffers, variable values and launch dimensions
  patched in between. `Device.Graph.t` gains `max_buffer_offset`, which keeps
  a call whose buffer view starts past 4 GiB out of Metal graphs, and
  `FIX_METAL_ICB` overrides the pre-M3 pipeline workaround.
- `Creation.clone` takes `?device` and copies a source that lives on another
  device across. `Op.scatter_indexed` places its buffers on the device of its
  operands.
- Add `Op.scatter_indexed`: a scatter along one axis whose kernel ranges over
  the updates instead of the destination, and writes the destination's
  storage in place, as `Op.assign` does. Duplicate updates land in index
  order (the last `` `Set `` wins, `` `Add `` accumulates), an index outside the
  axis writes nothing, and `~unique:true` lets the updates run in parallel:
  a position aimed at twice then holds an unspecified one of its updates, and
  every other position stays exact.
  `Op.scatter` and `Op.scatter_reduce` keep the reference lowering.
  `Creation.clone` is now exposed.
- Fix `contiguous` over a window that narrows a trailing axis, such as the
  first two columns of a 2x3 buffer. The view was taken for one range of the
  flat buffer and read the wrong elements; only a window whose earlier axes
  all have extent one is such a range.
- Add `Uop.custom_kernel` and `Tensor.custom_kernel`: a kernel written in uops
  runs from the tensor graph over realized sources, and each source is read
  back after the kernel. `Uop.placeholder` and `Uop.placeholder_like` build
  the storage a kernel body addresses, and `Creation.empty` allocates storage
  for a kernel to fill, on `?device` when given.
- A gather over 32768 rows or more compiles to one kernel with one gated load
  per output element. The reduce split used to fire before the gather collapse
  and the kernel read the whole table into an intermediate buffer, so
  `Op.gather`, tensor-index `Op.getitem` and rune's compiled `Nx.take` cost a
  pass over the table.
- The in-memory program cache keys on `Realize.program_config`: `NOOPT`,
  `NOLOCALS`, `TC`, `IMAGE`, `DISABLE_FAST_IDIV`, `TRANSCENDENTAL`,
  `ALLOW_TF32` and the default dtypes. A kernel compiled under one setting was
  served under another when the setting changed through `with_context`.
- Every context variable is declared once in `Helpers`, and a second
  declaration of a key raises. `PCONTIG` had two independent copies, so an
  override reached only one reader. `IMAGE`, `FLOAT16`, `TC`, `TC_SELECT`,
  `TC_OPT`, `NOOPT`, `TRANSCENDENTAL`, `DISABLE_FAST_IDIV` and `ALLOW_TF32` are
  now context variables read once at startup; override them with
  `Context_var.with_context`. `Heuristic.nolocals_var` is `Helpers.nolocals`.
- Float literals in rendered kernels are laid out by tolk rather than the C
  runtime, so they are identical on every platform; Windows printed their
  exponents with three digits.
- A copy between devices is scheduled as a kernel storing the source's
  flat view into a buffer on the target device, then turned back into a
  transfer once the schedule is linear. Multi-device graphs therefore
  emit one kernel per per-shard copy, as the reference does, instead of
  fusing copies into their consumers; a kernel that mixes devices without
  being a copy is rejected with `all buffers must be on the same device`.
- Multi-device sharding slices along a device range instead of a
  `_device_num` variable. The range is not a program axis: codegen lowers
  it to `_device_num` and keeps it out of range splitting and the
  optimizer's axes, and kernels number it first, so a sharded kernel's loop
  axes now start at 1 (`Lidx1`), as in the reference.
- Fix the symbolic division rules, which never fired: `x/x`, `(x*y)/y`,
  `0/0`, `(x*0)/0` and `(x/y)/z` were matched on `Fdiv`, an operation that
  only exists after the late decompositions, while a division in the graph
  is `x * recip y`. Chained divisions now fold to one division, so a kernel
  computing `(a/b)/c` renders `a/(b*c)`.
- CPU kernels run on Windows. Their entry now carries the Microsoft calling
  convention there, as the reference does: the object is compiled for a
  generic ELF target whose x86-64 convention is System V, so the host read
  the kernel's arguments from the wrong registers and larger kernels crashed.
- The CPU thread pool is sized from the runtime's recommended domain count
  instead of a `getconf` subprocess, which does not exist on Windows.
- CPU kernel timings come from a monotonic high-resolution clock. The wall
  clock moves in millisecond steps on Windows, which tied every fast kernel at
  zero and left beam search nothing to rank.
- The disk cache creates its temporary file exclusively under a random name
  instead of one derived from the process id. On Windows `Unix.getpid` is a
  handle value that sibling processes routinely share, so concurrent writers
  wrote into one temporary and tore the entry. A temporary whose final rename
  loses is also removed, so none accumulate next to the entries.
- The CUDA, NVRTC, comgr, and driver runtimes build on Windows. The vendor
  libraries load through `LoadLibrary`; the hcq layer maps anonymous memory
  through `VirtualAlloc`, so the queue builders run; the system layer's file
  primitives work, while the Linux kernel interfaces report themselves
  unsupported.
- New `Tolk_frontend.Linalg`: `qr`, `solve_triangular`, and `cholesky` unroll
  at graph-construction time into ordinary Tolk compositions, so they compile
  for every Tolk device. Large systems solve block-by-block as GEMMs, with
  the diagonal blocks inverted by one batched substitution.

- Driver-less NVIDIA (`NV_IFACE=PCI`) hardening: opening now waits for the
  GPU's boot firmware to report ready before sizing VRAM (a device opened
  mid-boot, or straight after the armed-region recovery reset, could read
  garbage), the GSP client is registered before the first object call,
  shutdown finalizes the boot layers in reverse bring-up order, and stalled
  waits back off to draining GSP events after 200ms instead of on every
  poll.

- `NV_DEBUG=4` traces every register write on the driver-less NVIDIA path
  (`wreg: 0x<addr> = 0x<value>`); `DEBUG>=2` logs the armed-region recovery
  reset and `DEBUG>=3` decodes incoming GSP RPCs by name.

- NVIDIA GPUs can now be driven over PCI with no kernel driver: setting
  `NV_IFACE=PCI` boots the GPU's GSP firmware directly — the falcon and
  chain-of-trust bring-up, the shared-memory RPC queues, the golden-image
  channel and context setup, and the object, control and memory calls as GSP
  remote procedures (`Tolk_nv.Pci_iface`), reading firmware from
  `NV_FW_PATH`. Opt-in and unvalidated on hardware so far; the kernel driver
  remains the default.

- A stalled AMD device wait now fails with the whole story — the timeout
  (expected and observed timeline values) folded with the driver's fault
  report — where it could previously raise `Failure("")` when the driver had
  no fault to report. Stalled waits also back off to the driver's event
  sleep after 200ms instead of 2s, surfacing faults sooner.

- NVIDIA GPUs are now a hardware-queue runtime target: the device
  `Nx_nv.device i` drives the kernel driver's channels directly, with
  kernels compiled straight to cubin by nvrtc, covering the Ampere, Ada,
  and Blackwell generations. The userspace CUDA backend
  (`Nx_cuda.device i`) remains available unchanged.

- Driver-less AMD devices can now sleep on interrupts instead of spinning:
  with `VFIO=1` (and the `vfio-pci` kernel module), the device's MSI vector
  is routed to an eventfd that stalled waits block on. Without it, waits
  poll as before.

- AMD GPUs can now be driven over PCI with no kernel driver: setting
  `AMD_IFACE=PCI` boots the GPU directly — firmware loading, memory hubs,
  security processor, engines (`Tolk_amd.Pci_iface`) — covering the
  RDNA3/RDNA4 consumer parts. Opt-in and unvalidated on real hardware so
  far; the kernel driver remains the default.

- AMD GPUs are now a runtime target: the device `Nx_amd.device i` drives
  the GPU through the Linux kernel driver's hardware queues, with kernels
  compiled by the ROCm comgr library. Supports gfx942, gfx950, and the
  gfx11/gfx12 generations (single-die), with DMA-engine host transfers and
  device-side execution timing.

- Fix the AMD renderer's `__ockl_get_local_id`/`__ockl_get_group_id`/
  `__ockl_get_local_size` declarations: the return and argument types were
  swapped (`unsigned int f(size_t)` instead of `size_t f(unsigned int)`), so
  kernels using launch indices declared the OCKL intrinsics with the wrong
  signatures.

- `Elf` now loads shared-object (`ET_DYN`) images in addition to
  relocatable objects, and builds the image from program sections rather
  than all allocatable sections, so GPU code objects load with the same
  layout their producers intended.

- The AMD/HIP renderer (`Cstyle.amd`) is now covered by the tinygrad parity
  and golden corpora on gfx1100, including tensor-core matmul cases across
  RDNA3/RDNA4 (WMMA f16/bf16), CDNA3 (MFMA bf16), and CDNA4 (scaled MFMA
  fp8).

- `Search.beam_parallel` is a `Helpers.Context_var`, so a caller can scope
  the `BEAM_PARALLEL` worker count to one compilation with
  `Context_var.with_context` instead of setting it process-wide.

- `Realize.pm_compile` takes `?beam`, stamping kernels that do not already
  carry a beam width, so a caller can enable beam-search autotuning for one
  compilation instead of process-wide through the `BEAM` environment
  variable.

- Free `BEAM`-search timing buffers as soon as each kernel's search finishes:
  they were reclaimed only when the GC ran, and then into the unbounded LRU
  allocator cache (which matches by exact size), so a model with many
  distinct kernel shapes accumulated their GPU memory until a failed
  allocation flushed the cache — OOM on large graphs with `BEAM>=1` where
  `BEAM=0` fits. Timing buffers now bypass the LRU cache and are released
  deterministically per kernel.

- Compile `BEAM`-search candidates in parallel domains (`BEAM_PARALLEL=N`,
  default off): the CPU-side compile of a step's candidates — optimize, lower,
  render, nvrtc — now overlaps across domains, while the GPU timing phase
  still runs one candidate at a time so timings never contend. Shared state
  is safe under domains: the hash-cons table, kernel naming, program cache,
  and disk cache take locks, and the per-node memo caches are domain-local.
  On the RNN grad repro (CUDA,
  `BEAM=2`, cleared tolk and driver JIT caches): 27s -> 6s with
  `BEAM_PARALLEL=8`; with warm caches ~11s -> ~7s.

- Stop `BEAM` search when progress falls below timer noise: the default
  `BEAM_MIN_PROGRESS` is now 5µs (env still overrides). The old 0.01µs
  default sat below the ~0.5µs device timer resolution, so search only
  stopped when a step brought no improvement at all — kernels kept searching
  on measurement noise, ~2.6× more candidate compiles on a CUDA `BEAM=2`
  pass.

- Test same-call WAR dependencies by node identity instead of structural
  equality in `fix_war_deps`. `Uop.t` nodes are hash-consed, so the two checks
  give the same verdict, but polymorphic equality on the `call_of` options
  descends into the full shared payload DAGs (exponentially on graphs with
  heavy sharing) where `==` is O(1).

- Reduce cold `BEAM`-search compile time by removing two sources of redundant
  CPU work: the device dispatch handle (driver module load, PTX → SASS JIT) is
  loaded once per compiled binary and reused across timed candidates, and
  candidates whose AST was already compiled are deduplicated up front via the
  hash-consed tag instead of only after `nvrtc`.

- Fix CUDA tensor-core kernels failing to compile. Every `__WMMA_*` primitive
  was emitted with scalar parameters and empty asm operand lists —
  `float f(half a, half b, float c)` — while the kernel body correctly built
  vector operands for it. The renderer rebuilt the three operand widths from
  the tensor-core upcast axes, which the expander clears once it has applied
  them, so the one-lane fallback always fired; the widths now come from the
  operands themselves. Metal was unaffected.
- Fix a gated image load falling back to a single zero instead of a full
  vector. The value a gated load reads when its gate is false was sized by
  re-deriving the access width from the index expression, which had no case for
  an image coordinate and defaulted to one lane; an image access reads four
  floats, so three lanes were left unwritten. The width now comes from the
  load's own lane count, exposed as `Uop.max_numel`.
- Fix multi-device programs failing to compile with
  `Invalid_argument "buffer copy: size or dtype mismatch"`. An operand that
  broadcasts along the shard axis — one of lower rank than the result, or with
  size one there — was split across devices as if it held a distinct slice per
  device, so a collective was sized from a fraction of its real shape. Such an
  operand now stays whole on every device. Separately, the scheduler refused to
  read a shape off a `Pad` or `Shrink` whose offset was symbolic, even though
  the shape is the size argument alone; the resulting unknown propagated into
  an elementwise node as a too-small shape and mis-sized the allreduce buffers.
- `Op.cat` joins operands with equal extents on the concatenated axis through a
  single stack node instead of padding each operand to the full width and
  summing them, so a concatenation of `n` inputs now selects on one loop range
  rather than emitting `n` pads and `n - 1` adds. It also rejects operands whose
  shapes differ off the concatenated axis, which used to broadcast silently.
- Move `stack` from `Op` to `Movement`, where it now builds a stack node
  directly rather than unsqueezing every operand and concatenating. Call sites
  change from `Op.stack` to `Movement.stack`.
- Fix an unsound simplification of `((hi << 32) | lo) >> 32`, which returned
  `hi` even when `hi` had bits above 32 that the shift had discarded. It now
  requires `hi` to come from a 32-bit source, matching what packing two `uint32`
  halves into a `uint64` actually produces.

- Fix a scheduler miscompile that made mixed-precision training steps fail to
  compile. Rangeify records per-node loop ranges while walking the tensor
  graph, and was carrying those records over to the nodes it rebuilt. Because
  rebuilding removes movement ops, two values that differed only in a
  broadcast collapse into one node — which then inherited the wrong rank, and
  its reader was indexed with fewer indices than the staged buffer had axes.
  The result was one store per element to a single address, rejected as
  `Coalesce: multiple stores to the same offset`, or silently wrong values
  where it slipped through. A shape shared between consumers of different rank
  is now mapped into each source's own axes, and a mismatched index raises
  instead of being tolerated.

- Rewrite the graph rewriter's traversal so nodes are visited in dependency
  order rather than plain depth-first order. Passes that number things as they
  go — accumulator registers, local buffer slots — now number them the same way
  the reference does, so generated kernel source matches it exactly instead of
  differing in register and buffer names. A call or function body is now left
  alone on every path that reaches it, not only on the edge through its call.
  This costs compile time: roughly 20% more wall time and 15% more allocation,
  concentrated in the codegen, schedule, and rangeify stages. The new traversal
  keeps a scheduled set and revisits each rebuilt node, which is inherently
  more bookkeeping than the depth-first walk it replaces.

- Tighten index arithmetic in tensor-core kernels. A scaled remainder and its
  quotient partner now recombine even when the quotient's divisor has absorbed
  an inner division, so two adjacent single-bit extracts of a thread id
  collapse into one multi-bit extract instead of being emitted separately.
- Fix a symbolic rewrite that dropped valid indices from a bounds test. A
  comparison of the form `x // d < c` with a non-positive `c` was rewritten to
  a bound one step too tight, so `x // 2 < 0` became `x < -1` and excluded
  `x = -1`, which satisfies the original. Any gate, mask, or loop bound reduced
  through that shape could exclude a valid element.
- Fix silently wrong results from any kernel holding both a group reduce and
  an ordinary loop reduce — the shape of most backward passes on a GPU. The two
  reduces were given the same accumulator register and the second's zero-init
  was dropped, so the closing add read one accumulator twice and the kernel
  computed `2(a+b)` where the answer is `a+b`. Affects every target with local
  memory (CUDA, Metal, OpenCL); CPU was never affected.
- Restore multi-threaded CPU kernels for large fused reductions. The
  ops-per-thread heuristic formed the iteration-space product in a machine
  integer, which wrapped on kernels fusing enough reduce axes (an unrolled
  recurrence's weight gradients reach 2^110), so the thread split was silently
  skipped and those kernels ran single-threaded.
- Multi-device programs with two outputs that both need a cross-device
  reduction — a data-parallel step returning both the loss and the gradient of
  a replicated parameter — no longer miscompile. Flattening a bufferize folded
  the source's own shape into the flattened extent, which overflowed to a
  negative size whenever that shape was unresolved, leaving the gradient with a
  one-element buffer that every lane wrote to at index zero. `Rune.pmap` steps
  that hit this failed to compile rather than returning wrong numbers.
- Tensor-core kernels now declare each `WMMA` result at the accumulator width
  the instruction actually returns, so tensor cores are usable again on Metal
  and CUDA. The renderer declared them scalar, emitting programs the target
  compilers reject (`cannot initialize a variable of type 'float' with an
  rvalue of type 'float2'` on Metal). Any tensor-core candidate therefore
  failed to compile, and `BEAM` search silently discarded all of them.
- Reading a tensor whose graph folds to a pure constant (`Run.data`,
  `Run.to_float_array`, `Run.item_float`, ...) now materializes it into a fresh
  buffer instead of raising: such a graph is placed on no device and owns no
  storage, so `Run.buffer_of` had nothing to return. Affected every
  constant-folded result, for instance `Rand.dropout ~p:1.`.
- A cast to an unsigned dtype now keeps the source's exact symbolic bounds when
  that source already fits the destination window; only a source that can wrap
  (negative, or wider than the destination) falls back to the full dtype range.
  `Uop.vmin`/`Uop.vmax` previously gave up on every unsigned cast.
- Reject the tensor-core optimisation when one of its X or Y axes is a reduce
  axis. Those axes index the WMMA accumulator tile, so a kernel that reduces
  over a matmul's output dimensions (`(a @ b)` summed over `M`, as a recurrent
  loss does) compiled to a kernel that computed the wrong values. `BEAM` search
  ranks candidates by runtime alone, so it could select it silently.
- Invalidate the on-disk cache, so a beam result cached before the tensor-core
  fix above is re-searched rather than replayed into an optimisation that is
  no longer legal.
- Normal `dune build` no longer runs the debug golden-test generator; its
  `DEBUG=6` AST diagnostics and `.actual` fixtures are confined to `runtest`.
- Codegen distributes the negation of a sum as a multiply by `-1` over its
  terms, so a later-negated constant-scaled subexpression folds its sign into
  the constant factor (`c*x` negated becomes `(-c)*x`) rather than re-negating
  the scaled product. Deep element-wise chains that reuse a scaled difference
  (e.g. an Euler Lorenz step) now lower to a single canonical form instead of
  keeping a redundant negate.
- Fix a compile-time blowup in schedule creation: `Rangeify.get_kernel_graph`
  re-derived tensor shapes without memoisation and rescanned the whole graph
  history once per kernel, so compile time grew super-linearly in graph size
  and deep element-wise folds (e.g. a 100-step Lorenz integration) stalled
  indefinitely. Shape derivation is now cached and kernel splitting stays within
  a single kernel, so compilation scales linearly.
- Constant-folding an integer `Floordiv`/`Floormod` by zero no longer raises;
  it folds to `0` / the dividend, matching the existing `Cdiv`/`Cmod` guards.
- Fix jitted random-number generation: the call transformation sized staging
  buffers for broadcast expands one rank short, so `Rand` under a captured
  jit failed with `buffer_like: unknown shape`.
- Fix reading realized contiguous views: a slice read back through `Run`
  returned its source's data at offset zero; views now alias the source
  allocation at the correct byte offset and share storage instead of
  copying.
- Full parity refresh against the reference compiler: dtypes are now a flat
  scalar enum (`Dtype.Val`/`Ptr`/`Image` and vector widths are gone — vector
  width comes from a value's shape, pointer provenance from its address
  space), reduces carry an op and leading-axes count, expands prepend leading
  dims with `Uop.broadcast_to` as the same-rank broadcast, and `dtypes.index`
  types shapes and loop bounds. Every generated kernel is verified
  byte-identical to the reference across the parity, codegen, renderer,
  kernel-graph, and debug golden suites.
- Fix kernel search on CPU: waited calls now report elapsed wall-clock time,
  so BEAM can rank CPU kernels — previously every candidate tied at infinity
  and selection was arbitrary.
- Fix kernel cost estimates: vector lanes count toward op/load/store volumes
  again, repeated reads of one buffer cap at its footprint, and tensor-core
  flops count the full matmul volume — beam decisions were skewed on all
  three.
- Fix multi-device scheduling: sharded elementwise graphs no longer hang
  kernel lowering, and ring allreduce no longer crashes on gated chunk
  reassembly.
- Scalar operands adopt their paired tensor's dtype: `float16_t + s` stays
  float16 instead of silently upcasting to float32, matching the reference's
  mixed-precision behavior.
- `Buffer.copy_from` is the canonical buffer copy, routed through the engine;
  `copyin`/`copyout`/`transfer` remain as the low-level allocator bridge.
- `State.safe_load` reads fp8 tensors (`F8_E4M3`, `F8_E5M2`), and
  `load_state_dict` only reconciles the exact scalar-to-one-vector shape
  pair, so stray unit-dimension mismatches fail loudly.
- Clang kernels declare half-precision buffers as `__fp16*`, matching the
  reference's C dialect.
- The renderer drops its unused pre-matcher hook, and `Renderer.make`'s
  emulated-floats option takes flat dtype pairs.
- CPU jit: bfloat16 kernels no longer fail with `Compiler.Compile_error` on
  hosts whose clang predates `__bf16` support (clang < 15 on x86-64). The
  CPU device now probes the compiler once and falls back to the float32
  storage-emulation path already used on riscv64; `Cstyle.clang`/
  `clang_no_abi` gained an optional `?native_bf16` flag.
- `Tolk_nn.Linear.create` and `Embedding.create` now randomly initialise
  their parameters (uniform `±1/sqrt in_features`, Glorot uniform) instead
  of zeros.
- The gpt2 example gains `--temperature` and `--seed`; at non-zero
  temperature it samples from the temperature-scaled softmax and reproduces
  the reference token stream at the same seed.
- Add `Rand`, a counter-based (Threefry) random number frontend:
  `manual_seed`, `rand`, `randn`, `randint`, `uniform`, `normal`,
  `scaled_uniform`/`glorot_uniform`/`kaiming_uniform`/`kaiming_normal`,
  `randperm`, `multinomial`, and `dropout`. Values are deterministic per
  seed, identical across devices, and keep advancing inside `Jit` captures.
  `rand` is limited to 32-bit floats for now. Also adds
  `Elementwise.threefry`, the Threefry-2x32 mixing primitive.
- Fix `Elementwise.neg` (and therefore `sub`) on unsigned integer tensors:
  negation now wraps at the operand's width instead of promoting to a wider
  signed type.
- Add `Compiler.cachekey`, exposing a compiler's disk-cache table name as a
  compiler/architecture fingerprint for callers keying their own caches.
- `Uop.export`/`Uop.import` serialize hash-consed graphs across processes;
  import re-interns every node so structurally-equal live nodes are reused
  physically. Export raises on graphs carrying gradient functions; import
  raises `Failure` on malformed input. `Uop.intern` is removed (it was a
  no-op on foreign graphs); `Schedule.fresh_internal_buffer_slot` is exposed
  for renumbering imported internal buffer slots.
- `Diskcache.put` writes atomically via rename; concurrent writers can no
  longer tear a cache entry.
- The gpt2 example supports `HALF=1`, storing weights and attention
  activations in float16; generated text matches the reference and every
  compiled kernel is byte-identical to it.
- Fix `layernorm` computing the epsilon add in float32 for reduced-precision
  inputs: the constant now follows the operand's dtype, so `float16`/
  `bfloat16` layer norms keep their variance and rsqrt in half precision
  instead of silently widening.
- Fix multi-device scheduling of keepdims-style reductions: the multi
  rewrite minted shape dimensions as `int32` constants while the rest of the
  compiler uses weak integer constants, so broadcasting a realized reduce
  buffer back against a sharded operand failed shape validation and jit
  raised `Failure "buffer_like: unknown shape"`.
- Memoize `Uop.axis` like `Uop.shape`: the unmemoized walk was exponential
  in residual depth, making multi-device compilation of deep networks
  appear to hang.
- `Realize.Buffers.seed_multi` binds a buffer node to a caller-provided
  multi-device buffer, mirroring `seed`.
- Fix multi-device scheduling of computed sharded outputs: buffer allocation
  sized MULTI-wrapped values per shard twice, and the store-revert rule
  cycled on multi-device store targets.
- Fix a rewrite cycle in rangeify when shard axes are realigned across
  devices (e.g. `x @ transpose x` on a sharded value): a symbolic variable
  parameter in a shard offset was mistaken for a buffer and indexed.
- Add a device registry: `Device.register` installs a backend opener per
  name prefix and `Device.get` opens and caches devices by canonical name;
  `Run` registers the CPU/CUDA/METAL backends there.
- The engine now executes multi-device schedules: buffers placed on a
  device tuple allocate one shard per device, kernels launch once per device
  with the `_device_num` variable bound, and copies transfer per shard pair
  (natively within a backend, via a host bounce across backends). Previously
  `Realize.resolve` silently read only the first shard of
  `MSELECT`/`MSTACK`.
- The memory planner suballocates multi-device internal buffers into
  per-placement arenas; single-device planning is unchanged.
- Fix scheduled `COPY`/`SLICE` calls dropping their destination buffer, and
  scalar (rank-0) values losing their flat index through staging and kernel
  splitting.
- Multi-device scheduling now matches the reference: `RING` defaults to 1
  (ring allreduce with >2 devices above the 256k-element threshold), and the
  `LATE_ALLREDUCE` toggle is supported (default 1 wraps allreduce into a
  precompiled function; 0 expands it inline during the multi rewrite).
- Fix several multi-shard rewrite bugs: `SHRINK`-before-`MSTACK` computed
  wrong sizes, pads/shrinks of sharded tensors were rejected or mis-shaped,
  sharded `PARAM`s were never resolved, and unsharding padded to the
  per-shard instead of the full size.
- Each allreduce output now gets a fresh buffer; identical allreduces no
  longer collapse onto one output allocation.
- Graph replay now repatches buffer arguments whose binding was reseeded
  between runs, not just input `PARAM` slots; previously such replays used
  stale device addresses. Changed arguments are diff-patched, so stable
  bindings cost one lookup. New `Jit.batch_graphs` exposes the JIT's graph
  batching to callers driving `Realize` directly.
- JIT replay now batches consecutive CUDA kernels and device copies into
  CUDA execution graphs, replaying each batch as a single `cuGraphLaunch`
  instead of per-kernel launches. Symbolic variable values, launch
  dimensions, and rebound input buffers are patched into the instantiated
  graph on every call. `JIT_BATCH_SIZE` controls the initial batch size;
  `JIT=2` disables batching.
- `Device.make` accepts a `?graph` batched-dispatch capability
  (`Device.Graph`) and `Device.prog` carries the backend kernel `handle`;
  the CUDA device provides both.
- CUDA device-to-host copies (`Device.Buffer.copyout`) now stage through a
  pinned host buffer, improving transfer bandwidth and releasing the OCaml
  runtime lock during the copy.
- Schedule-time bufferize removal now mirrors the reference cost rule:
  staged reads re-index through the producer regardless of range/index
  sizes, so `arange` embedding gathers fold completely and constant-index
  views of computed/assigned tensors (q/k/v selectors, KV-cache reads) no
  longer emit whole-buffer copy kernels. GPT-2 no longer copies the full KV
  cache per layer per decode step (~16% faster CUDA decode).
- Index arithmetic builds `a - b` as `a + b * (-1)` in schedule indexing
  and reduce-collapse rules, and collapsed range-sum clamps use the exact
  reference `min`/`max` structure, so gather offsets cancel symbolically.
- Weakint comparisons now lower with the index-dtype pass, keeping
  valid-bounded gather indices in 32-bit arithmetic instead of widening to
  int64.
- Fix C-style renderers duplicating upcast lane-0 load/store address
  expressions: `render_index` no longer re-orders `ADD` index chains at
  render time, so lane 0 reuses the shared named subexpression (`aluN`)
  instead of re-deriving it with a different term order.
- `examples/gpt2` now decodes through a per-layer key-value cache with
  symbolic positions and a captured JIT: one kernel set serves every decode
  step, taking greedy generation from ~2 tok/s to ~19 tok/s on CUDA
  (`--validate` reproduces the reference texts on CUDA and CPU).
- `Creation.full`/`zeros`/`ones` (and `_like` variants) now materialize a
  fresh buffer by default so in-place `Op.assign` has storage to write to;
  pass `~buffer:false` for the previous fold-into-consumers broadcast
  constant.
- `Run.realize` no longer fails on tensors whose graph folds to a constant
  expression (e.g. a realized `arange`); they stay lazy.
- Fix a shared-memory sizing miscompile in grouped reductions (the
  `GROUP`+`LOCAL`+`UPCAST` matvec path raised "invalid RESHAPE"); local
  staging buffers are now materialized by codegen.
- Kernel-source fidelity fixes: float `x + y*-1` renders as `x - y`; folded
  float constants keep full precision; kernels with two symbolic variables
  no longer emit invalid `make_void()`; dimensionless kernels are named
  `E`/`r` without a trailing underscore.
- Add `tolk.nn`: `Embedding`, `Linear`, and `Layer_norm` layers plus
  `State.safe_load`/`State.load_state_dict` for loading safetensors
  checkpoints into layer parameters.
- Add a GPT-2 (124M) text-generation example
  (`packages/tolk/examples/gpt2`) that reproduces reference greedy
  generations on CPU and CUDA.
- Add `Run.of_bytes` to create tensors from raw little-endian bytes and
  `Run.device_name` to inspect the realization device.
- Fix exponential graph walks in `Uop.ranges`, `Uop.addrspace`,
  `Uop.semantic_key`, symbolic lane counting, and the C-style renderer by
  memoizing them; realizing deep transformer graphs and reducing over
  prime-sized axes now takes milliseconds instead of minutes.
- Add `Jit`, a capture-and-replay JIT for tensor functions: the first call
  runs eagerly, the second records and compiles every kernel the function
  realizes, and later calls replay the compiled program on fresh inputs and
  symbolic variable values without rebuilding, rescheduling, or recompiling.
  Buffers backing live tensors (weights, KV caches, outputs) keep their
  storage across replays; other intermediates are folded into arena memory.
- `Run` now exposes the execution device and buffer storage registry
  (`device`, `buffer_of_node`, `buffer_nodes`), and `Tensor.live_tensors`
  lists the tensors currently reachable by the program.
- Kernels with symbolic sizes now render with reference-parity kernel names
  (e.g. `r_28start_pos2B129`), symbolic loop bounds, and symbolic GPU launch
  dimensions; new `Render.expr_to_string` renders scalar expressions
  compactly.
- Fix reductions over symbolically-sized axes silently collapsing: range
  creation now consults expression-level shapes.
- Support symbolic shapes in the tensor frontend: new `Movement.symbolic_shrink`,
  `symbolic_reshape`, and `symbolic_broadcast_to` entry points, and
  symbolic-shape handling through broadcasting, `dot`, `softmax`, reductions,
  and `Op.assign` — enough for a KV-cache transformer decode step where the
  sequence position is a bound variable. Operations that need concrete shapes
  (`pool`, `split`, strided indexing, ...) keep raising `Invalid_argument`.
- Add symbolic-integer helpers to `Uop`: `resolve`, `smax`, `smin`, `sprod`,
  a checked `broadcast_shape`, and `unbind` for splitting a `Bind` into its
  variable and value. `Uop.bind` now validates the value against the
  variable's range.
- Fix the engine so one schedule serves every bound value of a symbolic
  variable: callified `Bind` inputs keep their variable name and range,
  kernel graphs recover the canonical variable, and variable values are
  passed to kernels at launch instead of being resolved as buffers.
- Fix post-schedule parameter substitution to index call arguments including
  `Bind`s; previously a buffer argument following a `Bind` was bound to the
  wrong slot.
- JIT replay is now parameter-substitution based: capture substitutes input
  buffer nodes with slotted `Param`s, memory-plans the combined schedule
  once, and replays with per-call `input_uops` and `var_vals`, so replays
  with different variable bindings get correct kernel `vals` and launch
  dimensions. `Jit.call` takes input buffer nodes, and its `held_buffers`
  argument is honored: buffers that outlive the jitted computation (e.g.
  in-place caches) keep their allocation instead of being folded into
  arenas.
- The schedule capture hook moved to `Realize.capturing`;
  `Schedule.create_linear_with_vars` hands captured schedules over unplanned
  and its `?memory_plan` flag is removed (`Schedule.memory_plan_rewrite` is
  now exported). Capture can be disabled with `CAPTURING=0`.
- Add a CUDA runtime: tensor programs now compile through NVRTC and execute
  on NVIDIA GPUs. The driver and NVRTC libraries are loaded dynamically at
  run time, so builds do not require a CUDA toolkit and fail cleanly at
  device creation when no GPU is present.
- Fix gated vectorized loads: the masked fallback rendered a scalar zero
  for a vector access, which the CUDA compiler rejects; the zero is now
  stacked to the access width.
- Add in-place assignment: `Op.assign` records a buffer write in the graph,
  including writes through sliced views (e.g. a transformer kv-cache update),
  and repoints every live tensor aliasing the buffer so later reads observe
  the write. `Run.realize` now also rebinds assigned and previously realized
  nodes onto their computed buffers, so an assignment executes once and reads
  of a realized tensor reuse its buffer instead of recomputing.
- Add `Op.scaled_dot_product_attention` (optional additive or boolean
  `attn_mask`, or `is_causal` masking) and `Op.layernorm`.
- Fix a compilation- and schedule-cache collision: the cache key hashed node
  payloads with the polymorphic hash, which cannot distinguish constants such
  as `0` and `-1` (OCaml folds the halves of an `int64` with xor), so two
  kernels differing only in such a constant could silently share one compiled
  program. Constant payloads are now rendered exactly into the key.
- Fix buffer identity reuse across realizations: allocation slots are now
  drawn from one process-wide counter (`Uop.fresh_buffer_slot`), so two
  distinct allocations can no longer hash-cons onto the same node and
  read back each other's storage.
- CUDA source generation gains an `SM90` (Hopper) target tier in
  `Gpu_target.cuda`: `sm_90` uses the `sm_89` tensor-core table and fp8
  dtype support, and `CUDA_ARCH`/`CUDA_SM` values of 90+ resolve to `SM90`.
- Tensor-core (WMMA) kernels now lower end to end: fixed the WMMA output
  shape rule, upcast-axis deduplication, warp-lane decomposition, and WMMA
  devectorization, so tensor-core matmuls render byte-identical to the
  reference (covered by an fp8 `mma.sync` parity golden).
- New package: a minimal ML compiler for tensor computation. A tensor program
  is a graph of micro-operations; Tolk schedules it into kernels, lowers and
  optimizes them, renders C-style source, compiles, and executes the result —
  entirely from OCaml. Equivalent to tinygrad in Python.
- The `Tolk_frontend` library builds these graphs through a NumPy-style tensor
  surface: broadcasting and dtype promotion, movement ops, reductions,
  `matmul`, `conv2d` and pooling, cumulative scans, `sort`/`argsort`/`topk`,
  `gather`/`scatter`, `masked_select`/`nonzero`, `softmax` and friends, and
  NumPy-style advanced indexing — and executes them end to end
  (`Run.realize`) with host-data round-tripping.
- A capture/replay JIT (`Tolk.Jit`) records a traced program once and
  re-executes it against new inputs. Compiled kernels are first-class graph
  nodes carrying their rendered source and compiled binary.
- Runtimes: CPU (via Clang) and Metal. Renderers additionally target CUDA,
  AMD/HIP, and OpenCL for GPU code generation.

### Vega (new)

- The structural optimizers (`sgd_step`, `adam_step`, `adamw_step`) pass
  non-float leaves through unchanged instead of updating them, so a structure
  carrying an `Nx.Rng.t` alongside its parameters survives a step. Adam is
  where it showed: its direction runs each leaf through a square root and a
  division.
- Add `Loss_scale` for float16 training: static and dynamic loss scales
  with `scale`/`unscale`/`grads_finite`/`adjust`; all state is scalar
  tensors updated by `Nx.where` arithmetic, so it threads through
  `Rune.jit`/`pmap` steps and adapts across compiled calls.
- `sgd_step` with `momentum = 0.` (the default) no longer reads or updates
  the velocity state; under `Rune.jit` this stops parameter-sized zero
  velocities from being captured and transferred every step.
- New package: gradient-based optimizers and learning-rate schedules. Built
  on Nx with no autodiff dependency. Equivalent to Optax in JAX.
- The primary surface is structural: `sgd_init`/`sgd_step`,
  `adam_init`/`adam_step`, `adamw_init`/`adamw_step`, `global_norm`,
  `clip_by_global_norm`, and `clip_by_value` step whole parameter
  structures, any value with an `Nx.Ptree.t`, with optimizer state
  shaped like the parameters themselves.
- Schedules are plain functions of the step counter; loops evaluate one at
  the counter and pass it as `~lr`.

### Compress (new)

- Add the `compress` package: deflate with its zlib and gzip framings
  (`Compress_deflate`, moved out of `nx.io`), Snappy blocks
  (`Compress_snappy`), and LZ4 and Zstandard decompression (`Compress_lz4`,
  `Compress_zstd`), with no dependencies. A string compresses whole with
  `Compress_deflate.Zlib.compress ?level` and back with `decompress`; streams
  go through `Compress_deflate.Encoder` and `Decoder`, state machines that are
  given input and return output in bounded memory; and data held in memory,
  such as a page of a mapped Parquet file, decompresses directly between byte
  arrays with `decompress_into`.

### Nx

- `Nx.mean`, `Nx.var` and `Nx.std` at `float16`, `bfloat16` and the float8
  dtypes compute at float32 and round once. A float16 mean of 70000 ones was
  NaN and a variance whose squared deviations passed 65504 was infinite.
- Operations on empty tensors no longer offset the NULL address of an empty
  buffer, which C leaves undefined: a concatenation, a reduction over an empty
  axis, a gather from one, or a matmul with an empty inner dimension did.
- Copies, elementwise operations and reductions over views whose innermost
  axes are short, such as the windows of `Nx.sliding_window`, run 4 to 5 times
  faster on the CPU, `Nx.combine_patches` about 3 times and `Nx_wide.sum` 1.6
  times. A float sum of fewer than 8 terms per output over such a view adds
  its terms in order and may round differently.
- `Nx_device.Buffer.borrow` of a `Device_local` device's `Nx_device.signal_word`
  succeeds on the other devices of its machine, which map it as the device's
  pinned memory. It was refused, so one vendor's queue could not wait on
  another's work, as AMD's on NV's.
- `Nx_device.Profile` profiles nest and overlap, from any domains: each sees
  the events recorded while it is taken, so profiling a program no longer
  changes it. `Profile.start` no longer raises while another profile is
  taken. The new `Profile.take` profiles a function and cannot leave its
  profile taken. `Profile.counters` and `traced` answer for every profile
  taken, and each profile receives the counters and traces it asks for.
- **Breaking:** every index is an array of positions whose shape replaces its
  axis. `Nx.take ~axis` gives the axis its indices' shape, so a 0-d index
  drops it and an n-d one keeps its shape; it gave one axis of their count.
  The new index `T p` selects the positions held in a tensor in `slice` and
  `set`, where repeated positions keep the last write and positions outside
  the axis read zero and drop their write. `Nx.shrink` is gone for `slice` of
  `R`s, which cuts a range to its axis where `shrink` raised, and
  `Nx.compress ~axis ~condition` for `slice` with `M condition` at `axis`.
- **Breaking:** `Nx_dtype.kind dt` is the kind of number a dtype holds,
  `Float`, `Complex`, `Signed`, `Unsigned` or `Boolean`, and its `Float` arm
  makes a tensor of any dtype a float tensor. `Nx_dtype.is_float`,
  `is_complex`, `is_int` and `is_uint` are `Nx_dtype.is Float`, `is Complex`,
  a match on `Signed | Unsigned`, and `is Unsigned`.
- Add `Nx_device.Program.entry`, the address of a C function that a host
  program calls to call another host program, split as `Program.call` splits
  it, and that profiles record as a call.
- `Nx.logsumexp` and `logmeanexp` of a lane whose elements are all `-inf` are
  `-inf`, and of one holding `+inf` are `+inf`; both were NaN.
- Add the incomplete beta family: `Nx.betainc`, `Nx.betaincc`,
  `Nx.log_betainc` and `Nx.log_betaincc`, the beta distribution's laws,
  within stated bounds for shapes up to 2^20; the logarithms stay finite
  where a tail underflows. At a shape of 0 they take their limits.
- `Nx.diagonal` is built from movements, so its derivative under `Rune.jit`
  reads the cotangent in place. As a gather, its derivative was a reduction
  for every element of the matrix, which recomputed what the diagonal was
  read from, such as a Cholesky factor, for every term.
- Add the incomplete gamma family: `Nx.gammainc`, `Nx.gammaincc`,
  `Nx.log_gammainc`, `Nx.log_gammaincc`, `Nx.gammaincinv` and
  `Nx.gammainccinv`, the gamma distribution's laws, within stated bounds for
  concentrations up to 2^20; the logarithms stay finite where a tail underflows.
- **Breaking:** `cholesky`, `solve`, `inv` and `matrix_power` no longer raise
  `Invalid_argument` on a matrix that is not positive-definite or is singular,
  and `tensorsolve` and `tensorinv` no longer fall back to `pinv`. A matrix on
  which a linear-algebra function is undefined, including one holding NaN or an
  infinity given to `svd`, `eig` or `eigh`, has NaN in every element of its
  results, and the other matrices of its batch are unaffected. Where a raise is
  wanted, `Nx.check` that `all (isfinite r)`. `solve` and `inv` count only an
  exact zero pivot as singular, so `solve (s · a) b` is `solve a b / s` at any
  scale.
- **Breaking:** `matrix_rank` and `lstsq`'s `rank` are `Nx.int32_t` of the
  batch shape, one rank per matrix, where they were an `int` that summed the
  ranks of a batch, read back to the host. A matrix on which `svd` fails has
  rank `-1`, its rank being undefined.
- `cond` of a singular matrix is `infinity` under every norm, where `` `Two ``
  gave about `1 / ε` and `` `One `` and `` `Inf `` NaN; NaN is left for a
  matrix holding NaN or an infinity. `` `One `` and `` `Inf `` take each
  matrix's norm, where they took the largest of the batch.
- `pinv`, `cond`, `matrix_rank` and `lstsq` take each matrix's cutoff from its
  own largest singular value, where they read the largest of the whole batch:
  one matrix holding NaN gave every other a `pinv` of zeros and a `cond` of
  NaN, and `cond` divided the batch's largest by each matrix's smallest.
  `pinv` and `cond` read nothing back, so they compile under `Rune.jit`.
- **Breaking:** `lstsq` gives the least-squares solution of least norm for a
  matrix of any shape and rank, through its singular values, and `rcond` is
  relative to each matrix's largest singular value, defaulting to
  `max(m, n) · ε`. A tall matrix of rank below its column count gave NaN, and
  a wide one took as zero the singular values below `max(m, n)² · ε · σ²`, `σ`
  the largest.
- **Breaking:** `Nx_device.timeout`, `Nx_device.set_timeout` and
  `Nx_device.Driver.default_timeout` are removed: a wait lasts until the work
  signals, the driver reports a fault, or Ctrl-C (`Sys.Break`) interrupts it. A
  still timeline used to lose a healthy GPU after 30 s, such as one whose
  kernels ran long or waited for another process's. Under AMD's and NVIDIA's
  PCI interfaces, where the process drives the GPU, 30 s without progress is a
  hang.
- An NVIDIA channel that the driver stops, such as for a semaphore that
  faults, loses the device with the driver's error code, where it read as a
  hang after the timeout.
- **Breaking:** for driver authors, `Nx_device.Driver.sleep` names a sleep,
  which takes how long the timeline stayed still (`~still`), `signal`'s `wait`
  takes `~ms`, and `Driver.wait ~sleep ~timeline ready` waits for any
  condition while the device's sleep reports its faults.
- **Breaking:** a command buffer that Metal fails, such as one macOS ends for
  keeping the GPU from the display, loses the device with Metal's reason: it
  signaled its value as if complete. Submitters hand each command buffer to
  `Nx_metal_device.signaler` instead of encoding a signal on the event, and
  `Nx_metal_device.event` is removed.
- `Nx.take` of whole rows of `int4`, `uint4` and `bit` values on the host moves
  their bytes: a take of 1000 rows of a uint4 `[| 1000; 10000 |]` table takes
  0.8 ms instead of 48 ms, and the eager MXFP4 expert product it serves is no
  slower than over `uint8` codes.
- Add `Nx.Rng.peek`, the key `Nx.Rng.next_key` would return, taking none: the
  next draw returns it too.
- Add `Nx.Rng.next_root`, which takes the next key's place in the scope now
  and returns a function that computes that key: the draws after it are those
  after `Nx.Rng.next_key`, and no key is computed until the function is called.

- Add `Nx_quant.mxfp4_blocks`, which reads a GGUF file's MXFP4 tensor as views
  of its blocks: no byte is copied at load, and a compiled product reads each
  byte once.
- **Breaking:** `Nx_quant.Mxfp4`'s `codes` are `Nx.uint4_t` of shape
  `[| ...; n; k / 32; 2; 16 |]`, one code per value, where they were the
  checkpoint's `uint8` bytes. `Nx_quant.mxfp4` still takes those bytes.
- `Nx.bitcast` of a placed value is a view of its storage wherever the host's
  bitcast is one, so it runs on devices with no eager kernels, such as Metal,
  where it raised. `Nx.Repr.Placed.v` reads a storage's bytes as its dtype,
  where it required the dtype's format, and refuses a buffer not aligned to it
  and a bool over storage of another format.
- Add `Nx.Rng.with_root r f`, a key scope rooted at `r ()`, which runs at the
  first draw inside `f`, in the scope around. A scope that draws nothing takes
  no key, so the draws after it are unchanged. A root that raises raises at the
  draw, inside `f`. `with_key k` is `with_root (fun () -> k)`.

- Add `nx.wide`, double-word numbers: `Nx_wide.t` holds each number as two
  floats whose sum carries 106 significand bits at float64 and 48 at
  float32, with `add`, `sub`, `mul` and `div` of proven relative bounds, an
  exact `floor`, exact comparisons and a tree `sum`.
- **Breaking:** `Nx.check s ok data fail` carries data: it raises the
  exception `fail` builds from the first failing index and each leaf of `data`
  there, so a message can read a computed value. A check with no data passes
  `Nx.Ptree.unit` and `()`.
- **Breaking:** `Nx.Op.Check` carries `data` and `fail` in place of `msg`.
- `Nx.Rng`'s parameter refusals print the offending value, as in
  `Nx.Rng.gamma: concentration at [3] is -1, not in (0, inf)`. A sampler of
  two parameters checks each in turn.
- Add `Nx.i0e` and `Nx.i1e`, `e^-|x|` times the modified Bessel functions
  `I₀` and `I₁`, within 8 ulps: the von Mises distribution's normaliser and
  mean resultant length, finite where `I₀` overflows.
- `Nx.logspace` and `Nx.geomspace` compute each value in float64 and round it
  once to the dtype, and `Nx.geomspace` ends exactly on `start` and `stop`: at
  float32 they gave `9.999999` and `100.00001` for powers of ten.
- `Nx.linspace` ends exactly on `stop` with its endpoint, at every float dtype
  and count; its last value could miss `stop` by a rounding, and a range wider
  than the largest finite float gave infinities and NaN.
- `Nx.pp`, `Nx.to_string` and `Nx.print` print each float as the fewest digits
  that round to it at its dtype: a float32 `1.0000001` printed as `1`, and
  `0.1 +. 0.2` as `0.3`. NaN prints as `nan` on every platform, where glibc
  printed `-nan`. Error messages that name a float print it the same way.
- `Nx_dtype.precision`, `Nx_dtype.epsilon`, `Nx_dtype.min_normal` and
  `Nx_dtype.max_finite` give the significand width in bits, the gap above one,
  the least positive normal value and the largest finite value of each float
  dtype.
- An empty cut of a split axis of a placed value (`Nx.slice` to an empty range)
  is a view on the device of the shard where it starts. It was refused as
  moving elements between devices when each device held a single row, and so
  was a reshape of a placed value with an empty axis before its split one.
- **Breaking:** an integer `Nx.mean` is the exact mean rounded toward zero; it
  divided a wrapped sum by a count wrapped to the dtype, so an `int8` mean of
  200 threes was `-1`, and it raises past the count it computes exactly.
  `Nx.var`, `Nx.std` and `Nx.standardize` without a variance refuse integer
  tensors, whose variance need not fit their dtype.
- Eager operations queued back to back on an AMD GPU no longer stall it until
  the previous one's completion reaches memory: 20 dependent `Nx.add`s of 4K
  elements take 0.14 ms instead of 0.94 ms, and an `Nx.Rng.uniform` of 4K
  elements 0.32 ms instead of 0.73 ms.
- `Nx.erfinv` refines its guess with one Newton step on `erf` in the
  centre and on `erfc` in the tails, where it ran on `erf` near 1: its float64
  error near ±1 falls from about 135 ulps to 1.
- Add `Nx.erfc`, `Nx.ndtr`, `Nx.log_ndtr`, `Nx.ndtri`, `Nx.lgamma`,
  `Nx.digamma` and `Nx.lbeta`, the laws of the normal, gamma and beta
  distributions, each within a stated bound of its correctly rounded value and
  total on its domain.
- `Nx.erfinv` is within `4 + 8κ` ulps, κ its condition number, as `nx.mli`
  states.
- **Breaking:** `Nx.erf` takes and returns float tensors only; an integer or
  complex `erf` was refused at run time.
- Fix `Nx.erfinv` and `Nx.hypot` at `float16`, `bfloat16` and the float8
  dtypes, which computed at the narrow dtype: they compute at `float32` and
  round once, as every elementwise function does. `erfinv`'s first
  coefficient was below float16's least subnormal.
- Programs that link `nx.amd` carry 5.8 MB of AMD kernels instead of 8.7 MB,
  and their first eager operations on an AMD GPU load fewer code objects: the
  first use of 18 float32 operations spends about 9 ms loading instead of
  12 ms.
- `Nx_device.Program.load` finds a function of a binary its device holds
  without reading the binary. It hashed the whole binary on each call: 27 ms
  for each function of a 16 MiB binary.
- AMD GPUs of the gfx12 generation take sliding windows eagerly:
  `Nx.extract_patches` and `Nx.combine_patches` run on the GPU, bit for bit as
  on the host, overlapping windows summed in the host's order.
- AMD GPUs of the gfx12 generation scatter eagerly: `Nx.scatter` under `Set`,
  `Add`, `Max` and `Min` runs on the GPU, the last update winning under `Set`,
  an update outside the axis dropped, and the extremes and integer sums bit for
  bit as on the host. Narrow floats sum at float32 and round once.
- Eager operations on AMD devices allocate about half the OCaml memory they
  did and launch sooner: an `Nx.add` of 4K elements takes 33 us instead of
  38 us, and an `Nx.sort` of 1M elements allocates 76k words instead of
  238k.
- AMD GPUs of the gfx12 generation sort eagerly: `Nx.sort` and `Nx.argsort` run
  on the GPU for every dtype but the complex and sub-byte ones, stable in both
  directions, NaN last and `-0` below `+0`, bit for bit as on the host.
- AMD GPUs of the gfx12 generation hash with Threefry eagerly, so an `Nx.Rng`
  key placed on an AMD device splits, folds in and draws on the GPU, bit for bit
  as on the host, where `Nx.Rng.split` raised "no threefry".
- AMD GPUs of the gfx12 generation run scans eagerly: `Nx.cumsum`,
  `Nx.cumprod`, `Nx.cummax` and `Nx.cummin` run on the GPU, the running
  extremes and integer results bit for bit as on the host, running sums from
  `+0`. A running sum of 16M float32 elements takes 0.39 ms against 18 ms.
- AMD GPUs of the gfx12 generation pad, concatenate, gather and write windows
  eagerly: `Nx.pad`, `Nx.concatenate`, `Nx.take`, `Nx.take_along_axis` and
  `Nx.set` at a window run on the GPU for every dtype but the complex and
  sub-byte ones, bit for bit as on the host, an index out of range reading 0.
- AMD GPUs of the gfx12 generation multiply matrices eagerly: `Nx.matmul` runs
  on the GPU for every numeric dtype, operands of any layout and broadcast
  batches, integers bit for bit as on the host, floats within `k·eps·Σ|a b|`.
  A float32 product of 4,096 rows takes 13 ms against 284 ms on the host.
- AMD GPUs of the gfx12 generation reduce eagerly: `Nx.sum`, `Nx.prod`,
  `Nx.max`, `Nx.min`, `Nx.argmax` and `Nx.argmin` (and `Nx.all`, `Nx.any`) run
  on the GPU, the extremes and integer results bit for bit as on the host, float
  sums from `+0` within `n·eps·Σ|x|` of the exact sum.
- AMD GPUs of the gfx12 generation compute the elementwise operations eagerly:
  the unary functions (`Nx.neg`, `Nx.sqrt`, `Nx.exp`, `Nx.sin`, the
  roundings...), the binary ones (`Nx.add`, `Nx.div`, `Nx.pow`, `Nx.maximum`,
  the bitwise ones...), the comparisons, `Nx.fma` and `Nx.where`, bit for bit
  as on the host but for the transcendental functions, which keep `nx.mli`'s
  ulp bounds.
- NV devices queue a small copy in 5.1 µs instead of 8.9 µs, allocating 750
  words instead of 3,100: the runtime scanned every command segment its ring
  held on each copy, where it now tracks the segments in ring order.
- AMD GPUs of the gfx12 generation (gfx1200, gfx1201) copy and cast eagerly:
  `Nx.copy`, `Nx.contiguous` and `Nx.cast` on `Nx_amd` devices run on the GPU,
  bit for bit as on the host, for every dtype but the complex ones, `int4`,
  `uint4` and `bit`. The kernels are compiled ahead of time for
  `gfx12-generic` and carried by `nx.amd`; every other operation, dtype or GPU
  still raises naming `Rune.jit`.
- **Breaking:** a backend states the memories it owns (`Nx_backend.S.owns`),
  those its library computes its devices with, and `Nx.Device.name` shows a
  backend only on a memory it does not own: `Nx_amd` devices are `AMD:0`, and a
  backend that includes nx.cpu's kernels says `let owns _ = false` to keep its
  name.
- Add `Nx_amd_device.launch` and its `dispatch` record: kernels in order on an
  AMD device's compute queue, as one submission.
- Add `Nx_amd_packet.Pm4.dispatch`, the register writes, user SGPRs and
  dispatch of a code object's kernel (`Nx_amd_code_object.kernel`) on a PM4
  queue, which `Nx_amd_device.launch` and tolk's AMD queues share, and the
  term `Or`.
- Add `Nx_array.View.coalesce`, which merges the axes that a set of operands of
  one shape lets a kernel walk as one.
- `nx.mli` states the accuracy of the transcendental functions: each of
  `exp`, `log`, `log1p`, `expm1`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `tanh`, `erf`, `pow` and `atan2` gives its ulp bound at
  `float32` and `float64`, 1 ulp at the narrower floats, and Annex F's special
  values and signed zeros exactly, on every backend.
- An AMD device refuses a code object compiled for another GPU
  (`Nx_device.Program.load` is `Error "AMD: a code object for gfx1100; the GPU
  is gfx1201"`) instead of loading it to fault or compute garbage. It loads
  one for a generic processor that lists the GPU, such as `gfx12-generic`.
  `Nx_amd_code_object.target` and `runs_on` say which, and `Nx_device_elf.t`
  gains `flags` and `abi_version`.
- **Breaking:** `Nx_quant.apply` takes no `?ids`. Route positions to a stack
  of experts with `Nx.map_segments`, whose function applies
  `Nx_quant.apply (Nx_quant.take ~axis:0 ~indices:owners w) rows`, so a whole
  expert feed-forward sorts its positions once.
- **Breaking:** `Nx_quant.place` and `Nx_quant.walk` are removed: use
  `Nx.Ptree.place Nx_quant.ptree` and `Nx.Ptree.Walk.structure Nx_quant.ptree`.
  The constructors and `Nx.Ptree.place Nx_quant.ptree` refuse a split inside a
  block.
- Add `Nx_quant.take ~axis ~indices w`, a weight's rows or matrices gathered
  from its packed bytes, so an eager caller reads a large table's rows without
  decoding the table.
- `Nx_quant.dequant Nx.bfloat16` decodes MXFP4 by a bit shift, with the same
  values, and a product over it runs 1.5x faster on the host. `Nx_quant.apply`
  of a `float64` input accumulates at float64, where it rounded the input to
  float32.
- Add `Nx.map_segments ~segments ids f x`: each position's row of `x` goes
  through `f` with the owner its id names, in one call on the rows grouped by
  id, so a stack of `segments` matrices is read once per block of rows. An id
  out of range gives zeros, and its row never reaches `f`.
- Fix waits on AMD GPUs under the `amdgpu` driver that waited up to 200 ms
  past a run's end once they slept: the driver ignored the GPU's interrupt
  until the host armed its event's slot, which `Nx_amd_device` now does.
- Add `nx.amd.packet`: the PM4, AQL and SDMA packets of AMD GPUs' queues and
  the GC registers they write, polymorphic in the values they place
  (`Nx_amd_packet.Pm4`, `Aql`, `Sdma`, `Gc`). `nx.amd.device`'s copies and
  tolk's AMD queues encode their packets with it.
- Add `nx.amd.code_object`: AMD GPU code objects relocated as a loader
  relocates them (`Nx_amd_code_object.of_string`, `image`), and each kernel's
  descriptor (`kernels`, `kernel`). `nx.amd.device` and tolk read descriptors
  through it.
- Add `nx.nv.packet`, which encodes NVIDIA GPUs' commands once for the
  runtime and for tolk: methods, ring entries and launch descriptors
  (`Nx_nv_packet.Methods`, `Gpfifo`, `Qmd`), as data around values of any
  type, with the arithmetic on them recorded as terms (`Nx_nv_packet.term`).
  nx.nv.device's channels and tolk's NV queues write through it.

- Add `nx.nv.cubin`, which reads cubins: their image laid out and relocated
  (`Nx_nv_cubin.image`, `relocate`) and what a launch of a kernel needs, its
  code, registers, stack, shared memory and constant banks
  (`Nx_nv_cubin.kernel`). nx.nv.device and tolk read cubins through it.

- `Nx.take` and `Nx.take_along_axis` on the CPU walk their output without
  recomputing each element's position, and `Nx.take ~axis` copies the
  contiguous elements after `axis` at one index read: a take along axis 1 of
  `[8; 1024; 256]` runs 50 times faster, a `take_along_axis` of
  `[2000; 2000]` 7 to 9 times.
- `int4` and `uint4` compute as integers modulo 16, as every integer dtype
  computes modulo its width, where every function but casts and moves raised.
- `Nx.mod_` by zero is its dividend at every integer width, where it was 0, so
  `a = b * div a b + mod_ a b` holds for every `b`. `Nx.div` of a signed
  type's least value by -1 is that value.
- `Nx.lshift` and `Nx.rshift` by the width or more give 0, or -1 below zero
  for `rshift`: `lshift` of an `int64` by 64 returned its operand.
- Add the `bit` dtype (`Nx.bit`, `Nx.bit_t`): booleans eight to a byte, for
  keeping large masks. Every function that takes `bool` outside a condition
  takes `bit`; a condition stays `bool`, as `Nx.cast Nx.bool m`. `Nx.count`
  counts the `true` elements of a `bool` or `bit` mask. `Nx.any`, `Nx.all`,
  `Nx.max` and `Nx.min` of a `bit` mask read it a 64-bit word at a time and
  stop at the first word that decides them.
- **Breaking:** remove `Nx.itemsize`, whose bytes per element miscount every
  sub-byte dtype; `Nx.nbytes` gives sizes and `Nx_dtype.Scalar.bitsize`
  widths.
- `Nx.bitcast` counts widths in bits, so `bit`, `int4` and `uint4` read as
  their packed bytes, where it refused the 4-bit dtypes. `Nx.nbytes` counts
  bits too: an `int4` tensor reported twice its bytes.
- `Nx_io.save_npy`, `save_npz` and `save_safetensors` refuse `bit`, naming
  `Nx.cast Nx.bool` and the packed bytes as the ways to save it; `save_txt`
  writes its booleans.
- Add `Nx.Rng.binomial k n p`, exact by inversion below a mean of 10, by
  transformed rejection over a fixed 18 rounds above, and
  `Nx.Rng.von_mises k concentration`, whose concentration 0 is the uniform
  circle. Both take tensor parameters, check their domains and compile.
- Add `Nx.Ptree.dot`, `norm`, `scale` and `axpy`: the linear algebra of a
  structure's float tensors taken as one vector, in tensor arithmetic that
  traces and batches. Other tensors, such as counters and keys, are carried.
- **Breaking:** `Nx.relu` is removed; it lives in `Kaun.Fn.relu`. Outside
  kaun, write `Nx.maximum_s x 0.`, whose derivative at 0 is 1/2.
- `Nx.softmax` and `Nx.log_softmax` no longer overflow for a negative `~scale`
  (they returned NaN), and `Nx.softmax` raises `Invalid_argument` on an axis
  out of bounds, where an axis below `-rank` selected another axis.
- AMD GPUs under the kernel driver load a program in about 0.3 ms instead of
  8 to 12 ms, and freeing host memory mapped for the GPU no longer stalls the
  next work on every queue of the process for 5 to 10 ms.
- `Nx.Rng` samplers raise `Invalid_argument` on a parameter outside its
  domain, NaN included, where they returned NaN or a fixed draw (`bernoulli`
  true above 1, `poisson` 0 below 0). A traced one raises when the call returns.
- `Nx_amd_device.profiling` puts a GPU under the kernel driver in its stable
  power state itself, through a context of the amdgpu driver on the render
  node the device holds, and the process keeps it until it exits: counting or
  tracing no longer needs root to run `amd-smi set -l stable_std` first. It
  fails only when another process holds the state.
- A lost device (`Nx_device.Lost`) can be opened again: `Nx_cuda.get`,
  `Nx_metal.get`, `Nx_amd.get`, `Nx_nv.get`, `Nx.Device.cpu` and
  `Nx_remote_device.connect` give a fresh, unequal device, and the PCI
  interfaces refuse. Every eager operation, read and `Nx.place` of a value a
  lost device can reach raises `Lost`, empty values and host memory it mapped
  included. New `Nx_device.lost`; `Lost` prints as `NAME lost: why`.
- A GPU over PCI (`Nx_nv`, `Nx_amd` and their runtimes under `Pci`) opens
  without root once an administrator binds it to `vfio-pci` on a machine whose
  IOMMU is on and grants its `/dev/vfio/N`: `Nx_device_support.Pci` takes it
  through VFIO, maps its BARs and configuration space from the VFIO file, and
  maps system memory for it at device addresses the process allocates, so
  neither `/proc/self/pagemap`, `mlock` privileges nor huge pages are needed.
  Unbound or no-IOMMU functions are taken through `/sys/bus/pci` as before, as
  root. A missing binding, permission or memlock limit fails naming the
  `driverctl` command, udev rule or limits file that grants it. Behind an
  IOMMU, GPUs copy to each other through host memory, `Driver.dma` refuses
  their memory, and a remote server refuses such functions. New `Pci.access`,
  `state`, `addressing`, `Sysmem.map` and the `Vfio` module; **breaking:**
  `Pci.detached` returns how the function is taken, and an unbound function an
  IOMMU translates is no longer reported detached.
- `Nx_amd_device` uploads programs' code into the GPU's own memory through the
  copy engine, whatever the size of the memory BAR, and counts it there. On a
  GPU without Resizable BAR, programs now load under the kernel driver, where
  every load failed, and over PCI their code no longer runs from uncached
  system memory. Mapped memory there is pinned memory under both interfaces.
- `Nx.unique` over rows of several words, and `Nx_ragged.ids`, merge their
  blocks' groups without reading rows across the whole input. Ids of 10⁷ text
  rows of 12 bytes take 0.7× the time.
- **Breaking:** opening a GPU over PCI changes nothing on the machine.
  `Nx_nv.get_pci`, `Nx_amd.get_pci`, their `device_pci` forms, the runtimes'
  `get ~interface:Pci` and `Nx_rdma_device.get` no longer unbind kernel
  drivers, remove sibling functions, resize BARs, download firmware, reset
  GPUs or set `vm.compact_unevictable_allowed`; they fail naming the call that
  does. New `detach`, `attach`, `reset` and `fetch_firmware` in
  `Nx_nv_device` and `Nx_amd_device`, re-exported by `Nx_nv` and `Nx_amd`, and
  `Nx_rdma_device.detach` and `attach`, each documented with its privileges
  and what persists. In `nx.device.support`, `Firmware.get` becomes `find`,
  which downloads nothing, and `fetch`; `Pci.take` takes only a function that
  `Pci.detach` detached (`Pci.detached`), and `Pci.attach` gives it back.
  `Nx_amd_device` documents that AMD-PCI holds a GPU's clocks at their highest
  state while open.
- **Breaking:** `Nx_nv_device.get` and `Nx_amd_device.get` take the
  interface, `~interface:Kernel` or `~interface:Pci`, with no default. They
  took a GPU over PCI, detaching it from its kernel driver, whenever
  `/dev/nvidiactl` or `/dev/kfd` was absent.
- `Nx_nv_device` and `Nx_amd_device` number a vendor's GPUs in PCI bus order
  under both interfaces, so `NV:i` and `NV-PCI:i` (`AMD:i`, `AMD-PCI:i`) are
  one GPU, and taking a GPU over PCI renumbers none. The kernel interface
  followed the driver's list, which is in bind order and leaves out GPUs it
  does not hold. `count` is every GPU of the vendor, whatever driver holds it,
  and takes no interface. New `Nx_device_support.Pci.address` and
  `compare_address`; `Pci.scan` sorts addresses as numbers.
- Another machine's GPUs, which are driven over PCI, are named
  `NV-PCI@HOST:PORT`, `AMD-PCI:1@HOST:PORT`, and so on.
- `Nx.array_equal` is `false` for tensors of different shapes, as it states.
  It compared tensors whose shapes broadcast, so a `[|1; 2|]` and a `[|2; 1|]`
  tensor of one value were equal.
- `Nx_quant.apply ~ids` on a prompt groups positions in blocks of up to 16,
  the most that keep the padding below half the positions, where it took
  blocks of 4, and with `bfloat16` rows an MXFP4 weight is its `bfloat16`
  values widened, which hold them exactly. A compiled product of `bfloat16`
  rows then runs on a GPU's tensor cores: gpt-oss-20b's 512-token gate and up
  product on an RTX 5000 Ada takes 2.9 ms, from 13.3 ms. A product of 512
  tokens' `float32` rows, which no tensor core takes, pays the padding: 25.7
  ms on that GPU, from 23.2 ms.
- **Breaking** for backends: a backend is a value, `Nx_backend.t`, which its
  library makes once with `Nx_backend.v (module K)` and exports, as
  `Nx_cpu.backend`. Two backends are equal only when one `v` made them, so
  backends of one name no longer collide. `Nx_cpu` exports its kernels
  through `Nx_backend.kernels Nx_cpu.backend` alone.
- `Nx.svd` and `Nx.svdvals` bidiagonalize with vectorized panel products:
  a float64 128 x 128 SVD takes 1.65 ms on one kimchi core where it took
  2.43, and 512 x 512 56 ms where it took 130.
- `Nx.qr` forms Q only over the columns its reflectors change: a float64
  256 x 256 factorization takes 1.8 ms on one kimchi core where it took 2.9.
- `Nx.matmul` shares a large product among threads by row blocks when whole
  column panels would leave threads idle: a float32 2048 x 2048 product on
  6 cores takes 25 ms where it took 33.
- `Nx.matmul` and the products built on it run on AVX2 and FMA on x86-64
  CPUs that have them, picked at run time: a float32 512 x 512 product takes
  2.1 ms on one Lion Cove core where it took 14.4, and float64 4.1 ms where
  it took 11.8.
- A `Nx_device_support.Remote` command after a failed posted one reports the
  server's reason, such as "no memory of this connection", where it could
  report `connection lost: Broken pipe` once the server had closed the stream.
- Add GGUF's Q8_0, Q4_K and Q6_K block formats to `Nx_quant`: `Nx_quant.q8_0`,
  `q4_k` and `q6_k` take a tensor's bytes as `Nx_io.load_gguf` loads them.
  `dequant` gives ggml's values bit for bit, and `apply`, `~ids` included,
  multiplies by them, eagerly and under `Rune.jit`.
- `Nx.bitcast` on a device of a view at an offset, such as a slice, read the
  storage from its start: the result now keeps the view, as on the host.
- `Nx_ragged.ids` builds each round's words in the elements' memory order,
  which tells rows apart as well as the sort order does, and reads every row
  in place on its first round, where it reordered each word's bytes and
  gathered the rows' bounds.
- Add `Nx_array.View.within`, whether a view reaches only the first `n`
  positions of its storage, exact for every offset, stride and shape.
  `Nx_array.Elements.gather` and `contiguous` check views with it: a view
  whose positions or element count wrap past `max_int` read outside the
  buffer.
- **Breaking:** `Nx.t` is abstract, where it was `Nx_effect.t`, and the
  `nx.effect` library is gone: transformations match on `Nx.Op` and read and
  build representations with `Nx.Repr`.
- `Nx.take` along one axis of elements of 1 to 8 bytes moves them in a loop
  typed by their width, where it called a function per element, so
  `Nx_ragged.take` and the gathers of indices and bytes it is made of run
  faster.
- `Nx.take ~axis:0` of whole rows copies each row, now also from data whose
  rows overlap or lie apart, such as a window view, where it moved one element
  at a time: `Nx_ragged.ids` of 10⁷ twelve-byte strings takes 167 ms where it
  took 297 ms.
- `Nx.arange` past 1024 values adds a column of row starts to a row of
  offsets, one elementwise pass, where it summed every value in a running sum:
  10⁷ int64 values take 1.3 ms where they took about 6 ms.
- `Nx.scatter ~mode:`Add` of integers along one axis, and so
  `Nx.reduce_segments `Add` and `Nx.unique`'s counts, adds in a loop typed by
  the elements' width, where it called a function per update.
- `Nx.unique` groups its keys in hash tables where it sorted them, on every
  core, and `Nx_ragged.ids` groups each round's rows the same way: both now
  cost an expected pass over their keys, and their results are unchanged.
  `unique` of 10⁷ int64 keys takes 39 ms with 10² distinct values and 97 ms
  with 10⁶, where it took 316 ms and 513 ms, and `Nx_ragged.ids` of 10⁷
  twelve-byte strings 283 ms, where it took 1.5 s (Apple M1 Max).
- Add `Nx.Op.Group { by; x }`, the rows of a uint64 matrix numbered in order
  of first appearance, ids that depend on the rows alone. `Rune.jit` refuses
  it with a message that starts with `by`, as every caller reads the number of
  groups, and `Rune.vmap` numbers each lane's rows on their own.
  **Breaking** for backends: `Nx_backend.S` gains `group`.
- **Breaking:** files of named tensors are read and written as
  `Nx_io.Archive.t`, an immutable collection with distinct, non-empty names,
  in place of the mutable `Nx_io.archive` table. `load_npz`,
  `load_safetensors` and `Nx_io.Gguf.tensors` return one, and `save_npz` and
  `save_safetensors` take one; `Archive.of_list` and `union` build one.
  `Nx_io.Gguf.t` is abstract: `Gguf.info` replaces `tensor_infos`.
- Add `Nx_io.Archive.of_value` and `to_value`, which save and read back a
  value through its `Nx.Ptree.t`. Reading back fails, naming the entry, on a
  missing entry, another dtype or shape, or an entry under the structure's
  fields that no tensor names, so a 12-block model refuses a 24-block file.
  `Archive.tensor` and `float` read one entry by name; only `float` converts.
- Add `Nx.Ptree.field`, which puts a structure under a name, and
  `Nx.Ptree.prefix`, the path a structure's fields put it under.
- `save_safetensors` and `save_npz` name the entry whose dtype they refuse.
- `Nx_quant.apply ~ids` with at least twice as many positions as experts sorts
  them by expert and multiplies each expert once per pair of its positions,
  where it read every expert's weights once per position. A prompt's compiled
  expert products run about 4 times faster on CUDA, and faster on Metal and the
  host.
- **Breaking:** `Nx_quant.apply` and `Nx_quant.dequant` are compositions of
  `Nx` operations everywhere, and `Nx_quant.Effect` is removed. Their results
  live where their operands join, no longer on the host. Run eagerly,
  `apply ~ids` holds one float32 matrix per position, where it decoded a chunk
  at a time: run large weights under `Rune.jit`, which decodes inside the
  product.
- Add `Nx.Op.map_operands`, which rebuilds an operation over other operands,
  so an interpreter that substitutes values need not match every operation.
- `Nx_device.Driver.device` raises `Invalid_argument` on a name already made
  on its machine, so two callers cannot make two memories that print alike,
  such as a second `"CPU:1"`.
- `Nx_cuda_device` numbers GPUs in PCI bus order, read with
  `cuDeviceGetPCIBusId`, whatever `CUDA_DEVICE_ORDER` says, so `CUDA:i` is
  the GPU `nvidia-smi` numbers `i` when the driver sees every GPU.
- NVIDIA and AMD GPUs taken from their kernel driver (`interface:Pci`) are
  named `NV-PCI:i` and `AMD-PCI:i`, so a name says which interface reaches
  the GPU.
- **Breaking:** a device (`Nx.Device.t`) is a memory and the backend that
  computes on it eagerly, if any, as a plain value: `Nx.Device.equal`
  compares the two. `Nx.Device.host` and the test devices `Nx.Device.cpu k`
  compute with nx.cpu; a GPU computes eagerly with nothing until
  `Nx.Device.with_backend b d` pairs it with a backend `b`. Placing a value
  between devices over one memory is a view, so it changes only who computes.
  `Nx.Device.make` and `memory` cross to the `Nx_device.t` nx.device opens.
- Add `nx.metal`, `nx.cuda`, `nx.nv` and `nx.amd`, whose `device i` and
  `get i` open a vendor's GPU as an `Nx.Device.t`. `Nx_nv` and `Nx_amd` add
  `device_pci` and `get_pci`, the only calls that take a GPU from its kernel
  driver. nx links no vendor runtime: a program links the vendors it uses.
- **Breaking:** who computes an eager operation is read off its operands: the
  backend of their devices computes, in every domain and under every
  transformation, and a host operand joins the placed ones for the call. A
  device without a backend raises `Invalid_argument` before any work, naming
  `Rune.jit`, `Nx.Device.with_backend` and `Nx.Placement.host`, and a kernel
  a backend refuses raises `Invalid_argument` naming the backend, the device,
  the operation and the reason; nothing falls through to another backend.
  Operands on two devices over one memory raise, naming `Nx.place`.
- **Breaking:** `Nx.Placement.on d` is one device, `Nx.Placement.replicated
  ds` a full copy on each, and a placement names each memory once.
- **Breaking:** a constant (`Nx.full`, `zeros`, `ones`, `scalar` and their
  `_like` forms) is one element on each device of where it is made, the
  host's included, expanded to its shape as a view: it needs no kernel on any
  device, and a compiled call folds it. `Nx.copy` gives it storage of its
  own, which a compiled call can then lend to a result.
- Operations and `Nx.place` raise `Nx_device.Out_of_memory` when a device
  cannot allocate their result.
- Add `Nx.Ptree.place`, which places every tensor of a structure.
- The host's thread pool, which runs the blocks of `Nx_device.Program.call`
  and nx.cpu's parallel kernels, keeps a thread spinning for up to 100 us after
  the last job it took part in, instead of parking it after each. A compiled
  program's kernels, launched microseconds apart, no longer wake every thread:
  a jitted RNN step of 1,609 launches went from 52 to 29 ms on an M1 Max. The
  threads that a burst's jobs leave out park, so the burst spends no other core.
- A NaN result of `Nx.sum`, `Nx.prod`, `Nx.cumsum` or `Nx.cumprod` on the host
  is the first NaN term in index order along the reduced axes. Which NaN
  survived depended on the layout, the length and the path the reduction took.
- A NaN result of `Nx.add`, `Nx.sub`, `Nx.mul`, `Nx.div` or `Nx.fma` on the
  host is the first NaN operand, or for complex numbers the first NaN part.
  Given two NaNs, float `add`, `mul` and `fma` and complex `add` and `div`
  returned one or the other by the element's position, so bits changed with
  batching.
- `Nx.fma` with one operand broadcast from a single element runs about 3 times
  faster on the host. That operand is loaded once and the loop vectorizes,
  where it took the strided scalar loop before.
- `Nx.imag`, `Nx.angle` and `Nx.conjugate` keep non-finite components and
  signed zeros: `imag` of `1 + ∞i` was NaN, and `conjugate` of `1 + 0i` had an
  imaginary part of `0.`. Components are now read through `Nx.bitcast`.
- Add `Nx_io.load_gguf`, which loads a GGUF file (versions 2 and 3): its
  metadata as `Nx_io.Gguf.value`s and its tensors as values on the disk, as
  `load_safetensors` does. A block-quantized tensor loads as its bytes.
- `Nx.reshape` and `Nx.ravel` copy a tensor whose layout no view can express,
  such as a flattened transpose. They raised and asked for `Nx.contiguous`,
  while the same reshape under `Rune.jit` computed.
- `Nx_device.Buffer.create ~memory:Mapped` gives pinned memory when the
  window or the device's own memory cannot hold the buffer, and keeps the cache.
  It raised `Out_of_memory` once the device's own memory was full.
- `Nx_device.submit` waits on the host only for the work that touched the
  memory its buffers reach, as `Submission.waits` documents. It waited for all
  work of each device whose memory they reach, which serialized unrelated work.
- `Nx.complex` keeps each component as given: an infinite or NaN `~im` made
  the real part NaN, and a `-0.` component became `0.`. It now writes both
  components directly instead of adding `im * i` to `re`.
- `Nx.take` without an axis returns a value of the indices' shape. It raised
  for scalar or multi-dimensional indices.
- **Breaking:** `Nx_amd_device.counting` is `Nx_amd_device.profiling`, which
  also traces: each shader engine writes a thread trace of each kernel run, read
  as a `Profile.Trace` per engine, and on GFX11 and GFX12 each wave is a span on
  the GPU's timeline. `Nx_amd_device.Thread_trace` reads a trace's waves.
- `Nx_device.Profile.start ~trace` asks the devices that trace for a thread
  trace of each run of a program: a `Profile.Trace` event holds the raw trace of
  a part of the device, timed by its run, and `Profile.traced` tells the
  libraries that encode work, which keep encoded work for each value of it.
- Add `Nx.reduce_ranges`, the sum, maximum or minimum of each range of rows
  from `lo` to `hi`, for rolling and growing windows. Bounds clip to the rows,
  and a sum holds its range's terms alone. It costs `O((n + m) log n)` for `n`
  rows and `m` ranges, reads no value, and compiles and maps under `Rune.jit`
  and `Rune.vmap`.
- `Nx_device.staging h` is the host's staging memory, which `Buffer.copy`
  and the libraries that submit work now share: a copy fills a slot only once
  the queued work that uses it is done.
- `Nx_device.Buffer.reach d b access` gives device work its way to `b`: a
  borrow where `d` maps `b`, and for host memory under 64 KiB that `d` cannot
  map, such as a small tensor on a CUDA, AMD or NV GPU, a staged buffer that
  `submit` fills before the work and copies back after it when it writes.
- **Breaking:** pinned memory counts in no device budget. It is the host's RAM,
  so a GPU's `Nx_device.budget` counted it against VRAM it never used; its
  driver's refusal is its limit. `budget` covers the device's own memory,
  mapped memory included.
- **Breaking:** `Driver.memory`'s `Device_local.mapped` gives the size of the
  window mapped memory lies in. Mapped memory and loaded programs' code are held
  within that window, and `Stats.allocated` counts loaded code, which the
  collector now paces by the room left in its memory.
- A compiled `Nx_quant.apply` or `Nx_quant.dequant` decodes mxfp4 codes and
  scales with integer operations on their bytes. It looked them up in tables
  by an `int64` index, which a mixture of experts stored at 8 bytes per code
  byte between gathering the experts and multiplying.
- `nx` no longer depends on `bytesrw`. `Nx_io.gunzip` still decompresses in
  bounded memory, through a `Compress_deflate.Decoder`.
- Add `Nx.check ok msg`, which raises `Invalid_argument (msg i)` at the index
  of `ok`'s first false element. Inside a compiled function it reads nothing:
  the call raises when it returns, also from a staged scan or under `vmap`.
- Add `Nx.fma`, `a * b + c` rounded once on `float32` and `float64` and
  modular on integers, eagerly and compiled alike. `float16`, `bfloat16` and
  the `float8` dtypes take the `float32` multiply-add and round it.
- Add `Nx.log1p` and `Nx.expm1`, the C library's on the host, and compiled
  compositions within 4 ulps. `asinh`, `acosh` and `atanh` call
  `log1p` in place of a composition that lost accuracy near zero.
- Add `Nx.histogram`, counts or weighted sums of points in the cells of
  explicit edges in any number of dimensions, which compiles.
- Add `Nx.associative_scan`, the inclusive scan of an associative elementwise
  function over a structure, whose outputs keep their bits when the input
  grows, and `Nx.ewma`, the exponentially weighted moving average built on it.
  The errors of `cumsum`, `cumprod`, `cummax` and `cummin` name them, where
  they named an internal `associative_scan`.
- `Nx_device.Program.call ~split` runs a host program's iterations in blocks
  on the host's threads, and `Nx_device.Program.workers` is their number. Eager
  CPU kernels and compiled ones share this one pool.
- **Breaking:** `Nx_device.Profile.Counters` carries the counted run's
  `start` and `stop` in place of the time they were read, and
  `Profile.output_chrome_trace` shows each run's counters on its device's
  `counters` lane. Counters were matched to spans of the same name in order, so
  a run that went uncounted or was lost gave every later span another run's
  counts. Runs lost before a device read them are a `Profile.Overwritten` event.
- **Breaking:** `Nx_amd_device.counters` and `props.compute_units_per_array`
  are gone, and `Nx_amd_device.counting` keeps the counting of each set of
  counters, so that work encoded for one set is read whenever it runs.
- `Nx_device.Stats.bytes_in` and `bytes_out` count the copies that compiled
  work makes on CUDA, NV and AMD queues, which they missed: a compiled call
  reading a host value reported no upload. Submitters count such copies with
  `Nx_device.Submission.copied`.
- Add `Nx.Ragged`, ragged arrays as int64 offsets over a tensor of values
  (Arrow's large lists and strings): `v` and `of_lengths` check their offsets,
  `of_ids` groups rows by id and compiles, `sub`, `take`, `concat` and `map`
  transform them, `quantile` takes each row's quantiles, and `ids` and `rank`
  number rows by first appearance and by their order, byte strings as `memcmp`
  orders them.
- **Breaking:** a driver's `Driver.device ~peer` returns its mapping with how
  to unmap it, and the runtime keeps one mapping per device and memory, for
  borrows and copies alike, until the memory is released. AMD and NV kept peer
  mappings until the memory went back to their drivers, so cached memory stayed
  mapped on other GPUs.
- An NV device whose memory runs out while mapping a new allocation raises
  `Nx_device.Out_of_memory` and gives the memory back. It raised a driver
  fault, which lost the device.
- `nx.io` takes deflate from the `compress` package. Loading a deflated NPZ
  entry is about 3x faster, and the tensor uses the decompressed memory
  without a copy when its data is aligned; loading a PNG is about 2x faster.
  `Nx_io.gunzip` decompresses as it reads, in constant memory, where it
  mapped its whole input.
- Add the `nx.ragged` library: `Nx_ragged`, ragged arrays as int64 offsets over
  a tensor of values (Arrow's large lists and strings): `v` and `of_lengths`
  check their offsets, `of_ids` groups rows by id and compiles, `sub`, `take`,
  `concat` and `map` transform them, `quantile` takes each row's quantiles, and
  `ids` and `rank` number rows by first appearance and by their order, byte
  strings as `memcmp` orders them.
- Add `Nx.quantile`, linear quantiles along an axis or over a whole tensor, one
  sort for every probability, with NaN sorting last.
- CUDA modules and Metal pipelines are unloaded with their binary once no
  program of it and no launch that uses it is reachable; they were kept for the
  device's life. `Nx_device.Program.keep` ties a binary to a buffer that
  launches it, such as a word holding a function's handle.
- `Nx_metal_device.indirect_commands` releases its command buffer once its
  arguments' memory is released and the work that ran it is done. It was
  released when the arguments were collected, while queued work could still
  run it.
- `Nx_io`'s savers and `Nx_quant`'s eager kernels hold a read claim on the
  memory they read while they use it, so a compiled call on another domain
  can no longer lend it and tear a saved file.
- Reading a host value whose elements are not one run of its memory
  (`Nx.to_array`, `Nx.item` of a transposed value) holds a read claim while it
  gathers them, so a compiled call on another domain can no longer lend that
  memory and give a mix of old and new elements.
- **Breaking:** A device unloads a binary once no program of it and no buffer
  of its code (`Nx_device.Program.code`) is reachable, after the work it
  submitted until then. Devices kept every program they loaded for their life,
  so a long run that compiled many kernels never freed their code.
  `Program.load` finds a loaded binary's functions again with the same handles,
  in a new program record.
- **Breaking:** A driver's `load` takes a binary and returns a
  `Nx_device.Driver.image`: the region of its code, how it finds a function, and
  how it unloads. A driver with no memory for the code raises `Out_of_memory`,
  and `Program.load` collects unreachable programs and tries again, as
  `Buffer.create` does. NV refuses code larger than the memory the host can
  map at once, with an `Error` that names Resizable BAR.
- A complete `Nx.qr` of an m x 0 matrix and an `Nx.svd` with
  `~full_matrices:true` of a matrix with an empty dimension return the identity
  as their square orthogonal factors (Q, and U or Vt). They returned all zeros,
  which are not orthogonal, where compiled code returned the identity.
- `Nx_amd_device` counts a profile's counters on AMD GPUs: `Nx_amd_device.counters`
  lays out the GPU's counters by name, `Nx_amd_device.counting` gives the
  libraries that submit work the device's log and samples, and the device reads
  each run's values at its synchronizations. Under the amdgpu driver, counting
  needs the GPU's stable power state, which the error names how to set. `props`
  gains `compute_units_per_array`.
- `Nx_device.Profile.start ~counters` asks the devices that count to count those
  hardware counters during each run of a program: a `Profile.Counters` event
  holds a run's values, and `Profile.output_chrome_trace` adds their sums to the
  run's span. `Driver.device ~report` is how a device hands them over.
- `Nx.of_buffer` makes the value of a shape over a runtime buffer's elements
  without a copy, on the buffer's device, and `Nx.to_buffer` gives a buffer of
  exactly a value's elements in C order on its device: its own storage when the
  elements are one run of it. File formats read and write tensors through them
  instead of `Nx.Repr`.
- `Nx.shards` gives each device's buffer of a value and the view each device has
  of it, and `Nx.of_shards` makes a value at a placement over such buffers, for
  compiled calls that bind values' storage.
- **Breaking:** `Nx_device.Driver.on_free` is `Driver.depends`, which runs
  once an allocation is released and all work on it is done, its own device's
  included. A hook ran when the memory went back to its driver, so what it
  released outlived the allocation while the memory stayed cached.
- `Nx.contiguous` of a traced value is the value itself when its view is
  C-contiguous, as for any other value. It used to copy unconditionally, so
  under `Rune.jit` every call cut the program into another kernel. Use
  `Nx.copy` for a value that must be materialised.
- An allocation refused for lack of memory releases the borrows of that
  memory held by devices that run no operation, before it raises
  `Out_of_memory`. A borrow is released at its device's next operation, so a
  host buffer borrowed by a device that then stayed idle kept its memory out
  of reach.
- The garbage collector is paced by device memory: each device buffer counts
  against the room left in its device's budget, so dropped GPU buffers are
  found before the budget runs out. A device that filled its budget with
  garbage forced a full major collection on its next allocation, and one
  whose OCaml heap was small held its garbage until then.
- The live memory of host buffers is measured as each major cycle ends, in
  whichever domain ends it. It was measured by a finaliser of the domain that
  loaded `nx.device`, so while that domain was blocked, as in `Domain.join`,
  the pace stood still and the cache of collected host buffers was never
  trimmed.
- A device's released memory waits for the work that touched it, without
  blocking: each buffer of `Nx_device.submit`'s `touches` is stamped with the
  values its work signals. A GPU's buffer returns to its cache once other
  devices' work on it is done, where it was cached at once and could be reused
  while a peer still read it; a borrow is unmapped once its device's work on it
  is done, where every release waited for all of the device's work. Memory a
  lost device's unfinished work touched is retained, and the owner allocates
  on. A refused allocation collects garbage with its device free for other
  domains.
- The memory of a device buffer, a borrow's mapping or a disk file returns to
  its device from whichever domain's collection finds it unreachable. It
  returned only once the domain that made the buffer ran its finalisers again,
  so a domain that dropped buffers and then blocked, as in `Domain.join`, kept
  them.
- Disk buffers (`Nx_device.Buffer.of_file`, `create_file`) no longer hold a
  descriptor until they are collected, which ran out of descriptors
  (`EMFILE`) in loops of small file round trips. The disk keeps the descriptors
  of the 64 files it used last, and reopens a file by its path when needed,
  raising `Sys_error` if the path now names another file or one changed by
  another writer.
- `Nx.top_k` along an axis whose rows it can view no longer copies its
  selection keys first. Compiled, the ranking computes the keys where it
  compares them: a short row's positions take two kernels where they took
  three, as the router of gpt-oss's mixture of experts does.
- `Nx.positions c` repeats each index by its count, or lists where a mask
  holds. `compress`, `extract`, `nonzero` and `argwhere` are built on it: they
  read their length once and allocate nothing per element, where they read the
  whole mask and built OCaml lists of every position.
- **Breaking:** `compress` requires a 1-D condition with as many elements as
  the axis, or as the tensor without `~axis`. Without `~axis` a shorter
  condition was read as false past its end, and both forms accepted a condition
  of another shape with the right number of elements.
- `Nx.order_key dtype` gives each element an unsigned integer of `dtype`,
  `uint8` to `uint64` and at least as wide as the element, whose order is the
  sort order. `Nx.lexsort` sorts rows of keys stably, column 0 first;
  `Nx.searchsorted ~side` finds where keys or rows fall among sorted ones,
  comparing `-0.` equal to `0.`; `Nx.unique` numbers the groups of equal keys in
  order of first appearance, with each group's first position and size.
- `sort` and `argsort` order `int4` and `uint4` tensors, which they refused, and
  `cast` reads a strided view of them, which it refused unless contiguous.
- `scatter` takes `~mode:`Max` and `` `Min ``: each position holds the maximum
  or minimum of its element and the updates that reach it, NaN propagating and
  `-0.` below `0.`. A NaN result keeps the element's NaN, or else the first NaN
  update's, so its bits never depend on how the updates are grouped.
- `Nx.reduce_segments op ~segments ids x` sums or takes the extremes of `x`'s
  rows by segment id, from each op's identity; an id outside the segments drops
  its row.
- `scatter ~mode:`Add` on `float16`, `bfloat16` and the `float8` dtypes adds in
  `float32` and rounds once per position, as `sum` does. It rounded after every
  update, so 4096 additions of `1.` into one `float16` position gave 2048;
  gradients of gathers over these dtypes gain the same precision.
- Reductions and scans on the host give the same bits on every thread count,
  where a reduction keeping a long contiguous axis could change its association
  with the number of cores. Float sums over more than 1024 terms or over several
  axes, and float scans over more than 8192 elements, change their last bits and
  gain accuracy.
- `cumsum` and `cumprod` keep the bits of their earlier outputs when the input
  grows: the running sums of a prefix are the first running sums of the whole.
  Past 8192 elements, the running sums of non-negative terms can decrease by a
  rounding.
- `max`, `min`, `cummax` and `cummin` return the first NaN along an axis, the one
  `argmax` and `argmin` find, where a later NaN could replace it.
- Reductions across a contiguous axis longer than 4M elements (512K on eight
  threads) no longer fall back to a strided walk.
- `Nx_device.Buffer.Claim.with_` checks a call's donated buffers against its
  other buffers in `n log n` time, where a call donating a model's parameters
  and optimizer state made a quadratic number of overlap checks.
- `Nx_device.Buffer.spans` of a borrow judges the memory it maps: a borrow of
  all of a memory spans it even when its device maps whole pages around it, as
  Metal maps a file's pages, so `Claim.consume` accepts it as it accepts the
  memory itself.
- An operation and a read (`to_array`, `item`, `pp`, `map_item`, `iter_item`,
  `fold_item`) claim the memory of the host values they read while they read
  it, so a compiled call on another domain that consumes one of them computes
  from a copy instead of writing over it. An operation on a value whose memory
  such a call holds raises at once.
- **Breaking:** `Nx_device.Buffer.consume` is `Nx_device.Buffer.Claim.consume`,
  which needs a claim on the memory from `Claim.with_`. `Nx_device.Buffer.Claim`
  counts read claims on memory, shared by its views and borrows, and holds it
  exclusive for a writer. A buffer over a bigarray or a file is never
  exclusive, since the bigarray's holder and the file reach the memory outside
  the claims. A dead buffer names the memory's last consumption.
- Complex `div` and `recip` compute by Smith's algorithm, as OCaml's
  `Complex.div` does, on every platform. They used C's division, whose
  runtime gives a NaN part where the quotient's is zero when the other part
  overflows: on Linux, `(0 + 2^-50 i) / (0 - 2^-1074 i)` was `-inf + nan i`
  where it is `-inf + 0 i`.
- A host program on x86_64 that calls a function more than 2 GiB from its code,
  such as the host's own 16-bit float conversions, no longer crashes: the jump
  the loader writes for such a call read its target 2 bytes before the address
  it had stored.
- `Nx_device.runs_on_host` says whether a device's work is the host's: the
  host, or a device over host memory that loads no programs, such as a test
  device. nx.cpu computes on exactly those, so no longer on Metal.
- `Nx_io.encode_png` takes `?dpi`, written as a `pHYs` chunk so that viewers
  and printers show the image at its physical size, and `?srgb`, an `sRGB`
  chunk stating that the samples are sRGB. Without them the file is the same
  as before.
- On Linux, nx's thread pool has as many workers as CPUs the process may run
  on, its affinity mask bounded by its cgroup's CPU quota, where it counted
  every online CPU: under `taskset`, a cpuset or a container's CPU limit the
  extra workers took turns on the same cores. Pinned to 6 of 14 cores, a
  batch of 64 products of 32 x 32 takes 63 us instead of 73.
- Host buffers of 64 KiB or more reuse the memory of collected buffers of
  the same size instead of returning it to the C library, which on Linux
  trims or unmaps it, so the next eager operation faulted its pages in again
  and timings swung by 2-4x from run to run. The cache holds up to a major
  cycle's share of the program's memory (at least 32 MiB), counts against
  the host's budget, shows in `Nx_device.Stats.cached`, and
  `Nx_device.free_cache Nx_device.host` empties it. On Linux x86, vega's
  optimizer steps keep their medians with a spread of 0.3-8% where it reached
  120%, and eager random draws of 1M elements are 13-28% faster.
- An NV GPU under NVIDIA's kernel driver borrows host memory whose length is
  not a multiple of 4 KiB. `Nx_device.Buffer.borrow` refused it with
  `NV_ERR_INVALID_ADDRESS`, because unified memory takes ranges of whole pages
  and the range was the buffer's length; it now covers the buffer's pages.
- An NV device whose open fails while registering the GPU's device file no
  longer leaves that file open.
- An NV device opens through NVIDIA's kernel driver. Every open failed with
  `NV_ERR_INSUFFICIENT_PERMISSIONS`, because the driver gives a GPU only to a
  process that holds the GPU's device file open, and `Nx_nv_device` opened it
  only while it mapped memory. The file now stays open while the GPU is.
- NV devices open under release 615 of NVIDIA's kernel driver, which
  `Nx_nv_device` refused as unsupported. That release moved the bits of the
  cache flush the device issues and added 8 bytes to the parameters of a
  channel group, so it is described on its own rather than as 610. A release
  not described is still refused, naming it.
- Programs holding large tensors on the host run far fewer major collections:
  host buffers of 64 KiB or more pace the collector by a share of the OCaml
  heap and the live buffers together, where it took the heap alone. With 256 MB
  of tensors live, an MLP forward pass runs 8 major cycles where it ran 196
  (292 to 120 µs), and kaun's SGD train step takes 1.20 ms where it took 1.92.
- `Nx.top_k` ranks an axis of at most 32 entries in one pass, each entry
  placed by the number of entries before it, instead of one pass per entry
  taken or a sort. A compiled call runs 3 kernels for the positions of 4 of
  32 entries instead of 14, and 3 for 16 of 32 instead of 22; values, ties,
  NaN and the zeros keep their order. Eagerly the comparisons cost more over
  many rows: 0.77 ms against 0.20 for 4 of 32 over 512 rows.
- `Nx_device.Buffer.consume` of a buffer that was itself returned by
  `consume` no longer lets the memory be freed while the new buffer still
  uses it. The memory went back to the device at the next collection and was
  handed to other buffers, so a compiled call that consumes the state it
  returned on the previous call, such as a decoder's key-value cache, could
  read another buffer's bytes after a few steps.
- **Breaking:** a descending `sort`, `argsort` and `top_k` order by the exact
  reverse of the ascending order, so NaN comes first instead of last and the
  first index of `top_k` is `argmax`'s.
- `bitcast` reads between dtypes of different widths: a dtype `k` times wider
  consumes a last axis of `k`, and one `k` times narrower adds it, so a
  `[n; 8]` `uint8` value reads as `n` `uint64` words without a copy.
- **Breaking:** `Nx.Op.Read` is `Read { by; x }`, where `by` names the function
  that reads, such as `"Nx.item"`; an interpreter's refusal of a read starts
  with it.
- **Breaking:** indices are `int64`. `take`, `take_along_axis`, `scatter`, `D`
  and `Nx_quant.apply ~ids` take `int64_t`; `argmax`, `argmin`, `sort`,
  `argsort`, `top_k`, `nonzero`, `argwhere`, `lu`, `permutation` and
  `categorical` return it; `Nx_backend.int32_array` is `index_array`. Past 2³¹
  entries, `argmax`, `argmin`, `sort`, `argsort` and `lu` no longer raise, and
  `nonzero`, `argwhere`, `compress`, `extract` and `set` no longer wrap.
  `randint` still draws `int32`.
- `Nx.Rng` draws of more than 2³² words or `float32` values, or of more than
  2³¹ `float64` values or keys, no longer repeat their counters, which made the
  later part of such a draw repeat the earlier part. Smaller draws keep their
  values.
- `Nx.arange` raises `Invalid_argument` when a value does not fit its dtype,
  where it wrapped integers, took `float16` past 65504 to infinity, saturated
  the float8 dtypes at their largest finite value and filled `bool` by
  position. It no longer allocates per element: a 10⁷-element `int64` arange
  takes 6.2 ms instead of 586 ms, and `Nx.Rng` draws of 10⁶ `float32` take
  3.8 ms instead of 61 ms.
- `Nx.logical_and`, `logical_or` and `logical_xor` on `bool` tensors apply
  the bitwise operation directly instead of testing each operand against zero
  and casting the result back: one array per call instead of four, about four
  times faster on a 128 × 256 mask.
- `Nx.Ptree.Path.root`, `Nx.Ptree.Path.v` and `Nx.Ptree.Path.add` construct
  paths, which code outside nx could only receive from a walk. A path can now
  be written literally and compared with the one a walk gives, as in
  `Path.equal p (Path.v [ Field "out"; Field "w" ])`.
- **Breaking:** `Nx.correlate` and `Nx.convolve` change which values of the
  full correlation `` `Same `` and `` `Valid `` keep, as their documentation now
  states. With an even kernel of size `k`, `` `Same `` correlates each element
  `i` with the input from `i - k/2` to `i + k/2 - 1`, one element earlier than
  before: correlating `[|1.; ...; 7.|]` with `[|1.; 10.; 100.; 1000.|]` gives
  `[|2100.; 3210.; ...; 765.|]`, where it gave `[|3210.; ...; 765.; 76.|]`.
  With a kernel longer than the input along an axis, `` `Same `` keeps as many
  values as the kernel, where it kept as many as the input, and `` `Valid ``
  keeps those where the input lies within the kernel, where it gave an empty
  array. These are the windows numpy's `correlate` and `convolve` keep.
- `Nx.combine_patches` and `Nx.extract_patches` handle windows that do not fit:
  an axis shorter than its dilated kernel, even with padding, has no window.
  `combine_patches` of no window crashed the process by reading past its input,
  and now gives zeros. `extract_patches` counted a window that overhangs the
  padded axis by less than a stride, and now does not. Both now raise
  `Invalid_argument` for a geometry with a non-positive size, stride or
  dilation, a negative padding or mismatched lengths, and `combine_patches`
  for patches whose shape the geometry does not give, where invalid input
  could read or write out of bounds. A geometry whose sizes do not fit in 64 bits is
  refused too, below the frontend as well.
- `Nx_amd_device` allocates mapped memory (`Buffer.create ~memory:Mapped`) in
  the GPU's own memory, which the host writes through its memory BAR, and
  flushes the host data path before each copy of its own so that the copy
  engine sees those writes; `Nx_amd_device.flush_hdp` gives that flush to the
  libraries that submit copy-queue work. Without a BAR that covers the GPU's
  memory, or without the HDP flush register under the amdgpu driver, mapped
  memory is pinned memory, and a code object that needs host-visible memory is
  refused with the reason instead of raising.
- `Nx.eigh` and `Nx.eigvalsh` read only the real part of a complex matrix's
  diagonal, as their documentation now says and as `Nx.cholesky` already did.
  They used its imaginary parts too, so a complex matrix that was not exactly
  Hermitian gave a different decomposition from the Hermitian matrix its
  triangle names. Their range scaling also no longer reads the triangle they
  ignore, which made a tiny float32 matrix lose precision when that triangle
  held large values.
- An NV device gives mapped memory (`Nx_device.Buffer.create
  ~memory:Mapped`): GPU memory the host writes through BAR1. Under the
  driver-less interface with a 256 MiB BAR, or once the kernel driver's BAR1
  window is full, it is pinned memory; a full window no longer loses the
  device when it maps a program or other memory the host addresses.
- **Breaking:** `Nx_device.Buffer.create` takes `?memory:(Device | Pinned |
  Mapped)` in place of `?pinned`: `~pinned:true` is `~memory:Pinned`. `Mapped`
  is device memory that the host also writes through a window onto it (a
  BAR), which the device reads at the speed of its own. Where a device has no
  such window or it is full, mapped memory is pinned memory, and on a device
  the host addresses every memory is its own. Drivers describe the window with
  `Driver.Device_local`'s new `mapped` allocator.
- **Breaking:** `Nx_nv_device.kernel`'s `image` is the uploaded cubin as a
  buffer of the device rather than its address, so a compiled batch binds its
  program to the image nx.nv.device loaded and relocated.
- An NV device has room for a submission once each of its channels' rings is at
  most half full, as an AMD device's rings are, so `Nx_device.submit` waits for
  it and compiled code that writes the rings never overruns the entries the
  GPU has not fetched.
- **Breaking:** `Nx.Op.interpreter` gains `claims`, and `Nx.Op.intercept`
  hands its interpreter only the operations it claims. An unclaimed operation
  reaches the enclosing interpretation directly, without being performed
  again, so a transformation pays nothing for operations on values it does not
  own. An interpreter that takes every operation passes
  `claims = (fun _ -> true)`.
- `Nx.Op.shape` and `Nx.Op.dtype` give the shape and dtype of an operation's
  result without computing it, as nx allocates the result.
- `Nx_device.reaches d d'` says whether `d`'s work addresses the memory of
  `d'` once borrowed, without trying a borrow, and `Driver.device` takes the
  driver's `reaches` for the peers it maps. An AMD device reaches the GPUs of
  its machine that its copy engine reaches over a link or a large memory BAR.
- `Nx_device.submit` allocates about half as much per call: it gathers the
  devices a submission takes, and the latest work pending on their memory, in
  short lists instead of sorted copies and a hash table. A submission to one
  device touching six buffers allocates 250 minor words, from 478.
- `Nx_device.submit` waits, for at most each device's timeout, until the
  device's queues have room for a submission, as `Driver.device`'s new `room`
  says, and loses a device whose queues stay full. An AMD device has room once
  each of its rings is at most half full, so work submitted by compiled code
  never overwrites packets its engine has not read.
- **Breaking:** `Nx_amd_device.scratch` returns the scratch memory as a buffer
  of the device; the record of its address, size and `COMPUTE_TMPRING_SIZE`
  is gone, since the ring size depends on each kernel.
- **Breaking:** `Nx_amd_device.kernel`'s `code` is the uploaded code object as
  a buffer of the device, not its address, so a submitter names the program's
  memory as storage it touches.
- A scalar broadcast by `Nx_array.View.expand`, or by an `Expand` movement
  evaluated with `Nx.Op.eval`, is no longer reported C-contiguous by
  `View.is_c_contiguous`. A reshape of it read past its one element and failed.
- `Nx_cuda_device` gives compiled code what it needs to enqueue work itself:
  `driver_function`, the address of an entry point of the driver it loaded,
  and `stamp`, a host function that stamps a word with the host clock.
- `Nx_metal_device` gives compiled code what it needs to submit work itself:
  `msg_send`, the address of `objc_msgSend`, `selector`, which registers an
  Objective-C selector, and `indirect_commands`, which records dispatches of
  loaded programs into an indirect command buffer that lives as long as its
  argument buffer.
- **Breaking:** `Nx_device.Driver.mapping` is a variant: `Identity` for a
  device that addresses host memory at its host addresses, whose borrows are
  the memory itself at any address, and `Pages { map; unmap }` for a driver
  that maps whole pages. A test device over the host's memory borrows another
  one's small buffers and signal word, which the page rule refused.
- A float `Nx.sum` that is exactly zero is `0.`: a sum is `0.` plus its terms,
  on every path. The sum over axis 0 of a C-contiguous matrix started from its
  first row, so the column sums of a 2 x 3 matrix of `-0.` were `-0.` where
  every other layout gave `0.`; `Nx.scatter`'s `Add` mode added into a copy of
  the template, so `-0.` plus a `-0.` update stayed `-0.`; and `Nx.matmul` on
  macOS returned `-0.` from Accelerate where every product was `-0.`. `Nx.mean`
  and the products built on `matmul` follow.
- `Nx.maximum`, `Nx.minimum`, `Nx.max`, `Nx.min`, `Nx.cummax` and `Nx.cummin`
  are IEEE 754 maximum and minimum: NaN propagates and `-0.` is less than
  `0.`. A tie of zeros returned an operand by position, so `max` of
  `[-0.; 0.]` was `-0.` and of `[0.; -0.]` was `0.`, and an axis reduction's
  zero depended on the layout. `Nx.argmax` and `Nx.argmin` return the index of
  the element `max` and `min` return, and `Nx.sort`, `Nx.argsort` and
  `Nx.top_k` put `-0.` before `0.` ascending, where they kept the zeros in
  input order. Eager and compiled code agree on zeros.
- `Nx.Repr.Storage` has the claims a compiled call takes on the storage its
  arguments reach: `borrow` and `release` for reading, `upgrade` to an
  exclusive claim, `consume`, and `finish`; `pin` and `unpin` for the programs
  that bind a storage; and `live` and `pins` to read the state.
- New low-level section of `Nx`, for transformations and file formats: `Nx.Op`,
  the operations as values with their interpretation (`eval`, `intercept`,
  `intercepted`), where their results live (`placement`), and their `operands`,
  `name` and `pp`; and `Nx.Repr`, the representation of a value (host arrays,
  placed values over `Nx.Repr.Storage`, traced values with their `node`), whose
  constructors check that a view stays within its storage and that the
  buffers are of the value's format.
  `Nx.Placement.with_leading_axis` and `without_leading_axis` serve
  transformations over a leading axis.
- **Breaking:** `Nx_device.submit` takes a set of devices and the buffers the
  work touches, and runs `f` with an `Nx_device.Submission.t`: `value` is each
  device's value, `waits` is one `(device, value)` pair for each device of the
  submission and of its buffers, a fixed shape a compiled program can take as
  arguments, and `wait` blocks the host until a device signals a value, such
  as the previous run of a program whose arguments are rewritten. Work that
  the submission's queues cannot wait for is waited for on the host first.
- **Breaking:** `Nx_device.Profile.record` is `Nx_device.Submission.record`,
  and its stamps are two 16-byte slots, with the start stamp in the second
  word and the stop stamp in the fourth. `Nx_device.timeline` is
  `Nx_device.signal_word`, one word: the runtime no longer writes the
  submitted value into the device's memory, which nothing read and which cost
  a round trip per submission to another machine's device.
- **Breaking:** the vendors' low-level sections are a type `t`, `of_device`
  and accessors in place of the `handles` records, such as
  `Nx_metal_device.queue`, `Nx_cuda_device.compute`, `Nx_amd_device.sdma` and
  `Nx_nv_device.copy`. AMD's queue words and NV's channel words are
  `Nx_device.Buffer.t`s, and `kernel` returns an `option`.
- CUDA devices complete by `Sleep` rather than `Signal`, since their work
  stores its values into the signal word, which other devices' queues can
  then wait on. A fault is found up to 200 ms later than before.
  `Nx_device.Driver.Sleep` receives the region of the device's timeline, as
  `Signal` does.
- `Nx.cast` from `int64` or `uint64` to `bfloat16`, `float16` or a float8 type
  rounds once, from the integer's exact value. Past 2^53 it went through a
  double, which can round the integer onto a tie of the narrow type and then
  round again: bfloat16 of 9042383626829825 was 0x5a00 where one rounding gives
  0x5a01.
- `Nx_device.Buffer.borrow Nx_device.host b` maps system memory in place: a
  Metal buffer, a test device's buffer or any device's pinned buffer is
  readable through `Buffer.bigarray` once its device is synchronized. Only a
  GPU's own (`Device_local`) memory is refused. `Buffer.copy` through such a
  borrow waits for the device whose memory it maps.
- A host buffer is about twice as cheap to make: `Nx_device.Buffer.create` of a
  few bytes takes 75 ns where it took 158, so a one-element `Nx.add` takes
  161 ns where it took 237, and one of 64 KiB takes 0.85 µs where it took 3 µs.
- **Breaking:** vendor libraries describe their devices with
  `Nx_device.Driver.device` and `Driver.host`, whose memory is `Host_visible` or
  `Device_local` and whose work completes by `Poll`, `Sleep` or `Signal`; they
  replace `Nx_device.make`, `make_host` and their refusals of inconsistent knobs.
  The runtime names another machine's devices, `Driver.name` names one a
  vendor could not open, and every device starts at `Driver.default_timeout`.
- **Breaking:** `Nx_device.Buffer.host_address`, `Buffer.handle` and
  `Nx_device.external_buffer` are gone: `Buffer.address` is the host address on
  a host, `Driver.Region.of_buffer` is the memory a buffer lies in, and
  `Driver.buffer` wraps a vendor's region. `Buffer.dma` and `Buffer.on_free` are
  `Driver.dma`, which returns a `result`, and `Driver.on_free`.
- **Breaking:** `Nx_device.Buffer.create`'s `?host` is `?pinned`, and pinned
  memory is coherent on every vendor: the host and the device see each other's
  writes without a flush.
- `Nx_device.Buffer.borrow` borrows any buffer of the device's machine by where
  its memory lives: system memory (another device's pinned or host-visible
  memory) through the device's mapping, and an AMD or NV GPU's own memory
  through its driver's peer mapping. A buffer of the device borrows as itself.
- New `Nx_device.Buffer.overlaps`, which `Buffer.copy` and rune's check of
  consumed arguments use: a copy between a buffer and its borrow is refused.
- New `Nx_device.Driver.host_memory` and `Driver.host_programs`, what the host
  is built from: a few lines describe test devices over the host's memory.
- **Breaking:** a backend is kernels over arrays: `Nx_backend.S` has one
  function per operation, which writes a destination array that nx
  allocated. `Nx.Backend`, `Nx_array.Backend_intf` and a backend's `place`
  and `to_host` are gone, and nx places values itself.
  `Nx.Backend.Refused` is `Nx_backend.Refused`, which `Nx.place` no longer
  raises, and nx.cpu's allocating functions (`Nx_cpu.add`, `full`,
  `from_host`, `to_host`, ...) are kernels that write `~dst`.
- `Nx_array.Elements.fill` writes the element's bytes without a bigarray
  view.
- **Breaking (effect handlers):** `E_view` and `E_placement` are gone: a
  value's shape and placement are its own, read with no interpreter involved,
  and `Nx.Repr.Traced.v` takes the placement of the value it makes.
- **Breaking (effect handlers):** a transformation installs its interpreter
  with `Nx.Op.intercept { run; claims } f`, which hands `run` each operation
  of `f` that `claims` takes, and `Nx.Op.intercepted ()` tells whether one is
  installed around the caller. While no interpreter is installed on any
  domain, nx performs no effect and builds no operation: a one-element
  host `Nx.add` allocates 69 words instead of 241.
- Writing a `float` into a `float16` tensor and `Nx.cast` to `float16` from
  `float64` or a 64-bit integer round once to the nearest `float16`. They
  rounded to `float32` first, which moved a value next to a `float16` tie onto
  it and rounded it the wrong way; `bfloat16` and the float8 formats already
  rounded once.
- Reading a NaN of `float8_e4m3` or `float8_e5m2`, and
  `Nx_dtype.Scalar.decode`, keep its sign, as writing one already did: a
  negative NaN read as positive, so it did not survive a round trip.
- Converting to a float8 format saturates a finite value past the largest
  finite one to the largest finite value of its sign. `Nx.cast`, element
  writes and `Nx_dtype.Scalar.encode` gave NaN in `float8_e4m3` and the fnuz
  formats and infinity in `float8_e5m2`: `1e6` was NaN in `float8_e4m3` and
  is now 448. Infinities and NaNs are unchanged: an infinity stays one in
  `float8_e5m2` and is NaN in the formats without infinities.
- `Nx.shape` returns an array of its own. It returned the value's, so a
  caller that changed it changed the value's shape.
- **Breaking (effect handlers):** nx's operations are the constructors of one
  type, `Nx.Op.t`; the per-operation effects (`E_add`, `E_reduce_sum`, ...)
  are gone. `Nx.Op.eval (Binary (Add, a, b))` performs one, and `Read` reads
  a value to the host. New library `nx.backend` names the operations' kinds.
- `Nx.copy` always gives storage of its own, and `Nx.contiguous` returns a
  value whose bytes are already C-contiguous from its first element unchanged,
  a placed view included; it copied a placed view that did not cover its
  storage.
- A constant made at a placement is one element placed there and expanded,
  and a filled value of more elements is then copied into storage of its own;
  nx no longer holds one-element values itself. Reading one element of a
  placed value reads that element alone.
- **Breaking:** `Nx.Rng.fold_in_axis` is removed, and nx no longer declares
  a lane index: `Nx.Rng.fold_in_tensor k (Rune.lane_index ())` gives each lane
  of a map its own key.
- **Breaking:** `Nx_device.Profile.start` returns the profile, and
  `Profile.stop` takes it, so that only its holder stops it. The events
  `Memory` and `Program` are `Allocation` and `Load`, and `Profile.output` is
  `output_chrome_trace`.
- **Breaking:** a device lost to a hang, a driver fault or its machine's
  connection raises `Nx_device.Lost (d, why)`, printed `NAME: why`, where it
  raised `Failure`. A host without memory for a copy's staging raises
  `Nx_device.Out_of_memory`, where it raised `Stdlib.Out_of_memory`.
- **Breaking:** `Nx_device.Buffer.borrow`, `Buffer.of_file`,
  `Buffer.create_file` and `Nx_device.Program.load` return a `result`, as do
  `make`'s and `make_host`'s `?load`: a refused mapping, file or binary is an
  `Error`, and leaves the device usable.
- **Breaking:** the vendor libraries' `v` raises `Failure`, where it raised
  `Invalid_argument`, and `get`'s errors start with the device's name. `count`
  and `get` raise `Nx_device.Lost` for a host that cannot be reached.
- **Breaking:** `Nx_device.Buffer.file` and the `?file` argument of
  `Buffer.of_bigarray` are removed: a buffer over a file's bytes is a buffer on
  `Nx_device.disk`.
- `Nx_io.load_safetensors` returns each entry as a value on the disk over its
  bytes in the file, where it returned a host value over a mapping of the file.
  Placing an entry on the host or on a device whose memory the host addresses
  maps the file copy-on-write, as the load did, and placing it on another
  device reads its bytes into the device's memory without a host copy of the
  file. An entry at an offset that is no multiple of its element size is read
  from the file instead of copied out of the mapping element by element. The
  file must not be modified in place while its values are alive.
- `Nx_io.save_safetensors` writes each tensor into the file from its own
  storage, wherever it lives: a device writes its memory to the file, and a
  value on the disk is copied from its file.
- A value on the disk (`Nx.Device.make Nx_device.disk`) takes part in
  an operation as a host value: the operation reads it through a mapping of
  its file, or a copy when its bytes are not aligned to its elements, and
  computes on the host. A constant made beside it is the host's, and a
  movement of it stays on the disk and reads nothing. A placement onto the
  disk raises.
- `Nx.place` onto a device borrows a value on the disk
  when the device's memory is the host's, and otherwise copies each device's
  window straight from the value's buffer on another such device when the
  window is a contiguous run of it. It read the whole value to the host first.
- New `Nx_device.disk`, the file system as a device. `Nx_device.Buffer.of_file`
  and `Buffer.create_file` make buffers of files' bytes, and `Buffer.copy`
  reads and writes them: straight into or out of memory the host addresses,
  and through the host's staging memory for a GPU whose memory it does not
  address, the GPU copying one slot while the host reads or writes the other.
  On Linux the reads and writes go through io_uring where the kernel allows
  it. `Buffer.borrow` of a file's bytes by the host, or by a device whose
  memory is the host's (`Nx_device.shares_host_memory`), maps the file
  copy-on-write.
- A read of a strided view of a value on a runtime device, such as a
  transposed weight on Metal, gathers its elements with `Nx_cpu.copy`, about
  five times faster than element by element.
- `Nx.rfft` and the other transforms of no points are zero under every norm.
  Under `Forward and `Ortho they scaled the empty sum by 1/0 and gave NaN. The
  docs of `Nx.fft`, `Nx.rfft` and `Nx.irfft` now state what a transform of no
  points, and the inverse of one bin, return.
- An operation on a dtype it does not take raises `Invalid_argument`, where
  it raised `Failure`: every computation on `int4` or `uint4`, and arithmetic
  on `bool` (sums, products, negation, cumulative sums, `Nx.matmul`). The
  dtype documentation states which operations each takes.
- `Nx.argmax` and `Nx.argmin` raise `Invalid_argument` on an axis of more
  than `Int32.max_int` entries, which their int32 indices cannot reach, and
  say so. They raised `Failure` from the C backend, after flattening, which
  could copy the whole axis first.
- Storing an out-of-range integer into an int4 or uint4 element keeps its low
  four bits, as every wider integer dtype keeps its low bits. It clamped to
  the range, so `Nx.create Nx.int4 [| 1 |] [| 9 |]` held 7 where an int8 store
  of 300 wraps to 44. Casts from floats still hold at the range.
- `Nx_device.Buffer.copy` counts a copy through a borrow as traffic of the
  host, whose memory a borrow is. It counted it in the borrowing device's
  `bytes_in` and `bytes_out`, although borrowed memory is never counted in a
  device's statistics.
- `Nx_device.Buffer.borrow` refuses every host buffer `create` made of fewer
  than 64 KiB. It borrowed one when the allocator happened to place it on a
  page, so the same program borrowed on one run or machine and raised on
  another. A buffer of no bytes still always borrows.
- `Nx.svd`, `Nx.svdvals`, `Nx.qr`, `Nx.eigh` and `Nx.eigvalsh` keep their
  accuracy on a matrix of subnormal or huge entries. They now scale it into
  range by a power of two and scale the results back. `svd` of a float64
  matrix of subnormals gave singular values off by half, and `qr` lost
  digits in the subnormal range.
- New `Nx_device.Buffer.consume ~why b`: a buffer over `b`'s memory that
  kills every earlier handle to it, whose use then raises
  `Invalid_argument why`. `Buffer.spans` tells whether a buffer is all of its
  memory, which consumption needs.
- `Nx_device_support.Remote` bounds what it reads from a server: an answer
  announcing more than it can be fails the connection instead of allocating
  it, and a page size that is no power of two is refused at connection.
  `nx-remote` refuses configuration accesses that are not 1, 2 or 4 aligned
  bytes of the 4096, and moves large BAR reads and writes in pieces of 1 MiB.
- `Nx_amd_device` under `Pci` turns the GPU's bus mastering on in a partial
  boot too. A GPU left clean by the previous process boots partially, and one
  on another machine had its bus mastering turned off by `nx-remote` when that
  process left, so the next open could not reach system memory.
- `nx-remote` frees a client's memory only once the DMA of the functions it
  took is off, and leaks it, saying so, when that fails; releasing a function
  turns its bus mastering off. A client's system memory at an address must lie
  in its own reservations and outside its other memory, which `MAP_FIXED`
  would silently have replaced, and its reservations are released when it
  goes. `Nx_device_support.Sysmem` gains `unreserve` and `extent`.
- `nx-remote --key-file` reads the key with the new
  `Nx_device_support.Remote.read_key`: one open of the file, which must be a
  regular file of the server's user that no one else may read, of 16 to 4096
  bytes. An unreadable file crashed the server, a FIFO blocked it before it
  listened, and a file of another user was accepted.
- Taking a PCI function (AMD and NV under `Pci`, network adapters, and
  `nx-remote`'s clients) no longer follows a link planted at its lock file in
  the temporary directory, which created the link's target world-writable. It
  also takes the lock `nx_BUS.lock` besides the driver's own, so two processes
  of raven never drive one function, whatever driver name they give.
- `nx-remote` keeps serving whatever a connection does: a connection reset
  before the server set it up, or an exception a client's command raised there,
  stopped it from accepting anyone, and a peer that never spoke held the only
  client slot. A connection now holds the slot only once it proves the key,
  handshakes run at most 64 at once for at most 10 seconds each, and an idle
  client is probed, so one whose machine vanished is disconnected and its GPUs'
  DMA stopped. The protocol is at version 2: clients and servers of version 1
  refuse each other.
- Host operations allocate less: the host array holds its view, and binary
  operations, comparisons and `Nx.where` over operands of one shape skip
  broadcasting. A one-element `Nx.add` allocates 93 words (238 before) and
  `Nx.shape` 9 (40 before).
- **Breaking:** the `nx.c` library is `nx.cpu`, and its module `Nx_c` is
  `Nx_cpu`.
- **Breaking:** the `nx.core` library is `nx.array`, whose `Nx_array.t` is an
  array: typed elements of a runtime buffer through a strided view. `Nx_c.t`
  is that record, and loses its `context` field and accessors.
- **Breaking:** `Nx_core.Make_frontend` is removed: `Nx`'s surface is written
  once over its own values. Another implementation of the operations is a
  backend named in a placement.
- New `nx.rdma.device`: Broadcom BCM57608 RoCE adapters, which the runtime
  drives itself on this machine or another (`Nx_rdma_device.get ?host i`).
  Once an adapter is open on each of two machines, `Nx_device.Buffer.copy`
  between their GPUs' memory goes from GPU to GPU over RoCE, through the
  adapters closest to the GPUs, in sends and receives of up to 1 GiB, instead
  of through the hosts.
- `Nx_amd_device` and `Nx_nv_device` open another machine's GPUs:
  `get ~host ~interface:Pci i` and `count ~host ()`, given the host
  `Nx_remote_device.connect` gave, drive that machine's GPU `i` through its
  server, as `AMD-PCI@HOST:PORT`, `NV-PCI:1@HOST:PORT`, and so on. Their host
  memory is that machine's, and they copy directly only to GPUs of their
  machine. Under `Pci` both describe their memory to other PCI functions
  (`Nx_device.Buffer.dma`), such as a network adapter.
- New `nx.remote.device` and the `nx-remote` command. A machine runs
  `nx-remote --key-file FILE` (on `127.0.0.1:6667` unless `--listen` says
  otherwise), and `Nx_remote_device.connect ~key host` is that machine's host
  as a device, `CPU@HOST:PORT`: its buffers are the machine's memory, copies
  cross the network, and host programs run there. Both ends prove the key;
  nothing else protects the traffic, so use it on the machines' own network
  or through a tunnel. The server serves one process at a time and stops the
  DMA of the functions a process took once it disconnects.
- Devices of other machines. `Nx_device.host_of d` is the host of the machine
  `d` is attached to. A library that reaches another machine makes its host
  with `Nx_device.make_host`, whose memory the process reaches through an
  `io`, and that machine's devices with `make ~host`. `Buffer.copy` copies on
  another machine as on this one, staging in that machine's memory, and
  between machines over a link an opened device carries (`make ?link`), or in
  64 MiB chunks through the hosts otherwise; `Program.call` runs a program of
  another machine's host there. `Buffer.dma` is how other PCI functions reach
  a device's memory (`make ?dma`), and `Buffer.on_free` runs a function
  before a device frees memory another device mapped. A `synchronized` hook
  that raises now fails its device, and `Profile.record` refuses stamps of
  another machine than the device's.
- `nx.device.support` reaches another machine's PCI functions and memory
  through a server there. `Remote` is the client of its protocol: it proves a
  shared key, and both ends must prove it; `Remote_server` serves one client at
  a time and, when the client leaves, turns bus mastering off on the functions
  it took before freeing its memory. `Pci.take ~remote` takes a function of
  that machine, whose BARs are `Mmio` ranges reached through the connection,
  and `Pci.alloc_sysmem`, `free_sysmem`, `reserve`, `pin` and `unpin` give the
  system memory of a function's machine, which `Pci_memory` now uses.
  `Sysmem.alloc` without `~va` maps locked memory where the system chooses.
- **Breaking (effect handlers):** the `E_psum` effect and its entry function
  are removed. It had no caller, no batching rule and no backend operation,
  and it raised outside a map; `Nx.sum ~axes:[0] (Rune.lanes a x)` is the same
  sum over the lanes of the map named `a`.
- New `Nx_device.Profile`: between `start ()` and `stop ()`, devices record
  spans of their work, their allocated memory and the programs they load, all
  on the host clock (`now`). Host code adds spans with `span`, the libraries
  that submit work add them with `record` from timestamps their work writes,
  and `Buffer.copy` and `Program.call` record their own; AMD and NV clocks are
  calibrated against the host's when the profile is taken. `output` writes
  Chrome's trace event format, which Perfetto loads. With no profile taken,
  recording costs one atomic read and allocates nothing.
- **Breaking:** a vendor's `Nx_device.copy_queue` has a `stamp`, which writes
  a timestamp of the device's clock, and `Nx_device.make` takes that clock
  (`?clock`) and a `?resolve` hook for the timestamps the device's work does
  not write itself, which Metal's command buffers need.
- `Nx_device.host` loads programs on x86_64 and arm64: `Nx_device.Program.load`
  links an ELF relocatable object compiled for the machine, resolving its
  calls to the C and math libraries and to the compiler runtime's 16-bit float
  conversions, into memory that is executable and never writable, and frees it
  once the program is unreachable. The new
  `Nx_device.Program.call` runs a host program on buffers and integer values,
  with the OCaml runtime released.
- **Breaking:** the `nx.backend` virtual library is removed. `nx.c` is an
  ordinary library, `Nx_c` (it was `Nx_backend`), without `create_context`;
  nx always links it, and another implementation of the operations is a
  backend named in a placement (`Nx.Backend`) instead of a link-time
  replacement of nx.c.
- `Nx.Backend.S` has `place`, which makes a value at one of the backend's
  placements, and `Nx.place` asks the target placement's backend. A placed
  value is read by the backend that made its storage, so a backend can hold
  values in storage of its own and copy them in and out.
- A placement may mix `Nx.Device.host` with other devices, such as
  `Nx.Placement.replicated [ Nx.Device.host; d ]`: the host keeps its
  placed values in the same runtime buffers. It raised `Invalid_argument`.
- **Breaking (backends):** `Nx_core.Backend_intf.S` no longer declares `view`,
  `dtype` and `context`. They describe a value, which carries them, so a
  backend over nx's values would only repeat nx's answers. A kernel library
  keeps them as its own functions, and `Nx_core.Make_frontend` asks for them
  beside the signature.
- `Nx.qr` without `~mode` returns the reduced factorization, as documented.
  It returned the complete one: an `m × m` `Q` and an `m × n` `R` for a tall
  `m × n` matrix.
- `Nx.eigvalsh` converges on graded matrices, such as a tridiagonal whose
  diagonal runs from `1e-8` to `1e8`, which failed to converge from
  order 100 on while `Nx.eigh` succeeded. Its values-only iteration now picks
  the end it converges from, as LAPACK's `dsterf` does.
- `Nx.eigh ~uplo:`U` and `Nx.eigvalsh ~uplo:`U` read the upper triangle.
  `?uplo` was ignored and both always read the lower one, so a matrix given
  by its upper triangle gave the eigenvalues of another matrix.
- `Nx.eigh` and `Nx.eigvalsh` take complex Hermitian matrices, complex64 and
  complex128, where they raised `Invalid_argument`. The eigenvalues are real
  (float64) and the eigenvectors have the matrix's dtype.
- `Nx.eig` and `Nx.eigvals` hold on matrices whose largest entry is below
  about `1e-138` or above `1e138`, which failed to converge: the matrix is
  scaled into range first, as LAPACK's `xGEEV` does, and its eigenvalues
  scaled back.
- `Nx.svd`, `Nx.svdvals`, `Nx.qr`, `Nx.eigh` and `Nx.eigvalsh` hold on
  matrices whose entries are very small or very large. The Householder
  reflectors summed squares that under- or overflowed: a complex64 `svd` of a
  rank-deficient matrix failed, a float64 matrix scaled by
  `1e-170` came back as if it were another matrix, and one scaled by `1e170`
  gave NaN. The norms are now scaled, as LAPACK's `xLARFG` does.
- `Nx.eig` and `Nx.eigvals` converge on complex matrices with nearly equal
  eigenvalues, such as a Hermitian tridiagonal with paired eigenvalues, which
  failed to converge. The QR iteration's shift lost half its digits to
  cancellation there; it is now computed as LAPACK does.
- New `nx.nv.device` library: NVIDIA GPUs as `Nx_device.t`s on Linux without
  CUDA, through NVIDIA's kernel driver (releases 570, 580 and 610) or over PCI
  without it (`~interface:Pci`), booting the GSP with verified firmware.
- New `nx.amd.device` library: AMD GPUs as `Nx_device.t`s on Linux, through
  the `amdgpu` kernel driver or over PCI without it (`~interface:Pci`), which
  boots the GPU with firmware it verifies by digest and caches per user.
- New `nx.device.support` library: the PCI, memory, page-table and firmware
  support shared by runtimes that drive a GPU without its driver.
- New `nx.device.elf` library, `Nx_device_elf`: the layout of ELF objects into
  images, which host programs, GPU programs and firmware share.
- `Nx_device.make` takes `?sleep`, which lets a polling device block on its
  interrupts once its signal word stays still and report the fault it finds,
  and `?finalize`, which runs at exit on every device, failed or not.
- `Nx_device.Buffer.create`, `Buffer.view` and `Nx_device.external_buffer`
  take every element count whose bytes fit in `max_int`, as documented. They
  refused counts whose bits overflowed, from an eighth of that size (such as
  2^58 + 1 `float64` elements), with `Invalid_argument`; a host buffer too
  large to allocate now raises `Out_of_memory`.
- Every movement works on int4 and uint4 values: `Nx.take`,
  `Nx.take_along_axis`, `Nx.slice` with `L`, `M`, `D` or a step other than ±1,
  `Nx.pad`, `Nx.concatenate`, `Nx.extract_patches`, `Nx.set` and
  `Nx.scatter ~mode:`Set`. They raised "packed dtype not supported". Summing
  operations (`Nx.combine_patches`, `Nx.scatter ~mode:`Add`) still refuse
  4-bit dtypes, as arithmetic does.
- An int4 or uint4 value under any layout copies, prints and converts to an
  array. `Nx.copy`, `Nx.contiguous` and `Nx.to_array` raised
  `Failure "copy: packed dtype not supported"` on a transpose, a strided or
  offset view or a broadcast of one, and `Nx.slice` of an empty range raised
  from `gather`.
- `Nx.of_bigarray` and `Nx_device.Buffer.of_bigarray` raise
  `Invalid_argument` for a bigarray whose first element does not lie at a
  multiple of its size (of one component for complex kinds), such as a file
  that `Unix.map_file` maps from an unaligned `pos`. Such a tensor could not be
  read with `to_array`, `iter_item` or `pp`, and nx's kernels ran on the
  misaligned elements.
- **Breaking:** the `nx.buffer` library and its `Nx_buffer` type are removed,
  and with them `Nx.data`, `Nx.to_buffer`, `Nx.of_buffer`, `Nx.offset` and
  `Nx.strides`. A tensor's storage is a host `Nx_device.Buffer.t` that only
  nx's own libraries reach. Read elements with `to_array`, `item`,
  `iter_item` or `fold_item`, or loop over the bigarray that `to_bigarray`
  returns; build a tensor with `create`, `init` or `of_bigarray`, which still
  takes the bigarray's memory without a copy. A bfloat16, float8, uint32 or
  uint64 tensor is `bitcast` from the integers of its width.
- **Breaking (backends):** `Nx_core.Backend_intf.S.to_host` and `from_host`
  exchange host `Nx_device.Buffer.t`s, and `from_host` takes the dtype.
  `Nx_core.Elements` creates host buffers for a dtype and reads and writes
  their elements as its values.
- `Nx_io.save_safetensors` no longer holds a copy of the whole file in memory:
  it writes each tensor's storage to the file as it is.
- Values on devices keep a float's bits. A one-element result on a device, and
  a view of a value on a runtime device read back
  through a transpose, flip or other strided layout, passed their elements
  through OCaml floats, which quiet a signalling NaN.
- New `nx.cuda.device` library: `Nx_cuda_device.get 0` opens an NVIDIA GPU as
  an `Nx_device.t` named `CUDA` (`CUDA:1`, ... for the others) on its primary
  context, which it makes current only during its own driver calls, so other
  CUDA libraries keep theirs. Its buffers are GPU memory, copied on the
  device's copy stream; `Buffer.create ~host:true` gives page-locked memory,
  and `Buffer.borrow` page-locks host memory once for every GPU that borrows
  it. It loads CUDA modules (cubin, fatbin or PTX), moves bytes between GPUs
  with peer copies, and wraps memory other libraries allocated with
  `Nx_cuda_device.of_address`. The driver is loaded at run time from the
  library search path (`libcuda.so.1`, `nvcuda.dll`), so nx builds without
  CUDA; `count` is 0 on a machine without a driver. A device needs the
  driver's 64-bit stream memory operations on every platform, Windows
  included, and a GPU fault fails it with the driver's error.
- `nx.device` holds memory that the host does not address, for GPUs whose
  device copies it. `Buffer.copy` runs such copies on the device's copy queue
  as timeline work: directly between memory the device addresses, and in
  pipelined chunks through the host's staging memory otherwise, two 64 MiB
  slots that every device maps. It refuses overlapping ranges of one buffer.
  `Buffer.create ~host:true` allocates host memory that the device's work
  addresses, page-locked where that matters. `Buffer.borrow` now maps the
  whole host memory under a buffer once per device and shares the mapping
  among its borrows, and needs that memory to start on a page: host buffers
  of at least 64 KiB now do, and smaller ones are copied through staging.
  This also applies to Metal. `Buffer.host_address` raises `Invalid_argument`
  on memory the host does not address.
- `Nx_device.set_timeout` sets how long a wait for a device's work lasts
  before the device is failed (30 s by default), so that long kernels can
  run. A driver error while a copy is enqueued now fails the device, like a
  fault, and a transfer that could not be waited for keeps its destination
  out of reuse and in the failed device's reach.
- Vendor libraries describe their devices to `Nx_device.make` with allocator,
  mapping and copy queue records, and wrap memory another library allocated
  with `Nx_device.external_buffer`.
- New `nx.device` library: devices and their memory without the tensor layer.
  It provides the host (`Nx_device.host`) and device buffers of a storage
  format (`Nx_device.Buffer`) that are owned, borrowed or byte-offset views.
  A host buffer reads as a stdlib bigarray without a copy (`Buffer.bigarray`),
  and a bigarray of any storage kind is borrowed as a host buffer
  (`Buffer.of_bigarray`).
  Each device has a caching allocator with a budget (`budget`, `set_budget`,
  `free_cache`) and one `Out_of_memory`, which collects unreachable buffers
  before it raises. `Buffer.copy` synchronizes the devices it touches. There
  are also per-device statistics (`stats`), loaded programs (`Program`), and a
  timeline (`submit`, `synchronize`) for the libraries that submit work.
  A device that hangs or faults is failed for good: every later operation
  that takes it, and every copy or bigarray view of memory it can reach,
  raises its first error at once, while other devices stop waiting for its
  work.
- New `nx.metal.device` library (macOS): `Nx_metal_device.get 0` opens the
  Apple GPU as an `Nx_device.t`. Its buffers are memory the GPU shares with
  the host, kept resident through a residency set where Metal has one. It
  borrows host memory, loads metallib functions, and waits on its shared
  event.
- `Nx.sigmoid` of a large negative number is the subnormal its exact value
  rounds to. It computed `1 / (1 + exp(-x))`, whose `exp` overflows below
  about -88.7 at float32 and -709.8 at float64, and returned 0 there.
- `Nx_io.save_txt` writes uint32 and uint64 elements as their unsigned
  values, as numpy does, and `Nx_io.load_txt` reads them back. The largest
  uint32 was written as `-1`, and a uint32 of `2147483648` or more, or a
  uint64 of 2^63 or more, failed to load.
- `Nx_io.save_safetensors` refuses a name given twice with `Failure`. It wrote
  a file whose header named the tensor twice, which `load_safetensors` refuses.
- `Nx_io.save_safetensors` saves any view, and writes each element's bits as
  stored. A transposed, flipped, strided or broadcast tensor raised
  `Invalid_argument` from `reshape`, and a float32 signalling NaN was written
  quieted.
- `Nx_io.save_image` and `Nx_io.gunzip` raise `Unix.Unix_error`, as documented,
  when the directory of the file they write cannot be written. They raised
  `Sys_error` unless `save_image` was given `~overwrite:false`.
- `Nx_io.gunzip` reads every valid gzip member. It counted the bytes its
  DEFLATE decoder had read ahead as part of the stream, so a member whose data
  ended early enough in its last bytes failed with
  `Failure "truncated gzip member footer"`.
- `Nx.matmul` of batched operands with an empty result is an empty tensor. It
  raised "output has a broadcast (zero) stride", and a batch axis of 0 against
  one of 1 came out as 1.
- **Breaking:** `Nx.slogdet` returns the sign in the input's dtype (a complex
  number of modulus 1 for complex input) and the log magnitude as float64,
  where both were float32. Its type is
  `('a, 'b) t -> ('a, 'b) t * (float, float64_elt) t`.
- `Nx.det` of a matrix with an odd number of row exchanges has the right sign,
  and a float64 determinant has float64 precision. `det` took its sign from
  QR's `R` alone, dropping `det Q`, so a single row swap had determinant 1, and
  it went through a float32 `slogdet`. `det`, `slogdet`, `solve`, `inv` and
  `matrix_power` with a negative power now use `Nx.lu`.
- `Nx.solve` and `Nx.inv` solve complex systems correctly. The QR-based solve
  applied `Qᵀ` where a complex `Q` needs `Qᴴ`.
- `Nx.lu` factors a matrix with partial pivoting as `(perm, l, u)`, where row
  `i` of `l *@ u` is row `perm.(i)` of `a`, for rectangular and batched input; a singular matrix keeps its
  zero pivot in `U`. Rune differentiates it for square input and compiles it
  under `Rune.jit`.
- `Nx.rfftn` and `rfft2` refuse an `s` of another length than `axes`, as
  `fftn` and `irfftn` do. A longer `s` was ignored, and a shorter one raised
  `Invalid_argument "index out of bounds"`.
- `Nx.rfft`, `irfft` and every transform built on them (`rfft2`, `rfftn`,
  their inverses, `hfft`, `ihfft`, `stft`, `istft`) take and give float16,
  bfloat16 and float8 tensors, working at float32. They raised
  `Failure "unsupported bigarray kind"`, though their types accept any float
  dtype.
- `Nx.rfft` of an empty axis gives its one bin as zero, the empty sum, and so
  do `rfftn`, `rfft2` and `stft` when the last transformed axis is empty. The
  bin held whatever its fresh buffer did.
- `Nx.hfft` is the forward transform of the Hermitian signal its input
  describes, unscaled under the default norm, and `Nx.ihfft` is its inverse.
  `hfft` returned that signal's inverse transform, divided by `n` and in
  reverse order, and `ihfft` returned `rfft` unscaled and unconjugated.
- `Nx.Rng.beta` and `Nx.Rng.dirichlet` hold at small concentrations: every
  Dirichlet row sums to one and a Beta draw is the ratio its gammas stand for.
  Below a concentration of about 0.03 most float32 gammas underflow to zero,
  and a row of zeros came out as zeros, a fifth of the rows at 0.03.
- `Nx.Rng.truncated_normal` and `Nx.truncated_normal` at float64 reach 8.3
  standard deviations. A margin kept for infinite bounds stopped every draw at
  5.3, so an interval such as `[5, 6]` was drawn with the wrong mean and one
  such as `[7, 8]` gave its lower bound every time. Past the reach of `erf`, an
  interval still collapses onto its bound nearer zero, as now documented.
- `Nx.Rng.fold_in_tensor` agrees with `Nx.Rng.fold_in` on a negative counter
  too, as documented. It derived a different key from every negative int32.
- `Nx.Rng.shuffle` and `Nx.shuffle` return a tensor whose first axis is empty
  unchanged. They raised, asking `permutation` for a permutation of nothing.
- `Nx.Rng.randint` and `Nx.randint` over a range wider than 2^31, such as
  all of int32, draw across the whole range. The offset from `low` overflowed
  int32, and every draw past the middle of the range landed on one value.
- A complex element prints its imaginary part with its own sign: `(1-2i)`,
  where `Nx.pp` printed `(1+-2i)`.
- `Nx.rsqrt`, `log2`, `sigmoid`, `asinh`, `acosh` and `atanh` at `float16`,
  `bfloat16` and the float8 dtypes compute at `float32` and round once, as
  every other element-wise operation does. Composed of several operations,
  they rounded each step and could be a unit in the last place off.
- `Nx.cast` of an empty tensor to or from `int4` and `uint4` returns an empty
  tensor. It raised "packed dtype not supported".
- `Nx.cast`, `Nx.div` and `Nx.mod_` document what they did: a float cast to
  an integer truncates, holds at the range and takes NaN to 0; an integer
  divided by zero, or its remainder by zero, is 0; a remainder has the sign
  of the dividend.
- **Breaking:** `Nx.inner` of matrices is the inner product over the last
  axes of both, with the other axes of `a` then `b`, as `tensordot` over
  them gives. It was `vecdot`, pairing rows.
- `Nx.dot` of two tensors of rank above 1 contracts the last axis of `a`
  with the second to last of `b` and keeps both batches, as documented. It
  raised on a shape mismatch.
- `Nx.tensorsolve` without `axes` solves `tensordot a x = b` with `b` on the
  leading axes of `a`, as its equation says. It took the trailing ones.
- `Nx.norm ~ord:`Two` of a vector is its 2-norm. It crashed in `svdvals`.
- `Nx.Rng.uniform` and `Nx.Rng.normal`, and so `Nx.rand`, `Nx.randn` and the
  samplers built on them, return a draw with a buffer of its own. A compiled
  program that read a draw several times recomputed the generator for every
  read: a matmul of 512 rows against a drawn matrix took 305 ms instead of 2.
- `Nx.slice` and `Nx.set` clamp the start of a stepped range (`Rs`) into the
  axis, as they clamp its stop and both bounds of `R`. A start outside the
  axis read zeros or raised.
- `Nx.create`, `init`, `empty`, `full`, `zeros`, `ones` and `eye` refuse a
  negative dimension. `create int32 [| -2; -3 |]` with six elements made a
  tensor of that shape.
- `Nx.to_bigarray` always copies, as documented. It shared the storage of a
  contiguous tensor, so writing the bigarray changed the tensor.
- `Nx.is_c_contiguous` holds for a tensor whose only disorder is in axes of
  size 1, and for an empty tensor. It was false for a fresh empty tensor.
  `Nx.contiguous` documents that it shares a C-contiguous tensor only at
  offset 0.
- `Nx.one_hot` compares indices and classes as int64. In the index dtype the
  classes wrapped: a `uint8` index 5 among 300 classes also marked class 261.
- `Nx.concatenate` refuses an axis out of bounds, for one tensor too;
  `Nx.split` refuses zero parts instead of raising `Division_by_zero`;
  `Nx.array_split` counts a negative index from the end, as a range does;
  `Nx.extract` compares the sizes of the condition and the tensor, not their
  shapes; and `Nx.compress` without an axis refuses a condition longer than
  the tensor, where it read zeros past the end.
- `Nx.flatten` works on every layout, a view where the layout allows one and a
  copy otherwise. It raised on a transpose, and so did `argmax`, `argmin`,
  `roll`, `repeat`, `cumsum` and the other scans, `take`, `compress` and
  `extract` without an axis. `Nx.ravel` stays a view and still refuses.
- `Nx.mean` over an empty axis is NaN, not 0, and refuses an integer dtype.
- `Nx.var` and `Nx.std` refuse `ddof` at least the count, as documented.
- `Nx.logical_and`, `logical_or`, `logical_xor` and `logical_not` read
  non-zero as true on every dtype and give zero or one. They were bitwise, so
  `logical_not` of 5 was -4 and `logical_and` of 2 and 1 was 0.
- `Nx.rshift` of a negative integer is an arithmetic shift, rounding toward
  negative infinity. It divided, so `rshift (-1) 1` was 0.
- `Nx.asinh`, `acosh` and `atanh` keep their accuracy at the edges of their
  domains: `asinh` of a tiny value was 0, `asinh neg_infinity` was NaN, and
  `acosh` of a large negative value was `neg_infinity`, not NaN.
- `Nx.exp2` and `Nx.sigmoid` are accurate to a few units in the last place.
  Both rounded `x * log 2` before exponentiating, an error that grew with `x`.
- `Nx.hypot` is infinite when either side is infinite, as IEEE 754 requires.
  It was NaN beside a zero.
- `Nx.reshape` refuses a shape with another number of elements when either
  side is empty. It accepted `reshape [| 3 |]` of an empty tensor, a view of
  three elements over no storage.
- `Nx.shrink` accepts an empty range, as `Nx.slice` does, and `Nx.slice` by an
  index on a tensor with an empty axis no longer raises from the backend.
- `Nx.eye ~m n` is an `n × m` matrix, as documented. It was `m × n`, and
  with more rows than columns a lower diagonal missed its entries in the
  extra rows.
- `Nx.arange` over an empty range is an empty tensor, whichever the direction
  of the step. It raised for a positive step and made `Nx.tril` and `Nx.triu`
  of an empty matrix raise.
- `Nx.squeeze ~axes` of a scalar refuses its axes, as it does for every other
  rank, and `Nx.one_hot` refuses `num_classes <= 0`, as documented.
- `Nx.argwhere` of a non-zero scalar has shape `[| 1; 0 |]`, one coordinate
  with no axis, as documented. It was `[| 0; 0 |]`.
- `Nx.reshape` views every layout that a view can hold, such as a flipped
  tensor reshaped across its axes, and documents that it refuses the others.
  It refused some layouts it could view.
- `Nx.roll` without an axis keeps the tensor's shape when the shift is a
  multiple of the size or the tensor is empty. It returned the flattened
  tensor.
- Eager `Nx_quant.apply` of a weight split on its experts over rows copied on
  each device no longer raises "operands on ... place one of them": the eager
  product decodes each matrix, and reads the rows, on the host.
- `Nx.take` (and `Nx.slice` with a list of indices) along the split axis of a
  split value is a copy on each device, as a reduction over that axis is: each
  device selects among its own rows. It raised, although a compiled program
  lowers the same gather as a sum across the devices.
- Eagerly, `Nx.zeros_like`, `ones_like`, `full_like` and `fill` of a split
  value are split the same way, each device holding its slice. They made a full
  copy on every device, so an optimiser state built from split parameters
  (`Vega.adam_init`) took as many times their memory as there are devices.
- Preserve distinct tensor, view and device identities when they are created
  concurrently; lost counter increments could make unrelated values alias in
  identity tables used by tracing.

- Eager `Nx.matmul` chooses its small-product loop by rows: fewer than 8 rows
  up to about 5 million multiply-adds take it (4 x 4096 x 16 `bfloat16`: 28 us,
  not 85), and everything else the blocked kernel (100 x 100 x 10: 10 us, not 27).
- On macOS, eager `float32` and `float64` `Nx.matmul` with few outputs no
  longer goes to Accelerate: it is faster (2 x 300000 x 2 `float32`: 0.14 ms,
  not 1.35), and each output currently has the bits of its row-column dot.
- Eager `Nx.matmul` with few outputs over a long contraction splits the
  contraction across cores: 3 x 70001 x 5 at `bfloat16` takes 0.18 ms, not
  0.44.
- Eager `Nx.matmul` with fewer than half a register tile of outputs (48 at
  `float32`) and a long contraction is faster: 2 x 30000 x 2 takes 0.08 ms, not
  0.38, where it had packed mostly empty tiles.
- Eager `Nx.matmul` of small matrices is up to 8 times faster (16 x 64 x 100
  `float32`: 0.009 ms, not 0.074), and each output currently has the bits of
  `Nx.dot` of its row and column.
- Eager `Nx.matmul` of 2 to 7 rows, or of a few columns, is 15 to 50 times
  faster: 2 `bfloat16` rows times 2880 x 5760 take 3 ms, not 45 (7 rows: 3.2 ms,
  not 156).
- Eager float `Nx.sum` spreads every element of a run over sixteen partial
  sums by its position, strided or not, so a sum whose length is not a multiple
  of 16 can change in the last bit; currently a vector's sum ignores its stride.
- Eager `Nx.matmul` and `Nx.dot` of a vector and a matrix, in either order,
  are over 20 times faster at `bfloat16` and `float16` (a row times 2880 x 5760:
  0.95 ms, not 21.8), and each output currently has the bits of its row-column
  dot.
- Eager `Nx.dot` of two vectors, `Nx.vecdot` and `Nx.inner` are faster and
  more accurate, and give the same bits on any thread count: a dot of
  2^20 `bfloat16` elements takes 0.06 ms, against 0.39 ms before.
- `Nx.take` from a table on several devices at positions split across them
  runs where they live: each device takes its own positions' rows, and the
  result is split as the positions are. It raised as a gather along the split
  axis.
- `Nx.dot`, `Nx.vdot`, `Nx.inner`, `Nx.vecdot` and `einsum "i,i->"` of vectors
  compute as `Nx.matmul` does, so every product of the same vectors gives one
  answer. At `float16`, `bfloat16` and float8 they rounded each product first.
- Eager `Nx.argsort` runs the backend argsort alone. It sorted the values too
  and discarded them: 2^20 float32 entries take 7 ms, not 15.
- Eager `Nx.sort` and `Nx.argsort` share one stable radix sort: sort's values
  are its input's elements at its indices, bit for bit, where a descending sort
  reversed runs of `-0.` and `0.`. `Nx.sort` of 2^20 float32 takes 15 ms, not 145.
- Add `Nx_dtype.Scalar.encode` and `decode`, the bits of a float in float16,
  bfloat16 and the float8 formats, fnuz variants included, rounded once to
  nearest even. nx's stores round the same way, except a `float` stored into
  `float16`, which rounds through `float32` first. tolk folds constants with
  them.
- **Breaking:** dtypes move to a new library, `nx.dtype`, which depends on
  nothing: `Nx_core.Dtype` is `Nx_dtype`, and `Nx_buffer.to_stdlib_kind` is
  `Nx_dtype.to_bigarray_kind`, which returns an option. `Nx_dtype.Scalar`
  names storage formats without type parameters, for code that moves or
  compiles bytes. `Nx_core.Dtype.packed`, `pack` and `Packed` are gone.
  `Nx.dtype` is unchanged.
- Encoding a NaN as `float8_e4m3` or `float8_e5m2` keeps its sign, as
  infinities, overflow and the other float dtypes already did; it was always
  `0x7f`.
- Writing a `float` into a `bfloat16` or `float8` tensor, and `Nx.cast` to
  those dtypes from wider floats or integers, round once to nearest even. They
  rounded to `float32` first, so `Nx.scalar Nx.float8_e4m3 336.00000000000006`
  was 320 instead of 352.
- Operations over values on several devices keep their results on those
  devices: an elementwise result keeps its operands' split, which must be
  alike, a reduction over the split axis is a copy on each device, and an
  operation along the split axis (a sort, a pad, linear algebra) raises.
- **Breaking:** `Nx.Placement.t` is abstract. Build a placement with `host`,
  `device`, `replicated` and `sharded`, and take it apart with `devices` and
  `window` (the part a device holds); `equal` now holds for copies in any order.
- **Breaking:** the key type `Nx.Rng.key` is now `Nx.Rng.t`, a private type
  that only `Nx.Rng` builds (`Rng.key seed` still makes one), so arithmetic or
  slicing on a key, or an arbitrary int32 tensor, no longer type-checks as
  one. `(k :> Nx.int32_t)` reads a key's words, `Rng.of_tensor` turns loaded
  words back into a key, and `Rng.ptree` walks a key in a structure or a
  compiled function's signature, as `[@@deriving ptree]` does for a part of
  type `Nx.Rng.t`.
- Add `Nx.Rng.split_batch ~n key`: the keys of `split ~n key` as one batch,
  whose lanes each see one key under `Rune.vmap`. It replaces stacking split
  keys into an `[n; 2]` tensor by hand.
- **Breaking:** an index outside the axis, negative included, drops the update
  in `Nx.scatter` and reads zero in `Nx.take` and `Nx.take_along_axis`, eagerly
  as under `Rune.jit`. Eager calls used to raise `Invalid_argument`, so a
  function behaved differently once compiled.
- A slice of a split value inside one shard (`Nx.slice [ I i ]`, `Nx.item`) is
  a view of that shard on its device: `Nx.item` reads one element where it read
  the whole value, and `Rune.jit` refuses to consume it (pass `Nx.copy` of it).
- Moving a value split over several devices (`Nx.transpose`, `reshape`,
  `slice` by indices and unit-step ranges, `flip`, `broadcast_to`,
  `sliding_window`) keeps it split and copies nothing, where it read the value
  to the host; a movement that would cross devices raises `Invalid_argument`.
- **Breaking:** so do `Nx.roll` along the split axis (but by one shard of a
  value split in two: copies on both devices, as compiled), `Nx.array_split`
  and `split` across shards, and `Nx.flatten`, `ravel`, `diagonal` or
  `reshape [| -1 |]` of a value split on a later axis; the error names the
  refused movement (a reshape, cut, flip or window of the split axis). Read the
  value with `Nx.to_array`, or place it on one device first.
- `Nx.repeat` along an axis copies once instead of concatenating a slice per
  index: 1024x256 twice along axis 0 allocates 840 words, where it allocated
  313,496.
- Add `Nx.Ptree`, structures of tensors. A structure is a type `'a t` with one
  function, `walk`, that visits its parts with a `Walk` cursor: `leaf` for the
  parameter's positions, `tensor` for tensors of a fixed type, `int` and `case`
  for the data a compiled program depends on, and `field`, `index`, `option`
  and `list` for the rest. `instantiate`, `nest`, `tensor`, `unit`, `pair`,
  `option`, `list` and `iso` build a structure at one type, `'s Nx.Ptree.t`,
  which transformations, optimisers and checkpoints take; `@->`, `consumes`
  and `returns` build the signature of a compiled function.
- `Nx.Ptree.map`, `map2` and `fold` pass each tensor its path, whose typed
  segments (`Path.segments`) make a mask a pattern match. `cast` and
  `Payload.map`, `map2` and `fold` change the payload's type, for casts and
  for metadata shaped like a model. `map2` raises naming the first path at
  which its two values differ and what each holds there. `visits` lists each
  tensor and each report a walk makes, with its path, for a structure's
  tests; `pp_visit` prints one. `flatten` returns a value's tensors in walk
  order and its `Skeleton.t`, which compares and hashes; `rebuild` puts
  tensors back into a value.
- `Nx_quant.walk` walks a weight with a `Walk` cursor, reporting its format as a
  case, and `Nx_quant.ptree` is its structure; `map2` and `iter` go, as
  `Nx.Ptree.map2` and `fold` over `Nx_quant.ptree`.
- **Breaking:** `Nx_io.packed` and `Nx_io.to_typed` become `Nx.packed` and
  `Nx.unpack`, one packed tensor type for files and checkpoints. `unpack`
  raises `Invalid_argument` naming both dtypes, where `to_typed` raised
  `Failure`. `Nx_io.packed_dtype` and `packed_shape` are removed: match on
  `Nx.P`.
- `Nx.top_k` on `float8_e4m3` with `k` up to 8 returns the greatest entries.
  It returned the first entry repeatedly: `Dtype.min_value` was `-infinity`,
  which that dtype encodes as NaN; its bounds are now `-448` and `448`.
- Add `Nx_quant.place`, which places a quantised weight part by part and
  refuses a split of its inputs that would cut a 32-value group.
- Add `nx.quant`: `Nx_quant.mxfp4` builds a weight from a checkpoint's MXFP4
  codes and scales without a copy, and `Nx_quant.apply ?ids` multiplies by it
  with `Nx.matmul`'s shapes, decoding a bounded chunk at a time eagerly.
- `Nx.top_k` with `k` above 8 no longer takes one entry per pass, and over an
  axis longer than 2048 no longer sorts it: a radix select on the bits of each
  entry finds the `k`th greatest and only the `k` entries kept are ordered.
  Compiled on Metal, 40 of 50257 float32 take 2 ms instead of 890 ms and 512
  of 32768 0.9 ms instead of 3.7 ms; eagerly it costs about 2.5 times the old
  sort at sampling sizes, and 64 rows of 131072 peak at about 480 MB instead
  of 200 MB. Compiled, NaN now comes last as documented.
- Add `Nx.bitcast`, which reads each element's bits as another dtype of the
  same width without converting it, NaN payloads and subnormals included. It
  compiles under `Rune.jit`, except to or from float8, which the compiler
  emulates and `Rune.jit` refuses; it maps under `vmap` and has zero
  derivative.
- Add `Nx.Device`, `Nx.Placement`, `Nx.place` and `Nx.placement`: a value
  can live on a device a runtime opens, and where it lives is a value. An
  operation on placed operands returns a placed result, host operands join
  them, and operands on two devices raise `Invalid_argument`.
- A read of a placed value (`item`, `to_array`, `to_bigarray`, `pp`) copies
  the elements it reads and leaves the value where it is.

- Fix `Nx_io.load_safetensors` and `save_safetensors` corrupting Unicode and
  control characters in tensor names. Decode JSON Unicode escapes and surrogate
  pairs, emit valid JSON escapes, and reject malformed string escapes.

- `Nx_io.load_safetensors` maps the file instead of reading it: loading reads
  the header only, and each tensor is a view of the file whose pages are read
  when first used. It used to hold the file twice in memory and copy every
  tensor out element by element; a 2.5 GB checkpoint that took 3 s to load
  takes 0.05 s. Entries at an address their dtype cannot be read from are
  copied. A loaded file must not be modified in place while its tensors are
  alive; `Nx.copy` detaches a tensor from its file.
- `Nx_io.load_safetensors` loads `F8_E8M0`, `F4`, `F6_E2M3` and `F6_E3M2`
  entries as their `uint8` bytes instead of dropping them with a warning,
  loads 16-bit entries at odd offsets instead of raising, and rejects a file
  whose length disagrees with its header, a header that names a tensor twice,
  and anything that is not a regular file. Its errors name the file.
- `Nx_io.save_safetensors` no longer truncates its destination in place: it
  writes a temporary file beside it, syncs it and renames it, so a crash or a
  failed save leaves the previous file whole. If the rename is refused the
  written file is kept and the error names it. Saved files now have mode
  `0o640`, as the other `Nx_io` writers give theirs.
- `Nx.cast` at the tensor's own dtype is the tensor itself and no longer a
  copy: only a change of dtype allocates. Use `Nx.copy` for fresh storage.
- Fix `einsum` with a repeated index that does not sit at the end of its
  operand (for example `abnb->an`): the surviving label stayed where the index
  first appeared instead of moving with the diagonal to the end of the
  operand, so the result took the wrong shape and values.
- Add `Nx.top_k ~k ?axis`, the `k` greatest entries along an axis and their
  positions, as `(values, indices)`: the first `k` of a descending `sort`, ties
  lowest position first, NaN last. Up to 16 entries it costs `k` passes and no
  sort, which is what a mixture-of-experts router needs under `Rune.jit`, where
  `argsort` is quadratic in the axis. `values` differentiates.
- `sort`'s documentation said NaN sorts first in descending order. It sorts
  last in either direction, as the backend contract states.
- `scatter` states what a broken `unique_indices` promise leaves: a position
  selected more than once holds an unspecified one of its updates under
  `` `Set `` and an unspecified value under `` `Add ``, and every other position
  is exact. The whole result used to be undefined. Eager and `Rune.jit` both
  keep the narrower promise.
- Add `sliding_window`, a zero-copy view framing a tensor into windows of a
  given length along an axis. Framing without a copy was reachable only through
  `stft`, which bundles a taper and a transform with it; a reduction, a filter
  or an overlap-save convolution wanting the frames themselves had to gather
  them with `extract_patches`.
- **Breaking**: tensors are values (RFC 0001, `doc/rfc/0001-tensors-as-values.md`).
  The in-place writers `blit`, `set` (int-list form), `set_slice`, `set_item`,
  `put`, `index_put`, `put_along_axis` and the `.%{}<-` / `.${}<-` operators
  are gone, with `empty` and `empty_like` (a value has no uninitialized form
  to fill in; use `zeros`) and the `?mode` of `take`. One
  functional `set specs v t` returns `t` with `v` at the selected positions,
  and a new index form `D (start, len)` selects a run from a run-time start,
  so a KV-cache write traces once for every position. Views, broadcasts and
  overlapping windows are ordinary tensors; `of_bigarray` takes ownership.
  Tensor-valued indices (`take`, `scatter`) must lie in range: the C backend
  no longer wraps negatives and raises `Invalid_argument` instead of
  `Failure`. Build tensors element by element with `create`, `init`, `stack`
  or a filled bigarray.
- The backend contract gains `update`, the pure window write `set` lowers to
  for single indices, unit-step ranges and run-time runs; a compiler can
  perform it in place.
- Add `Nx.erfinv`, the inverse error function, beside `erf`. At float64 it
  carries double precision, a seven-digit polynomial refined by Newton steps
  against a series for `erf` that does not depend on the backend's own. It
  was an internal helper of `Nx.Rng.truncated_normal`, whose float64 draws now
  carry double precision as well instead of seven digits.
- `Nx.Rng.poisson` runs at its rate's compute dtype instead of always at
  float64, so a float32 rate compiles on Metal and every other device without
  double precision. The rejection test now evaluates the log pmf in a form
  that does not cancel terms of size `rate log rate`, which is what had forced
  float64. Draws for a float64 rate are unchanged; a float32 rate gives a
  different stream than before.
- Add `Nx.Rng.bits k shape`, the generator's raw uniformly random 32-bit
  words, so a distribution the module does not provide can be built on the
  same generator with the same purity and transform guarantees. `uniform` at
  float32 is the low 24 bits of these words scaled into `[0, 1)`.
- Add `Nx_io.encode_png`, the PNG bytes of a `uint8` image tensor as a
  string, for embedding images in documents or sending them over a socket
  without going through a file.
- The C backend compiles with mingw-w64 on Windows. It no longer relies on the
  C11 `CMPLX` constructors or `aligned_alloc`, which mingw's headers lack, and
  registers no fork handlers there. The I/O layer writes through Windows file
  handles, so `save_npy`, `save_npz`, `save_txt`, the image encoders, and gzip
  output work there.
- `save_txt` writes exactly the requested `newline`. The channel was in text
  mode, which turned every line ending into CRLF on Windows.
- `save_txt` formats floats itself, correctly rounded to numpy's 19
  significant digits on every platform. A C runtime only has to round 17,
  and Windows' stops there and prints three-digit exponents.
- **Breaking:** the samplers take their distribution parameters as tensors,
  elementwise, and the draw has the parameters' shape and dtype:
  `Nx.Rng.bernoulli k p`, `poisson k rate`, `gamma k concentration`,
  `beta k a b`, `truncated_normal k lower upper` and `dirichlet k
  concentration` (components on the last axis), with the keyless `bernoulli`
  and `truncated_normal` following. A tensor of rates now gives one count per
  rate in a single draw, and a parameter that is a jit input or a vmap axis
  traces or batches the draw with it. A scalar parameter is spelled
  `Nx.broadcast_to shape (Nx.scalar dtype v)`. Parameter values are no longer
  checked, since under a transform they are not known when the program is
  built; each docstring states its domain. `Nx.Rng.uniform` loses its unused
  `?low`/`?high`, and `categorical` loses `?shape`: its result is the shape of
  the logits with the axis removed, so broadcast the logits for more draws.
  `randint` keeps its host-int bounds, as every size in Nx is a host int.
  `truncated_normal` differentiates through both bounds; `gamma`, `beta` and
  `dirichlet` differentiate through their concentrations with the bias of
  the accepted proposal. Draws from `gamma`, `beta`, `dirichlet`,
  `truncated_normal` and `categorical` differ from before for a given key.
- `Nx.Rng.poisson` samples at any rate; the cap at 100 is gone. Below rate
  10 the count is read off the cumulative distribution with a single uniform,
  from 10 up it comes from a transformed rejection sampler with a fixed round
  count, so the work per element no longer grows with the rate. Draws for a
  given key differ from before.
- The inverse error function behind `Nx.Rng.truncated_normal` now has a finite
  derivative at zero. Its tail branch took `sqrt` of a quantity that is zero
  at that point, and although the branch is never selected there, its infinite
  derivative turned the gradient into NaN under `Rune.grad`. Values are
  unchanged.
- Every keyed draw (`Nx.Rng.uniform`, `normal`, `permutation`, `split`, and
  the samplers built on them) no longer copies its key once per Threefry
  block before hashing. The copy was as large as the draw itself; the kernel
  reads the key through a stride-0 view instead. Values are unchanged.
- Add `Nx.solve_triangular ?upper ?transpose ?unit_diag a b`, a first-class
  triangular solver named after its scipy analog. It skips the factorization
  cost of `solve` for a pre-triangularized `a`; `b` is a vector or a stack of
  right-hand sides, batched like `a`.
- **Breaking:** the backend operation `triangular_solve` and its effect
  `E_triangular_solve` are renamed `solve_triangular` and `E_solve_triangular`.
  Out-of-tree backends and effect handlers must follow.
- `Nx.diag` no longer reads its operand back to the host, so it traces under
  `Rune.jit` and construction differentiates through `scatter`. It now raises
  for inputs of rank above 2, as documented; the undocumented reading of the
  row-major flattening as a matrix is gone. The packed `int4` and `uint4`
  dtypes, which the graph operations reject, are no longer accepted either.

- Speed up batched `fft`, `rfft` and `irfft` in the default C backend: the
  worker count was picked as though a transform line cost one pass over its
  samples, so a stack of a few dozen medium-length lines ran on a single core.
  Lines are now weighted by the `n log n` work a transform actually does; short
  stacks still run serially.
- Speed up the C backend's real FFTs: even-length `rfft` and `irfft` pack into
  a half-size complex transform, and `irfft` no longer stages a serial copy of
  its input. Even-length `rfft` now returns exactly real DC and Nyquist bins.
- Pad Bluestein lengths (a prime factor above 13) to the nearest 7-smooth size
  instead of the next power of two, so `fft` at 4099 runs on 8232 points, not
  16384. Results there change in the last bits (error within ~1.1x of before).
- `irfft` at an odd Bluestein length now discards `Im X[0]` like every other
  length, so a non-Hermitian input no longer leaks ~1e-16 of it into the output.
- `irfftn` and `irfft2` now honour `s` along every transformed axis: the
  leading, complex axes are cropped or zero-padded to the requested lengths,
  as in `ifftn`. Previously only the last axis was resized while the
  normalization still divided by the full product of `s`, so a leading-axis
  size change returned the wrong shape and scale.
- **Breaking**: `Rng.run ~seed f` is replaced by `Rng.with_key (Rng.key seed) f`,
  and `Rng.split_off` is renamed `Rng.next_key`. The two entry points were one
  handler differing only in what it rooted at, and `~seed` was the one that
  silently rooted at a constant — the case `Rune.jit` must refuse. A scope now
  visibly inherits its root key's properties: root it at a jitted function's
  input leaf and the keyless samplers inside compile.
- Add `Rng.beta` and `Rng.dirichlet`, built from `Rng.gamma` and inheriting its
  approximation. `dirichlet` puts its components on a new trailing axis, so a
  draw of `shape` gives `shape @ [| n |]` with every row on the simplex.
- Add `Rng.gamma` and `Rng.poisson`. `gamma` is the keystone
  for statistical work — beta is `g1 /. (g1 +. g2)`, a Dirichlet is a vector of
  gammas over its own sum, chi-square and Student's t follow in turn — and its
  docstring records those derivations. It is the one sampler in `Rng` that is
  not exact: every gamma algorithm rejects and a rejection loop cannot be
  traced, so eight attempts are drawn and the first acceptance taken, leaving
  about one element in `1e14` to fall back to the mean. `poisson` is exact,
  expressing Knuth's count as a cumulative product rather than a loop that stops
  when the draw says so; its cost is `O(rate)` per element, so `rate` is capped
  at 100.
- Add `Rng.gumbel` and `Rng.exponential`. `gumbel` is the noise
  behind `categorical`, which now builds on it; adding it to log-probabilities
  and taking a softmax instead of an argmax gives the relaxed, differentiable
  form. `exponential` is `-log (1 - u)` — built from `1 - u` because a draw can
  be exactly `0`, where `-log u` would diverge.
- float64 draws carry 53 random bits instead of 24. `uniform` built every draw
  in float32 and widened it, and `normal` ran the whole Box-Muller transform in
  float32, so a float64 sample was a double holding float32 noise — 2^24
  distinct values instead of 2^53. This matters most to `Norn`, whose HMC and
  NUTS samplers draw momenta with `randn f64` and accept with `rand f64`.
  float64 streams therefore change; narrower dtypes are unaffected.
- Add `Rng.fold_in_tensor`, deriving a subkey from a value known only at run
  time — a step counter carried through a compiled loop, a device index.
  `Rng.fold_in` takes a host `int`, so under `Rune.jit` it freezes whatever the
  counter held at trace time; only the mapped-axis specialisation
  (`Rng.fold_in_axis`) was reachable before.
- `permutation` and `shuffle` order 64-bit random sort keys instead of a
  24-bit uniform draw. A uniform carries at most 24 significant bits, so at
  60,000 elements — one MNIST epoch through `Kaun.Data` — about 107 pairs
  collided and `argsort` resolved every one of them towards the input order.
- Unscoped draws (`Nx.rand` and friends outside `Rng.run`/`Rng.with_key`) are
  seeded from system entropy and so differ from run to run, as the docs always
  claimed. They came from OCaml's default `Random` state, which is seeded
  deterministically, so they repeated exactly on every run — and would have
  started varying if any linked library called `Random.self_init`. Open a scope
  for reproducibility.
- `Rng.truncated_normal`, `Rng.categorical`, `Rng.permutation` and
  `Rng.shuffle` complete the keyed sampler set: every distribution now has a
  pure form that composes with `Rune.jit`, `vmap` and `pmap`, and each keyless
  sampler is that form applied to a subkey of the ambient scope. The keyless set
  at the top level of `Nx` is closed at the reflexive draws (`rand`, `randn`,
  `randint`, `bernoulli`, `truncated_normal`, `categorical`, `permutation`,
  `shuffle`); the distributions added since take a key, which also leaves
  `gamma` and `beta` free for the special functions of those names.
- `truncated_normal` draws by inverting the conditioned distribution instead of
  rejecting out-of-range samples. The rejection loop read its stopping
  condition back to the host, so it could not be traced or compiled at all, and
  it gave up after 1000 rounds on a narrow interval; inverting costs one draw
  per element whatever the bounds. `Kaun.Init.glorot_normal`, `he_normal` and
  `lecun_normal` go through it, so they are now compilable.
- **Breaking**: `truncated_normal` takes `~lower` and `~upper` before its dtype,
  matching the keyed form.
- **Breaking**: `randint` and `Rng.randint` take their bounds as `?low` and
  `~high` and always return `int32`, replacing the trailing positional `low`,
  the arbitrary `?high` default of `10`, and the dtype argument. The dtype
  argument accepted float dtypes and raised at run time, and the draw was
  computed in `int32` before being widened, so ranges beyond `int32` were
  silently wrong; both are now impossible. Cast the result for another integer
  width. Bounds outside `int32` raise `Invalid_argument` (#196).
- Sliding-window extraction is 10-77x faster on the C backend. The kernel used
  to divide ten times per output element; it now walks contiguous runs, so
  convolution (`correlate`, `convolve`) and the pooling filters
  (`maximum_filter`, `uniform_filter`) move at memory bandwidth instead of
  around 0.4 GB/s.
- `relu`, `sigmoid`, `clamp`, `hypot`, `tril`/`triu`, `logical_not`, and the
  boolean reductions build their constant operand as a scalar instead of a
  full-size tensor, making them 1.4-3.1x faster. Values are unchanged.
- Element-wise ops against a scalar operand are 1.4-9.5x faster. A broadcast
  0-d operand (every `_s` variant, and any operand broadcast along the innermost
  axis) used to fall onto a scalar code path; the C map, comparison, and `where`
  kernels now hoist it out of the loop and stay vectorized. Results are
  unchanged bit for bit.
- Speed up `Nx.Rng.uniform` about 2x and `Nx.Rng.normal` about 4x (4M float32
  draws: 550ms to 283ms and 1153ms to 301ms). A Threefry row yields two words
  and `uniform` discarded one of them; `normal` then drew two uniforms per
  sample and threw away the sine half of Box-Muller. Both halves are now kept.
  Values for a fixed key change.
- Fix `Nx.Rng.uniform` and `Nx.Rng.randint` returning `high`, which both
  document as excluded: the draw is now built from the random bits so it is
  half-open at every float dtype, rather than by float32 rounding that reached
  `1.0` about once in 2^24 draws (once in 4000 at float16). `Nx.rand` also
  samples at its result dtype instead of narrowing a float32 draw.
- Fix `Nx.Rng.randint` skewing towards zero for a negative `low`: it truncated
  the shifted float, so `low` was never drawn and `0` was drawn twice as often.
  Values for a fixed key change.
- **Breaking:** the real FFT family (`rfft`, `irfft`, `hfft`, `ihfft` and their
  2-D/N-D variants) and `fftfreq`/`rfftfreq` now take the output dtype first,
  like the constructors. It selects storage precision independent of the
  input's, so a float32 signal can keep a `complex64` spectrum.
- Add complex accessors `Nx.real`, `Nx.imag`, `Nx.magnitude`, `Nx.angle` and
  the `Nx.complex ~re ~im` constructor, taking the result dtype first so a
  `complex64` spectrum can yield a `float32` magnitude. `Nx.conjugate` now runs
  the same element-wise kernels instead of reading every element back to the
  host one at a time; it returns NaN components where the input has a
  non-finite one, which the boxed version handled exactly.
- Add `Nx.stft`, `Nx.istft`, and `Nx.hann` for short-time Fourier analysis.
  `stft` frames through a view rather than materializing, returns time-major
  `[frames; bins]`, and tapers with a periodic Hann by default; `istft`
  overlap-adds and divides by the windows' own envelope, so any
  `step <= window` inverts.
- **Breaking:** `Nx_core.Backend_intf.S` gains `sliding_window`, a pure-view
  movement producing overlapping windows. Out-of-tree engines implementing the
  `nx.backend` virtual library must add it.
- **Breaking:** consolidate tensor formatting around the compact `Nx.pp`,
  `Nx.to_string`, and `Nx.print`. Remove `pp_data`, `data_to_string`,
  `print_data`, `format_to_string`, `print_with_formatter`, `dtype_to_string`,
  and `shape_to_string`; add `Nx.pp_shape` alongside `Nx.pp_dtype`.
- The default C backend and `nx.io` codecs now build cleanly with strict GCC
  warnings and single-pass ELF linkers. Empty DEFLATE streams also avoid
  allocating the encoder's match tables.
- Add float32- and float64-preserving `dct`, `idct`, `dst`, and `idst`
  transforms of types I–IV, including N-D variants and forward, backward, and
  orthonormal scaling modes.
- `truncated_normal` now rejects integer dtype witnesses at compile time,
  matching the other normal samplers.
- `rand` and `randn` now reject integer dtype witnesses at compile time instead
  of accepting them and raising `Invalid_argument` at runtime.
- Fix elementwise arithmetic on non-contiguous views: `mul_s`, `div_s`, and
  tensor `div` now honor the view's offset and strides instead of reading
  out-of-view values from the underlying buffer.
- Replace the vendored camlzip and stb image libraries with owned ISC codecs in
  `nx.io`. NPZ no longer creates a temporary NPY file for each entry, image and
  archive decoding writes directly into Nx buffers, and Nx no longer needs zlib
  or pkg-config.
- Speed up `load_npy`, stored `load_npz`, compressible `save_npz`, and `gunzip`
  by removing redundant checksum passes, processing stored data in larger
  batches, and bounding DEFLATE match searches.
- Add `Nx_io.gunzip` with checksum validation and atomic destination replacement.
  **Breaking:** `save_image` now supports only modern PNG and JPEG output; BMP
  and TGA output and the public `nx.zip`/`nx.io.stb_image*` libraries are removed.
- Replace the default `nx.c` backend with the self-contained C implementation.
  FFT and dense linear algebra no longer require PocketFFT, OpenBLAS, LAPACKE,
  libomp, or platform depext configuration; macOS automatically uses Accelerate
  for eligible matmuls and every platform retains the owned GEMM fallback.
  **Breaking:** the public `nx.pocketfft` vendored library is removed.
- Preserve Nx semantics across the backend cutover for detailed `matmul` shape
  errors, empty `all`/`any`, vector right-hand sides, explicit-size real FFTs,
  and complex pseudoinverses.
- Prevent parallel `nx.c` operations from hanging in a forked child by rebuilding
  the backend worker pool after `fork`.
- Correct the backend interface docs: `reshape` never copies — it raises
  `Invalid_argument` when the existing strides cannot express the new shape —
  and `triangular_solve`'s `transpose` solves with the conjugate transpose
  (`Aᴴ`) for complex dtypes.
- Unify random number generation on one splittable Threefry generator, reached
  through `Nx.Rng`. The explicit samplers `Nx.Rng.uniform`/`normal`/`randint`/
  `bernoulli` are pure, order-independent functions of a key; the implicit scope
  `Nx.Rng.run`/`with_key` still drives the keyless `Nx.rand`/`randn`/… with no
  key argument, and now shares the explicit stream by construction (`Nx.rand` is
  `Nx.Rng.uniform` on a subkey). A key is now a transparent `[|2|]` int32 tensor
  (previously an opaque host int), so it flows as a parameter-tree leaf, a jit
  input and a `vmap`/`pmap` axis; `Nx.Rng.fold_in_axis` derives a per-lane key
  under a transform. `Nx.Rng.to_int` is removed — read a key with `Nx.to_array`.
  Breaking: the same seed now yields different `Nx.rand`/`randn`/… values.
  Migration: re-bless any exact-value goldens; keyless call sites are unchanged.
- Split the backend contract's `eig`/`eigh` (each a `vectors:bool -> ... option`
  returning an optional vectors component) into four total functions matching
  the public API: `eigvals`/`eigvalsh` return values only, `eig`/`eigh` return a
  non-optional `(values, vectors)` pair. The values-only variants drive the
  cheaper LAPACK no-vectors path. Removes a representable invalid state (a
  runtime flag steering an option) from the contract. `Nx.eig`/`eigh`/`eigvals`/
  `eigvalsh` are unchanged.
- Make the backend contract's `scatter` take required `~mode` and
  `~unique_indices` labels instead of optionals, adopting the rule that
  backend-contract operations carry no optional arguments (user-facing defaults
  live on the frontend). `Nx.scatter` keeps its `?mode`/`?unique_indices`
  defaults.
- Collapse the backend contract's four reductions (`reduce_sum`, `reduce_prod`,
  `reduce_max`, `reduce_min`) into a single `reduce ~op ~axes`, matching
  `associative_scan`. The op always returns the result with the reduced axes
  removed; the frontend reinserts size-1 axes for `~keepdims:true`, so backends
  no longer implement `keepdims`. `Nx.sum`/`max`/`min`/`prod` are unchanged.
- Split the backend contract's `div` into `fdiv` (IEEE 754 float/complex
  division) and `idiv` (truncated integer division), matching the effect
  layer's existing dtype dispatch. Backend implementors now provide two
  domain-specific primitives instead of one that branches on dtype; the
  frontend selects between them. `Nx.div`'s behavior is unchanged.
- Support boolean-mask indexing in `slice` and `set_slice`: an `M mask` spec
  selects (or writes at) the positions where the rank-1 boolean `mask` is true
  along the axis it addresses. The mask length must equal that axis.
- Require the labeled argument `~indices` in `take` and `take_along_axis`, for
  consistency with `put`, `scatter`, and the other indexing functions.
- Require `~axis` in `concatenate`. The old axis-less form silently flattened
  every input; ravel the inputs first to recover it.
- Remove `squeeze_axis` and `unsqueeze_axis`; use `squeeze ~axes:[ i ]` and
  `unsqueeze ~axes:[ i ]`.
- Remove the per-element tensor-form `map`, `iter`, and `fold` (each scalar
  presented as a scalar tensor); use the faster `map_item`, `iter_item`, and
  `fold_item`, which pass raw scalars.
- Remove the numpy stack shorthands `vstack`, `hstack`, and `dstack`, along
  with the `Nx.Infix` concatenation operators `( @= )` and `( @|| )`. Use
  `concatenate`/`stack` directly, reshaping 1-D inputs as needed.
- Remove the commutative reverse-scalar aliases `radd_s`, `rmul_s`,
  `rmaximum_s`, and `rminimum_s`; use `add_s`, `mul_s`, `maximum_s`, and
  `minimum_s` (the operands commute). The non-commutative `rsub_s`, `rdiv_s`,
  `rpow_s`, and `rmod_s` remain.
- Remove redundant property and conversion aliases: `size` (use `numel`),
  `dims` (use `shape`), `astype` (use `cast`), `clip` (use `clamp`), `invert`
  (use `bitwise_not`), `expand_dims` (use `unsqueeze ~axes`), `identity` (use
  `eye`), `stride i t` (use `(strides t).(i)`), and `lerp_scalar_weight` (use
  `lerp` with a `scalar_like` weight).
- Remove the duplicate `cmp*` comparison family (`cmplt`, `cmpne`, `cmpeq`,
  `cmpgt`, `cmple`, `cmpge`). Use the named spellings `less`, `not_equal`,
  `equal`, `greater`, `less_equal`, `greater_equal` instead.
- Declare the licenses of the vendored components in `nx.opam`: the package
  now advertises `ISC AND LGPL-2.1-or-later WITH OCaml-LGPL-linking-exception
  AND BSD-3-Clause AND (MIT OR Unlicense)` covering camlzip, pocketfft, and
  stb_image, instead of claiming plain ISC.
- Remove the `?out` parameter from the backend `fft`/`ifft`/`rfft`/`irfft`
  operations. It was the only destination-passing parameter in the backend
  interface and the frontend never passed it; the FFT ops now allocate their
  result like every other compute operation.
- `einsum` failures now raise `Invalid_argument` with an `einsum:`-prefixed
  message like every other frontend error, instead of bare `Failure`.
- Remove the unused scalar-arithmetic surface from `Nx_core.Dtype`: `add`,
  `sub`, `mul`, `div`, and `bits`. Element arithmetic is performed by the
  backend kernels; these host-side helpers had no callers.
- Remove the unused validity-mask machinery from `Nx_core.View`: the `?mask`
  argument of `View.create` and the `mask`, `is_valid`, `linear_index`,
  `pad`, `strides_opt`, `can_get_strides`, and `is_materializable` functions.
  No view ever carried a mask (eager `pad` copies into a fresh buffer), so
  every view now has well-defined strides and `View.strides` is total.
  `View.create` validates the length of explicit `?strides` eagerly.
- Fix `float8_e4m3` conversions: the top binade was broken (256–448 saturated
  to 448 on write and decoded as 240 or NaN on read) and values below `2^-6`
  underflowed to zero instead of using the format's subnormals down to
  `2^-9`. Infinities now convert to NaN instead of saturating to ±448: an
  infinity did not overflow, and no finite value stands for it. `float8_e5m2`
  subnormal rounding now keeps the sticky bits, so round-to-nearest-even
  resolves ties correctly. Both conversions apply to reading and writing
  single elements and every C kernel operating on float8 tensors.
- The element types `int4_signed_elt`, `int4_unsigned_elt`,
  `int8_signed_elt`, `int8_unsigned_elt`, `int16_signed_elt` and
  `int16_unsigned_elt` are renamed `int4_elt`, `uint4_elt`, `int8_elt`,
  `uint8_elt`, `int16_elt` and `uint16_elt`.
- Reductions along non-innermost axes stream rows instead of striding a cache
  line per element: `sum ~axes:[0]` on 512×512 is ~9.6× faster (and `mean`
  with it), with bit-for-bit identical results.
- Vectorize `sum` along the contiguous axis (`sum ~axes:[1]` on a C-contiguous
  matrix): ~12.5× on 512×512.
- Contiguous elementwise ops stay serial-SIMD below 16M elements instead of
  parallelizing at 32768: a single vectorized core saturates memory bandwidth,
  so `add`/`mul` on 1M floats are ~7× faster on Apple Silicon.
- `copy`, `contiguous`, and axis-aligned `concatenate` collapse to a single
  `memcpy` when source and destination regions are contiguous (~31× on a
  512×512 `concatenate`); results are bit-identical.
- Speed up full-array `sum`: the reduction is now vectorized instead of
  parallelized (fork/join overhead dominated the bandwidth-bound sum) — up to
  125× at 128×128 and 20× at 1M elements on Apple Silicon.
- Benchmark suites across the workspace now run under a dedicated `bench`
  alias with a shared lock instead of `runtest` (`nx` and its
  `matmul`/`conv2d`/`einsum` suites, `norn`, `talon`, `vega`, `fehu`,
  `brot`, `sowilo` — matching `rune` and `kaun`, which already
  did this). `dune runtest` no longer runs perf regression checks (which could
  fail an ordinary test run on measurement noise); run them with
  `dune build @bench`.
- Expand `bench_nx` to ~30 cases across `binary`, `unary`, `reduce`, and
  `structural` groups — adding `sub`/`div`, unary elementwise, axis-wise
  reductions, broadcasting, non-contiguous inputs (transposed-view operands,
  strided-axis reductions, transpose materialization), and
  `cast`/`copy`/`concatenate`. A `lab` tag marks a fast representative subset
  (select with `--tag lab`).
- Fix unary `-` from `Nx.Infix` to negate tensors with `neg`. It previously
  performed `logical_not`, unexpectedly turning zero into one and nonzero
  values into zero.
- **Breaking.** Remove the `^` logical-XOR operator; use `logical_xor`
  directly. Its concatenation-level precedence misgrouped comparisons.
- **Breaking.** Remove the `<.>` dot-product operator; use `dot` directly.
  Its comparison-level precedence grouped mixed arithmetic unexpectedly.
- **Breaking.** Rename the infix matrix-multiplication operator from `@@` to
  `*@`, giving it multiplication precedence in mixed arithmetic expressions.
- Fix `cast` and all float16 compute: the float32-to-float16 conversion
  corrupted any value with an odd biased exponent that needed mantissa
  rounding (e.g. casting `0.274` to float16 returned `0.5`), converted `inf`
  to `nan`, and flushed subnormals to zero. Conversion is now IEEE
  round-to-nearest-even with subnormal support, matching numpy. Casting a
  signaling NaN to `bfloat16` no longer returns `inf`.
- Add `scatter`, the pure counterpart of `put_along_axis`: returns a new
  tensor with `values` placed along `axis`, with `` `Set``/`` `Add`` modes
  and a `unique_indices` hint. Works under `Rune.jit` and differentiates
  with respect to both inputs.
- Fix `` `Add``-mode scatter on the C backend to accumulate updates into the
  template's values instead of a zeroed buffer, matching the jit lowering
  and the autodiff rule.
- Fix `rfft` and `irfft` bypassing the effect-based backend dispatch: they
  called the C backend directly, making them invisible to every effect
  handler (autodiff, vmap, jit). They now perform `E_rfft`/`E_irfft` like
  the other FFT operations, with the target `dtype` carried in the effect.
- Require OCaml >= 5.5.0 (module-dependent functions are used by the
  `Ptree.S`-based APIs downstream).
- Fix `flatten` raising on rank-0 tensors; it now reshapes them to `[|1|]`.
- Extend safetensors I/O: cover the remaining dtypes with SafeTensors
  equivalents (float64, int64, int8, uint8, bool, ...) and support rank-0
  tensors. Dtypes with no SafeTensors equivalent (complex, int4) fail with a
  clear error.
- Route `to_host` through a new `E_to_host` effect so effect handlers observe
  value reads. Transformations can now materialize or reject concretization
  deliberately (a JIT tracer needs this; reading a batched tensor inside a
  vectorizing map is now detectable instead of silently exposing the physical
  buffer).
- Fix the scatter effect dropping its mode: `scatter ~mode:\`Add` was silently
  executed as a `Set`-mode scatter whenever an effect handler (autodiff, vmap)
  intercepted the operation. `E_scatter` now carries `mode` and
  `unique_indices`.
- Remove `~out` parameter from all backend compute operations. Operations now
  allocate and return their result instead of writing to a caller-provided
  buffer. This simplifies the effect system, fixes vmap, and prepares the
  architecture for JIT compilation.
- Add `Shape.reduce_output_shape` for computing output shapes after axis
  reduction.
- Add machine learning examples: PCA, K-Means, DBSCAN, and t-SNE implemented
  from Nx primitives.
- Fix incorrect results for views and slices in binary, unary, ternary, cast,
  and shape C stubs. The `iterate_inner_dims` helpers did not account for the
  ndarray offset, producing wrong results when the data starts at a non-zero
  offset in the underlying buffer.

### Rune

- `Rune.vmap` now maps `Nx.matmul` correctly when the other operand carries
  leading batch dimensions of its own: the map's axis was aligned against the
  operand's first batch dimension, which raised a shape mismatch or silently
  paired the wrong matrices.
- `grad` through `Nx.solve_triangular ~unit_diag:true` no longer assigns a
  gradient to the diagonal, which the solve never reads; it disagreed with
  both the function and `jvp` there.
- `grad` and `jvp` through a batch of matrices are now correct for
  `Nx.cholesky`, `Nx.qr`, and `Nx.solve_triangular`: the rules reversed every
  axis and extracted, rather than built, their diagonal terms, so a stack of
  matrices gave wrong gradients or a shape error.
- `Rune.jit` compiles `Nx.qr`, `Nx.solve_triangular`, `Nx.cholesky`,
  `Nx.solve`, and `Nx.inv`: the factorizations unroll at trace time into the
  fixed number of steps their shapes imply (see `Tolk_frontend.Linalg`), and
  `grad` through them compiles as well. A singular or non-positive-definite
  input yields infinities or nans in the compiled program rather than an error.

- `Rune.jit` compiles `Nx.qr` and `triangular_solve` — Householder QR and
  forward substitution unrolled at trace time into the fixed number of steps
  their shapes imply, so the whole factorization lowers to ordinary Tolk
  compositions and compiles for every Tolk device. Compiled results match the
  eager kernels, including the LAPACK reflector sign and the zero-tail
  no-reflector convention. A linear solve inside jit is the same composition
  written out by hand (`Nx.solve` itself still refuses to trace, because its
  singularity check reads a traced value); a singular system yields infinities
  rather than an error. Wide right-hand sides (nrhs ≥ 32, n > 64) solve
  block-by-block — 32-row blocks, diagonal blocks inverted once, one GEMM per
  block against the rows solved so far — instead of unrolling one thin matmul
  per row, which cuts the compiled solve's O(n²·nrhs) concatenation copying to
  O(n²·nrhs/32): replay at 256×256 drops ~80× (58 ms to 0.7 ms) and the
  compiled solve now beats the eager C kernel 3-5× from 128×128 up. Compile
  time grows linearly in the matrix dimension, and `grad` inside a jitted
  function now also differentiates through `qr` and `cholesky`: the tape
  pullbacks' `diag` use read host bytes and refused to trace, so the QR and
  Cholesky pullbacks (and the triangular-solve JVP rule) form the diagonal
  terms from the identity instead.

- `Rune.jit` and `Rune.pmap` run on `Rune.device "NV"` — NVIDIA GPUs driven on
  the kernel driver's hardware queues (Linux), with kernels compiled straight
  to cubin. `"CUDA"` keeps selecting the userspace CUDA driver API backend.

- `Rune.jit` and `Rune.pmap` run on `Rune.device "AMD"` (`"AMD:n"` for a
  specific GPU), running compiled programs on AMD GPUs on Linux. Previously the device
  factory rejected the name with `Invalid_argument: unknown device AMD:0`
  even though the runtime had landed; an AMD device that cannot open now
  reports `device AMD:0 unavailable` with the reason, like CUDA.

- `Rune.jit` compiles `Rune.scan` as a loop in the compiled program — the fold
  step compiles once and runs per slice — instead of unrolling every step into
  the trace, and `grad` through a jitted scan compiles a reversed loop over the
  step's pullback. Compile time for a scanned recurrence (an RNN, a sampling
  loop) no longer grows with its sequence length. A carry that changes shape
  across steps, and a scan reached through `vmap` or `pmap`, unroll into the
  trace as before.

- `grad`, `vjp`, and `jvp` now differentiate `Nx.rfft` and `Nx.irfft` — both
  are linear, so each rule is an exact transpose — and `vmap` batches all four
  FFT transforms (`fft`, `ifft`, `rfft`, `irfft`), so spectral losses built on
  real FFTs train end to end.
- `jit`, `jit'` and `pmap` take `?beam_parallel`: the
  number of domains compiling a beam-search round's candidates, scoping the
  `BEAM_PARALLEL` setting to one compiled function. It only changes compile
  time, never the compiled code.

- `jit`, `jit'` and `pmap` take `?beam`: beam-search
  autotuning of the compiled function's kernels, equivalent to compiling under
  `BEAM=n` but scoped to that one function. The width is part of the
  persistent compile-cache key, so tuned and untuned compilations of the same
  trace do not collide.
- A parameter structure may now hold leaves that are not parameters — an
  `Nx.Rng.t` threaded through a compiled step, a step counter, a batch of
  indices. `grad`, `value_and_grad` and `vjp` carry them instead of raising:
  they are not tracked, and their slot in the gradient structure holds zeros.
  One structure can therefore serve both `grad` and `jit`, where before a key
  had to be captured in a closure and the parameters kept in a second structure
  — the workaround the GPT-2 example spells out across four hand-written ptree
  modules. The single-tensor `grad'`/`vjp'` still reject an integer argument,
  having nowhere to carry one.

- Gradient rules that combine with a constant (`asin`, `atan`, `tanh`, `sqrt`,
  `pow`, `max`/`min`, `where`, `cholesky`, `cumprod`) no longer materialize a
  full-size tensor of ones or zeros to do it.
- Fix the gradient of `Nx.fft` and `Nx.ifft`: reverse mode pulled cotangents
  back through the opposite transform, which reversed the frequency index. Each
  transform is its own transpose and now pulls back through itself.
- Fix the derivative of `abs` on complex tensors for cotangents that are not
  real: the modulus is real-valued, so the rule now keeps only the real part of
  the cotangent and pushes forward to a real tangent.
- Fix the derivative of `abs` on complex tensors, in both forward and reverse
  mode: it pulled back through `sign z` instead of its conjugate, which negated
  the imaginary part's contribution. Gradients of a real-valued function that
  passes through a complex magnitude were wrong; real dtypes are unaffected.
- Reverse mode differentiates the sliding-window movement through `fold`
  rather than a materialized scatter, so the backward pass is a single
  overlap-add instead of allocating a `window`-fold copy of the cotangent.
- `vmap` now batches `Nx.extract_patches` and `Nx.combine_patches` instead of
  raising; both preserve leading dimensions, so the batch axis passes through.
- `jacfwd'` and `jacrev'` support float32 and float64 inputs without an
  implicit float64 specialization. Forward-mode Jacobians keep the output
  dtype, reverse-mode Jacobians keep the input dtype, and both evaluate the
  differentiated function only once.
- `pmap` now decorrelates per-device randomness: under `Rune.pmap`,
  `Nx.Rng.fold_in_axis key` folds each device's own index into the key, so a
  replicated key yields an independent draw per device (device `i` draws
  `Nx.Rng.fold_in key i`). Data-parallel dropout masks now differ across
  devices instead of replicating. Previously every device drew the identical
  values from a replicated key.
- The `rune` bench suite now covers `jit`: a `Jit` group times compiled
  execution of the MLP forward pass and the deep elementwise chain against
  their eager equivalents, with compilation hoisted out of the measured region.
- Random number generation lives in `Nx.Rng`: `rune` no longer declares a `Rng`
  module or a `type key`. The unified splittable keys and samplers are reached
  as `Nx.Rng.*`; rune's transforms only answer the generator's effects (jit
  lowers threefry, `vmap` batches per-lane keys with `Nx.Rng.fold_in_axis`),
  adding no RNG vocabulary of their own. Migration: rename `Rune.Rng.*` to
  `Nx.Rng.*`.
- `jit` now compiles random number generation: threefry lowers to the
  compiler's primitive with bit-exact parity to eager execution. A key that
  does not depend on the jitted function's inputs (`Nx.rand` and friends, or
  a captured key) raises `Jit_error` at trace time instead of silently
  replaying one frozen draw per call — thread an `Nx.Rng` key through the
  inputs.
- `jit` compilations now persist across processes: compiled kernels are
  stored on disk (`$XDG_CACHE_HOME/tolk/rune_jit`) keyed on the traced
  computation and compile environment, so a warm process skips scheduling,
  lowering, and kernel compilation (gpt2 train first step ~17 s -> ~4.4 s,
  results bit-identical). Set `JITCACHE=0` to disable; `pmap` compilations
  are not persisted.
- `pmap` now differentiates through `~keepdims:true` reductions
  (`max`/`sum`/`mean`), unblocking softmax, layer norm, attention, and the
  stock losses in data-parallel training.
- Add `pmap`: compile a function to run in parallel across a device tuple.
  `in_axes` shards or replicates each argument; the function
  observes global shapes and reductions over a sharded axis become
  cross-device allreduces automatically, so differentiating a mean loss
  inside `pmap` yields data-parallel gradients. Outputs stay resident per
  device and feed back into matching placements with no transfer.
- Fix `jit` raising `Jit_error` ("not scheduled to a buffer") when a
  function returned an input leaf of rank 2 or higher unchanged.
- `jit` compiled programs now replay through device execution graphs (CUDA
  graphs): consecutive kernels batch into single graph launches, honoring
  `JIT` (>= 2 disables) and `JIT_BATCH_SIZE`. GPT-2 training drops ~116 to
  ~110 ms/step and decode rises ~320 to ~380 tok/s on an H100.
- **Breaking.** `jit` closure captures are now compile-time constants on
  every device: they are bound once when the trace compiles and never
  refreshed, and a jitted function that assigns to a capture (`assign`,
  `blit`) raises `Jit_error` at trace time instead of writing state back.
  Thread mutable state through the input structure instead; assigning to an
  input leaf still writes back on every call. Mutating a captured tensor
  between calls now has unspecified visibility (the CPU device may observe
  it through zero-copy aliasing; other devices never do).
- Tensors captured by a jitted closure are uploaded to the device once per
  closure and shared by all compiled signatures, instead of once per
  signature — halving resident weight memory for prefill+decode closures.
- `jit` outputs on CUDA and Metal now stay resident on the device until
  read: metadata reads never transfer, and an unread output fed back as an
  input of a jit call on the same device seeds the compiled program directly
  with its device buffer. Iterated jitted calls (training steps, decode
  loops) no longer round-trip state through the host: GPT-2 decoding goes
  from ~142 to ~320 tok/s at 100 tokens and training steps from ~380 to
  ~117 ms on an H100, with bit-identical results. Device memory is reclaimed
  when outputs are read or collected (budget via
  `RUNE_JIT_RESIDENT_BUDGET`).
- Add `Rune.jit_stats`/`Rune.reset_jit_stats` transfer counters, a
  `RUNE_JIT_DEBUG=1` per-call transfer log, and `RUNE_JIT_FORCE_COPY=1` to
  exercise the device copy path on the CPU device.
- Inside `jit`, `Nx.full`/`Nx.zeros`/`Nx.ones` (and `*_like`) now trace as
  broadcast scalar constants instead of captured host tensors re-uploaded on
  every call, scalar constants fold into kernels as immediates, and replay
  reuses its transfer staging buffers — a jitted GPT-2 124M train step on
  CUDA drops from ~2.1 s to ~0.35 s.
- `jit` uploads closure-captured tensors to non-CPU devices once per
  compilation instead of on every call (captures the function assigns to are
  still re-read each call, so in-place state carries across calls). Jitted
  functions capturing large weights no longer pay a full re-upload per call.
  CPU behavior is unchanged.
- `jit` runs on `Rune.device "CUDA"`: jitted programs compile through NVRTC and
  run on NVIDIA GPUs.
- `jit` takes a `?devices` argument selecting where kernels compile and run:
  the CPU device (the default) or Metal on macOS.
- On the CPU device, jitted programs now run on the tensors' own memory:
  contiguous inputs and captured tensors are read in place and outputs are
  computed directly into the returned tensors' storage, removing the byte
  copies previously made on every call. Non-contiguous tensors and other
  devices still go through copies.
- Add just-in-time compilation: `jit` and `jit'` trace a function
  once per input signature (leaf dtypes and shapes), compile the trace into
  fused native kernels through the Tolk compiler, and replay the compiled
  program on subsequent calls. Differentiating inside a jitted function
  compiles the forward and backward passes together — a whole training step
  (forward, backward, parameter update) compiles into one program; under an
  enclosing transformation (`grad`, `vmap`, `with_debug`) the wrapped
  function runs eagerly so results never change. Sliding-window operations
  (`extract_patches`/`combine_patches`, and convolution built on them)
  compile too. Reading a traced tensor's value or using an operation the
  compiler cannot express (FFT, linear algebra, RNG, complex dtypes) raises
  `Jit_error` at trace time instead of compiling a wrong program.
- **Breaking.** Ground-up rewrite. Transformations now operate over typed
  parameter structures: `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`,
  `hvp`, and friends take the structure of their arguments
  and return gradients with the same structure and leaf dtypes as the
  parameters — mixed-dtype parameters differentiate in a single forward
  and backward pass. Functions of a single tensor use the primed variants
  (`grad'`, `value_and_grad'`, `vjp'`, `jvp'`, `vmap'`, `jacfwd'`,
  `jacrev'`, `hessian'`, `hvp'`).
- New transformation surface: reusable pullbacks (`vjp_fun`), gradient
  checkpointing (`remat`), Hessian-vector products (`hvp`), custom
  differentiation rules (`custom_vjp`, `custom_jvp`), directional gradient
  checking (`check_grads`), staging-ready control flow (`scan`, `cond`,
  `while_loop`), and operation logging (`with_debug`, replacing `debug`).
- Removed: the list-based variants (`grads`, `value_and_grads`, `vjps`,
  `jvps`, ...) — use a structure instead; the finite-difference
  `check_gradient` API — use `check_grads`; and `jit`/`trace_graph` —
  JIT compilation via Tolk will return as a transformation in a later
  release.

### Kaun

- **Breaking:** `Fn.gelu` takes and returns float tensors only, as `Nx.erf`
  does.
- **Breaking:** `Fn.sigmoid`, `Fn.tanh`, `Fn.softmax` and `Fn.log_softmax` are
  removed: use `Nx.sigmoid`, `Nx.tanh`, `Nx.softmax` and `Nx.log_softmax`, whose
  `~axes:[ a ]` is the old `~axis:a`.
- **Breaking:** `Kaun.Checkpoint` is removed. A model's structure saves and
  reads it back through `Nx_io.Archive.of_value` and `to_value`, sections are
  `Nx.Ptree.field`s joined with `Archive.union`, and an importer reads entries
  with `Archive.float` and `tensor`. Reading back now refuses an entry under
  the structure's fields that the value does not name.
- **Breaking:** `Kaun_hf.load_checkpoint` is `Kaun_hf.load_safetensors`, which
  returns an `Nx_io.Archive.t`; `kaun.hf` no longer depends on `kaun`.
- The `kaun` library depends on nx alone, no longer on rune: add rune to your
  project to differentiate. `Batch_norm.apply` no longer detaches the running
  statistics; `Rune.value_and_grad_aux` returns them undifferentiated.
- Add `Cache_index.draft ~slots ~sees` to verify a tree of draft tokens in one
  call: each stores at a chosen slot and sees the cache and the draft tokens
  `sees` names. A kept path's slots go in the next call's table.
- Add `Cache_index.packed ~seq lens` for several sequences per lane: positions
  restart at each sequence and a token sees only its own.
- **Breaking:** `Rope` schedules hold the cosines and sines of every position
  below a context, and `Rope.apply` reads the rows at its positions. The
  constructors take `~context`; `Rope.context` returns it. Compiled attention
  no longer computes sines and cosines, and long positions turn by their angle
  rounded once instead of a float32 product.
- **Breaking:** `Embedding.apply`'s ids, the labels of
  `Loss.softmax_cross_entropy_sparse`, `Metric` and `Kaun_datasets`,
  `Rope.apply`'s positions, `Fn.keep_top_k`'s `k` and every tensor of
  `Cache_index` are `int64`, nx's index type, so `Nx.argmax` and `Nx.top_k`
  feed them without a cast. Widen brot's `int32` token ids once with
  `Nx.cast Nx.int64`.
- `Kaun_datasets` downloads and extracts on Windows. It found curl and ran
  curl and tar through the shell with `command -v`, which `cmd.exe` lacks; it
  now runs them directly. `Kaun_datasets` and `Kaun_hf` find the cache under
  `USERPROFILE` when `HOME` is unset, where they raised `Not_found`.
- The gpt-oss example's `main.exe` and `validate.exe` take `--devices` in place
  of `--jit` and run expert-parallel over several devices
  (`Gpt_oss.expert_parallel`): each device holds an equal share of the
  experts, the rest a copy on each. `Layer_loop.cached` and `greedy` take
  `~devices`, a list.
- The Llama example's `--devices` (a device, a CPU count or a comma-separated
  list) replaces `--jit` and compiles the decode step tensor-parallel over
  several devices: projections split into and out of the heads, caches on
  their kv-heads, the rest a copy on each device.
- **Breaking:** `Embedding.apply` returns a row of zeros for an id outside
  `[0, vocab)`, and `Loss.softmax_cross_entropy_sparse` takes zero loss for a
  label outside the classes (still counted by `` `Mean ``), eagerly as under
  `Rune.jit`, following `Nx.take`. Both used to raise `Invalid_argument`
  eagerly.
- The GPT-2, Llama and gpt-oss examples' structures (`Params`, and gpt-oss's
  `Block` and `Moe`) are one `walk` each; their `Cache` modules and
  `Gpt_oss.map`, `ptree` and `block_ptree` go. `generate` takes one `?device`,
  for which the step compiles and on which the caches are placed.
- **Breaking:** a layer's structure, and `Attention.Cache`'s, is its one `walk`,
  which walks an `Nx.Ptree.Walk` cursor. `Kaun.ptree` is `Nx.Ptree.instantiate`
  and `Attention.Cache.List` is `Nx.Ptree.list`.
- **Breaking:** a layer's `map`, `map2`, `iter`, `fold`, `fold2` and `names` are
  `Nx.Ptree.Payload.map`, `map2` and `fold` over `(module L)`, whose functions
  take each payload's path, or `Nx.Ptree.map`, `map2`, `fold` and `cast` for
  tensors.
- **Breaking:** `Checkpoint.of_params (module P)` and `of_packed` are
  `Checkpoint.of_value p`, `to_params` and `to_packed` are `to_value p`, and
  `find` and `get` return `Nx.packed`. Names are unchanged, except that a fixed
  tensor now has an entry.
- **Breaking:** `Cache_index.map`, `map2` and `iter` are `Nx.Ptree.map`, `map2`
  and `fold` over `Cache_index.ptree`, which also reports the tokens' case,
  `every`, block sizes, window and selection, so a compiled step sees them.
- The GPT-2, Llama and gpt-oss examples' importers and cache builders take
  `?placement : role -> axis:int -> Nx.Placement.t` and place each leaf and
  cache pool with it as they build it, naming each leaf's tensor-parallel cut.
- `Metric` functions read placed predictions, labels and scores to the host
  once and compute there. They ran on the device and placed every
  intermediate, and raised on Metal, which cannot hold the float64 they sum.
- Add `Cache_index.every m index` for layers that keep one entry per block of
  `m` positions, such as a compressed key: `make ~every` and `rows ~every` give
  the index a table of blocks, and a block is stored by its last token and seen
  from that token on.
- Add `Cache_index.select columns index`: each token reads only the columns it
  chose, so `extend` returns `[batch; seq; k; ...]` rows and a sparse or
  windowed layer's read costs `k` rows per token whatever the context.
- `Kaun_hf.load_checkpoint` no longer asks the Hub for a shard index when the
  repository is already cached as a single `model.safetensors`: a cached model
  used to make one network request on every start, and to stall without a
  connection. `Kaun_hf.download_file` now runs `curl` directly instead of
  through a shell: the previous detection was a POSIX shell command, which
  `cmd.exe` cannot run, so downloads failed on Windows with "curl not found".
- The gpt-oss example takes text: `--prompt` (with `--system`, `--reasoning`
  and `--show-analysis`) renders a harmony conversation with the checkpoint's
  tokenizer, streams the model's final answer as it decodes and stops when the
  model closes its turn. Its `Harmony` module renders and parses the format,
  checked against `openai-harmony` by `validate_text.exe`.
- Remove `Kaun_hf.rename`, `transpose` and `split`. They existed because a
  template found entries by its own paths; an importer now asks for each entry
  by the file's name, so a rename is the name at the field, a transpose is
  `Nx.matrix_transpose` and a fused tensor is `Nx.split ~axis n`, all views.
- Add `Checkpoint.to_tensor ~shape dtype name` and `Checkpoint.to_float ~shape
  dtype name`, which read one entry by name and check its shape. A model's
  importer is now an ordinary function that builds the parameter record from
  them, so no template is allocated, leaves may have different dtypes, and a
  wrong configuration fails at import with the entry's name. `to_tensor` is
  strict and returns the entry as stored, a view of the file; `to_float` casts
  between `float16`, `bfloat16`, `float32` and `float64` and refuses anything
  else.
- `Checkpoint.to_value` raises on any dtype mismatch, so a restart that names
  the wrong dtype fails instead of narrowing its state. To convert, read the
  entry with `Checkpoint.to_float`.
- `Checkpoint.load` and `Kaun_hf.load_checkpoint` map their files, as
  `Nx_io.load_safetensors` now does: loading reads headers only, entries are
  views of the file, and entries whose dtype nx lacks arrive as `uint8` bytes
  instead of being skipped. A loaded file must not be modified in place while
  its entries are alive; `Checkpoint.save` replaces its destination atomically.
- `Kaun_hf.download_file` downloads to a uniquely named temporary file beside
  the cache path and renames it once complete. An interrupted download used to
  leave a partial file at the cache path, which later runs served as cached,
  and two processes fetching one file wrote over each other.
  `Kaun_hf.clear_cache` runs a major collection and retries once when a file
  cannot be removed.
- The pieces `Attention.apply` and `Attention.cached` are made of are public:
  `Attention.split` projects and splits into heads, `Attention.attend` is
  grouped-query attention with `?mask`, `?scale` and `?sinks`,
  `Attention.merge` concatenates heads through the output projection, and
  `Attention.Cache.extend` stores a call's keys and values and returns what its
  tokens attend over. A model with its own attention variant composes a layer
  in a dozen lines. `apply` and `cached` compute what they did, bit for bit.
- `Attention.scaled_dot_product_attention` takes `?scale`, which replaces
  `1 / sqrt d`, and `?sinks`, attention-sink logits that join each query's
  softmax as one more key of value zero, as gpt-oss needs. With sinks a query
  that sees no key yields zero. Without the options the computation is
  unchanged.
- Add `Rope.of_frequencies`, a schedule from one head's inverse frequencies,
  for schedules the module does not name, and `Rope.yarn`, the YaRN
  long-context frequencies with an untruncated correction range, as gpt-oss
  uses them. `Rope.apply` keeps norms: YaRN's attention temperature is a
  number the model passes to the attention core, not part of the schedule.
- **Breaking**: the sampling masks `Fn.top_k` and `Fn.top_p` are now
  `Fn.keep_top_k` and `Fn.keep_top_p`. They return the logits with everything
  outside the kept set at negative infinity, where `Nx.top_k` returns the `k`
  greatest entries: one name, one meaning.
- **Breaking**: a sliding window is part of the cache index.
  `Cache_index.window w index` is `index` seeing the last `w` positions, and
  `Cache_index.extend` and `Cache_index.mask` lose `?window` and read the
  index's, so a layer can no longer zero with one window and mask with
  another. `Attention.cached` loses `?window` too: a layer with a window is
  `cached p cache (Cache_index.window w index) x`.
- `Attention.apply` and `Attention.cached` no longer copy the keys and values
  to group them under their query heads. The decode step of
  `kaun/bench/decode` runs 24 fewer kernels and allocates 9% fewer host words;
  on Metal it takes 8.0 ms against 8.5 ms at a cache of 256, and the same 8.9
  ms at 1024.
- **Breaking**: a decoder has one forward pass. `Attention.cached` takes a
  `Kaun.Cache_index.t`; over `Cache_index.whole`, which
  reads and keeps nothing, it is plain causal attention and returns its cache
  untouched. A model's `hidden` is
  `fst (cached ... (Cache_index.whole ~batch ~seq ()) ids)`, the second fold
  over `Attention.apply` is gone from the `04-gpt2` and `05-llama` examples,
  and its compiled gradient costs what that fold did.
- **Breaking**: `Kaun.Cache_index` replaces `Attention.Span`, the `route` type
  and `Attention.route ~slots`; a model resolves nothing. A cache index is
  opaque: `Cache_index.make ?row ~pos ~table ()` takes the tokens' positions
  and the slots holding each sequence, and `?row` names the sequence of each
  lane. A token stores at the slot its table names at its position.
  `Cache_index.rows`, `advance` and `positions` keep the meaning they had on
  `Span`. A layer calls `Cache_index.extend index values pool`, which stores
  the call's values and returns what its tokens attend over, and attends under
  `Cache_index.mask`.
- `-1` addresses nothing everywhere, so the cache write is one
  `Nx.scatter ~unique_indices:true` over the call's tokens with no pass over
  the pool: a token that stores nothing targets `-1`, whose store is dropped,
  and an unallocated column reads zero. A pool of `slots` slots has `slots`
  rows. On Metal, measured with an extra row after the slots that such
  writes used to target, a two-layer decode step compiled with `Rune.jit` and
  consuming its caches takes 3.6 ms at a context of 256 over 4096 slots and
  3.8 ms over 131072, where it took 3.8 ms and 24.6 ms; the GPT-2 124M shaped
  step of `kaun/bench/decode` takes 8.4 ms and 8.7 ms at caches of 256 and
  1024 against 9.1 ms and 10.3 ms.
- Attention is total. A query whose mask hides every key yields zero from
  `Attention.scaled_dot_product_attention`, `Attention.apply` and
  `Attention.cached`, with zero gradients, where it yielded `nan`.
- **Breaking**: `Attention.causal_mask ~valid` no longer keeps the diagonal of
  a padded query. The kept key existed to avoid that `nan`; a padded query's
  output is now the projection of zero.
- The `05-llama` example's `validate` checks the residual stream after every
  block, a ragged batch through the key-value caches, half precision
  (`--dtype`) and compiled runs (`--jit`), and ships a second fixture,
  TinyLlama 1.1B, which covers an untied head and the standard rotary
  schedule on real weights.
- New example `05-llama`: Llama 3.2 1B written on the decode contract
  (grouped-query attention, rotary positions with the Llama 3 schedule,
  RMS norm, SwiGLU), loaded from an ungated mirror whose weights are
  byte-identical to Meta's, with sampled generation through key-value caches
  and a `validate` program that checks the import against the reference
  implementation's float32 logits, block by block.
- `Kaun.Fn.keep_top_k` and `Kaun.Fn.keep_top_p` mask next-token logits for
  sampling: entries outside the kept set become negative infinity and the
  shape is unchanged, so they compose with a temperature division and
  `Nx.Rng.categorical` in any order and compile. `k` and `p` are tensors, a
  scalar or one entry per row, so a batch can mix requests.
- `Loss.softmax_cross_entropy` and `softmax_cross_entropy_sparse` compute
  their log-probabilities and reduction in a float32 island for half and
  quarter precision logits and cast the result back: a bfloat16 log-sum-exp
  over a large vocabulary biases the gradient.
- **Breaking**: cached decoding is addressed by positions and slots.
  `Attention.cached ~head_dim ?rope p cache route x` replaces `apply_cached`
  and its single scalar position. A cache is a flat pool of slots with no
  batch axis, so `Attention.Cache.make ~slots ~kv_heads ~head_dim` replaces
  `?batch ~num_heads ~head_dim ~len`, and a model's per-block caches are a
  list of them. An `Attention.Span.t` (`make`, `rows`, `advance`,
  `positions`) carries each token's position and the slot holding each
  position of each row's sequence, and `Attention.route ~slots span` resolves
  it once per call for every block. Rows at different positions share a
  batch; paging, shared prefixes and beams are values of the slot map rather
  than cache types; an address outside its range addresses nothing (`-1` is
  padding); a prompt fed whole or in chunks gives the same outputs; and
  admitting a sequence changes values, never the compiled program. See RFC
  0002.
- **Breaking**: `Attention.apply` requires `~head_dim` and takes `?mask` in
  place of `?num_heads` and `?causal:bool`, so causality and padding
  intersect; `Attention.causal_mask ~seq ?valid ()` builds the mask and keeps
  the diagonal, so a padded query never yields `nan`. `?rope` rotates queries
  and keys, and `Attention.make ?q_dim ?kv_dim` sizes the projections for
  grouped-query attention, where the keys broadcast over their group.
- `Kaun.Rope` adds rotary position embeddings: a schedule is the inverse
  frequencies of one head (`Rope.make`, and `Rope.llama3` for the Llama 3.1
  long-context bands) and `Rope.apply` rotates queries or keys at per-token
  positions, forming the angles at float32 whatever the activation dtype.
- `Kaun.Rms_norm` adds root mean square normalization, the norm of
  Llama-class models, with the float32 island `Layer_norm` has for half and
  quarter precision inputs.
- Seeded `glorot_normal`, `he_normal` and `lecun_normal` initialisers produce
  different values for a given key: they draw through
  `Nx.truncated_normal`, whose bounds are now tensors and whose draw changed
  with them. `Dropout` masks are unchanged.
- `Loss.huber`, `Loss.sigmoid_bce`, and `Fn.leaky_relu` compare against a
  scalar rather than a materialized constant tensor.
- The MNIST CNN example now saves and restores model parameters with both
  AdamW moment trees and the step counter, demonstrating how to resume
  momentum-based optimization without resetting its history.
- The `kaun` bench suite is broader: alongside the MLP Adam train step and
  forward pass it now covers an SGD train step, a small CNN train step
  (conv + max-pool blocks with `Conv`/`Pool`), and a single `Linear` layer
  forward and forward+backward in isolation.
- `Dropout.apply` takes an optional `?key:Nx.Rng.t`: the mask becomes a
  pure function of the key and the input's shape, so dropout composes with
  `Rune.jit` (pass the key as an input leaf; keyless dropout under jit
  raises `Jit_error`) and with `vmap` via per-lane keys.
- The gpt2 example trains stochastically: `Gpt2.logits` takes
  `?dropout:(rate, key)` enabling the canonical dropout sites, and
  `train.exe` gains `--dropout` and `--seed`, deriving per-step mask keys
  with `Rune.Rng.fold_in` for seed-reproducible runs.
- The gpt2 example is dtype-generic: `main.exe --dtype float16|bfloat16`
  for half-precision generation (float16 greedy tokens match float32 at
  half the weight memory), `train.exe --compute-dtype bfloat16|float16` for
  mixed-precision training with float32 master weights (bfloat16 engages
  tensor cores).
- Add `astype` to every layer (`Linear`, `Embedding`, `Conv`, `Layer_norm`,
  `Attention` and its `Cache`, `Batch_norm` and its `Stats`): cast parameter
  trees to another float dtype; gradients flow back at each leaf's original
  dtype, so casting float32 parameters inside a loss yields float32
  gradients.
- Half-precision inputs now compute attention scores/softmax and
  layer/batch-norm statistics in float32 islands; float32 and float64
  graphs are unchanged.
- `Batch_norm` is now dtype-generic (`'b params`, `'b Stats.stats`) like the
  other layers; `Batch_norm.t` and `Stats.t` remain the float32 aliases.
- The GPT-2 training example gains `--devices` for data-parallel training
  through `Rune.pmap` — a CPU device count (`--devices 4`) or an explicit
  tuple (`--devices CUDA:0,CUDA:1`). Parameters replicate, the batch shards
  on axis 0, gradients allreduce automatically; per-step losses match the
  single-device step within fp32 reduction order.
- `Attention.apply_cached` updates the cache with a gather instead of a
  one-hot matmul, cutting the per-step update from O(len*seq*head_dim) to
  O(len*head_dim).
- **Breaking.** The attention KV cache moved into an `Attention.Cache`
  submodule: `Attention.cache` and `map_cache` are now `Attention.Cache.make`
  and `Attention.Cache.walk` on `'b Attention.Cache.t`.
- New GPT-2 training example (`examples/04-gpt2/train.ml`): jitted
  forward+backward+SGD via `Rune.jit` and `Vega.sgd_step` with the tied
  `wte` LM head, exporting per-step metrics and final weights as
  safetensors.
- Add key-value cache decoding to `Attention`: `cache`, `apply_cached`, and
  `map_cache`/`map2_cache`/`iter_cache`. The cache is functional and
  addresses slots with tensor arithmetic on the position, so a single-token
  decode step compiles once under `Rune.jit`; the GPT-2 example decodes
  through it (~85x faster jitted CUDA decode).
- The GPT-2 example loads a local safetensors checkpoint when one is cached
  (`Gpt2.from_file`, `Gpt2.of_checkpoint`), tokenizes via `tokenizer.json`,
  and can compile its forward pass with `Rune.jit` on CPU or CUDA
  (`--jit DEVICE`).
- **Breaking.** Ground-up rewrite on typed parameter structures. There is
  no `Layer.t` and no `Train` driver anymore: a layer is a plain record of
  tensors with a pure `apply` function (`Linear`, `Conv`, `Embedding`,
  `Attention`, `Layer_norm`, `Batch_norm`, `Dropout`, ...), a model is a
  record of layers with a hand-written `Nx.Ptree.S` traversal, and a
  training step is code you own: `Rune.value_and_grad` composed with a
  structural Vega optimizer update. Losses, initializers, activations
  (`Fn`), data batching, metrics, checkpoints, and HuggingFace Hub
  integration (`kaun.hf`, `kaun.datasets`) are provided as plain functions
  over these records.

### Brot

- OLMo 2 and Phi-4 take the cl100k scanner too: they ask for the pattern's
  matches as `Removed` with `invert`, which is now recognised as the same
  pieces (Phi-4 24 → 45 MB/s single-threaded; not through the fused kernel).
- Llama 3, GPT-4 (cl100k), Qwen2 and Qwen3.5 tokenizers encode through the
  fused C kernel that GPT-2 uses, with a walker for their pattern: Llama 3
  43 → 79 MB/s and Qwen2.5 40 → 90 MB/s single-threaded on OpenWebText
  (GPT-2 is unchanged at 155 MB/s).
- Llama 3, OLMo, GPT-4 (cl100k), Qwen2 and Qwen3.5 tokenizers pre-tokenize
  about twice as fast: their `Split` patterns are recognised and run by a
  scanner checked against the regular expression (Llama 3 27 → 43 MB/s
  single-threaded). Such a pipeline can also be cut across domains by
  `encode_batch_ids`. `Pre_tokenizer.pp` shows `walker=cl100k(...)` for them.
- Tokenizer files whose `Split` pre-tokenizer carries a regular expression now
  load and tokenize as HuggingFace does: Llama 3, Qwen2.5, DeepSeek-V3 and
  gpt-oss (o200k) where `Brot.from_file` used to fail with "regular expression
  'pattern' is not supported". `Pre_tokenizer.split_regex` builds one directly.
- Regular expressions from tokenizer files now accept the case-insensitive
  option (`(?i)`, `(?i:..)`) and a lookahead (`(?=..)`, `(?!..)`) that ends the
  pattern or one of its alternatives, so `Normalizer.replace_regex` and
  `Replace` normalizers using them load instead of being rejected.
- **Breaking:** the stage modules' internal plumbing is no longer exported:
  `Brot` is now published from a single signature, so
  `Pre_tokenizer.plan`/`fill`/`lead_class`,
  `Encoding.token`/`of_run`/`with_overflowing` and `Post_processor.affixes`
  are gone from the public API. The documented API is unchanged.
- `encode_batch_ids` is ~3–5% faster: the fused C kernels now write each
  chunk's ids straight into its int32 result buffer instead of filling an
  int buffer that was copied per document (GPT-2 batch 179 → 186 MB/s
  single-threaded, 810 → 845 MB/s across domains; Mistral 58 → 60 MB/s).
- SentencePiece-style BPE tokenizers (Llama 1/2, Mistral, Gemma 2) encode
  another ~1.2–1.35× faster on native code (llama 33 → 40 MB/s, mistral
  44 → 58 single-threaded): the `▁`/punctuation unit walk, the pretoken-cache
  probe and the short merge run fused in the C kernel, as the byte-level path
  already does. Bytecode and js_of_ocaml keep the OCaml path.
- SentencePiece word units also end before the eight frequent punctuation
  bytes no vocabulary piece or merge reaches across, shrinking the distinct
  units the pretoken cache holds: another ~1.3× on Llama and Mistral. The
  same split applies to any non-byte-level BPE vocabulary that passes the
  safety scan, `▁`-free ones included, on pretokens longer than a cache key.
- SentencePiece-style BPE tokenizers (Llama 1/2, Mistral, Gemma 2) encode
  4–5× faster: when a creation-time vocabulary scan proves no piece or merge
  can cross a `▁`-opened word boundary, `encode`/`encode_ids`/
  `encode_batch_ids` cut a whole-document span into `▁`-run word units that
  take the pretoken cache and the linear merge instead of one whole-document
  heap merge. Ids, offsets and tokens are unchanged; a vocabulary that fails
  the scan (Gemma 3) keeps the previous path.
- Honour a `stride` when truncation overflows: `truncation` gains a `stride`
  field and `?stride` argument, successive overflow windows overlap by it,
  and windows now match HuggingFace exactly — a pair's windows are the same
  combinations `tokenizers` produces (they were dropped before), and windows
  cover the tokenized pretokens rather than the whole excess, since
  HuggingFace stops tokenizing once truncation's `max_length` is reached.
- `Encoding.truncate`'s `~stride` and `~direction` are now optional,
  defaulting to `0` and `` `Right``; a stride at or past `max_length` raises
  `Invalid_argument` when the encoding is actually truncated.
- **Breaking:** Tidy the public API: `encode_pairs_batch` is now
  `encode_batch_pairs`, `from_json` is `of_json` and `train_wordlevel` is
  `train_word_level`; `Pre_tokenizer.whitespace`, `whitespace_split`, `bert`,
  `unicode_scripts` and `Decoder.byte_level`, `byte_fallback`, `fuse` are
  plain values; `Post_processor.bert` drops its `unit`; `Encoding.concat`
  takes a list, replacing `concat_list`; `save_model_files` is
  `?prefix -> t -> folder:string -> string list`, dropping its `unit`.
- **Breaking:** Remove the trainer options that did nothing: `?show_progress`
  on all four trainers, and `?shrinking_factor`, `?max_piece_length` and
  `?n_sub_iterations` on `train_unigram`.
- `Encoding.create` now validates that its seven arrays share one length and
  raises `Invalid_argument`, instead of crashing later on mismatched arrays.
- Add `Encoding.pp`, formatting an encoding as a table of tokens, ids,
  offsets, word ids and masks.
- The BPE pretoken cache gained a small resident front table (128 KB,
  direct-mapped, filled by promoting the main table's hits) probed before the
  8 MB main table, cutting its memory traffic on large corpora.
  `cache_capacity 0` disables both tables.
- Faster GPT-2 byte-level encoding: the C kernel now classifies text in
  64-byte batches (NEON on arm64, SWAR elsewhere) and derives pretoken
  boundaries with bitmask algebra, walking only non-ASCII neighbourhoods and
  batch edges byte by byte. `encode_ids` on wiki-64k drops ~12.7 to
  ~10.3 ns/pretoken; single-domain OpenWebText throughput rises ~10%.
- Faster byte-level batch encoding on real text: the C kernel's
  pretoken-cache probe is software-pipelined, fetching each span's cache line
  while the next span is walked — ~12% faster single-domain and ~7%
  multi-domain `encode_batch_ids` (GPT-2/RoBERTa over OpenWebText), at a
  small cost on single small documents.
- Tokenizers with added tokens configured encode faster: the scan for
  added-token occurrences now seeks candidate first bytes a word at a time
  instead of probing a table per byte, lifting single-domain GPT-2
  `encode_batch_ids` from ~99 to ~122 MB/s, with similar gains for RoBERTa and
  BERT.
- Tokenizers with added tokens configured encode faster: the scan for
  added-token occurrences now seeks candidate first bytes a word at a time
  instead of probing a table per byte, lifting single-domain GPT-2
  `encode_batch_ids` from ~99 to ~122 MB/s, with similar gains for RoBERTa and
  BERT.
- On native code, byte-level BPE (GPT-2, RoBERTa) now encodes through a fused
  C kernel: the byte-level pattern walk, the pretoken-cache probe and the
  merging of pretokens up to 15 bytes run in one pass over the text, roughly
  halving `encode_ids`/`encode_batch_ids` time on English text. Results are
  identical to the pure-OCaml path, which bytecode and js_of_ocaml keep using.
- BPE's pretoken cache is now two-way set-associative in the same 8 MB: a
  colliding pair of pretokens no longer evict each other, cutting the miss
  rate from 7.9% to 4.8% on real text (`encode_batch_ids` on OpenWebText runs
  ~9% faster). `cache_capacity` keeps its meaning — entries, rounded up to a
  power of two, `0` disables.
- BERT-style normalization (`Normalizer.bert`) runs as one pass with an ASCII
  fast lane: 24× faster on English text (18 → 440 MB/s), 2–3× on non-Latin
  scripts, identical output and offsets. `Brot.encode` with offsets on
  bert-base goes from 14.7 to 3.4 ms per 64 KB document, `encode_ids` from
  5.0 to 1.6.
- Unicode normalization (`nfc`, `nfd`, `nfkc`, `nfkd`) is streamed with a fast
  lane for ASCII and for characters already in normal form: 6–7× faster on
  mostly-ASCII text and 10–40% faster on Cyrillic, Greek, Vietnamese, Korean,
  Arabic and CJK; `lowercase`, `strip_accents`, `nmt` and `strip` are 3–5×
  faster; `apply_aligned` costs 1.3–1.5× `apply` instead of 2–7×, so offsets
  on normalized text are 3–35× cheaper, and `prepend`/`strip` track
  alignments for free.
- Fixed `apply_aligned` under `nfc`/`nfkc` composing an LV Hangul syllable
  with the following U+11C3 (a plain starter, not a trailing jamo); `apply`
  and HuggingFace never did.
- Offsets are exact through pre-tokenizer sequences that rewrite after
  splitting (`Sequence [WhitespaceSplit; Metaspace]` as in T5/ALBERT/XLNet,
  `Sequence [Split; ByteLevel]`): each token now reports its own bytes instead
  of the whole word's, and `Pre_tokenizer.pre_tokenize` places the pieces of
  later members exactly. Such pipelines also encode faster (T5 `encode_ids`
  ~1.6×) and take part in cut-document parallel batches.
- `Pre_tokenizer.metaspace ~prepend_scheme:`First` is honoured: the marker is
  prepended only to the piece that opens the document — not after white
  space, an added token, or bytes a normalizer removed — as HuggingFace does
  (`Always` was used before).
- Offsets of unknown tokens and byte-fallback runs match HuggingFace: a fused
  unknown run and every byte token of a fallback run stand for the run's
  bytes, so the tokens after them are no longer shifted (`Encoding.offsets`
  on Unigram/BPE models).
- A Unigram, WordPiece or WordLevel model behind a byte-level pre-tokenizer
  (`Pre_tokenizer.byte_level`, alone or in a sequence) is now handed
  byte-level-encoded pieces and matches its vocabulary in that form, as
  HuggingFace does; before, its ids were wrong.
- Add `Brot.encode_batch_ids`, the throughput path: the ids of a whole batch
  in one `int32` Bigarray (`Brot.ids`) plus per-row lengths, straight into
  `Nx` via `Nx.of_bigarray`; no `Encoding.t` and nothing allocated per token,
  and a long text is spread over domains when its pipeline allows. 7–8× over
  one domain on 8+ cores.
- `Brot.encode_batch` and `Brot.encode_pairs_batch` take `?domains` and split
  work by bytes rather than by document count, so one long text no longer
  leaves the other domains idle and small batches no longer pay for spawning
  domains (a 32-document batch went from 3.7 ms to 0.6 ms).
- Unigram tokenization now finds the segmentation whose scores add up to the
  most, as SentencePiece and HuggingFace do, instead of the longest match at
  each position, which mis-cut e.g. `traces` as `▁trace`+`s`; a run of
  characters the vocabulary does not hold is one unknown token, or its bytes
  under `byte_fallback`, and white space inside a pretoken is no longer
  dropped. `Brot.unigram` gains `?unk_id` and `?byte_fallback` (`unk_token`
  names the unknown entry when the vocabulary holds it; without one an
  uncovered character raises `Failure`); T5-style `tokenizer.json` (`unk_id`,
  `byte_fallback`) loads and saves faithfully.
- Unigram encoding is at least as fast as the greedy encoder it replaces
  (about 1.2×) and model loading about twice as fast, through a double-array
  trie.
- `Decoder.byte_level` now decodes as HuggingFace does: a token whose
  characters are not all in the byte-level alphabet stands for its own bytes as
  a whole rather than character by character, and the bytes of every token are
  read as one text. A character spelled across two tokens now decodes, and
  every maximal ill-formed byte sequence becomes one `U+FFFD` instead of being
  returned as invalid UTF-8.
- `Pre_tokenizer.pre_tokenize` now reports offsets into the text it was given
  for every pre-tokenizer. A `metaspace`, alone or in a `sequence`, reported
  offsets into the marked text, which could run past the end of the input —
  `Sequence [WhitespaceSplit; Metaspace]` placed the second piece of
  `"Hello world"` at `(6, 14)` instead of `(6, 11)`.
- `Pre_tokenizer.metaspace ~split:false` now gives every token exact byte
  offsets. Its pre-tokenizer joined the walking path, where before it fell
  back to whole pieces and every token of a document reported the document's
  span.
- `Encoding.offsets` reports byte spans of the text as it was passed in rather
  than of the normalized text: a token of `"café"` under an accent-stripping
  normalizer spans the accented bytes. They were in normalized coordinates
  before, which for BERT- and LLaMA-style pipelines pointed at the wrong bytes.
  Reading `offsets` on a pipeline with a normalizer costs a second
  normalization pass; reading only `ids` costs none.
- `Encoding.word_ids` is `Some` for every content token, numbering the
  pretokens of a sequence from `0`, an added token counting as one. It was
  `None` throughout.
- `Encoding.tokens`, `Encoding.offsets` and `Encoding.word_ids` are worked out
  when first read, so `encode` costs no more than `encode_ids` for a caller
  that only wants the ids.
- `Brot.encode` truncates before the post-processor runs, on a budget of
  `max_length` minus the special tokens it will add, and a pair gives up
  tokens the way HuggingFace's `LongestFirst` does. Truncation ran on the
  finished encoding before, so special tokens pushed content past
  `max_length`.
- `Brot.encode` truncates from the left by keeping the last `max_length`
  tokens, matching HuggingFace; it previously kept the first and left the rest
  in `overflowing`.
- `Brot.encode_ids` is ~5.9× faster on a 64 KB GPT-2 document and allocates
  about a thousandth of what it did (330 → 56 ns per pretoken, 778k → 0.9k
  minor words); `Encoding.with_type_id`, `Encoding.with_overflowing` and
  `Post_processor.affixes` are new.
- **Breaking:** `Normalizer.replace ~pattern ~replacement` now replaces a
  literal string with a plain scan (2.3× faster and 2.2× less allocation than
  through the regex engine). Regular expressions move to the new
  `Normalizer.replace_regex`, which reads the Unicode-aware dialect of
  tokenizer files (`\s`, `\d`, `\w`, `\p{..}` by short or long general
  category name; `.` and negated classes match characters, not bytes) and
  rejects unsupported constructs (`(?i)`, lookaround, backreferences, `\b`,
  ...) with a message saying which.
- **Breaking:** `Normalizer.byte_level` is a plain value: it dropped
  `add_prefix_space`, which HuggingFace's `ByteLevel` normalizer has no field
  for, so the JSON is canonical and the behaviour identical.
- Add `Normalizer.nmt`, the `Nmt` control-character cleanup, matching
  HuggingFace character for character.
- Normalizer JSON now round-trips `Replace` patterns written as `{"String":..}`
  or `{"Regex":..}`, `{"type":"ByteLevel"}` and `{"type":"Nmt"}`, so
  CLIP-style tokenizer files load; an unsupported regex is reported as
  `invalid regular expression "...": <why>`.
- Fix regex `Replace` splitting a multibyte character when stepping over an
  empty match: empty matches now advance by whole characters and one right
  after a match is skipped, as HuggingFace does.
- `Pre_tokenizer.punctuation ~behavior:`Merged_with_previous`` and
  `Pre_tokenizer.split ~behavior:`Contiguous`` returned the whole text as one
  piece instead of splitting it. They now match HuggingFace: the first keeps
  each delimiter with the text before it, the second keeps neighbours that are
  both delimiters, or both not, as one piece.
- `Pre_tokenizer.split ~invert:true` treated every byte outside the pattern as
  a delimiter of its own. It now inverts whole segments as HuggingFace does, so
  `~pattern:","` inverted splits `"a,,b"` into the two commas rather than into
  single bytes.
- `Pre_tokenizer.split ~pattern:""` returned the whole text. An empty pattern
  now makes every character a piece, and no pieces at all with `~invert:true
  ~behavior:`Removed``.
- `Post_processor.process` keeps both sequences of a pair when
  `~add_special_tokens:false`. It used to return only the first, silently
  dropping the second sentence. The second sequence gets type ID `1`, except
  under `roberta`, which has a single segment and puts every type ID at `0`
  with or without special tokens.
- A `template` post-processor applies its template when
  `~add_special_tokens:false`, dropping only the special pieces, so the order
  and type IDs a pair template assigns to `$A` and `$B` are honoured. Building
  the processed encoding is 8× faster.
- `train_bpe`, `train_wordpiece`, `train_wordlevel` and `train_unigram` count
  the pre-tokens their own pipeline produces: every text goes through
  `?normalizer` then `?pre`, and each piece is one word, as in HuggingFace.
  Training used to split on spaces whatever the pipeline was, so a byte-level
  model learned merges over words it would never meet. With no `?pre` a whole
  text is one word — pass `~pre:(Pre_tokenizer.whitespace_split ())` for the
  old behaviour. Training from `` `Files`` keeps each line's newline, as
  HuggingFace does, so a byte-level model trained from a file learns a token
  for it and blank lines count as a `"\n"` word.
- `initial_alphabet` entries are code points, not bytes:
  `train_bpe ~initial_alphabet:["é"]` now puts `é` in the vocabulary instead
  of the raw byte `"\xc3"`. Each string contributes the code point it starts
  with; empty or invalid entries are dropped.
- `train_bpe` and `train_wordpiece` no longer cap the alphabet at 1000
  characters when `limit_alphabet` is omitted, matching HuggingFace and the
  documented default of keeping every character.
- `train_wordlevel` numbers words after the special tokens instead of reusing
  ids `0..n-1` for both, which produced a vocabulary with two tokens per id;
  special tokens now count against `vocab_size`, as in HuggingFace.
- The `train_*` documentation now matches the code: `init` carries added and
  special tokens but never the model, `show_progress` displays nothing,
  `max_token_length` counts characters and holds a merge back once the joined
  run reaches it, and `train_unigram` states that its EM training is not
  implemented.
- `save_pretrained` writes a post-processor HuggingFace can read. The
  `ByteLevel` post-processor was missing `add_prefix_space`, which HuggingFace
  requires alongside `trim_offsets`, so a saved GPT-2 tokenizer failed to load
  at all; `TemplateProcessing` wrote `"pair": null`, which HuggingFace also
  rejects, and now writes the pair template.
- `Post_processor.roberta` honours `trim_offsets` and `add_prefix_space`, which
  it stored and ignored, and `Post_processor.byte_level` takes
  `?add_prefix_space`. Trimming now matches HuggingFace: it counts the space
  marker and whitespace in the encoded token, so a byte-level encoded tab or
  newline keeps its offsets, and a token that is only whitespace loses both
  ends.
- `Post_processor.template` without `~pair` uses HuggingFace's default
  `$A:0 $B:1` instead of raising when a pair is processed, `to_json` writes it,
  and `added_tokens ~is_pair:true` counts that pair rather than the single
  template's special tokens.
- Pre-tokenizers walk byte spans instead of building intermediate pieces:
  `Pre_tokenizer.pre_tokenize` is 1.2–2.1× faster and the byte-level (GPT-2)
  split runs at ~245 MB/s allocating nothing.
- `Pre_tokenizer.metaspace`'s `?replacement` is a `string` defaulting to `"▁"`
  (U+2581), which a `char` could not hold, and must be exactly one character.
  The marker is prepended only when the marked text does not already start
  with one, and `~split:false` reports the offsets of the text as given — both
  matching HuggingFace, which brot diverged from on text already containing the
  marker.
- `Pre_tokenizer.char_delimiter` takes a `string` of one character, so a
  multi-byte delimiter such as `"▁"` now works; HuggingFace's
  `CharDelimiterSplit` allows one.
- `Pre_tokenizer.pre_tokenize` returns no piece for an empty text, as
  HuggingFace does; `Byte_level ~use_regex:false`, `metaspace ~split:false` and
  `split ~pattern:""` used to return one empty piece.
- `Pre_tokenizer.pre_tokenize` no longer raises or reads past the input on
  malformed UTF-8: a truncated sequence, a byte that cannot lead one, and a
  surrogate encoding are each one byte of no category. It used to raise
  `Invalid_argument` on WTF-8 input.
- `Pre_tokenizer.split ~behavior:`Merged_with_previous`` reports the offsets of
  a delimiter that follows another delimiter instead of repeating the previous
  piece's; this matches HuggingFace.
- `Pre_tokenizer.to_json` writes a `Split` pattern as `{"String": …}` and a
  `Metaspace` `prepend_scheme` in lower case, the shapes HuggingFace requires;
  it used to write a bare string and `"Always"`, which HuggingFace refused to
  load. `of_json` reads both, defaults `prepend_scheme`, `split` and
  `Punctuation`'s `behavior` when absent, and reports a clear error for a
  `{"Regex": …}` pattern, which has no equivalent in brot.
- `Normalizer.to_json` writes `BertNormalizer`, the type name HuggingFace uses,
  so a saved tokenizer round-trips unchanged; it previously wrote `Bert`, which
  HuggingFace accepts but rewrites. Reading a `Strip` normalizer with a missing
  `strip_left` or `strip_right` now defaults it to `true`, matching
  `Normalizer.strip`, instead of stripping only on the right.
- `Normalizer.apply_aligned` returns the normalized text together with an
  `alignment` mapping its bytes back to the input, and
  `Normalizer.original_span` reads a span off it. Inserted characters take the
  span of the character they were placed next to and removed ones take none,
  matching what HuggingFace reports, so token offsets can be given in the
  coordinates of the original text.
- BPE tokenization now answers a repeated pretoken from a direct-mapped cache
  seeded with the whole vocabulary. `bpe`'s `cache_capacity` is the slot count
  of that cache (32 bytes a slot, one table per domain), default `262144`
  instead of `10000`; short words merge by a linear rank scan rather than a
  binary heap, which is 2x faster on a miss.
- Fixed a heap overflow in `bpe` with `byte_fallback` and
  `continuing_subword_prefix` or `end_of_word_suffix`: byte fallback spells a
  character out with its affixes, so one source byte becomes several tokens
  and the merge buffers were sized for the bytes. Words of more than a few
  characters corrupted the heap.
- Fixed the placement of `unk_token` around byte fallbacks in `bpe`. A
  fallback no longer lets a pending unknown token out first, so `"za"` with
  `<0x7A>` absent gives the fallback tokens of `a` followed by `<unk>`, as
  HuggingFace does, rather than the reverse.
- Fixed `bpe` reading past the end of a pretoken whose last UTF-8 sequence is
  cut short; with `continuing_subword_prefix` or `end_of_word_suffix` this
  raised `Invalid_argument`. The bytes that remain are now taken as one unit
  and fall through to the byte fallback or the unknown token.
- Loading and saving a tokenizer now carry the BPE model's `byte_fallback`,
  `fuse_unk`, `ignore_merges` and `dropout`. LLaMA and other SentencePiece
  models were dropping `byte_fallback` on load, so every character outside the
  vocabulary became `<unk>` instead of its `<0xNN>` byte tokens, and
  `save_pretrained` wrote the flags back as `false`.
- Decoders rewrite each token instead of joining the list first.
  `Decoder.replace`, `Decoder.strip`, `Decoder.wordpiece` and `Decoder.ctc`
  were collapsing, which stopped a later `Decoder.byte_fallback` in a
  `Decoder.sequence` from ever seeing a byte token — LLaMA decoded `<0x0A>`
  literally.
- `Decoder.ctc` cuts `pad_token` out of a token wherever it occurs rather than
  only dropping tokens equal to it, and drops the tokens left empty:
  `["x<pad>y"]` decodes to `"xy"`.
- `Decoder.strip` takes `~content:string ~start:int ~stop:int`, the counts
  HuggingFace serializes, instead of `~left ~right` booleans; `~content` was a
  `char` and could not hold a marker like `▁`.
- `Decoder.metaspace` takes `~replacement:string` and `~prepend_scheme` instead
  of a `char` and `~add_prefix_space`, and drops every marker in the first
  token rather than one leading space.
- `Decoder.bpe` turns every occurrence of its suffix into the space that
  follows the word, and `~suffix` now defaults to `"</w>"`. It no longer
  inserts a space after a token that has no suffix.
- `Decoder.byte_fallback` decodes a run of byte tokens that is not valid UTF-8
  as one U+FFFD per byte, instead of returning invalid bytes.
- `Decoder.to_json` writes the type names and the `Replace` pattern shape
  HuggingFace reads; the files it produced before were rejected by
  `tokenizers` for `byte_level`, `byte_fallback` and `wordpiece` decoders.
- `Decoder.of_json` reads only HuggingFace's spellings; the brot-only
  `"Byte_level"`, `"Byte_fallback"`, `"Word_piece"` and bare-string `Replace`
  pattern are gone. A tokenizer saved by an earlier brot carries them and must
  be re-saved to load again — those files were never readable by `tokenizers`
  either.
- `Brot.id_to_token` and `Brot.decode` give an added token matched against
  normalized text its normalized form, as HuggingFace does: `id_to_token t 2`
  on LLaMA is `"▁</s>"` while `token_to_id` still takes `"</s>"`, and a `<s>`
  written literally in the input round-trips through encode and decode.
- `train_bpe` now applies `end_of_word_suffix` and `continuing_subword_prefix`
  while learning: the vocabulary gains the affixed characters (`w</w>`) and the
  merges are written over them (`lo w</w>`). A model trained with a suffix
  previously held no suffixed entry at all, so every word-final character
  missed at encode time.
- `train_bpe` now drops characters excluded by `limit_alphabet` from the words
  instead of merging them, counts `max_token_length` in characters rather than
  bytes, settles equally frequent pairs by vocabulary id, and can merge a pair
  a second time when it reappears, recording it once at its later rank. Trained
  vocabularies and merges match HuggingFace's `BpeTrainer` exactly wherever
  that trainer is deterministic.
- `train_bpe` learns merges from an incremental pair index instead of
  recounting every pair each round: a 1 MB corpus trains in 0.03 s instead of
  3.0 s.
- A merge pair listed twice in a `merges.txt` or `tokenizer.json` now takes the
  rank of its last occurrence, as HuggingFace does; the first occurrence used
  to win.
- `Brot.special` is now `Brot.added_token` and the record it builds is
  `added_token`, its `token` field renamed to `content`; `?specials` is
  `?added_tokens` on every constructor and `specials` is `added_tokens`. The
  type covers HuggingFace's added tokens, of which special ones are a subset,
  so the old names described only half of what it holds.
- `add_tokens` takes an `added_token list` and works for every model, not just
  the word-level one: it registers added tokens exactly as passing them at
  construction would, and no longer raises `Invalid_argument`. Added tokens no
  longer enter the model's own vocabulary — they are numbered from the end of
  it, as HuggingFace does — so registering the same token twice no longer
  drifts the identifiers it hands out. To build a model vocabulary, pass
  `vocab` to the constructor.
- `Pre_tokenizer.unicode_scripts` now matches HuggingFace on whitespace and
  unknown scripts. A leading run of spaces used to be emitted as a piece of its
  own; it is now dropped, since the first piece opens at the first script
  change. Only U+0020 and characters of no known script join the surrounding
  run — the other whitespace characters (`\t`, `\n`, U+00A0, U+3000, …) carry
  their own script and split.
- `encode` now splits added and special tokens out of the input ahead of the
  pre-tokenizer and the model, matching HuggingFace: `"a<|endoftext|>b"` with
  GPT-2 gives `[64; 50256; 65]` instead of tokenizing the marker as text. At a
  given position the longest token wins; `~single_word`, `~lstrip`, `~rstrip`
  and `~normalized` on `Brot.added_token` all take effect.
- `Brot.added_token` takes `?special` (default `true`); `?normalized` now
  defaults to `not special`, so `added_token c` matches HuggingFace's
  `add_special_tokens([c])` and `added_token ~special:false c` matches
  `add_tokens([c])`. `decode ~skip_special_tokens:true` drops only the tokens
  with `special` set, so a plain added token survives decoding.
- A `bos_token`, `eos_token` or `pad_token` is now a special token in its own
  right: matched atomically in the input, numbered from the end of the
  vocabulary when the model does not hold it, and skipped when decoding. This
  makes `padding` work with a pad token that is not in the model vocabulary.
  `unk_token` is unaffected — it configures the model's unknown handling and is
  never matched in the input.
- `token_to_id`, `id_to_token`, `vocab` and `vocab_size` now cover added tokens
  the model does not hold; those are numbered from the end of the model
  vocabulary, as HuggingFace does. `added_tokens` reports the same set that
  `to_json` writes, with real ids and `special` flags.
- BPE `end_of_word_suffix` is now appended to the last character of a word
  instead of the first, and a one-character word takes it too;
  `continuing_subword_prefix` goes on every character but the first. Models
  such as CLIP that set a suffix previously produced wrong tokens for every
  word. Byte fallback now covers the affixed character, prefix and suffix bytes
  included.
- `Decoder.wordpiece ~cleanup:true` now applies HuggingFace's detokenization
  cleanup: the space before `.`, `?`, `!`, `,` and the English contractions is
  taken back and `" do not"` becomes `" don't"`, so `["hello"; ","; "world"]`
  decodes to `"hello, world"`. It no longer trims or collapses whitespace.
- `Decoder.ctc ~cleanup:true` now applies the same cleanup to each token before
  replacing the word delimiter, as HuggingFace does. Previously only the
  delimiter was replaced.
- Fix BERT tokenization to match HuggingFace: the `Bert` and `Punctuation`
  pre-tokenizers now treat all 32 printable ASCII non-alphanumerics as
  punctuation. `$ + < = > ^ ` | ~` were missing, so `==` tokenized as one word
  instead of two.
- `Normalizer.bert` now strips only nonspacing marks after NFD, keeping spacing
  and enclosing marks. Stripping every mark dropped the vowel signs of abugidas,
  so `नमस्ते हिन्दी` lost two characters.
- `Normalizer.lowercase` and `Normalizer.bert ~lowercase:true` now apply the
  Unicode lowercase mapping instead of case folding: `ß` and `ﬁ` lowercase to
  themselves rather than expanding to `ss` and `fi`.
- `Normalizer.strip_accents` no longer decomposes to NFD on its own, matching
  HuggingFace's `StripAccents`. Compose it after `Normalizer.nfd` to strip the
  accents of precomposed characters.
- `Normalizer.bert ~clean_text:true` keeps unassigned codepoints instead of
  discarding them as control characters, so they reach the model and become its
  unknown token.
- Fix `encode` returning stale tokens from a previously encoded word, or
  crashing, when a word is made only of characters with no id and the model
  has no `unk_token` or `byte_fallback`. Such a word now yields no tokens,
  matching HuggingFace.
- Fix `encode_batch` returning wrong tokens in rare cases: domains racing on
  the BPE word cache could pair one word's key with another word's tokens, so a
  cache hit returned the wrong ids. Merge scratch buffers are now held per
  domain, so parallel encoding no longer allocates a fresh word and merge queue
  per token.
- Fix GPT-2 (`ByteLevel`) pre-tokenization of whitespace: a run followed by
  text keeps all but its last character, so `"\n\nNot"` splits as `"\n"`,
  `"\n"`, `"Not"` and `"x  y"` as `"x"`, `" "`, `" y"`, as HuggingFace has it.
- Fix the optional leading character of a letter, number or symbol run in that
  same pattern: it is a space, not any whitespace, so `"\tab"` splits as `"\t"`
  then `"ab"`. `add_prefix_space` likewise only skips a leading space.
- Fix letters in that pattern being the Alphabetic property instead of the
  Unicode Letter category, which wrongly joined a combining mark to the letter
  it follows.
- Fix the `Whitespace` pre-tokenizer's word class, which now holds for
  combining marks and connector punctuation and no longer for numbers such as
  `"½"`, matching `\w`.
- Fix `Bpe` emitting any pre-token found in the vocabulary as a single token.
  That shortcut is what `ignore_merges` selects, and it never applies under
  `dropout`; without it the merges decide, and a vocabulary entry no merge can
  build comes out as its decomposition.
- Fix `Bpe` reusing cached merges under `dropout`, which replayed the first
  result for every later occurrence of a word instead of drawing again.
- `Bpe.create` now reads `""` for `continuing_subword_prefix` and
  `end_of_word_suffix` as no affix, which is how tokenizer files spell it: the
  accessors return `None` for those models, and GPT-2 no longer takes the
  allocating affix path when encoding.

### Talon

- `talon.parquet` decodes definition levels and booleans straight into bits
  and puts a page's values on its rows in one C pass: a 2^20-row file of
  nullable columns reads in 0.6 of the time.
- **Breaking:** a `bool` column stores its values as `Nx.bit`:
  `Column.to_tensor Nx.bit` and `Column.ragged Nx.bit` read them, and
  `Nx.cast Nx.bool` gives bytes. `Column.of_tensor` shares a 1-D `bit`
  tensor and packs a `bool` one.
- A column's validity is an `Nx.bit_t` (`Column.validity`, `of_tensor`,
  `of_ragged`, `layout`), and its null count is read at the first
  `Column.null_count`, then kept: derived columns, filters, takes and joins
  read no count, and a filter gathers each validity once. A validity may have
  no null. The values under a null are unspecified, so a reader of
  `Column.layout` masks them; `Query.optimize` changes no value and no null.
- **Breaking:** talon is rewritten, and the previous API (`Col`, `Row`, `Agg`,
  `pp_display`, `to_html` and the old `Talon_csv`) is removed. A `Talon.t` holds
  typed columns in Arrow layouts over nx buffers, with nulls as validity, never
  as sentinels. A `Talon.Query.t` is built by verbs (`select`, `filter`,
  `sort`, `slice`, `aggregate`, `join`, `append`, `derive`) over expressions
  `('a, 's) Talon.Expr.t`, whose type says whether they hold a value per row or
  per group. Each verb checks its input's schema when applied and reports every
  problem at once, before any data is read; `Query.run` runs the plan as nx
  operations over batches.
- The rewrite fixes the old talon's wrong answers: duplicates and pivots
  compared rows by their printed text, `sort_values` put nulls first,
  `cumsum`, `diff` and `shift` dropped nulls, and the CSV reader split lines
  before quotes, read `0` and `1` as booleans and read a bad field as null.
  `talon.csv` now reads RFC 4180 strictly, fails with the line and column of a
  bad field, and writes with `Talon_csv.encode`.
- Add `talon.parquet`, which reads flat Parquet files mapped in memory, with
  row-group pruning from statistics.
- Later releases add windows, `ewm` and `cut`; ordered, as-of and interval
  joins; zoned time; hashing, sampling and folds; nested columns, `unnest`
  and records; Parquet writing; Arrow IPC and JSON; Unicode case mapping; and
  HTML display in Quill.

### Quill

- The kernel no longer turns markdown data URIs in the toplevel's output into
  displays. Figures display through display tags, as `Hugin.pp` prints them.
- `Quill_top.install_printer` returns the toplevel's report when a printer does
  not install, and quill prints it instead of ignoring it. Quill installs
  `Talon.pp` and `Talon.Query.pp`, and loads `talon.parquet`.
- A cell displays the display tags it prints with `Format.printf` or
  `Format.eprintf` where it prints them, so one cell can show several values.
- A display with a non-empty id replaces the session's display with that id in
  place, in any cell (`Quill.Cell.Display`'s `id`, `Quill.Doc.add_output`).
- Printers display by opening a `Format.String_tag` display tag
  (`Quill.Cell.output_of_tag`), without linking Quill.
  `Quill.Cell.Display_tag` is removed.
- The web notebook shows SVG displays holding any UTF-8: `Quill.Cell.Display`
  holds images in base64, and the notebook no longer encodes SVG with `btoa`.
- The terminal notebook's footer shows `Ctrl-C Interrupt` again while a cell
  runs, in place of the run action. The footer redesign had dropped it, so
  interrupting was only discoverable from the help screen.
- Building quill no longer needs a node toolchain. The bundling rule for the
  server frontend was a target, so a directory build such as
  `dune build packages/quill` ran esbuild and failed without `node_modules`;
  the rule now lives under the `assets` alias only, and
  `dune build @assets --auto-promote` still refreshes the committed `dist/`.
- Allow `quill file.md` without requiring `quill -- file.md` or `quill run file.md`.
  The CLI now detects file arguments and routes them to the default TUI command.
- Fix image Display outputs showing raw base64 text in markdown files. Images now
  render as inline `<img>` tags with data URIs, visible in any markdown viewer.
- Add `--figures-dir` flag to `quill run` for writing images to disk and
  referencing them by path instead of inlining base64 data.
- Improve table styling in the web notebook and book build with clean borders,
  monospace font, and proper header treatment.
- Resolve relative notebook paths to absolute and change into the notebook
  directory before execution, so that relative file references in code cells
  work correctly.
- Add `vega` to the default Raven packages loaded in Quill kernels.
- Remove `Quill_top.install_printer_fn`. It was unused and relied on
  `Toploop.install_printer`, which was removed in OCaml 5.5. Use
  `Quill_top.install_printer` instead.

## [1.0.0~alpha3] - 2026-03-14

This release reshapes raven's foundations. Every package received API
improvements, several were rewritten, and two new packages — nx-oxcaml and
kaun-board — were built as part of our Outreachy internships.

### Highlights

- **Unified tensor type** — `Nx.t` and `Rune.t` are now the same type.
  Downstream packages no longer need to choose between them or convert at
  boundaries. Rune is now a pure transformation library (grad, vjp, vmap)
  over standard Nx tensors.
- **nx-oxcaml** (new, Outreachy) — Pure-OCaml tensor backend using OxCaml's
  unboxed types and SIMD intrinsics. Performance approaches the C backend —
  in pure OCaml.
- **kaun-board** (new, Outreachy) — TUI dashboard for monitoring training
  runs in the terminal. Live metrics, loss curves, and system stats.
- **quill** — Rewritten from the ground up with two interfaces: a terminal UI
  with syntax highlighting and code completion, and a web frontend via
  `quill serve` with a CodeMirror 6 editor, WebSocket-based execution,
  autocompletion, and diagnostics.
- **brot** — The tokenization library formerly known as saga. Complete rewrite
  with a cleaner API. [1.3-6x faster than HuggingFace Tokenizers](packages/brot/bench/)
  on most benchmarks.
- **nx** — Redesigned backend interface, RNG with effect-based scoping.
  Einsum **8-20x** faster, matmul dispatch at BLAS parity with NumPy.

### Breaking changes

- **nx**: Redesigned backend interface with new `Nx_buffer` type. Removed
  `nx.datasets` library. Moved NN functions to Kaun (use `Kaun.Fn`). Renamed
  `im2col`/`col2im` to `extract_patches`/`combine_patches`. RNG uses
  effect-based implicit scoping instead of explicit key threading. Removed
  in-place mutation operations (`ifill`, `iadd`, `isub`, `imul`, `idiv`,
  `ipow`, `imod`, `imaximum`, `iminimum` and `_s` variants). Removed
  `Symbolic_shape` module; shapes are concrete `int array` throughout.
  Removed `Instrumentation` module.
- **rune**: `Rune.t` no longer exists — use `Nx.t` everywhere. `Rune` no
  longer re-exports tensor operations; use `open Nx` for tensor ops and
  `Rune.grad`, `Rune.vjp`, etc. for autodiff. Remove any `Rune.to_nx` /
  `Rune.of_nx` calls. Removed `enable_debug`, `disable_debug`, `with_debug`;
  use `Rune.debug f x` instead.
- **rune**: Removed JIT/LLVM backend. This will come back in a future
  release with a proper ML compiler.
- **kaun**: Rewritten core modules API, datasets, and HuggingFace integration.
  Removed `kaun-models`.
- **brot**: Renamed from saga. Rewritten API focused on tokenization.

### Nx

- Unify `Nx.t` and `Rune.t` into a single tensor type. A new `nx.effect` library (`Nx_effect`) implements the backend interface with OCaml 5 effects: each operation raises an effect that autodiff/vmap/debug handlers can intercept, falling back to the C backend when unhandled. `Nx.t` is now `Nx_effect.t` everywhere — no more type conversions between Nx and Rune.
- Make transcendental, trigonometric, and hyperbolic operations (`exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `sigmoid`) polymorphic over all numeric types including complex, matching the backend and effect definitions.
- Make `isinf`, `isfinite`, `ceil`, `floor`, `round` polymorphic (non-float dtypes return all-false/all-true or no-op as appropriate).
- Redesign backend interface with more granular operations (e.g. dedicated unary and binary kernels). This improves performance by letting backends optimize individual ops directly, and prepares for the JIT pipeline which will decompose composite operations at the compiler level instead of the frontend.
- Rewrite `Nx_buffer` module with new interface. The backend now returns `Nx_buffer.t` instead of raw bigarrays.
- Add new C kernels for unary, binary, and sort operations, and route new backend ops to C kernels.
- Add scipy-style `correlate`, `convolve`, and sliding window filters.
- Generalize `unfold`/`fold` to arbitrary leading dimensions.
- Remove neural-network functions from Nx (softmax, log_softmax, relu, gelu, silu, sigmoid, tanh). These now live in `Kaun.Fn`.
- Rename `im2col`/`col2im` to `extract_patches`/`combine_patches`.
- Remove `nx.datasets` module. Datasets are now in `kaun.datasets`.
- Simplify `Nx_io` interface. Inline vendor libraries (safetensors, and npy) directly into nx_io.
- Move the `Rng` module from Rune into Nx with effect-based implicit scoping. Random number generation uses `Nx.Rng.run` to scope RNG state instead of explicit key threading.
- Reduce matmul dispatch overhead to reach BLAS parity with NumPy.
- Fix Threefry2x32 to match the Random123 standard.
- Fix `save_image` crash on multi-dimensional genarray.
- Pre-reduce independent axes in einsum to avoid OOM on large contractions.
- Make Nx backends pluggable via Dune virtual libraries. The new `nx.backend` virtual library defines the backend interface, with the C backend (`nx.c`) as the default implementation. Alternative backends (e.g., `nx-oxcaml`) can be swapped in at link time. The `Nx_c` module is renamed to `Nx_backend`.
- Fix `.top` libraries failing to load in utop with "Reference to undefined compilation unit `Parse`".
- Fix OpenMP flag filtering in `discover.ml`: strip `-Xpreprocessor -fopenmp` as a pair on macOS to prevent dangling `-Xpreprocessor` from consuming subsequent flags and causing linker failures. (@Alizter)
- Add missing bool→low-precision cast support (f16/bf16/fp8) in the C backend.
- Add UInt32/UInt64 dtypes, rename complex dtypes to Complex64/Complex128, and drop Complex16/QInt8/QUInt8/Int/NativeInt as tensor element dtypes.
- Remove in-place mutation operations (`ifill`, `iadd`, `isub`, `imul`, `idiv`, `ipow`, `imod`, `imaximum`, `iminimum` and `_s` variants). Use functional operations instead.
- Remove `Symbolic_shape` module; shapes are now concrete `int array` throughout.
- Remove `Instrumentation` module. Nx no longer wraps operations in tracing spans. Debugging tensor operations is handled by Rune's effect-based debug handler.
- Fix critical correctness issue in fancy slicing (`L`) where permutations were ignored if the number of indices matched the dimension size (e.g., `slice [L [1; 0]] x` returned `x` unmodified).
- Rewrite `slice` implementation to use `as_strided` for contiguous operations, reducing overhead to **O(1)** for view-based slices and separating gather operations for better performance.
- Optimize `set_slice` by replacing scalar-loop index calculations with vectorized coordinate arithmetic, significantly improving performance for fancy index assignments.
- Improve `einsum` performance **8–20×** with greedy contraction path optimizer (e.g., MatMul 100×100 f32 207.83 µs → 10.76 µs, **19×**; BatchMatMul 200×200 f32 8.78 ms → 435.39 µs, **20×**)
- Rewrite `diagonal` using flatten + gather approach instead of O(N²) eye matrix masking, reducing memory from O(N²) to O(N)
- Improve error messages for shape operations (`broadcast`, `reshape`, `blit`) with per-dimension detail and element counts.

### nx-oxcaml (new)

New pure-OCaml tensor backend that can be swapped in at link time via Dune virtual libraries. Uses OxCaml's unboxed types for zero-cost tensor element access, SIMD intrinsics for vectorized kernels, and parallel matmul. Performance approaches the native C backend — in pure OCaml. Supports the full Nx operation set: elementwise, reductions, matmul, gather/scatter, sort/argsort, argmax/argmin, unfold/fold, pad, cat, associative scan, and threefry RNG. (@nirnayroy, @tmattio)

### Rune

- Unify tensor types: `Rune.t` is now `Nx.t`. Rune no longer re-exports the Nx frontend — it is a pure transformation library exporting only `grad`, `grads`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `no_grad`, `detach`, and debugging/gradcheck utilities. All tensor creation and manipulation uses `Nx` directly.
- Remove `Tensor` module and `Nx_rune` backend. Effect definitions moved to the new `nx.effect` library shared with Nx.
- Remove `Rune.to_nx` / `Rune.of_nx` (no longer needed — types are identical).
- Remove `Rune.enable_debug`, `Rune.disable_debug`, `Rune.with_debug`. Use `Rune.debug f x` to run a computation with debug logging enabled.
- Remove JIT compilation support from Rune. The `Rune.Jit` module and LLVM/Metal backends have been removed and will be re-introduced later as a standalone package.
- Update to new `Nx_buffer.t` type.
- Propagate new backend operations through effects and autodiff.
- Rewrite `Autodiff` module to fix critical JVP correctness issues, enable higher-order derivatives (nested gradients), and introduce `vjp` as a first-class primitive.
- Fix pointer-based hashing in autodiff, correcting nested JVP handler behavior.
- Add autodiff support for `as_strided`, enabling gradients through slicing and indexing operations
- Add autodiff support for `cummax` and `cummin` cumulative operations
- Add autodiff support for FFT operations
- Add autodiff support for some linear algebra operations: QR decomposition (`qr`), Cholesky decomposition (`cholesky`), and triangular solve (`triangular_solve`).

### Kaun

- Simplify and redesign the core API for better discoverability and composability. Layers, optimizers, and training utilities now follow consistent patterns and compose more naturally.
- Add `Fn` module with `conv1d`, `conv2d`, `max_pool`, `avg_pool` — neural network operations that were previously in Nx now live here with a cleaner, more focused API.
- Redesign datasets and HuggingFace integration with simpler, more composable APIs.
- Remove `kaun-models` library. Pre-built models now live in examples.
- Reinitialize dataset each epoch to avoid iterator exhaustion (#147, @Shocker444, @tmattio)

### kaun-board (new)

TUI dashboard for monitoring training runs in the terminal. Displays live metrics, loss curves, and system stats. Extracted from kaun's console module into a standalone package. (#166, #167, #170, @Arsalaan-Alam)

### Brot

- Rename the library from saga to brot.
- Simplify brot to a tokenization-only library. Remove the sampler, n-gram models, and I/O utilities. The sampler is rewritten with nx tensors and moved to `dev/mimir` as the seed of an experimental inference engine.
- Merge `brot.tokenizers` sub-library into `brot`.
- Remove dependency on Nx.
- Use `Buffer.add_substring` instead of char-by-char loop in whitespace pre-tokenizer.
- Compact BPE symbols in-place after merges, avoiding an intermediate array allocation.
- Replace list cons + reverse with forward `List.init` in BPE `word_to_tokens`.
- Use pre-allocated arrays with `Array.blit` instead of `Array.append` in encoding merge and padding, halving per-field allocations.
- Avoid allocating an unused `words` array in post-processor encoding conversion.
- Reduce WordPiece substring allocations from O(n²) to O(n) per word by building the prefixed candidate string once per position.
- Add `encode_ids` fast path that bypasses `Encoding.t` construction entirely when only token IDs are needed.
- Add ASCII property table for O(1) character classification in pre-tokenizers, replacing O(log n) binary search for `is_alphabetic` (600 ranges), `is_numeric` (230 ranges), and `is_whitespace` (10 ranges). Yields 12-27% speedup on encode benchmarks with ~30% allocation reduction.
- Add inline ASCII fast paths in all pre-tokenizer loops, skipping UTF-8 decoding and using `Buffer.add_char` instead of `String.sub` for single-byte characters. Combined with the property table, yields 20-30% total speedup and 36-55% allocation reduction vs baseline.
- Parallelize batch encoding with OCaml 5 domains.
- Optimize BPE merge loop with open-addressing hash, flat arrays, and shift-based heap.
- Add trie-based WordPiece lookup and normalizer fast path.
- Remove dependency on `str` library.
- Generate unicode data offline, removing runtime dependency on `uucp`.
- Remove unused `Grapheme` module. Grapheme cluster segmentation is not needed for tokenization.
- Remove `uutf` dependency in favour of OCaml `Stdlib` unicode support.

### Fehu

- Simplify and redesign the core API. Environments and training utilities now follow consistent functional patterns that are easier to use and compose.
- Remove `fehu.algorithms` — fehu now only depends on rune, and users bring their own algorithms. Examples provided for well-known RL algorithms like DQN and REINFORCE.

### Sowilo

- Cleaner public API — internal implementation split into focused submodules while the public surface stays small.
- Faster grayscale conversion, edge detection, and gaussian blur.

### Quill

Rewritten from the ground up. Terminal UI with syntax highlighting, code completion, and a compact single-line footer. Web frontend via `quill serve` with a CodeMirror 6 editor, WebSocket-based execution, autocompletion, and diagnostics. Markdown notebook format shared across both interfaces.

Interactive REPL: `quill` with no file argument launches a toplevel with syntax highlighting, tab completion, persistent history, smart phrase-aware submission, and piped mode.

### Hugin

Rewritten from the ground up with a declarative, composable API. Plots are
built by combining inert mark descriptions (`line`, `point`, `bar`, `hist`,
`heatmap`, `contour`, `errorbar`, etc.) with `layers`, decorating them
(`title`, `xlabel`, `legend`, etc.), and laying them out (`grid`, `hstack`,
`vstack`). A compilation pass resolves data to a Scene IR that separate
backends render.

- New declarative specification API replacing the imperative figure/axes/artist
  architecture. Marks compose with `layers`, decorations chain functionally,
  and grid layouts nest arbitrarily.
- **ucairo** — Minimal Cairo FFI bindings (36 C stubs) replacing the `cairo2`
  opam dependency.
- Dual-backend rendering: Cairo (PNG, PDF, interactive SDL window) and SVG from
  a shared Scene IR.
- OKLCH perceptual color space with `Color.oklch`, `Color.hex`, named CSS
  colors, and alpha support.
- Curated colormaps (`Cmap.viridis`, `plasma`, `inferno`, `magma`, `cividis`,
  `turbo`, `coolwarm`, `spectral`).
- Theme system with `light`, `dark`, and `minimal` presets.
- Linear, log, and symlog axis scaling with automatic tick generation.
- Legend placement with configurable location and multi-column layout.
- Interactive `show` with SDL window resizing, Escape/Q to close.
- Rewritten examples and documentation.

### Talon

- Remove `jsont`, `bytesrw`, and `csv` dependencies from Talon. CSV support is now built-in via the `talon.csv` sub-library with a minimal RFC 4180 parser.
- Remove `talon.json` sub-library.

## [1.0.0~alpha2] - 2025-11-03

We're excited to announce the release of Raven 1.0.0~alpha2! Less than a month after alpha1, this release notably includes contributions from Outreachy applicants in preparation for the upcoming _two_ internships.

Some highlights from this release include:

- NumPy-compatible text I/O with `Nx_io.{save,load}_text`
- Lots of new functions in Nx/Rune, including neural-net ones `dropout`, `log_softmax`, `batch_norm`, `layer_norm`, and activation functions like `celu` and `celu`, and generic ones like `conjugate`, `index_put`, and more.
- Addition of `.top` libraries for `nx`, `rune`, and `hugin` that auto-install pretty-printers in the OCaml toplevel. You can run e.g. `#require "nx.top"`.
- Addition of a visualization API in Fehu via the new `fehu.visualize` library, supporting video recording.
- Redesign of Kaun core datastructure and checkpointing subsystem for complete snapshotting.
- Many, many bug fixes and correctness improvements.

We've also made numerous performance improvements across the board:

- Nx elementwise ops: 5–50× faster (e.g., Add 50×50 f32 88.81 µs → 1.83 µs, **48×**; Mul 100×100 f32 78.51 µs → 2.41 µs, **33×**).
- Nx conv2d: **4–5×** faster on common shapes; up to **115×** on heavy f64 batched cases (e.g., B16 C64→128 16×16 K3 f64 1.61 s → 13.96 ms).
- Rune autodiff: **1.2–3.7×** faster on core grads (e.g., MatMulGrad Medium 34.04 ms → 11.91 ms, **2.86×**; Large 190.19 ms → 50.97 ms, **3.73×**).
- Talon dataframes: big wins in joins and group-bys (Join 805.35 ms → 26.10 ms, **31×**; Group-by 170.80 ms → 19.03 ms, **9×**; Filter 9.93 ms → 3.39 ms, **3×**).
- Brot tokenizers: realistic workloads **4–17%** faster (e.g., WordPiece encode single 136.05 µs → 115.92 µs, **1.17×**; BPE batch_32 24.52 ms → 22.27 ms, **1.10×**)

We're closing 8 user-reported issues or feature requests and are totalling 30 community contributions from 8 unique contributors.

### Nx

- Fix einsum output axis ordering for free axes (e.g., `i,jk->jki`, `ij,klj->kli`) by correcting final transpose permutation and intermediate left-axis reordering.
- Add `Nx_io.Cache_dir` module with consolidated cache directory utilities respecting `RAVEN_CACHE_ROOT`, `XDG_CACHE_HOME`, and `HOME` fallback, replacing project-specific cache logic across the whole raven ecosystem (#134, @Arsalaan-Alam)
- Add `Nx_io.save_txt` / `Nx_io.load_txt` with NumPy-compatible formatting, comments, and dtype support (#120, @six-shot)
- Optimize `multi_dot` for matrix chains, reducing intermediate allocations and improving performance
- Add public `index_put` function for indexed updates
- Clarify `reshape` documentation to match its view-only semantics
- Provide `nx.top`, `rune.top`, and `hugin.top` libraries that auto-install pretty printers in the OCaml toplevel and update Quill to load them
- Add `ifill` for explicit in-place fills and make `fill` return a copied tensor
- Speed up contiguous elementwise ops via vectorized loops
- Fast-path contiguous single-axis reductions to avoid iterator fallback
- Speed up float reductions with contiguous multi-axis fast paths
- Fast-path padding-free `unfold` to lower conv2d overhead
- Move neural-network operations (softmax, log_softmax, relu, gelu, silu, sigmoid, tanh) from Kaun to Nx
- Add public `conjugate` function for complex number conjugation (#125, @Arsalaan-Alam)
- Fix complex vdot to conjugate first tensor before multiplication, ensuring correct mathematical behavior (#123, @Arsalaan-Alam)
- Update comparison and conditional operations to use boolean tensors (#115, @nirnayroy)
- Add support for rcond parameter and underdetermined systems to `lstsq` (#102, @Shocker444)
- Fix `matrix_rank`/`pinv` Hermitian fast paths to use eigen-decomposition and match NumPy for complex inputs (#96, @six-shot, @tmattio)
- Optimize matmul BLAS dispatch for strided tensors, improving matrix multiplication performance
- Fix slow builds reported since alpha1 (#88, @tmattio)
- Fix macOS ARM crash when loading extended bigarray kinds
- Add float16 and bfloat16 support to safetensors I/O, including precise conversions that preserve denormals/NaNs (#84, @six-shot, @tmattio)
- Refined `View` internals for leaner contiguity checks and stride handling, cutting redundant materialization on hot paths
- Merge `Lazy_view` into the core `View` API so movement ops operate on a single composed view
- Documented the reworked `View` interface
- Documented the `Symbolic_shape` interface
- Added Accelerate framework flag when compiling on macOS, fixing issues in some environments (#129, @nirnayroy)

### Hugin

- Fix random `SIGBUS`/bus errors on macOS when closing `Hugin.show` windows by
  destroying SDL windows with the correct pointer in the finalizer.
- Let `Hugin.show` windows close cleanly via the window button or `Esc`/`q`, avoiding frozen macOS REPL sessions

### Rune

- Add `Rune.no_grad` and `Rune.detach` to mirror JAX stop-gradient semantics
- Improve gradient performance slightly by replace the reverse-mode tape's linear PhysicalTbl with an identity hash table
- Fix `Rune.Rng.shuffle` flattening outputs for multi-dimensional tensors; the
  shuffle now gathers along axis 0 and keeps shapes intact
- Replace `Rune.Rng.truncated_normal` clipping with rejection sampling so
  samples stay inside the requested interval without boundary spikes
- Add support for categorical sampling with `Rune.Rng.categorical` (#89, @nirnayroy)
- Allow plain `llvm-config` in discovery, fixing build in some platforms (#71, @stepbrobd)

### Kaun

- Added Similarity and Polysemy analysis to the BERT example (#137, @nirnayroy)
- Support attention masks via the new `Kaun.Attention` module
- Support loading sharded Hugging Face safetensors
- Fix BERT and GPT‑2 model loading
- API simplification: removed type parameters from public types; `Ptree` now supports mixed‑dtype trees via packed tensors with typed getters.
- Checkpointing overhaul: versioned `Train_state` with schema tagging, explicit `Checkpoint.{Snapshot,Artifact,Manifest,Repository}` (retention, tags, metadata), and simple save/load helpers for snapshots and params.
- Overhaul dataset combinators: derive tensor specs from Rune dtype, fix sampling/window bugs, validate weighted sampling, and respect `drop_remainder`
- Make dataset `prefetch` truly asynchronous with background domains and allow reusing an external Domainslib pool via `parallel_map ~pool`
- Use `Dataset.iter` for epoch batches to reduce overhead
- Update BERT and GPT-2 tokenizer cache to use `Nx.Cache` for consistent cache directory resolution (#134, @Arsalaan-Alam)
- Honor text dataset encodings via incremental Uutf decoding (#122, @Satarupa22-SD).
- Preserve empty sequential modules when unflattening so indices stay aligned for checkpoint round-tripping
- Prevent `Training.fit`/`evaluate` from consuming entire datasets eagerly and fail fast when a dataset yields no batches, avoiding hangs and division-by-zero crashes
- Allow metric history to tolerate metrics that appear or disappear between epochs so dynamic metric sets no longer raise during training
- Make `Optimizer.clip_by_global_norm` robust to zero gradients and empty parameter trees to avoid NaNs during training
- Split CSV loader into `from_csv` and `from_csv_with_labels` to retain labels when requested (#114, @Satarupa22-SD)
- Implement AUC-ROC and AUC-PR in Kaun metrics and simplify their signatures (#124, #131, @Shocker444)
- Add mean absolute percentage error, explained variance, R² (with optional adjustment), KL-divergence, and top-k accuracy to Kaun metrics
- Add NDCG, MAP, and MRR ranking metrics to Kaun metrics
- Add BLEU, ROUGE, and METEOR metrics to Kaun for pre-tokenized sequences, removing tokenizer dependencies
- Add SSIM, IoU, and Dice metrics for vision workloads in Kaun

### Talon

- Remove automatic sentinel-based null detection for numeric columns; explicit masks (via [_opt] constructors) now define missing data semantics
- Replace join nested loops with hashed join indices, cutting lookup from O(n·m) to near O(n)
- Reuse a shared Nx-based column reindexer so filter/sample paths avoid repeated array copies
- Fix `fillna` to honor column null masks and replacements, restoring expected nullable semantics
- Preserve null masks when reindexing during joins so sentinel values remain valid data
- Handle numeric index columns in `pivot`, preventing distinct keys from collapsing into a single bucket
- Respect null masks when serializing numeric columns to JSON, emitting JSON `null` instead of sentinel values
- Detect big integers as int64 in Talon CSV loader (#121, @Arsalaan-Alam)
- Allow forcing column types in Talon JSON loader (#104, @nirnayroy)
- Add documentation to compare Talon and Pandas (#154, Satarupa22-SD)

### Saga

- Remove legacy `Normalizers.nmt` and `Normalizers.precompiled` constructors (and their JSON serializers) so the public surface only advertises supported normalizers
- Tighten template processor JSON parsing: require integer type ids, drop the legacy special-token list format, and ensure multi-id special tokens round-trip with the new record fields
- Make tokenizer JSON loading tolerant of HuggingFace quirks (missing `model.type`, string-encoded merges), restoring compatibility with upstream `tokenizer.json` files
- Cache byte-level encode/decode lookup tables to avoid rebuilding them during tokenization, trimming avoidable allocations
- Skip BPE dropout sampling when dropout is disabled, removing redundant RNG work on common hot paths
- Fix Unigram tokenization so longest matches are emitted without aborting the sequence when a vocab hit occurs
- Recompute pad token ids when the pad special string changes, preventing padding with stale ids
- Fix Unigram `token_to_id`/`id_to_token` vocabulary lookups (#117, @RidwanAdebosin)
- Optimize `Pre_tokenizers.whitespace` to reduce allocations and improve tokenization performance
- Simplify tokenizers interface

### Sowilo

- Add `resize` (nearest & bilinear) that works for 2D, batched, and NHWC tensors
- Update grayscale conversion and RGB/BGR channel swaps to run entirely on Rune ops, keeping batched inputs compatible with JIT backends
- Make `median_blur` compute the true median so salt-and-pepper noise is removed as expected
- Fix `erode`/`dilate` so custom structuring elements (e.g. cross vs. square) and batched tensors produce the correct morphology result

### Fehu

- Added snapshot-based save/load for DQN and REINFORCE agents (#127, @RidwanAdebosin, @tmattio)
- Added typed `Render` payloads with enforced `render_mode` selection in `Env.create`, auto human-mode rendering, and vectorized `Env.render` accessors so environments consistently expose frames for downstream tooling
- Introduced the `Fehu_visualize` library with ffmpeg/gif/W&B sinks, overlay combinators, rollout/evaluation recorders, and video wrappers for single and vectorized environments, providing a cohesive visualization stack for Fehu
- Added a `Fehu.Policy` helper module (random/deterministic/greedy) and sink `with_*` guards so visualization sinks handle directory creation and cleanup automatically
- Added `Buffer.Replay.sample_tensors` to streamline batched training loops and exploration handling
- Reworked `Fehu_algorithms.Dqn` around `init`/`step`/`train` primitives with functional state, warmup control, and snapshotting helpers
- Rebuilt `Fehu_algorithms.Reinforce` on the same `init`/`step`/`train` interface with optional baselines, tensor-based rollouts, snapshot save/load, and updated tests/examples/docs using the new workflow
- Upgraded the GridWorld environment to return ANSI and RGB-array frames using the new render types, and updated the DQN example to optionally record pre- and post-training rollouts via `FEHU_DQN_RECORD_DIR` using `Fehu_visualize` sinks
- Reworked space sampling to return `(value, next_rng)` and split keys internally, fixing correlated draws in Box/Multi-discrete/Tuple/Dict/Sequence/Text samplers while adding `Space.boundary_values` for deterministic compatibility checks
- Extended vectorized environments to reuse space boundary probes and now store structured `final_observation` payloads in `Info`, improving downstream consumption
- Added `Buffer.Replay.add_many` and `Buffer.Replay.sample_arrays`, preserved backing storage on `clear`, and exposed struct-of-arrays batches for vectorised learners
- Tightened `Env.create` diagnostics with contextual error messages and an optional `~validate_transition` hook for custom invariants
- Enriched `Wrapper` utilities with `map_info`, Box `clip_action`/`clip_observation`, and time-limit info reporting elapsed steps
- Upgraded `Info` values to carry int/float/bool arrays with stable JSON round-tripping (handling NaN/∞) and sorted metadata serialization for deterministic diffs
- Improved training helpers: Welford-based normalization with optional unbiased variance, documented `done = terminated || truncated`, and returned `nan` when explained variance is undefined
- Treat time-limit truncations as terminals when computing rollout advantages and expose the `truncated` flag in buffer steps
- Require callers of `Training.compute_gae` to pass final bootstrapping values and ensure `Training.evaluate` feeds the current observation to policies
- Allow `Space.Sequence.create` to omit `max_length`, keeping sequences unbounded above while preserving validation and sampling semantics
- Validate vectorized environments by round-tripping sample actions/observations across every instance, preventing incompatible spaces from slipping through
- Finish clipped value loss support in Fehu.Training (#119, @nirnayroy)

### Nx-datasets

- Migrate to `Nx.Cache` for cache directory resolution, enabling consistent behavior. (#133, @Arsalaan-Alam)
- Fix cache directory resolution to respect `RAVEN_CACHE_ROOT` (or fall back to `XDG_CACHE_HOME`/`HOME`), allowing custom cache locations. (#128, @Arsalaan-Alam)
- Switch CIFAR-10 loader to the binary archive so parsing succeeds again
- Add a CIFAR-10 example
- Standardize dataset examples on `Logs`
- Use `Logs` for dataset loader logging (#95, @Satarupa22-SD)

## [1.0.0~alpha1] - 2025-10-02

This release expands the Raven ecosystem with three new libraries (Talon, Saga, Fehu) and significant enhancements to existing ones. `alpha1` focuses on breadth—adding foundational capabilities across data processing, NLP, and reinforcement learning—while continuing to iterate on core infrastructure.

### New Libraries

#### Talon - DataFrame Processing
We've added Talon, a new DataFrame library inspired by pandas and polars:
- Columnar data structures that support mixed types (integers, floats, strings, etc.) within a single table (aka heterogeneous datasets)
- Operations: filter rows, group by columns, join tables, compute aggregates
- Load and save data in CSV and JSON formats
- Seamless conversion to/from Nx arrays for numerical operations

#### Saga - NLP & Text Processing
Saga is a new text processing library for building language models. It provides:
- Tokenizers: Byte-pair encoding (BPE), WordPiece subword tokenization, and character-level splitting
- Text generation: Control output with temperature scaling, top-k filtering, nucleus (top-p) sampling, and custom sampling strategies
- Language models: Train and generate text with statistical n-gram models (bigrams, trigrams, etc.)
- I/O: Read large text files line-by-line and batch-process corpora

#### Fehu - Reinforcement Learning
Fehu brings reinforcement learning to Raven, with an API inspired by Gymnasium and Stable-Baselines3:
- Standard RL environment interface (reset, step, render) with example environments like Random Walk and CartPole
- Environment wrappers to modify observations, rewards, or episode termination conditions
- Vectorized environments to collect experience from multiple parallel rollouts
- Training utilities: Generalized advantage estimation (GAE), trajectory collection and management
- RL algorithms: Policy gradient method (REINFORCE), deep Q-learning (DQN) with replay buffer
- Use Kaun neural networks as function approximators for policies and value functions

### Major Enhancements

#### Nx - Array Computing
We've significantly expanded Nx's following early user feedback from alpha0:
- Complete linear algebra suite: LAPACK-backed operations matching NumPy including singular value decomposition (SVD), QR factorization, Cholesky decomposition, eigenvalue/eigenvector computation, matrix inverse, and solving linear systems
- FFT operations: Fast Fourier transforms (FFT/IFFT) for frequency domain analysis and signal processing
- Advanced operations: Einstein summation notation (`einsum`) for complex tensor operations, extract/construct diagonal matrices (`diag`), cumulative sums and products along axes
- Extended dtypes: Machine learning-focused types including bfloat16 (brain floating point), complex16, and float8 for reduced-precision training
- Symbolic shapes: Internal infrastructure for symbolic shape inference to enable dynamic shapes in future releases (not yet exposed in public API)
- Lazy views: Array views only copy and reorder memory when stride patterns require it, avoiding unnecessary allocations

#### Rune - Autodiff & JIT
We've continued iterating on Rune's autodiff capabilities, and made progress on upcoming features:
- Forward-mode AD: Compute Jacobian-vector products (`jvp`) for forward-mode automatic differentiation, complementing existing reverse-mode
- JIT: Ongoing development of LLVM-based just-in-time compilation for Rune computations (currently in prototype stage)
- vmap: Experimental support for vectorized mapping to automatically batch operations (work-in-progress, not yet stable)
- LLVM backend: Added compilation backend with support for LLVM versions 19, 20, and 21
- Metal backend: Continued work on GPU acceleration for macOS using Metal compute shaders

#### Kaun - Deep Learning
We've expanded Kaun with high-level APIs for deep learning. These APIs are inspired by popular Python frameworks like TensorFlow, PyTorch, and Flax, and should feel familiar to users building models in Python:
- High-level training: Keras-style `fit()` function to train models with automatic batching, gradient computation, and parameter updates
- Training state: Encapsulated training state (TrainState) holding parameters, optimizer state, and step count; automatic history tracking of loss and metrics
- Checkpoints: Save and load model weights to disk for model persistence and transfer learning
- Metrics: Automatic metric computation during training including accuracy, precision, recall, F1 score, mean absolute error (MAE), and mean squared error (MSE)
- Data pipeline: Composable dataset operations (map, filter, batch, shuffle, cache) inspired by TensorFlow's `tf.data` for building input pipelines
- Model zoo: Reference implementations of classic and modern architectures (LeNet5 for basic CNNs, BERT for masked language modeling, GPT2 for autoregressive generation) including reusable transformer components
- Ecosystem integration: Load HuggingFace model architectures (`kaun.huggingface`), access common datasets like MNIST and CIFAR-10 (`kaun.datasets`), and use standardized model definitions (`kaun.models`)

### Contributors

Thanks to everyone who contributed to this release:

- @adamchol (Adam Cholewi) - Implemented the initial `associative_scan` native backend operation for cumulative operations
- @akshay-gulab (Akshay Gulabrao)
- @dhruvmakwana (Dhruv Makwana) - Implemented `einsum` for Einstein summation notation
- @gabyfle (Gabriel Santamaria) - Built PocketFFT bindings that replaced our custom FFT kernels
- @lukstafi (Lukasz Stafiniak) - Major contributions to Fehu and FunOCaml workshop on training Sokoban agents
- @nickbetteridge
- @sidkshatriya (Sidharth Kshatriya)

## [1.0.0~alpha0] - 2025-07-05

### Initial Alpha Release

We're excited to release the zeroth alpha of Raven, an OCaml machine learning ecosystem bringing modern scientific computing to OCaml.

### Added

#### Core Libraries

- **Nx** - N-dimensional array library with NumPy-like API
  - Multi-dimensional tensors with support for several data types.
  - Zero-copy operations: slicing, reshaping, broadcasting
  - Element-wise and linear algebra operations
  - Swappable backends: Native OCaml, C, Metal
  - I/O support for images (PNG, JPEG) and NumPy files (.npy, .npz)

- **Hugin** - Publication-quality plotting library
  - 2D plots: line, scatter, bar, histogram, step, error bars, fill-between
  - 3D plots: line3d, scatter3d
  - Image visualization: imshow, matshow
  - Contour plots with customizable levels
  - Text annotations and legends

- **Quill** - Interactive notebook environment
  - Markdown-based notebooks with live formatting
  - OCaml code execution with persistent session state
  - Integrated data visualization via Hugin
  - Web server mode for browser-based editing

#### ML/AI Components

- **Rune** - Automatic differentiation and JIT compilation framework
  - Reverse-mode automatic differentiation
  - Functional API for pure computations
  - Basic JIT infrastructure (in development)

- **Kaun** - Deep learning framework (experimental)
  - Flax-inspired functional API
  - Basic neural network components
  - Example implementations for XOR and MNIST

- **Sowilo** - Computer vision library
  - Image manipulation: flip, crop, color conversions
  - Filtering: gaussian_blur, median_blur
  - Morphological operations and edge detection

#### Supporting Libraries

- **Nx-datasets** - Common ML datasets (MNIST, Iris, California Housing)
- **Nx-text** - Text processing and tokenization utilities

### Known Issues

This is an alpha release with several limitations:
- Quill editor has UI bugs being addressed
- APIs may change significantly before stable release

### Contributors

Initial development by the Raven team. Special thanks to all early testers and contributors.

@axrwl
@gabyfle
@hesterjeng
@ghennequin
@blueavee

And to our early sponsors:

@daemonfire300
@gabyfle
@sabine

[1.0.0~alpha0]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha0
[1.0.0~alpha1]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha1
[1.0.0~alpha2]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha2
