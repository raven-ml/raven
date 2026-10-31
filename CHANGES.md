# Changelog

All notable changes to this project will be documented in this file.

- Only document user-facing changes (features, bug fixes, performance improvements, API changes, etc.)
- Add new entries at the top of the appropriate section (most recent first)

## [1.0.0~beta1] - Unreleased

### General

- **Breaking:** nx and the packages built on it require OCaml 5.5 or later;
  alpha3 required 5.2.
- `nx-oxcaml` moves to its own repository,
  <https://github.com/raven-ml/nx-oxcaml>, and raven no longer ships it. Raven
  now uses OCaml 5.5 syntax and the newest OxCaml compiler is based on OCaml
  5.4, so the backend cannot build in this tree. It builds against the `nx`
  1.0.0~alpha3 release until OxCaml supports OCaml 5.5.
- **Breaking:** a structure of arrays is an OCaml type with one `walk` over
  an `Nx.Ptree.Walk` cursor, described by `Nx.Ptree.instantiate (module M)`
  and combined with `Nx.Ptree.tensor`, `pair`, `list`, `option` and `nest`.
  `[@@deriving ptree]` (`ppx_ptree`) derives the walk.
  - It replaces `Kaun.Ptree.t`, a dynamic tree of tensors, lists and dicts,
    whose `map`, `map2`, `fold` and `flatten` are `Nx.Ptree`'s.
  - Rune's transformations take descriptions: `Rune.grad p f x`, `vjp p q`,
    `jvp p q`, and `jit` and `vmap` on a signature
    `Nx.Ptree.(p @-> returns q)`.
  - Kaun's layers export their walk, as `Kaun.Linear.walk`, and a decoder's
    caches are `Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))`.
  - `Kaun.Checkpoint.save` and `load` are `Nx_io.Archive.of_value p` and
    `to_value p ~like` with an `Nx_io` save or load. `load` cast entries and
    ignored extra keys; `to_value` raises `Failure` on both.
  - Vega's optimiser states are structures: `Vega.sgd_ptree p`,
    `adam_ptree p` and `lbfgs_ptree p`, and `Vega.Loss_scale.walk`.
- `fehu` and `sowilo` move to `contrib/`. Each is its own
  dune project with its own version, builds against `main`, and sits outside
  the 1.0 API commitment. Changes to `fehu` and `sowilo` are now
  recorded in `contrib/<package>/CHANGES.md`. The `raven` package no longer
  installs `fehu` and `sowilo`; install them by name.

### Rig (new)

- `Rig.Buffer.free` takes `?why`, the reason `Rig.Buffer.dead` then answers,
  so a library that frees memory it consumed for an operation keeps that
  operation's name in later refusals.
- Add `rig`, which runs work on GPUs from OCaml through Metal, CUDA, NVIDIA's
  and AMD's kernel drivers, or no kernel driver on Linux, and orders the work
  of every device by the memory it uses (`Rig.submit`, `Rig.Buffer.copy`).

### Norn

- Add `Diag.psis`, Pareto smoothed importance weights with each set's `k̂`,
  and `Diag.loo`, leave-one-out cross-validation of pointwise log likelihoods
  (`Norn_model.pointwise` over draws) with each point's `elpd_i` and `k̂`.
- Add `Summary.of_weighted`, the summary of `Weighted.t` draws; its
  `Pareto_k` finding, new in `Summary.finding`, judges the weights.
  `Summary.concat` raises on summaries whose columns differ.
- `Dist.quantile` computes the quantiles of `Dist.gamma` and
  `Dist.inverse_gamma`, which raised, with first derivatives in `p` and the
  parameters that stay finite wherever the exact ones are.
- Add `Dist.binomial ~trials ~logits`, the count of successes in `trials`
  trials of probability `sigmoid logits`, `trials` an `int32` tensor.
- Every type carries its values' device set: `('u, 'f, 'd) Nuts.state`,
  `('f, 'd) Bij.t`, `('x, 'f, 'd) Dist.t` and the like. Samplers make their
  state beside the position; `Dist.placement` is where a distribution makes
  its values, and a model lies where its run's values do, refusing a site or
  a factor on another set.
- `Nuts` runs chains cut across devices. `Hmc`, `Ensemble`, `Nested` and
  `Smc` raise on a position cut across devices, since they read across chains.
- `Gaussian.diagonal`, and `of_precision` without `~low_rank`, raise when the
  mean holds no tensor, as they have no set to make the variances on.
- `Summary.columns` are host values.
- **Breaking:** a density from `with_gradient` whose function reads its values
  no longer runs in any sampler. Samplers trace their transitions, and a rule
  reads values only where `Rune.custom_jvp` runs it on values.
- `Dist.support` of `Dist.transform (Bij.interval ~low ~high)` with
  per-element bounds holds every element's values. It mapped the base's
  support through the lowest `low` and the highest `high`.
- `Norn.Nuts`, `Norn.Hmc`, tempering's moves and nested sampling's slice
  moves work on one row per chain instead of per tensor of the position: a NUTS
  leapfrog step compiles to 210 kernels instead of 760 on eight schools, and
  compiles and runs faster. The momenta's random draws change.
- Add `Norn.Evidence.expectation`, a posterior expectation under an
  evidence's sample with its standard error: the shrinkage noise of nested
  sampling's weights and the sampling error of independent groups of draws,
  the lineages `Norn.Nested.state` now tracks or tempering's chains.
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

- `dune pkg lock` and `dune build` in a project that depends on raven's
  packages from source build them: `lz`, `rig` and the other packages
  failed with "Module Marks doesn't exist" in a hugin test library.
- **Breaking:** the tensors a figure keeps are host values: `num`, `cat`, the
  `?valid` masks, `image` and `Picture.image` take `Nx.host` tensors, which
  `resolve` and `draw` compute on with nx. A tensor on another set is given
  as `Nx.place Nx.Host.on x`. `Field2.v` reads a tensor of any set once.
- **Breaking:** `Stats.histogram` and `Scale.missing` compute where their data
  lies and give results on its set: `Stats.bins` is `'d Stats.bins`. A custom
  scale's transform, an OCaml function, reads its tensor on the host.
- Channels and images take `Nx.float4_e2m1fn` tensors, and `Number.decimals`
  that dtype, as every other float dtype.
- PDF streams are compressed by `lz`'s zlib encoder: `Hugin` PDFs have
  other, slightly smaller stream bytes and `/Length` values, and read the
  same.
- Raster PNGs and the PNGs embedded in SVGs, written through `Nx_io`, have
  other bytes and decode to the same pixels.
- `Hugin_vg_raster` inks where a stroke's subpaths overlap a dot, the round caps
  of a subpath shorter than a twentieth of a pixel. The dot wound the other way,
  so the overlap cancelled to a hole, as at a line's piece cut short by its
  domain's end.
- `Hugin_vg_raster` paints nothing of a clipped primitive whose box misses its
  clip's. It painted the product of their coverages in the pixels both reach, a
  faint line beside the clip, which stamps drew for some instances only.
- On arm64, `Path.crop` drops a closed subpath that the box cuts to points on
  one line, as elsewhere: ocamlopt fused the collinearity test's products into
  one multiply-add, whose rounding could keep a subpath of two points.
- hugin writes floats the same on every system: its printers, error messages
  and the numbers of its SVG and PDF output round a float's exact value once,
  ties to even. On Windows they wrote three exponent digits (`1e+016`) and
  rounded ties up (`135.5625` as `135.563`).
- `Nx.bit` values are not real, as `Nx.bool` ones are not: `Stats.histogram`
  refuses them.
- **Breaking:** hugin is redesigned: a figure (`Hugin.t`) is marks that bind
  channels (`num`, `cat`, `dim`) to roles over named `Scale.t`s, composed with
  `layer`, `grid` and facets. `imshow`, `hstack`, `Cmap` and decorations go.
- `hugin` no longer depends on `bytesrw`: the PDF writers deflate their
  streams with `Lz.Zlib.encode_string`.
- Hugin no longer depends on cairo or SDL2. Figures render through
  `hugin.vg` with the bundled Inter font, so `opam install hugin` needs no
  system libraries and a figure produces the same bytes on every machine.
  PDF output keeps text as text with the font embedded, and SVG output embeds
  the font too.
- Remove `Hugin.show`. There is no interactive window any more; view figures
  in Quill or save them to a file.

### Vega (new)

- `lbfgs_step` steps a complex leaf as its real coordinates where a part is a
  signed zero, which it could flip.
- `lbfgs_step` computes a float16 or bfloat16 leaf at float32 and rounds it
  once, when stored, as the other steps do. It rounded its recursion, rate and
  move to the leaf's dtype at every operation.
- **Breaking:** states and structures carry the parameters' device set, as
  `('p, 'd) adam_state` over an `('p, 'd) Nx.Ptree.t`. Each `*_init` makes
  the step counter beside the parameters and raises when they hold no array.
- **Breaking:** a step's `~lr` is a function of its state's counter, which the
  step applies: pass a `Vega.Schedule` or `Vega.lr v`, no longer
  `sched st.step`. `Vega.Schedule.t` takes the set, `'d Schedule.t`.
- **Breaking:** `Vega.Loss_scale.static` and `dynamic` take the placement of
  the scale, and `Loss_scale.ptree` is `Loss_scale.walk`.
- Optimizers step complex parameters: each part is a real coordinate with its
  own state, and a step descends along the conjugate of `Rune.grad`'s
  gradient. They passed complex leaves through unchanged.
- `Vega.clip_by_global_norm` returns a structure of no array unchanged.
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

### Rune

- The derivative budget of `Nx.gammaincinv` and `Nx.gammainccinv` in `p` is
  stated at the smaller tail `min (p, 1 - p)`.
- A compiled `Nx.eigh` of a matrix whose norm passes its dtype's range, such
  as `float32` `[[2.5e38, 1e38], [1e38, -2.5e38]]`, returns its eigenvalues
  and eigenvectors instead of NaN: the rotations run on the matrix scaled by
  a power of two, as a compiled `Nx.svd`'s do. An eigenvalue past the range
  is an infinity.
- **Breaking:** transformations take a structure (`Nx.Ptree.t`) for each value
  whose arrays they read, of mixed dtypes in one pass: `grad p f params`,
  `value_and_grad p f`, `jvp p q f params tangents` and `vjp p q f params`.
- **Breaking:** `value_and_grad_aux p x f` takes the auxiliary value's
  structure `x`. `grad'`, `value_and_grad'`, `jvp'` and `vjp'` take functions
  of one array, as alpha3's `grad` did.
- **Breaking:** `vjp` and `vjp'` return the result and its pullback, which
  maps the result's cotangents to the arguments' and may be applied many
  times, in place of taking the cotangent as an argument.
- **Breaking:** `custom_jvp p q rule` and `custom_vjp p q rule` take a `rule`
  returning the result with its tangent map or its pullback, closing over its
  residuals, in place of `~fwd`, `~jvp_rule` and `~bwd`.
- **Breaking:** reverse mode transposes a `custom_jvp`'s tangent map, and
  forward mode of a `custom_vjp` with a float result raises unless a reverse
  mode around it applies the pullback; both differentiated `fwd`.
- **Breaking:** `vmap s f` maps axis 0 of every array of its arguments, `s`
  being `f`'s signature (`Nx.Ptree.fn`, built with `@->` and `returns`).
  `Nx.item` of a lane raises.
- **Breaking:** `vmap`'s `?in_axes`, `?out_axes` and `?axis_size` go: a value
  every lane shares is captured, another axis is `Nx.moveaxis`, and
  `?axis_name` is `?axis` from `Rune.axis ()`.
- **Breaking:** a complex array is differentiated as its two real parts: the
  gradient of a real function of `z = x + iy` is `∂f/∂x - i ∂f/∂y`, against
  whose conjugate a step descends.
- **Breaking:** where the operands of `Nx.maximum` or `Nx.minimum` tie, each
  takes half of the derivative; the second took all of it. The `Rune`
  interface states each operation's derivative at its kinks.
- **Breaking:** `grads`, `value_and_grads`, `value_and_grads_aux`, `vjps`,
  `jvps`, `custom_vjps`, `custom_jvps` and `vmaps` are removed: use the
  structured functions with `Nx.Ptree.(list tensor)`.
- **Breaking:** `jvp_aux` is `jvp` with the auxiliary arrays in the result's
  structure. `no_grad` is removed: `detach` the values whose derivative must
  be zero.
- **Breaking:** `finite_diff`, `finite_diff_jacobian`, `check_gradient`,
  `check_gradients` and their types are removed: `check_grads p f params`
  compares a gradient with central differences.
- **Breaking:** `debug` is removed: `program` and `pp_program` print what a
  function computes.
- **Breaking:** `scan`, `scan'`, `iterate` and `iterate'` take the step as a
  positional argument, `scan c x y f ~init xs`, where they took `~f`.
- **Breaking:** a `scan` or `iterate` step is traced once, eagerly too: an
  OCaml `if` on `Nx.item` in it raises, so branch with `switch` or `Nx.where`,
  and a draw without `~key` raises, so draw from a key in the carry or rows.
- **Breaking:** a `custom_jvp` or `custom_vjp` rule and its derivative are
  traced at each call under `grad` of `grad`, a Hessian-vector product, a
  loop's step or a map capturing an argument, so `Nx.item` of one raises.
- **Breaking:** `lane_index a x` takes the map's name, which a map without
  `~axis` lacks, and an array whose devices hold the index. `lanes` and
  `lane_index` raise outside a map of that name, where they saw one lane.
- **Breaking:** `Nx.item` in a `root`'s `solve` of a value a transformation
  around tracks raises, since `solve` stages its operations; it read the plain
  value. Read it before the `root`, where `Nx.item` gives its primal.
- **Breaking:** a `scan` step returns a carry of the shapes it received; one
  whose shapes change raises at the trace.
- **Breaking:** under `vmap`, the gradient of an array an `iterate`'s step
  captures may be NaN where its lanes stop apart, where it was an exact zero:
  a held lane's zero cotangent meets its donor's infinite partial derivative.
- **Breaking:** `Total.add` of a value on another device set than its
  `collect`'s `init` raises, naming both sets; the addition was moved to the
  total's placement. Place the value with `Nx.place` first.
- **Breaking:** `Total.collect`'s `~zero` is `~init`.
- **Breaking:** `grad` of a complex scalar raises; it differentiated the real
  part. Take the real part with `Nx.real`, or use `vjp`.
- **Breaking:** `Jit_error` is removed: a function `jit` cannot trace raises
  `Invalid_argument`, as `Nx.item: the value is traced by Rune.jit`.
- **Breaking:** a `jit` program the compiler declines, such as one holding
  `lanes` or an operation it does not lower, is evaluated by nx. On a set
  without kernels, such as a GPU set, it raises
  `Invalid_argument "… has no kernels …"`, where one holding `lanes` compiled.

- Every operation nx computes has a derivative, `Nx.svd`, `Nx.eigh`,
  `Nx.eig`, `Nx.lu` and the special functions (`Nx.erfinv`, `Nx.lgamma`,
  `Nx.gammainc`, `Nx.betainc`, ...) included.
- rune differentiates the special functions in every argument and at every
  order, within budgets the `Rune` interface states.
- The complete factors of a tall `Nx.qr` or a non-square `Nx.svd`, and a
  block scale of `Nx.contract` that depends on the parameters, have no
  derivative: their tangents raise.
- Add `linearize`, a function's value and a tangent map that evaluates one
  forward pass's record without running `f` again, and `linear_transpose`,
  the transpose of a linear function.
- Add Jacobians `jacfwd'` and `jacrev'`.
- Add `custom_transpose`, a linear function with a given transpose, and
  `check_transpose`, which tests that transpose numerically.
- Add loops `scan c x y f ~init xs` and `iterate c ~max ~until f init` (and
  `scan'`, `iterate'`), branches `switch` and `cond` (and `cond'`), where only
  the taken branch computes.
- Add `root`, a solution differentiated through its residual by the implicit
  function theorem.
- A transformation of a loop or branch is a loop or branch of its function's
  transformation, and under `vmap` each lane stops or branches on its own.
- Add `remat s f`, whose reverse mode computes `f`'s intermediates again in
  the backward pass instead of keeping them.
- `scan`, `iterate`, `switch`, `root`, `remat` and `jit` trace their functions
  at the call, eagerly too: a loop's step runs once per trace, and `Nx.item`
  of a traced value or a draw without `~key` inside it raises.
- Add `jit s f` and `jit'`, which compile `f` through `tolk`, raven's
  compiler. A compiled function traces once per key (structure, dtypes,
  shapes, placements) and compiles once per memory layout.
- `jit`'s `?beam` searches kernels for speed and `?parallel` compiles
  candidates on several domains. Calls under `jvp`, `vmap` and reverse mode
  run programs derived from the traced one.
- A compiled call consumes the arguments donated to it (`Nx.donate`,
  `Nx.Ptree.donate`), and a result may take their memory. A failed `Nx.check`
  raises as the call returns.
- A compiled call's results match eager within the accuracy the `Rune`
  interface states; a program the compiler declines is evaluated by nx.
  `RUNE_JIT_DEBUG` reports why a call traced again.
- Transformations and compiled functions run over nx's device sets: a lane
  computes on the devices holding its row, and every tangent and cotangent
  lies at its array's placement, so a gradient lies where its parameter does
  and a backward pass runs wherever nx routed the forward pass. A tangent or
  cotangent given at another placement raises.
- Add map names `axis`, with `lanes`, every lane's value, and `lane_index`,
  the calling lane's index.
- Add `Total`, a sum code anywhere in a function adds to, and `Probe`, values
  code emits from inside a function, which the caller reads as it returns.
- Add `program`, `pp_program` and `eval_shape`, which trace a function without
  computing it and print its program or give its results' shapes.
- Rune depends on `tolk` and `rig`.

- `grad'` of `Nx.sum (Nx.cumprod x)` keeps the terms past a zero: at
  `[2; 3; 0]` it is `[4; 2; 6]`, where it was `[1; 2; 6]`.
- `Nx.cummax` and `Nx.cummin` pass each running extremum the derivative of
  the first element that reached it: `grad'` of `Nx.sum (Nx.cummax x)` at
  `[3; 1; 2]` is `[3; 0; 0]`, where it was `[1; 0; 0]`.
- `jvp` of `Nx.pow_s x 2.` at `0.` is `0`; it was NaN.
- Derivatives of `Nx.sign`, `Nx.qr`, `Nx.cholesky`, `Nx.solve` and `Nx.inv`
  are right on complex arrays. `Nx.sign`'s was zero.

### Jera (new)

- `System.solve`, `System.lanes` and `Minimize.solve` measure the steps of
  unknowns below about `1e-140` without underflow in their convergence test.
  Such a lane could stall at its zero, or converge farther from it than `tol`
  allows.
- `Root.newton`, `System.solve`, `System.lanes` and `Minimize.solve` end a lane
  whose step cannot move its estimate `Stalled` unless its error meets `tol`.
  It used to converge with its error past `tol` under a tolerance near an ulp
  or a slowly contracting search.
- Every function is polymorphic in its data's device set, which its types
  carry: `('a, 'd) Solution.t`, `('x, 'f, 'd) Minimize.t`, `('b, 'd) Grid.t`
  and the like. Constants are placed whole on each device the data lies on.
- `Sde.Brownian.v` takes a key of the path's set and `Brownian.walk`
  replaces `Brownian.ptree`; `Quad.qmc` takes such a key. A Rule's
  `Quad.Rule.nodes` are host values, and `Quad.Rule.nodes_at` places them.
- `Linear.dense` and `banded` bound the residual of a subnormal solution by
  `c n η (1 + ‖A‖)`, `η` the least subnormal, and `cg` and `gmres` by
  `rel ‖r‖ + n η`, so tiny well-posed systems no longer end `Stalled`.
- `System.lanes` and `System.solve` with `newton` converge a lane whose line
  search finds no decrease over a step at the floats' limit when the error
  bound of that step meets `tol`. A zero with an unknown at 0 ended `Stalled`:
  at `f`'s evaluation level a step moves that unknown by more than its ulps.
- `Minimize.solve` takes a box as `?lo` and `?hi`, an absent side no bound:
  gradient methods take Bertsekas's projected steps, `levenberg_marquardt`
  solves its damped system on the free coordinates and rejects a projected
  step its model predicts no decrease for, and `nelder_mead` clips its
  vertices; a coordinate held at a bound has the bound's derivative. A side
  unlike the start in structure or a float tensor's shape raises, naming the
  path.
- Add `System.lanes`, Newton's method over the leading axes of a tensor:
  each lane is a system of the last axis's unknowns with its own status, a
  `k × k` direct solve of the caller's Jacobian per step, and its own implicit
  derivative, so many small systems solve without `Rune.vmap`.
- Add `Ode.kvaerno5 ?mass ~linear derivative`, Kværnø's L-stable ESDIRK
  5(4) for stiff fields and, with a singular mass, index-1 DAEs. Each attempt
  prepares `M − γhJ` once with `~linear`; each stage is a simplified Newton
  solve, stated as a root in the answer. Its paths use the cubic Hermite
  extension, or with a mass the polynomial through its stage values.
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

- Add the `ppx_ptree` deriver: `[@@deriving ptree]` writes the `walk` of an
  `('a, 'd) t`, `'d t` or `t` (payload, then set), one of `Nx.Ptree.S`, and no
  description: use `Nx.Ptree.instantiate (module M)` or `Nx.Ptree.v M.walk`.
- A structure nested at another's payload, `(('a, 'd) Mlp.t, 'd) M.t`, is
  walked with `Nx.Ptree.Walk.under M.walk Mlp.walk`; a tensor, key or packed
  part lies on the set parameter.
- `[@ptree.int]` marks an integer a compiled program depends on,
  `[@ptree.skip]` data no program reads, and `[@ptree.walk f]` a part walked
  with `f`, after a field or a constructor's single argument.
- Add a runnable linear-regression example that trains a derived structure
  with `Rune.grad` under `Rune.jit`.

### Munin (new)

- `Munin_sys.Stat` samplers raise `Sys_error` where they cannot read the
  system, as network, disk I/O, per-core CPU, process table and load average
  do on Windows, instead of returning zeros or empty lists. `Munin_sys.start`
  and `munin watch` leave out the metrics a system lacks.
- `Store.list_runs` and `Store.find_run` take a run's status from its event
  log where `index.json` is behind it, as a process killed between the two
  writes of `Session.finish` or `Session.resume` leaves it: such a run read as
  running for good, or as finished and not resumable. A run directory without
  a manifest is no longer listed.
- Runs that several processes or domains start and finish in one store at
  the same time keep their status in `Store.list_runs`. A change to
  `index.json` holds a file lock and replaces the file whole. Where the file
  system refuses locks, a change removes `index.json`, and runs are read
  from their manifests.
- A change to a damaged or missing `index.json` rebuilds it from the runs on
  disk. It replaced the index with one holding only the changed run.
- `Store.list_runs` and `Store.find_run` take a store's runs from its run
  directories, and `index.json` only caches their headers. A run the index
  lacked, as a killed process or an index of another shape leaves, was
  hidden, and a run whose directory was removed was still listed.
- `Session.start` never gives two runs one directory. Runs started within
  one second drew ids that collided about once in 65,536 pairs, and the
  second wrote into the first's run.
- `Munin_sys.stop` wakes the monitor instead of waiting out the rest of the
  sampling interval, up to 15 s by default, at the end of every run.
- `Store.latest_run` and `Store.list_runs` order runs by `Run.started_at`.
  A run id stamps only the second, so runs started within one second came
  back in random order.
- `Munin_sys.start` samples every 15 s by default, down from 2 s, which
  logged about 16,000 events an hour. `Stat.Proc.Self.compute` gives the
  mean CPU percentage over any interval past 10 ms: it read 0 past 10 s, and
  800 at most without `~num_cores`.
- Artifact digests are BLAKE2b-256 from the standard library instead of
  SHA-256, so `munin` no longer depends on the `sha` package. Blobs live
  under `blobs/blake2b` in the store.
- The system statistics stubs build on Windows. CPU times come from
  `GetSystemTimes`, memory from `GlobalMemoryStatusEx`, process memory from
  `GetProcessMemoryInfo`, uptime from `GetTickCount64`, partitions and disk
  space from the fixed drives, page size from the Win32 calls.

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

- `Schedule.create_linear_with_vars` called from several domains at once
  schedules each body once, the others waiting for it, as `Codegen.to_program`
  compiles each program once.
- `Device.renderer` gives each device's one renderer, and a target naming another
  fails with `D has no renderer 'R'`, no longer followed by a suggested spelling.
  `Helpers.select_by_name` and `Helpers.select_first_inited` are removed.
- `Ops.pp_arg` prints the record arguments that `Ops.pp_param_arg`,
  `pp_kernel_info`, `pp_program_info`, `pp_bufferize_opts` and `pp_call_info`
  printed. `Dtype.sum_acc` and eight test-only `Bigint` functions are removed.
- **Breaking:** `Ops.hcq_info`, `Ops.hcq_kernel`, `Ops.pp_hcq_info`,
  `Ops.getaddr`, `Spec.hcq`, `call_info.aux` with the `?aux` of `Ops.call` and
  `Shape.call_with_outputs`, and the `Queue` and `Region` arguments are gone:
  rig encodes the queues they described. `Spec.full` no longer accepts a
  program over a buffer. A `Linear`'s name is a `String` argument.
- `Rangeify.get_kernel_graph` schedules two pairs of outputs that each share a
  computation and each read an output of the other pair, where it raised
  "graph_rewrite does not terminate" and rune evaluated the program uncompiled.
- A host sum of an unrolled product multiplies no vector for its first term
  alone (`Codegen.to_program`): rune's compiled 64 by 64 matmul replays in
  8.9 us where it took 12.4 us, and tolk lowers a matmul kernel 25% faster.
- `Compiler_cpu.clang` compiles without the straight-line vectorizer under a
  Clang of LLVM 18, which wrote invalid code for clamped gathers on AVX-512
  processors and crashed ("SmallVector unable to grow"), as Ubuntu 24.04's did.
- A compiled minimum of a minimum against one value, or maximum of a maximum,
  picks what eager evaluation picks at a tie or a NaN, where LLVM folded it to
  the inner one: `Nx.where (Nx.less m c) m c`, `m = Nx.minimum c x`, gave `-0.`
- On riscv64 a selection against a float zero keeps the zero's sign, as
  `Cstyle.clang` already did on arm64, and on x86_64 at float16 with any C
  library.
- `Helpers.Diskcache.clear` empties every table at one step: a `put` or `get`
  meanwhile comes wholly before or after it, where a put could outlive it. An
  older tolk sharing the cache reads later entries as writes in progress.
- `Rangeify.get_kernel_graph` stores a value whose copies, one per index it is
  read at and one per reduction trip that computes an element again, would add
  over 1024 nodes. A compiled `Nx.cumsum` of `Nx.log` of uniform draws computes
  each once, not up to 256 times; an `Nx.associative_scan` solve compiled in
  62 s, now 2.4 s.
- On Windows, `Helpers.Diskcache.put` replaces an entry that a `get` holds
  open, where it kept the old value. It renames its entry into place with
  POSIX semantics, which replace a file open shared for deletion.
- A maximum over a weak integer compiles. rune's Fold of a padded float window
  takes one where no term reaches an element. Its identity was a value no
  integer holds, and is the least value of its accumulator's width
  (`Ops.identity_element`).
- A reduction to which no lane adds, as an int16 fold that sums no term into
  some elements, gives its identity. Its program read the accumulator's
  registers without storing the identity there (`Codegen.to_program`).
- `Rangeify.get_kernel_graph` no longer runs two outputs as one kernel where
  one reads a storage before a write over it and the other after. The kernel
  read one state for both, a wrong value with no error, as for `v + a` and
  `v * v + a'`, where `a'` is `a` once assigned.
- `Decomp_op`'s division rules read a weak divisor in the dividend's type: on
  int8, `x // 128` skipped the sign correction for -128, and `1 // 128` gave 0
  for -1.
- A kernel that reads a buffer state and depends on a kernel that overwrites
  it reads a copy of the state made before the overwrite
  (`Schedule.create_schedule`). It ran after the overwrite and read the new
  values, as `b + (a * 3)` did where `b` read `a` before `a` was assigned.
- A compiled integer product or quotient by a power of two shifts only by less
  than its type's width, and folded constants wrap to it. On int32,
  `(y * 65536) * (y * 65536)` was a shift by 32, which C leaves undefined: its
  values varied by run where they are 0.
- A compiled integer maximum of products moves a factor out of the maximum only
  where no product wraps: the uint32 maximum of a running sum over the greatest
  value twice was one less than that value.
- A compiled sum over a selection with a conjunct free of its range, as rune's
  `Fold` of a padded float builds, is `+0` where that conjunct is false. The sum
  times the conjunct's cast gave `-0` at a negative sum and NaN at an infinity.
- The C-style renderers (`Cstyle`) declare a value read once when its
  expression would nest past 64 brackets, so a long chain of operations, such
  as rune's float64 `fmod`, compiles where Clang and Metal rejected it.
- `Compiler_cuda.nvrtc` compiles in processes of its own, which serve compiles
  in turn, so a source that crashes NVRTC 13.3 or 13.4's assembler raises
  `Compile_error` naming the signal, which a search skips, where it ended the
  program. NVRTC's own cache of binaries is disabled there: its key ignores
  `--use_fast_math`, `-I` and other options, so it could serve a binary built
  under others.
- `Tolk_engine.link` links a copy between a GPU and the host whose offset
  into storage made at link depends on the run's values, or whose offset into
  an input moves with a range's trips, which raised. Such a copy is a move of
  the program (`Rig_program.Move`), which takes offsets each step computes.
- A kernel reducing over a variable size computes the reduction's identity
  where the size is 0. A sum read its accumulator unwritten; over a value the
  range does not change, a sum of an infinity was NaN and a maximum the value.
- `Codegen.made` is the program `to_program` has made of a kernel in this
  process, if any, without waiting; `Realize.lower_and_compile` compiles only
  the kernels without one.
- The disk cache keeps a `Codegen.to_program` program up to its source and
  takes its binary from the compiler's table, so each binary is stored once.
- A `Tolk_engine.link` that raises ends the holds it took on bound buffers
  (`Rig.Claim.release`), so a donation right after it reuses their memory.
- `Tolk_engine.link` holds the memory of every bound buffer a kernel or
  copy accesses (`Rig.Claim.hold`) while the schedule is reachable, so no
  call consumes it between runs. A bound buffer that died between two runs
  made the next one raise.
- `Tolk_engine.run`, `time` and `link` raise `Invalid_argument` for a buffer
  a kernel accesses that does not start at its parameter's phase, such as
  `Rig.Buffer.view b ~first:4` for a phase-0 parameter, whose vector accesses
  faulted on x86. `Tolk_engine.slots`, storage made at link and
  `Memory.memory_plan_rewrite` start buffers at their phase.
- `Tolk_engine.time` sums only the work of the run it times, which it finds
  by the points the run's steps report: a kernel of the same name that
  another domain ran meanwhile added to the time, so a search under load
  could pick a slower kernel. A submission of several kernels now counts,
  where it counted nothing, and one of copies alone still counts nothing.
- `Transcendental.payne_hanek_reduction` takes the words of `1/(2pi)` by the
  bits of their index, in 111 selections from 345 in float64: a scalar float64
  sine or cosine compiles in 0.05 s, from 0.08 s. A host kernel of float64
  sines over an array now takes 8 lanes: it runs 4.5 times as fast and
  compiles longer, 2.2 s from 1.4 s for 8 nested sines over 1024 values.
- `Helpers.Diskcache` holds at most `Setting.cachesize` bytes (`CACHESIZE`,
  5 GiB by default, `0` for no bound), removing the entries used least
  recently. It grew without end: each change of tolk's sources leaves its
  programs and schedules unread.
- On x86, kernels with a float16 negation, absolute value or copysign of
  more than 16 lanes (`Nx.neg`, `Nx.abs`, a minimum) compile with Clang 18 for
  processors with AVX-512, where Clang crashed. The Clang renderer marks x86
  kernels `__attribute__((min_vector_width(512)))`.
- `Simplify.pm_simplify_ranges` merges two ranges in every reduction that
  closes them or in none, at the kernel's sink. It merged them in one of the
  reductions that share their loops, which the linearizer then refused.
- On Windows, a CUDA or AMD compile whose library does not load names the
  library's file and Windows' reason, as elsewhere. It said only
  `LoadLibrary failed`.
- `Helpers.system` runs its program directly, without a shell. On Windows
  cmd.exe ran its built-in `echo` in place of the program and printed the
  quotes around each argument.
- On Windows, tolk reads an empty environment variable as empty, where it
  read it as unset: an empty `DEBUG` was refused elsewhere and took the
  default there.
- `Ops.exec_alu` computes `Op.Sin` correctly rounded on every system, by its
  own exact argument reduction. It called the C library's sine, whose
  mingw-w64 version folded `sin 1e300` to a value of the wrong sign.
- `Dtype.pp_const` prints a float's shortest round-tripping digits on every
  system. It used the C library's conversions, which printed
  `-9.564275832644383e-25` for `-0x1.28p-80` on Windows.
- On Windows, the Clang renderer no longer marks kernels and calls through
  function pointers `__attribute__((ms_abi))`: rig calls host code by
  System V on every system, so a called function pointer got the wrong
  convention.
- On Windows, `Helpers.Diskcache.get` reads an entry while another domain or
  process renames one into place, where it raised `Sys_error "...: Permission
  denied"`. The cache's entries and comgr's worker program are read shared for
  deletion.
- On Windows, compiles on several domains at once no longer raise
  `Sys_error "...: Permission denied"` removing Clang's or comgr's temporary
  files. A process another domain started meanwhile inherits them, so they are
  opened shared for deletion.
- `Shape.symbolic` keeps a selection that guards a float-to-integer
  conversion or an integer divisor. Removing it let a compiled vector
  kernel on x86 read any value in lanes it discarded.
- On x86 hosts without AVX512-BF16, compiled bfloat16 kernels that select
  between vectors, merge a gated load or bitcast stacked lanes compile and
  link: the Clang renderer holds bfloat16 as `unsigned short` there. They
  aborted clang or failed to link `__truncsfbf2`.
- **Breaking:** `Tolk_engine` runs compiled schedules as programs of rig
  (`Rig_program`), over rig's devices and buffers, and tolk depends on rig
  instead of nx. The queue encoders (`Ops_metal`, `Ops_cuda`, `Ops_amd`,
  `Ops_nv`) and `Hcq2`'s batches and host programs are gone;
  `Tolk_engine.time` sums the kernels' spans in rig's profile on every
  device. AMD counters and thread traces are not collected meanwhile.
- Compiled kernels that pad or concatenate an integer quotient no longer die
  with SIGFPE on x86, as `Norn.Nuts`'s compiled transition did: the padding
  divided by the zeros it loads, and now divides by 1.
- Beam search accepts positional scalar arguments, fixing `PARAM has no name`
  when `Rune.jit ~beam` searches a loop that reads its index. `Search` links
  complete calls through `Tolk_engine.link_call`, which replaces `link_program`.
- Reductions over one domain that read none of each other share their loop
  and compute what their sources share once. A compiled gradient in a
  parameter that many operations broadcast sums one reduction per broadcast,
  and each recomputed the whole graph: over a 104 x 104 grid such a gradient
  took 650 ms compiled against 130 ms eagerly, and takes 31 ms.
- A schedule kept on disk now serves a later process that builds the same
  function: its key no longer holds the slots a process counter gives a staged
  loop's carry or renamed call-local storage. A body that hits the schedule
  cache under other range numbers keeps its own loops.
- On the host, a compiled product of MXFP4 codes in a checkpoint's layout
  (`Nx_quant.mxfp4`) decodes them as vectors, 16 at a time, where it decoded
  each code alone and branched on it on x86: its time no longer depends on the
  codes, and on an M1 it runs 1.7-3.6 times as fast, as fast as GGUF's blocks.
- `Tolk_engine.time` always times a schedule on its own profile and its
  devices' stamps, so a profile taken around a search no longer changes what
  it measures or the kernels a beam search picks. `Tolk_engine.clock` no
  longer depends on whether a profile is taken.
- The host runs a run of its calls, with the ranges and back edges around
  them, as one host program that walks them as data, calling each kernel
  through `Nx_device.Program.entry`: no OCaml runs per trip or per kernel, and
  the program does not grow with its calls or their trips.
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
  simplification and moves to the new module `Shape`, with shapes and
  simplification, which call each other: `Shape.shape`, the movements
  (`Shape.reshape`, `Shape.mop`, ...), `Shape.simplify`, `Shape.resolve`,
  `Shape.Sint`, divisibility and `Shape.sym_infer`, then what calls take and
  the calls: storage (`Shape.param`, `Shape.alloc`, ...), sharding, the
  bindings of variables, `Shape.call_with_outputs` and
  `Shape.program_info_of_sink`.
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
  unset. The search's `BEAM_*` variables and `HCQ_NUM_SDMA` are declared, and
  `NO_MEMORY_PLANNER` reaches results, so the caches key on them too.
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

### Lz (new)

- Add `lz`, compression between bigarrays by raven's own C kernels, held to
  the reference implementations' single-threaded speed: deflate, zlib and
  gzip (levels 1-9), Zstandard (levels 1-12), LZ4 blocks and frames, Snappy,
  and `Lz.Crc32`. `decode` writes into a destination the caller sizes and
  `encode` into one `bound` sizes; `Lz.Decoder.step` and `Lz.Encoder.step`
  stream, lending each run out of the codec's own memory. Errors are
  `Lz.error` values with a kind and the input offset. `lz.bytesrw` adds
  bytesrw filters. It replaces the unreleased `compress` package.

### Nx

- `Nx.Rng.ptree` describes a key at every set, for signatures such as
  `Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)`. A binding of
  `Nx.Ptree.v Nx.Rng.walk` had its set fixed by its first use.
- Add `Nx_io.encode_txt`, the text `save_txt` writes, as a string, so rows can
  be appended to a file opened with `Out_channel.open_gen [ Open_append; … ]`.
- `Nx.attention` states its error bound for each element of `o` and for `lse`.
- A compiled `Nx_wide.sum` of 10^6 float64 numbers makes its first call in
  1.8 s, from 2.2 s: it stores its tree's partial sums every five levels, from
  three. The association changes with them, so a sum may differ in its last
  bits, within the same bound.
- `gammaincinv` and `gammainccinv` state their error bound at the smaller
  tail `min (p, 1 - p)`, the precision they keep as `p` nears `1`.
- `#require "nx.top"` no longer loads `compiler-libs.toplevel`, which the
  toplevel already holds: loading it printed an `Invalid_argument`
  exception on OCaml 5.5. When it cannot install `Nx.pp`, `nx.top` raises
  `Failure` naming the printer, where it printed an error and loaded.
- Other packages' C compiles as nx.cpu's does by including
  `%{lib:nx.cpu:vector_flags.sexp}` and `%{lib:nx.cpu:v3_flags.sexp}`.
- **Breaking:** a value is `('v, 's, 'd) Nx.t` on the device set `'d` that
  `Nx.devices` mints over `Rig.t` devices, and `Nx.place` moves it. A value
  made from numbers is on `Nx.host`; the `_at` forms make it at a placement.
- **Breaking:** aliases take the set: `Nx.float32_t` is `'d Nx.float32_t`.
- **Breaking:** nothing changes a value. `?out`, `blit`, `set_item`,
  `set_slice`, `put`, `index_put`, `put_along_axis`, `.%{}<-` and `.${}<-`
  go: `Nx.set` and `Nx.scatter ?combine ~axis p u x` return the result.
- **Breaking:** `Nx.set idx x v`, which wrote `v` into `x`, is
  `Nx.set idx v x`, value before target, over an `index` list (`I i` for a
  position). `fill v x` is `full_like x v`.
- **Breaking:** the libraries change: `nx.core` and `nx.buffer` are
  `nx.array` (`Nx_array`, dtypes as `Nx.Dtype`), `nx.backend` is `nx.kernel`
  (`Nx_kernel.S`), and `nx.c` is `nx.cpu` (`Nx_cpu`).
- **Breaking:** `nx.effect`, `nx.zip`, `nx.pocketfft`, `nx.io.stb_image` and
  `nx.io.stb_image_write` are removed. nx depends on `rig` and `lz`,
  and needs neither zlib, pkg-config nor logs.
- **Breaking:** on x86-64, `nx.cpu` requires x86-64-v3 (AVX2, FMA, F16C and
  BMI2), and raises `Failure` when it initialises on a host without it.
- **Breaking:** dtype constructors are `Nx.Dtype`'s, no longer re-exported
  by `Nx`: `Nx.Dtype.Float32`, `Bfloat16`, `Uint8`, `Float8_e4m3fn`. The new
  `Float4_e2m1fn` and `Bit` break exhaustive matches.
- **Breaking:** element types are `int8_unsigned_elt`, `int8_signed_elt`,
  `int16_signed_elt`, `int16_unsigned_elt` and `float8_e4m3fn_elt`, and the
  dtype `float8_e4m3` is `float8_e4m3fn`.
- **Breaking:** positions are `int64`: `take` and `take_along_axis` take
  `'d Nx.int64_t`, and `argmax`, `argmin`, `sort`, `argsort`, `argwhere`,
  `nonzero` (now a list), `matrix_rank` and `lstsq`'s rank return it.
- **Breaking:** a position held in data outside its axis reads zero and
  writes nothing; `take`'s `?mode` goes.
- **Breaking:** `extract ~condition x` is `extract c x`, and `compress` goes
  for `extract` or `take`. `concatenate`'s `~axis` and `unsqueeze`'s `~axes`
  are required.
- **Breaking:** `Nx.Infix`'s `~-` is `neg`, where it was `logical_not`, and
  `&&` and `||` take booleans only. `@@` is `*@`.
- **Breaking:** the operators `<.>`, `/@`, `**@`, `@=`, `@||`, `%`, `mod`,
  `%$` and `^` go: use `dot`, `solve`, `matrix_power`, `concatenate`,
  `mod_`, `mod_s` and `logical_xor`.
- **Breaking:** draws are `Nx.Rng`'s, from `~key` or the scope's key:
  `Nx.rand`, `randn`, `randint`, `bernoulli`, `categorical`, `permutation`
  and `truncated_normal` are `Nx.Rng.uniform`, `normal` and the same names.
- **Breaking:** `Rng.randint ?low ~high shape` draws `int32` without a dtype,
  `Rng.bernoulli p` takes its probabilities as an array, and `shuffle x` is
  `take ~axis:0 (Rng.permutation (dim 0 x)) x`.
- **Breaking:** a key is `'d Nx.Rng.key`, two 32-bit Threefry-2x32 words,
  where it was an `int`. `Rng.run ~seed f` is `Rng.with_key (Rng.key seed) f`
  and `Rng.to_int` is `Rng.to_tensor`. A seed draws other numbers.
- **Breaking:** `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh`,
  `slogdet`, `lstsq`, `cond`, `rfft`, `irfft`, `fftfreq` and `rfftfreq` take
  their real or complex result's dtype first, where some returned `float64`.
- **Breaking:** `norm` is `vector_norm` and `matrix_norm`.
- **Breaking:** `lstsq ?rcond` is `?rtol`, `matrix_rank` keeps `?rtol` alone,
  dropping `?tol` and `?hermitian`, and `cond ?p` drops `Fro` and the
  negative norms.
- **Breaking:** options are variants: `cholesky ?upper`, `eigh ?uplo` and
  `eigvalsh ?uplo` are `?triangle:(Lower | Upper)`, `qr ?mode` and
  `svd ?full_matrices` are `?factors:(Reduced | Complete)`, and `fft_norm`
  is a variant.
- **Breaking:** `fft ?axis ?n` is `fft ?axes ?s`, and `fft2`, `fftn`,
  `ifft2`, `ifftn`, `rfft2`, `rfftn`, `irfft2` and `irfftn` are `fft`,
  `ifft`, `rfft` and `irfft` over `~axes`.
- **Breaking:** `einsum "ij,jk->ik" [| a; b |]` is
  `einsum (Nx.Pattern.v "i j, j k -> i k | j") [ a; b ]`, naming summed
  axes after `|`.
- **Breaking:** `multi_dot` takes a list, and `meshgrid` a list with a
  required `~indexing:(`Xy | `Ij)`.
- **Breaking:** `erf`, `sigmoid`, `softmax`, `log_softmax`, `var`, `std` and
  `lerp` take float values only.
- **Breaking:** `of_bigarray` copies its bigarray, which it shared;
  `Nx.Repr.of_array Nx.Host.v (Nx_array.of_bigarray dt b)` reads it in place.
  `to_bigarray` takes the kind, and `map_item` returns a host value.
- **Breaking:** `Nx_io` reads named arrays as `Nx_io.Archive.t`, one array as
  `Nx.packed`: `packed`, `archive`, `to_typed`, `packed_dtype` and
  `packed_shape` are `Nx.packed`, `Nx.unpack` and `Archive.find` or `tensor`.
- **Breaking:** `Nx_io` reads and writes PNG and JPEG images only:
  `save_image` is `save_png` and `save_jpeg`, for `[| h; w; c |]` images, and
  `load_image` keeps the file's channels, without `?grayscale`.
- **Breaking:** saves drop `?overwrite` and replace their path atomically.
  `load_npz_entry` goes. `load_txt` takes the dtype before the path and drops
  `?max_rows`; `save_txt` takes `?sep` alone. In place of `~append:true`,
  write `encode_txt`'s text to a file opened with `Open_append`.
- **Breaking:** removed, with their replacements:
  - `dims`, `size`: `shape`, `numel`. `itemsize`:
    `Nx.Dtype.bits (Nx.dtype x)`, in bits. `data`, `strides`, `stride`,
    `offset`, `is_c_contiguous`, `contiguous`, `to_buffer` and `of_buffer`:
    `Nx.Repr` over `Nx_array.t`, and `copy`.
  - `empty`, `empty_like`, `identity`: `zeros`, `zeros_like`, `eye`.
  - `expand`, `expand_dims`, `unsqueeze_axis`, `squeeze_axis`, `ravel`,
    `unflatten`, `shrink`, `broadcasted`, `astype`: `broadcast_to`,
    `unsqueeze`, `squeeze`, `flatten`, `reshape`, `slice`,
    `broadcast_arrays`, `cast`.
  - `vstack`, `hstack`, `dstack`, `array_split`: `concatenate`, `stack`,
    `split`, `split_at`.
  - `cmplt`, `cmpne`, `cmpeq`, `cmpgt`, `cmple`, `cmpge`, `clip`, `invert`:
    `less`, `not_equal`, `equal`, `greater`, `less_equal`, `greater_equal`,
    `clamp`, `bitwise_not`.
  - `radd_s`, `rmul_s`, `rmaximum_s`, `rminimum_s`: `add_s`, `mul_s`,
    `maximum_s`, `minimum_s`. `rmod_s s x`: `mod_ (scalar_like x s) x`.
    `lerp_scalar_weight`: `lerp` of a `scalar_like`.
  - `relu`: `Kaun.Fn.relu`, or `maximum_s x 0.`. `standardize`: `mean`,
    `var` and `rsqrt`. `convolve`: `correlate` with a `flip`ped kernel.
    `maximum_filter`, `minimum_filter`, `uniform_filter`: `sliding_window`
    with `max`, `min` or `mean`. `tensorsolve`, `tensorinv`: `reshape` with
    `solve` or `inv`.
  - `map`, `iter`, `fold`, `iter_item`, `fold_item`: `to_array`.
  - `pp_data`, `print_data`, `data_to_string`, `format_to_string`,
    `print_with_formatter`, `dtype_to_string`, `shape_to_string`: `pp`,
    `print`, `to_string`, `pp_dtype`, `pp_shape`, `Nx.Dtype.name`.

- An operation computes where its operands lie and returns once its work is
  submitted; operands that do not lie together raise, naming the `Nx.place`
  to write. `Nx.sync`, a read and a `place` onto `Nx.Host` wait.
- Without `?kernels`, a set of GPU devices computes only under `Rune.jit`.
- Placements cut values across devices: `Nx.Placement.on`, `device`,
  `split ~axis` and `mesh` over an `Nx.Mesh`. An operation moves bytes
  between devices only for a fold over a cut axis: its partial results and
  the parts of the result they close into.
- `Nx.donate x` gives `x` up to the one operation that reads it, which may
  write its result into `x`'s memory.
- Add `Nx.Ptree`, structures of arrays described by their `walk`: map, fold,
  place, cast and flatten a model's parameters or an optimiser's state.
- Add `Nx.check`, which raises the caller's exception where a condition on
  data fails, and `Nx.associative_scan`, which scans a structure.
- Add the `float4_e2m1fn` and `bit` dtypes, and `Nx.bitcast` between widths.
- Add the `T` and `D` indices, positions held in data.
- Add `Nx.contract` and `Nx.rearrange` over `Nx.Pattern`, `Nx.conv` and
  `conv_transpose`, and `Nx.attention` over dense or paged keys and values.
- Add special functions: `erfc`, `erfinv`, `ndtr`, `log_ndtr`, `ndtri`,
  `lgamma`, `digamma`, `lbeta`, `gammainc` and `gammaincc` with their
  logarithms and inverses, `betainc` and `betaincc` with their logarithms.
- Add `i0e`, `i1e`, `expm1`, `log1p`, `fma`, `softplus`, `rms_norm`,
  `logcumsumexp` and `ewma`.
- Add complex parts: `real`, `imag`, `magnitude`, `angle` and `complex`.
- Add `lu` and `solve_triangular`; `dct`, `idct`, `dst`, `idst`, `stft`,
  `istft` and `hann`.
- Add `top_k`, `lexsort`, `searchsorted`, `unique`, `quantile`, `histogram`,
  `positions`, `count` and `order_key`; and `reduce_segments`,
  `reduce_ranges`, `map_segments` and `sliding_window`.
- Add `Nx.Rng` samplers `exponential`, `gumbel`, `gamma`, `beta`,
  `von_mises`, `poisson`, `binomial` and `dirichlet`; `split_batch`,
  `fold_in_tensor`, `Rng.walk` and `Rng.refusing`.
- `Nx_io` maps the files it loads where the system can, and adds
  `load_gguf` (`Nx_io.Gguf`), `encode_png`, `Archive.float`, and
  `Archive.of_value` and `to_value`, which save and read back a structure.
- `Nx_io` loads a PNG whose image data has bytes after its zlib stream, which
  the PNG specification says to ignore, and its decode errors give the failing
  byte's offset and the format's message.
- Add `nx.quant`: weights in GGUF's Q8_0, Q4_K and Q6_K block formats and in
  MXFP4, with `Nx_quant.dequant`, `apply`, `apply_experts` and `take`.
- Add `nx.ragged` (`Nx_ragged`), rows of varying lengths as int64 offsets
  over an array of values, and `nx.wide` (`Nx_wide`), double-word numbers
  of 106 significand bits at float64.
- An integer `Nx.mean` is the exact mean truncated toward zero, so it never
  wraps. It divided a sum that wraps in the dtype.
- An error names the function the user called and the operands as
  `dtype shape on set`, as in
  `Nx.add: shapes [2; 3] and [4] do not broadcast`.

### Kaun

- **Breaking:** layers carry their arrays' device set, as `('a, 'd) Linear.t`.
  Drawn layers' `init` and `make` take `~key` and place the parameters where
  the key lies; `Init.t` draws from a key.
- **Breaking:** `Layer_norm`, `Rms_norm` and `Batch_norm.init`,
  `Attention.Cache.make` and `Attention.causal_mask` take the placement of
  the arrays they make.
- **Breaking:** `Rope.t` and `Cache_index.t` take their set, are built on the
  host, and move whole onto a model's devices with `Rope.place` and
  `Cache_index.place`. `Cache_index.ptree` is `Cache_index.walk`.
- **Breaking:** `Fn.gelu_approx`, `silu` and `softplus` take float tensors
  only, as the `Nx.exp` they compute with does.
- **Breaking:** `Kaun_datasets` loaders and `Metric.confusion_matrix` return
  host values; `Data`'s shuffle permutes where the data lies, and
  `Dropout.apply`'s `?key` is a key at the input's set.
- `Kaun_datasets` decompresses in bounded memory and removes a partial file
  when decompression fails.
- `Kaun_datasets` creates its cache directory under an absolute Windows path
  such as `C:\Users\…`; loaders raised `Unix_error (EINVAL, "mkdir", ".\C:")`.
- `Kaun_datasets.cifar10` raises `Failure` naming the file when a batch does
  not hold 10,000 entries. A longer batch wrote past its arrays, and a shorter
  one returned uninitialised bytes.
- `Kaun_hf.download_file` writes a download through the descriptor that created
  its temporary file. curl reopened it by name, so a link put there meanwhile
  redirected the write, and a umask without the owner's write bit failed it.
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

- Tokenizer files with a `Precompiled` normalizer (T5, ALBERT, XLNet and other
  SentencePiece models) load: `Normalizer.precompiled` reads the SentencePiece
  charsmap and normalizes, with offsets, as Hugging Face's tokenizers does,
  cutting grapheme clusters as UAX #29 does at Unicode 17.0.0.
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

- Queries run under `Rune.grad`, and under `Rune.jit` and `Rune.vmap` where
  every size a step computes follows from its inputs' sizes; `Query.run` and
  `Talon.column` raised there. Steps such as `filter` raise at a trace.
- **Breaking:** `Column.of_tensor`, `to_tensor`, `validity`, `ragged`,
  `of_ragged`, `layout`, `Talon.take`, `Talon.to_tensor`, `Kind.tensor`,
  `Type.tensor` and `Expr.fn` use `Nx.host` values, as columns are host data.
- `Expr.nx` and `nx2` run each map nx performs in one pass, so fused functions
  such as `Nx.fma`, `Nx.hypot` and `Nx.acosh` lift. A refusal names the
  function: `performs Nx.sum`.
- `Talon_parquet` reads compressed pages with `lz`. A page that fails to
  decompress reports the kind of error (malformed, truncated, output too
  long, unsupported) and the byte of the page's compressed data where it
  happened; a page that decodes short of its header's size says how many
  bytes it produced.
- On Windows, `Column.parse`, `Expr.Str.parse` and `talon.csv` read a float as
  the nearest value of its type, as elsewhere. msvcrt's strtod misread some
  float64 texts near a power of two, and texts close to halfway between two
  `float32` or `float16` values rounded the wrong way. A float text of more
  than 15 digits or a far exponent no longer goes through strtod, and reads
  in about 0.6 of the time.
- On Windows, a table shows its floats with two exponent digits and rounded
  once, ties to even (`1.00000e+16`, not `1.00000e+016`), as on other systems.
  `Expr.pp` writes a float literal in the text `Column.print` writes, with a
  point when it has neither point nor exponent: `1e20` is
  `100000000000000000000.`, which was `1e+20`.
- **Breaking:** `Talon_parquet.sniff` and `source` take a `Rig.Buffer.t`; open
  a file with `Rig_disk.of_file`.
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

- **Breaking:** A notebook's cells are its top-level code blocks marked to run,
  as in ```` ```ocaml {.run} ````. Every other block, plain ```` ```ocaml ````
  included, is text and never runs. A notebook from older Quill fails to load
  until migrated with
  `` sed -E -i.bak 's/^( {0,3})(`{3,}|~{3,})[[:blank:]]*ocaml[[:blank:]]*$/\1\2ocaml {.run}/' notebook.md ``.
- **Breaking:** `Quill_markdown.of_string` returns a `result`, an error where it
  raised for a notebook that stores figures as files. It and
  `Quill_markdown.load` fail on a block that asks to run in another shape, such
  as ```` ```{.ocaml .run} ````, or inside a list or quote, naming its line.
- **Breaking:** `Quill.Kernel.t` is abstract and runs one language. Kernels are
  built with `Kernel.make ~language`, and run cells with `Kernel.execute`, which
  gives a cell in another language an `Error` output and a failed `Finished`.
- **Breaking:** `Quill.Cell.code` takes a required `~language`, checked by the
  new `Quill.Cell.is_language`, `Quill.Cell.is_ocaml` is removed, and
  `Quill.Session.set_cell_kind` takes `` `Code language ``.
- Saving a notebook that stores figures as files keeps a figure its text still
  names, where it deleted the file under an output left in text.
- `quill build` frames a code block in text like a code cell, with a copy
  button. The terminal and web notebooks highlight only OCaml cells.
- **Breaking:** `Quill_book.Build.build`, `Quill_book.Build.build_file` and
  `Quill_tui.run` return `(unit, string) result`, an error for a notebook that
  cannot be loaded.
- `quill serve` and the terminal notebook refuse to save a notebook changed on
  disk since they loaded or saved it, where they wrote over the change. Every
  save replaces the file whole, so a crash leaves the old file or the new one.
- Add `Quill_markdown.load` and `Quill_markdown.save`, which read and write a
  notebook with the image files it stores beside it.
- The web notebook starts sections at headings of the rendered text, never at
  a `#` line in a code block, and no longer splits a text cell whose edits it
  lost.
- A `<!-- quill:cell -->` comment at the start of text stays in the notebook,
  where loading took it as the text cell's id and saving dropped it. Text cell
  ids saved before March 2026 now show as literal lines in the terminal UI.
- Saving a code cell with a line of three or more backticks fences it with more,
  where the line cut the cell on the next load.
- In the terminal notebook, Ctrl+C interrupts the running cell without
  quitting quill, Shift+Enter runs a cell in terminals that report it through
  the kitty keyboard protocol, and F1 opens help while editing, where `?` is
  now typed into the cell.
- A cell that raises `Failure` or `Invalid_argument` shows its message as
  written, as in `Exception: Failure: msg`, where the toplevel printed it as
  a string literal with escaped newlines and quotes.
- The server refuses a port another socket listens at on Windows, where a
  second server could bind it.
- The server's 30 s bound on a connection's reads and writes holds on Windows,
  where it was 30 ms.
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
- Add `--figures` to `quill run --inplace`, which stores images as files in
  `figures/<stem>/` beside the notebook. The notebook's first line records it,
  so later runs, `quill clean`, `quill serve` and the terminal notebook keep it.
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
