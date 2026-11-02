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

- AMD GPUs copy a box of two rows or more in one SDMA sub-window packet
  per 2^14 rows (2^16 on GFX12) where it took one packet per row: on an
  R9700, `Rig.Buffer.submit_copy` of 65,536 rows of 4 bytes takes 153 µs
  instead of 7.5 ms.
- `Rig.Buffer.free` takes `?why`, the reason `Rig.Buffer.dead` then answers,
  so a library that frees memory it consumed for an operation keeps that
  operation's name in later refusals.
- Add `rig`, which runs work on GPUs from OCaml through Metal, CUDA, NVIDIA's
  and AMD's kernel drivers, or no kernel driver on Linux, and orders the work
  of every device by the memory it uses (`Rig.submit`, `Rig.Buffer.copy`).

### Norn

- Norn's messages and `Support.pp` write each float as the shortest decimal
  that reads back; `Summary.pp`, `Summary.pp_finding` and `Evidence.pp` keep
  their digits, rounded once, ties to even. The text is the same on every
  system, where Windows wrote `2e+006`, and 1.125 to two decimals as `1.13`.
- Eager `Nuts.step` allocates half the words and runs half the operations per
  leapfrog step: a leaf merges only where it completes a subtree. Compiled
  steps and builds measured faster, times pending. Draws are unchanged.
- `Dist.quantile` computes the beta's quantile, by `Nx.betaincinv`, where it
  raised, and `Dist.truncated` takes the beta.
- `Evidence.expectation`'s nested-sampling error, which counted the volumes'
  noise twice, comes from runs resampled by lineage, and each error widens by
  Student's t over the new `Evidence.effective_groups`.
- Add `Dist.von_mises`, `Dist.lkj_cholesky`, `Dist.truncated` and `Dist.hmm`,
  a hidden Markov model's observations with its states summed out.
- `Dist.membership` replaces `Dist.mixture_membership` and gives an HMM's
  per-step state posteriors as well as a mixture's components'.
- `Dist.quantile` of `Dist.cauchy`, `half_cauchy`, `half_normal` and `laplace`
  keeps its digits in the tails: cauchy's at `p = 1e-12` was off by 1.4e-5.
- Add `Support.Less` and `Bij.less`, the reals below a bound. `Norn_model`
  names such a support `(-inf, b)`, where it read as the reals.
- `Dist.gamma`, `Dist.inverse_gamma`, `Dist.beta`, `Dist.dirichlet` and
  `Dist.student_t` compute their log densities in Loader's form, within a
  stated bound: a float32 gamma(1e7, 1e7) read 0 at 1 for 7.14.
- `Dist.dirichlet` reads a point within rounding of the simplex as `x / Σ x`.
- `Summary.v`, `Summary.of_weighted`, `Diag.mcse_mean` and `Diag.loo`'s `se`
  compute spreads over the draws' scale: draws 1e-200 apart no longer get an
  `sd` of 0 and draws 1e200 apart an infinite one, the error scales with the
  draws and `ess` no longer depends on their scale.
- Add `Gaussian.pathfinder` (Pathfinder), `Gaussian.vi` (ADVI over a factor
  covariance of any rank) and `Gaussian.importance`, whose weighted draws'
  `Diag.psis` `k̂` says whether a fitted Gaussian holds.
- The log masses of `Dist.poisson`, `Dist.binomial` and `Dist.neg_binomial`
  use Loader's saddle-point form and state their bound: at float32 and
  counts near 10^6 the old sums of log-gammas were off by up to 0.8 nats.
  float16 and bfloat16 compute them at float32. They stay exact at logits
  whose sigmoid underflows, and their scores finite at means down to the
  least subnormal.
- `Dist.quantile` of `Dist.laplace` has the derivative `2 scale` in `p` at
  the median, where it was `0.`.
- The gradient of `Dist.binomial` and `Dist.bernoulli` in a logit of
  exactly `0.` is the score, where it was `0.`.
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

- A malformed CFF charstring's message writes its number with two exponent
  digits under Windows too, as `9e+08`.
- `Font.of_string` reads fonts with CFF and CFF2 outlines, running their
  Type 2 charstrings with bounded stacks, nesting and work so that a hostile
  font is rejected, and a variable CFF2 font as its default instance.
  `Font.outlines` tells the formats apart; `Font.subset` writes a CFF font's
  subset with a CID-keyed CFF table, which PDF embeds as a `CIDFontType0`
  OpenType program and SVG as `font/otf`.
- `legend ~ncol` sets a legend's entries in rows of that many columns, and
  `` `Inside `` takes the middle of each edge and `` `Center `` besides the
  corners (`Hugin.inside`), centring the legend along the edges it does not
  meet.
- `time r counts` lifts integer counts of a `Time.resolution` as instants,
  read by a temporal scale: axes tick at calendar boundaries and label them in
  the calendar, zooms take instants, the `text` role writes RFC 3339, and `m4`
  reduces lines over time. `Hugin.Time` is the kit's `Time`.
- `rect ~stack` and `area ~stack` stack bars and areas, and `Mark.bind ~stack`
  stacks the length of any mark: rows that share the other position and panel
  are laid end to end, positive and negative lengths apart, with the offsets
  of `Stack.offset`, and the scale fits the stacked intervals.
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

- Vega's argument errors write each float as the shortest decimal that reads
  back, the same text on every system: `got -1000000` where they wrote
  `got -1e+06`, or `-1e+006` under Windows.
- Steps store float16 and bfloat16 leaves with stochastic rounding keyed by
  the state's counter and the leaf's path, so an update below half a spacing
  moves the value in expectation instead of being lost. Thread the state.
- **Breaking:** `adam_step`, `adamw_step`, `radam_step`, `lamb_step`,
  `rmsprop_step`, `adagrad_step`, `adan_step` and `adafactor_step` raise on a
  float16 leaf, whose average of squared gradients underflows to zero.
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

- `Rune.Probe.collect` raises `Out_of_memory`, `Stack_overflow` and
  `Sys.Break` again where it returned them as an `Error`, which a caller could
  drop.
- `Rune.jit` on a GPU whose compiler's library is missing (NVRTC, comgr or
  MTLCompiler) evaluates each call uncompiled, as for any program tolk
  declines, and says why under `RUNE_JIT_DEBUG`; its first compile raised
  `Compile_error` naming `NVRTC_PATH` or `COMGR_PATH`.
- A `Rune.jit` call allocates 63 fewer words on a hit and 66 fewer where
  its program checks: it builds no closures for checks it does not make or
  for reading its arguments, no table of argument pieces or list of leaves
  for a program over one device, and no list of the lanes it compares with
  each traced program's.
- The gradients of `Nx.gammaincinv` and `Nx.gammainccinv` run through their
  last step alone, as the beta's inverses' do, where they ran through every
  Halley step: the compiled gradient is half the program and compiles in
  half the time. Their derivatives past the second are no longer the
  quantile's, and are 0 where the quantile is subnormal.
- `Rune.jit`'s first call lowers its program once where it donates arguments
  (`Nx.donate`) or passes strided views, such as transposed weights. It
  lowered it a second time, at that call's layouts and donation.
- `Rune.jit` multiplies MXFP4 weights by `bfloat16` activations on tensor
  cores: `Nx_quant.apply` and `Nx_quant.apply_experts` enter each scaled value
  as the `bfloat16` it is.
- `Rune.jit` converts `float16` and `bfloat16` by tolk's casts, so a product of
  these dtypes, or of operands cast from them as kaun's attention scores are,
  takes the tensor cores of CUDA, NV and AMD GPUs.
- `Rune.jit` compiles a program over several devices that places a cut
  argument whole on each device, as fully sharded training's forward and
  backward do, where it evaluated it: each call holds within RFC 0005's bound.
  The kernels such a program makes no longer stay the same over 2 devices as
  over more; they do not grow with the number of devices.
- A compiled program over several devices assembles a value from its pieces
  in storage of its own, one store per piece, a copied piece copied into its
  region, so each piece may go once it is stored; it read every piece in one
  kernel, all held beside the result. A device's window of a value gathered
  whole on it reads only the pieces within the window: the cotangent of a
  weight cut across devices and placed whole on each no longer copies every
  device's sums to every other.
- `Rune.jit` over several devices compiles a function that reads a table
  computed from constants, which it evaluated: a sum or product over a
  sub-byte `Nx.bitcast` (`Nx_quant.apply` of Q4_K and Q6_K weights by a
  vector) and an `Nx.exp2` or `Nx.sin` of an arange, as rotary embeddings use.
- A compiled real contraction (`Nx.matmul`, `Nx.dot`, `Nx.einsum`,
  `Nx.contract`) promises `Nx.contract`'s bound, `γ(K + 1, 2u)`, which a
  matrix unit that truncates its sums meets. rune stated `γ(n + 1 + s)` at
  `u`.
- `Rune.scan` over rows `[t; 1; f]` split along their batch axis runs
  eagerly, and `Rune.jit` compiles it: eager computation raised at `Nx.get`
  of a row, and a compiled call evaluated.
- `Rune.jit` over cut values compiles as many kernels whatever the number of
  devices where a loop reads its window's coordinates (`Rune.lane_index`,
  `Nx.tril`, `Nx.eye_at` at a cut placement, `Nx.argmax` along an axis no
  device cuts): each device made kernels of its own. A fold, scan, gather or
  scatter over a cut axis passes each tile's start so too (`Nx.argmax` and
  `Nx.take_along_axis` along it, the positions of `Nx.cummax`'s derivative),
  so each device's own partial kernels are one program. `Nx.argmax` along a
  cut axis and `Nx.scatter` over one, of `Add` or into a value cut along it,
  compile programs that do not grow with the devices; a sum, a scan or a
  gather over a cut axis still closes its partials by programs that do.
- Under `Rune.vmap` over values cut along the mapped axis, `Rune.lane_index`'s
  indices lie where the lanes' rows lie, cut as they are; they were whole on
  every device.
- `check_grads` and `check_transpose` write their values as the shortest
  decimal that reads back, the same text on every system: `-1000000` where
  they wrote `-1e+06`, or `-1e+006` under Windows.
- `Rune.jit` compiles `Rune.scan` and `Rune.iterate` over values on several
  devices of one set: one loop whose trips run each device's kernels and
  moves of the step, with eager's placements and bits. Such programs were
  evaluated, trip by trip.
- `Rune.jit` compiles programs over devices of several targets, such as a GPU
  beside a memory device: each device runs kernels compiled for its own
  target. Such programs were evaluated, and on a set without kernels raised.
- A compiled call over several devices given a donated argument lends each
  device's piece of it to the result's piece there, so an update of state cut
  across devices writes in place; such calls were evaluated. A program over
  several devices holds no memory between calls: what it computes on each
  device is made per call and returned once that work is done.
- A compiled program over the memory devices of one set compiles kernels
  that do not grow with the number of devices, where the cut is even: each
  piece is stored into a view of its run of storage, wherever the run lies. A
  data-parallel step over 16 devices made 73 programs and makes 13.
- `Rune.jit` compiles programs over several devices of one set, values cut
  across them, whole on each or whole on one: each device computes its
  windows in kernels of its own, compiled once per target, with eager's
  placements. Such programs were evaluated. `switch`, `remat`, custom rules,
  checks and totals over several devices are still evaluated, as are
  programs that gather a cut argument, as fully sharded training gathers its
  weights.
- A loop's step that runs a construct over none of the step's values, such as
  a scan over a capture, executes its `Rune.callback`s and offers its
  `Rune.Probe` emissions every trip: the construct ran once, so a callback ran
  for one trip, and `Rune.jit` offered one trip's emissions.
- A compiled call of nested loops no longer raises or evaluates its program
  depending on how much the process compiled before it.
- A compiled loop whose step reads the result of a loop before it runs that
  loop once: it ran again in every trip, so a chain of loops took time
  exponential in its length, and the gradient of an `Ode.kvaerno5` solve in
  jera never finished.
- An evaluated `Rune.jit` call, as one over values on several devices,
  gives each donated argument, and in a program of operations alone each
  value it makes, to the operation that reads it last as a donation, so an
  optimiser update writes its state in place; a result over a donated
  argument's memory is no longer copied. A fully sharded step holds its
  share of the state, its activations and about two layers.
- An evaluated program finds the values it drops with fewer words: the eager
  gradient of a scan whose step holds a root allocates 3.2% fewer.
- Reverse mode records each operation with fewer words: kaun's MLP train
  step with Adam allocates 27,157 words where it allocated 28,073.
- A `custom_jvp` tangent map run on values, under `grad` or `jvp` with no map
  or traced function between, adds to a `Total` as one sum from `-0.`, as
  `rune.mli` states and a traced rule does: its additions reached the total one
  by one, which can round otherwise.
- A `root`'s `linear_solve` run by a held lane, in an `iterate` whose lanes
  stop apart or a `switch` whose lanes take different branches, treats it as
  held: `lanes` of the map raises, its checks pass and a callback skips it.
  It read every lane's value, raised on a check of the held lane's zero
  cotangent and ran the callback for the held lane.
- An evaluated program, such as `Rune.jit` over values on several devices,
  drops each value after its last use, as eager code does: it held every value
  until the call returned, 3.5 times the memory of a sharded optimiser update.
- A compiled loop's step that writes rows into its carry with `Nx.scatter`,
  or `Nx.set` by positions, stores those rows alone, and so does a compiled
  scatter into a donated argument whose memory its result takes, where a
  scatter of up to 16 updates along its axis rewrote every element of the
  target. It holds for `Set`, `Max`, `Min` and an `Add` other than a float
  narrower than 32 bits, at dtypes of 8 bits or more, into the carry itself
  (`rune.mli`'s Loops and rules state when). `Rune.iterate`'s docs give the
  pattern for a loop that writes a row per step.
- `Rune.lane_count a` is `Some n` inside the map named `a` of `n` lanes and
  `None` outside it, where `lanes` raises. A `jit` keys its programs on each
  answer its trace got about the maps around it, `None` included, so a
  compiled function whose trace asked about a map, traced outside it and
  called inside it, traces again.
- **Breaking:** a `jit` call's checks (`Nx.check`, a sampler's parameters, a
  `switch` index, an `iterate` past `max`) are one check of the call. Off the
  host the call compiles, returns without waiting for its device, and its
  failure raises at the next wait on its set, as an eager `Nx.check`'s does; an
  eager `switch` or `iterate` still raises at the call.
- Derivative rules allocate fewer words per operation: a reduction's
  derivative keeps its fused program, and a broadcast's and a contraction's
  transposes skip unused work. `Rune.value_and_grad` of a three-layer MLP
  allocates 13,760 words where it allocated 16,024.
- Derivatives taken on several systhreads of one domain no longer race on
  the domain's tables of derived plans: a thread that took over while another
  grew a table could miss plans and derive them again, or loop in a bucket
  being relinked.
- A function traced by `scan`, `remat`, `custom_jvp`, `custom_vjp` or
  `custom_transpose` that calls `Rune.lanes` or `Rune.lane_index` of a `vmap`
  no longer raises when a `vmap`, `grad` or `jvp` around that map holds its
  arguments. A traced `custom_jvp`'s result holds the map's lanes only where it
  is computed from a value the map holds or from `Rune.lane_index`; a traced
  `custom_vjp`'s or `custom_transpose`'s results always hold them.
- A compiled `Nx_quant.apply_experts` contracts each row with its own
  expert's matrix alone, where it computed every expert's product for every
  row and kept one: 32 experts cost 32 times the work. Its gradient in `x`
  compiles alike.
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
  and a draw without `~key` raises. A step no longer draws from a key of its
  own taken from the scope around the loop, which shifted every draw after
  the loop by one key: it draws from a key in the carry or rows, or in a
  scope it opens at one with `Nx.Rng.with_key`.
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

- `Minimize.solve` under a tolerance finer than the floats resolve ends a lane
  `Stalled` near the minimum instead of `Budget_spent`: Newton and Newton–CG
  stepped between floats for up to 31 000 evaluations. Newton's search now
  asks the gradient's norm to fall where `f` cannot show a decrease.
- `Minimize.bfgs` and `lbfgs` bound their error by the step plus the
  gradient's bound whatever their readings: BFGS stalled at an answer
  `1e-101` from a zero component under `Tol.v ~rel:1e-10 ~abs:1e-14`, its
  readings above 1.
- `Minimize.solve` counts an error of at least `ε κ |x|∞` per component, `κ`
  the condition its newest moves measured: BFGS and `levenberg_marquardt`
  converged under `Tol.ulps` with a component 4e-17 from its zero at 0,
  where the gradient rounded to 0. Under a tolerance finer than that, lanes
  now end `Stalled` even where they land on the minimum, coupled or not: a
  relative tolerance `r` fails at every component below `ε κ |x|∞ / r`, and
  an absolute term of `ε κ |x|∞` avoids it.
- `Tol.pp` and Jera's messages write each float as the shortest decimal that
  reads back, the same text on every system: `abs 1000000` where they wrote
  `abs 1e+06`, or `1e+006` under Windows.
- `Root.newton`, `System` and `Minimize.solve` converge where `|δ| / (1 − q)`
  meets `tol` at the answer `x + δ`, `q` in `tol`'s scaled norm: Broyden ended
  38× past `tol`; a map whose rate rises near its zero 245×, now at most 99×.
- `Minimize.bfgs` and `lbfgs` also bound their error by the gradient over the
  least curvature their steps measured (L-BFGS converged 264× past `tol`); they
  stall more lanes under tolerances near the floats' resolution.
- These solves resolve a step below the least normal float to the subnormals'
  spacing: one that underflowed to 0 counted as a zero, so a lane converged
  under `Tol.ulps` at a subnormal fifty spacings from its zero.
- `Root.newton`'s, `System.newton`'s and `System.lanes`' docs no longer promise
  that a wrong derivative cannot end a solve early: one wrong by orders of
  magnitude at the answer can.
- `Ode.path` takes again only its accepted steps, whatever its `budget`: it
  took a step of zero length in each slot past them. Compiled, its search
  writes each attempt's records alone, where it wrote them through a mask
  over every slot.
- `Minimize.solve` and `System.solve`'s methods and `System.lanes` add the
  matrix products that steer them, `Jᵀ J`, `Jᵀ r`, `H ∇f`, `B s` and their
  like, in a fixed pairwise order, so a compiled solve rounds them as an eager
  one: a compiled Levenberg–Marquardt fit differed from eager from its first
  step. The factorisations they call, `Nx.solve`, `Nx.qr` and `Nx.inv`, still
  round as the compiled program allows; `Tol`'s doc names the solves whose
  compiled decisions match eager.
- `System.lanes` allows for the subnormals' absolute rounding when it checks
  a lane's linear solve. A lane whose zero has an unknown below the normal
  floats failed the check and ended `Stalled`.
- `System.newton`, `System.broyden` and `Minimize.levenberg_marquardt` measure
  their merit, Jacobian update and cost on the residual or step divided by its
  largest magnitude. Below about `1e-154` the squares underflowed: Newton
  accepted every full step, Broyden never updated its Jacobian and the
  Levenberg–Marquardt ratio read `0 / 0`.
- `Sde.march` queries the Brownian path once per step instead of twice,
  carrying its value and integral from each step's end to the next step.
- `Sde.milstein` gives no NaN on a step whose span rounds to zero, as 4 steps
  over `[1, 1 + 2^-52]` make.
- `Sde.Brownian.v` raises on ends that round together in its dtype, as `1` and
  `1 + 1e-9` in float32, or whose span overflows it. Such a path made every
  query NaN.
- Each interval's last step changes in `Ode.march`, `Split.march` and
  `Sde.march`, so their results change in their last bits. Step `j` starts at
  `at.(i) + j h`, held at `at.(i+1)` where that rounds past it; every step but
  the last has size `h`, and the last runs from its start to `at.(i+1)`. It
  used to end a rounding error past `at.(i+1)`, so `Sde.march` to the path's
  end could raise, as over `[0, 0.3]` in 10 steps.
- `Ode.march`, `Split.march` and `Sde.march` accept monotone times a tiny gap
  apart. Their check multiplied each gap by the first, which underflowed to
  zero, and raised, for gaps of `1e-162` in float64.
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

Local experiment tracking for Raven: runs, their metrics, readings of the
machine and devices, files and versioned artifacts, kept as plain JSON and
text files that `Munin.Store_format` specifies, so `jq`, `tail -f` and
shell scripts read a store as the library does. No server.

A run is one append-only log written by its one `Session`, under a lock
the kernel releases when the writer dies, so a run reads running, crashed
or, from another machine, silent. `Session.metric` declares a metric and
`Session.log` logs it; `Session.sample` samples readings on a thread that
`Session.finish` stops; `Session.resume` names the step it restarts from,
and the samples after it become orphans that readers leave out.
`Store.runs` lists runs from their first and last records, and `History`
folds a log and follows it at the cost of its new bytes, with statistics
over finite samples, so a NaN or an infinity is counted and never chosen
as best. An ended run keeps its statistics in a checked hint, so a listing
shows best values without folding long logs. Tags and notes change during
and after a run. Artifacts are numbered versions of content-addressed
blobs with aliases, never renumbered. Provenance records the command, the
code with its uncommitted diff, the toolchain, the machine and the seeds.

`munin.sys` gives readings of the machine and of this process, and a
device-reading constructor that fixes a device's keys; munin reads no
device itself and depends on no rig. The `munin` command lists, shows,
compares, tags, annotates, deletes, collects and checks, printing
tab-separated lines.

`munin.tui` is a terminal dashboard of a store, opened by bare `munin` on a
terminal and by `munin watch [RUN]`: the store's runs under a header per
experiment, with the params that differ and each run's best value, and one
run on one dominant chart drawn from every sample, its other metrics as
strips aligned under it, up to two marked runs as level lines with rows of
their difference from the focus, and its machine's readings. A run's state
word comes from its lock and from samples the dashboard saw arrive.

### Tolk (new)

- `Compiler_cpu.clang` compiles for x86_64 without AVX10.1 and AVX-512's half
  arithmetic under a Clang of LLVM 18, which crashed on float16 casts for such
  processors, as Ubuntu 24.04's did with `native`. `Compiler_cpu.flags` states
  the arguments.
- `Rangeify.get_kernel_graph` counts no copy for outputs that read a value at
  one index through a quotient or remainder of their range, as rows in blocks
  or merged with a product's columns do: a decoded weight stays in its product.
  On an RTX 5000 Ada, gpt-oss-20b's 512-token prefill takes 0.45 s from 1.19,
  an MXFP4 product of 64 bfloat16 rows past L2 1.5 ms from 6.0; of float32
  rows 1.3 times slower, and on the host 2 to 5 times slower.
- A kernel's stores, calls and code in the target's language
  (`Op.Set.effectful`) run as often, and in the order, they are written: two
  loops merge (`Simplify.pm_simplify_ranges`) only where each runs as often
  as before, a loop shrinks to its guards only where every effect is guarded
  and never past its own end, a sum holding one is not collapsed, and
  `Coalesce.memory_coalescing` no longer merges stores whose values wait on
  each other.
- An int64 shift right by 32 or more, and an int64 remainder of the least
  int32 by -1, are no longer computed at int32, where C gives them no value.
- A store whose value reads another buffer after that buffer's own store no
  longer stores its value in a kernel of its own; a kernel of a buffer and a
  variable is no copy between devices.
- A gather of a comparison of two elements, `b[i] + 1 = a[i]`, compiles:
  `Simplify.load_collapse` no longer rewrites it without end.
- On the host, `Heuristic.hand_coded_optimizations` counts a kernel's lanes in
  the target's vector registers (`Renderer.registers`, read from Clang by
  `Compiler_cpu.registers`): a reduce kernel's stored axis takes a 64-byte
  vector first, and its accumulators then take up to half the registers;
  `Coalesce.memory_coalescing` merges accesses up to one register, doubles
  included. A float32 matmul on fudge runs 2.3 times as fast, a conv 1.7.
  `Heuristic.held_registers` states the count.
- A sum gated on an index read at its own range, as `b.(r) = r`, compiles and
  adds each trip it selects; `Simplify.load_collapse` raised
  `Invalid_argument`, failing rune's blocked contractions.
- On a GPU, `Heuristic.hand_coded_optimizations` streams a contraction's
  operand of least reuse: lanes along its unit-stride range, its reuse held in
  registers, before the tensor cores up to a reuse of 32. A Q8_0 token's
  product on an RTX 5000 Ada takes 27 us from 408, Q6_K 0.57 ms from 3.4.
- Making a compiler finds its library: `Compiler_cuda.nvrtc`,
  `Compiler_amd.hip` and `Compiler_metal.compiler`, and so `Cstyle.cuda`,
  `Cstyle.hip`, `Cstyle.metal` and `Device.renderer`, raise `Failure` naming
  `NVRTC_PATH`, `COMGR_PATH` or `MTLCOMPILER_PATH` on a machine without
  NVRTC, comgr or MTLCompiler.
- `Search.beam_search` times each candidate as its compilation ends, on every
  device, while other domains compile the rest (`Worker.iter`, which hands
  each result to its consumer in the order the applications end). A
  candidate's program is kept nowhere (`Codegen.linearize`, then
  `Codegen.compile`), and its binary is kept in its compiler's disk cache,
  which `CACHESIZE` bounds.
- An operation on an emulated narrow float rounds its result only at the
  store, and a cast of a float64 or a 32- or 64-bit integer to a narrow float
  without native arithmetic goes through a float32 rounded to nearest. On a
  target without the type, such a cast still rounds once.
- Compiling a long program allocates about 30% less: float64
  `Nx.betainccinv`'s gradient, one kernel of 42k instructions, 424 M words
  against 608 M. `Renderer.Estimates.of_uops`, `Linearizer.linearize`,
  `Codegen.split_blocks`, `Ops.key` and `Graph.to_string` did more work per
  node than their outputs need; the outputs are unchanged.
- `Rangeify.get_kernel_graph` stores a value whose copies past the first, one
  per index it is read at and one per reduction trip that computes an element
  again, would compute more than 360 nodes, and counts once the trips of
  a sum that selects one, as a routed product's rows select their expert. A
  compiled `Nx.cumsum` of `Nx.log` of uniform draws computes each once, not up
  to 256 times; an `Nx.associative_scan` solve compiled in 62 s, now 2.4 s; a
  compiled `Nx.sin` read by two kernels is computed once; rows of few per
  expert decode their expert's weights in place.
- `Memory.memory_plan_rewrite` plans a buffer that a call reaches through a
  view, that call one of its uses, where it left the buffer memory of its own
  for the whole schedule: a store into a run of a buffer stores through a view
  of it. `Schedule.create_linear_with_vars` holds the buffers its call views.
- A store into a contiguous run of a buffer stores into a view of the run:
  stores of one shape at other offsets of one buffer are one program.
- `Schedule.create_schedule ~by_storage` and
  `Schedule.create_linear_with_vars ~by_storage` run ready kernels in the
  order the program reaches them, and first one that frees storage without
  making any, then one that frees what it makes, for a schedule whose storage
  is made at each run: a value is read soon after it is made, so a run holds
  less at once. Other schedules keep their order.
- A copy stored into a run of consecutive elements of storage on its device
  copies into the run, where it copied into storage of its own and a kernel
  stored that: the two are no longer live at once.
- `Memory.memory_plan_rewrite` frees a copy's buffer from the call after its
  last, as a kernel's: rig runs a later write of its bytes after the copy, so
  a schedule of copies needs less memory.
- `Codegen.to_program` compiles a tensor-core product that one step of the
  core sums plus a scalar, such as a 16 by 16 float16 matmul plus its initial
  value on CUDA, where it raised `Invalid_argument` in lowering.
- A value of no device that a schedule stores, such as an arange, lives on the
  device of each kernel that reads it, stored once per device; it lived on the
  sink's first device, and kernels on the others refused to read it.
- `Compiler_cpu.clang` compiles a source of more than 400 000 bytes without
  Clang's loop vectorizer: float64 `Nx.betainccinv`'s gradient takes Clang
  5.7 s, from 19.7 s, its loop unpacked. Cached objects compile again once.
- A CPU or Metal kernel loads its arguments from one byte parameter, eight
  bytes each, a buffer's address read as an `Op.Refparam`
  (`Renderer.t.packed_args`, `Codegen.pack_args`).
- **Breaking:** `Ops.param_arg` and `Ops.variable` take no `~multiple_of`,
  and the `param_arg` record has no such field; a value known to be a multiple
  of `k` is written `v * k`.
- A float constant cast to an integer type is 0 for NaN and the type's least
  or greatest value for an infinity, where `Dtype.const` raised.
- `1 ** x` folds to 1 of the power's type, where it took the base's.
- A kernel whose loop is bounded by a value read from memory is ordered as
  one bounded by its type: run counts are exact, where their product
  overflowed and placed loads inside the loops that use them.
- An index of a computed value by scalars compiles, where rangeify raised.
- Estimates count a range whose end reads another range at its greatest
  trips, and every count exactly, where one past an `int` raised.
- `-(x + y)` and `(x & m) // c` are no longer rewritten, `x // d < c` folds
  for positive `d` only, and like terms regroup integers, not booleans.
- `(x * -1) * y` regroups floats, and `where (a < b) c (a + (c - b))` of a
  signed `a` is a maximum where `a + (c - b)` does not wrap.
- A CPU kernel on x86_64 Windows calls the process's functions, those it
  names and those Clang calls itself, by Windows' convention, where `frexp`
  crashed, and rig calls the kernel by System V's: `Tolk_engine.target` names
  Windows (`x86_64-windows,native`).
- `Schedule.create_linear_with_vars` takes variables of one name bound to
  several values: each kernel that reads one takes it from its own call, so
  kernels that differ in that value alone are one program. It raised "bind
  mismatch".
- A folded `Op.Where` or `Op.Neg` keeps a NaN's payload, a signalling NaN's
  included, where it quieted one: a selection and a negation convert nothing.
- A CUDA conversion to `Fp8e4m3` or `Fp8e5m2` saturates an infinity to the
  format's greatest finite value, where it kept it an infinity or a NaN.
- An 8-bit float's NaN folds as its format's NaN of its sign, the positive one
  in e4m3 and the fnuz formats, and an emulated cast to a narrow float gives
  its quiet NaN: a NaN result's sign and payload are unspecified.
- `Heuristic.hand_coded_optimizations` upcasts a tensor-core kernel only
  while it keeps 100 workgroups: a product of a few hundred rows, such as a
  128 by 128 half-precision matmul, ran in a few workgroups and left most of
  a GPU's compute units idle.
- A Metal or CUDA kernel that calls a body compiles: it defines the body's
  function, where it dropped it. A call of a compiled program inside a kernel
  raises `Invalid_argument`.
- A fold that would make a signed constant its type does not hold, of two
  constants, like terms or `(x + y) * c`, is no longer made.
- A call body inside a kernel is named in its lowering, numbered apart from
  other bodies of its name, and a call of a program not yet compiled is
  inlined too.
- A reduction of a gated value selects its identity in symbolic, and storing
  `where g (load index) alt` stores `alt` where `g` does not hold.
- `(x ^ y) ^ y` folds to `x` only for a weak `y`, where a typed `y` committed
  a weak `x` to its type.
- **Breaking:** `Ops.eq` builds `Op.Cmpeq`, and `Ops.ne` and `Ops.logical_not`
  are equalities to false; a negated equality renders as `!=` where the
  renderer has `Op.Cmpne`.
- **Breaking:** `Op.Floordiv`, `Op.Floormod`, `Op.Cdiv`, `Op.Cmod` and
  `Op.Reciprocal` are `Op.Div`, `Op.Mod`, `Op.Truncdiv`, `Op.Truncmod` and
  `Op.Recip`, and printed graphs name them so.
- A CPU program split into blocks over many buffers gives its values on
  x86_64 Windows, where they were garbage: its entry copies its parameters
  inline, where Clang called the process's `memcpy`, of another convention.
- `Tolk_engine.link ~storage:Per_run` makes a schedule's storage at each run,
  each buffer before the first call that takes it and returned after the last,
  so the schedule holds none between runs and a run holds only what its calls
  need at once. `Jit.jit_lower ~arenas:false` leaves the buffers out of arenas
  for it. A run keeps none of the buffers it was given once it returns.
- `Dtype.const` wraps an integer to a signed type too: 300 as an `Int8` is
  44, as it was through `Ops`' reads at the width.
- A HIP kernel that calls a body compiles: its program defines the body's
  function, as a device function, where it dropped it.
- A store whose value reads its destination elsewhere, as a flip, a roll
  or a row of it, stores the value staged in `Rangeify`, by indices, where
  `Prepare` staged it by movements.
- **Breaking:** `Decomp_dtype.f2f_clamp` is gone: `Decomp_dtype.f2f`
  narrows with its overflow and saturation itself. Without saturation,
  e4m3's greatest value plus half an ulp now rounds to the greatest value.
- `Gater` no longer moves a selection into a converted gated load, and no
  longer has rules for two-index image accesses.
- A reduction over ranges its value ignores folds once over the product of
  their sizes, selecting its identity where the product is not positive.
- A stack of `Invalid` lanes folds to an `Invalid` expanded to its width.
- The high half of `(x << 32) | y` folds to `x`'s low half for any `uint64`
  `x`.
- `Codegen.program_key` gives the key a program is kept under, and
  `Codegen.programs` counts the programs made. `Realize.lower_and_compile`
  compiles one kernel of each key, so kernels that differ in their
  parameters' devices alone no longer wait for one another's compilation.
- Scheduling a graph of many copies between devices walks less of it: a
  store's check for reading its destination enters only the nodes after the
  destination.
- `Codegen.to_program` gives a kernel the program made for another device of
  its target, in memory and on disk: a program's key no longer holds its
  parameters' devices, so the devices of a set compile each kernel once.
- `Ops.vmin` and `Ops.vmax` of a cast into a weak float round to doubles, as
  the value does: a `uint64` cast held `2^64 - 1` where its value is `2^64`.
- **Breaking:** nodes have no tag: `Ops.Tag`, `Ops.tag`, `Ops.rtag`,
  `Ops.remove_all_tags` and the `?tag` of `Ops.v`, `replace`, `sink`, `index`,
  `load`, `store`, `after`, `ins`, `Ops.Upat.v` and `Ops.Upat.op` are gone, and
  the graph format has no `tag=`.
- **Breaking:** a permute holds its order in its second source, a stack of
  weak integer constants; the `Axes` argument is gone.
- **Breaking:** `Shape.placeholder` takes no `?device`, `Shape.placeholder_like`
  no `?addrspace` and `Shape.param` no `?multiple_of`.
- `x // -1` stays a division when `x` is unsigned, where a weak `-1` is the
  type's greatest value; it was `-x`, wrong for every `x` but `0` and the
  greatest.
- `-x < -y` and `-x < c` become `y < x` and `-c < x` for signed integers only
  where no negation wraps at the comparison's width, and the late rules that
  did so without asking go.
  `not (x < c)` becomes `c - 1 < x` for unsigned integers too, unless `c` is
  the type's least value.
- Two stage constant folding, `(x + 1) + 2` to `x + 3`, runs in
  `Shape.symbolic_simple`, so the late decompositions fold too.
- A kernel lowers the body of a call once per renderer and settings.
- `Dtype.const` wraps an integer to an unsigned type: -1 as a `Uint8` is
  255. The negation and masked-shift rules read -1 at their operand's type,
  so they apply to unsigned constants.
- `c0 + x < c1` (`Shape.symbolic`) and the comparisons `pm_reduce_collapse`
  solves (`x + y < c`, `x * y < c`) apply to signed and index integers only,
  `(x // c + a) // d` to index integers only, and `Symbolic.reduce_mul_chain`
  moves no factor out of an unsigned maximum: unsigned arithmetic wraps.
- C-style sources compute an addition, subtraction, product, negation, left
  shift or multiply-add on an unsigned char or short in an unsigned int and
  narrow after: C computes them in `int`, where `65535 * 65535` overflows,
  which is undefined.
- HIP writes bfloat16 as `__bf16` on every AMD target.
- `Coalesce.memory_coalescing` merges the CPU device's loads and stores as if
  every buffer started on a 16-byte boundary, since its vector types are
  aligned to their element; the start a buffer is known at decides the merges
  of the GPU targets only.
- Repeated stores into a shrink of their own destination, such as
  `x[0] += 1`, each store in place, without a copy.
- `Ops.storage_base` remembers what it finds: the storage of n stores into one
  buffer takes n steps in all, where it took n²/2.
- `Jit.jit_lower` makes each buffer it does not hold storage of the program
  (`Shape.alloc`), which `Tolk_engine.link` allocates.
- **Breaking:** the `ALIGNED` setting (`Setting.aligned`) is gone: Clang's
  vector types are aligned to their size, a bool vector's to one byte, and on
  the CPU device to their element, so that they load from any element.
- **Breaking:** `Ops.kernel_info` has `optimized`, which
  `Postrange.Scheduler.get_optimized_ast` sets and `Postrange.apply_opts` and
  `Codegen.linearize` read, where the optimiser tagged the kernel's sink `1`.
- **Breaking:** a call ends none of its arguments' ranges (`Ops.range_start`):
  it runs inside them.
- A store that folds away, of a load of its own element or through an invalid
  index or value, keeps its sources as dependencies (`Shape.symbolic_simple`,
  `Symbolic.sym`), and a constant ordered after a store no longer folds to a
  bare constant (`Shape.symbolic`). A kernel dropped the stores either waited
  on.
- `C.findlib` no longer looks for `p.dll` on Windows.
- **Breaking:** a program takes its parameters in slot order, buffers and
  scalars interleaved. `Ops.program_info.vars` is `params`, every parameter by
  slot; `Shape.kernel_params` is the order a kernel takes them;
  `Device.Tiny_elf.param` carries its `addrspace` in place of a compact slot;
  `Realize.get_call_var_uops` is `get_call_kernel_args`, one value per
  parameter; `Ops.vals` is gone. A custom kernel's variable argument stays a
  variable in its own slot (`Shape.custom_kernel`), and the engine lays out a
  launch's parameters in that order.
- **Breaking:** `Jit.jit_lower` takes `~input_vars`, the capture's variables:
  it unbinds them, so each run reads the value it binds, and keeps any other
  binding.
- **Breaking:** `Ops.param_arg.addrspace` is a `Dtype.addr_space`, no option.
- `Dtype.can_lossless_cast` decides by the types' bounds and widths: `Int8`,
  `Uint8` and the 8-bit floats now cast to `Bfloat16` without loss.
- A bool sum or product over a range its value ignores is the value, where it
  multiplied the value by the range's size (`Simplify.pm_reduce_unparented`).
- `Spec.shared` refuses a store whose value is not of its target's type, and a
  weak operand of another kind than its result's.
- A comparison of a sum of multiples with a constant folds for any constant
  and any common factor (`Shape.symbolic`); an exclusive or of non-negative
  integers is bounded by the wider operand's next power of two.

- **Breaking:** `Op.Flip`, the argument `Ops.Flips` and the movement `Flip`
  are gone. `Shape.flip` reverses an axis with pads, reshapes, shrinks and
  permutes, whose index folds to `n - 1 - i` on an axis of known size; a store
  through a padded view gates its destination index (`Prepare.pm_mops`).
- HIP on RDNA 4 multiplies int8 matrices into int32 on its tensor cores
  (`Tc.amd_rdna4`).
- `Codegen.full_rewrite_to_sink` decomposes the types a target lacks after the
  transcendental functions, so those they make, such as Payne-Hanek's
  `uint64`, are decomposed too.
- `Linearizer.linearize` places a node in more loops after one in fewer that
  runs as often.
- `Shape.alloc` takes `?vmin_vmax`, the bounds of the values it holds.
  `Shape.view_as` reshapes a flat node to a 1-d shape too, and refuses a size
  other than the shape's greatest.
- `Shape.repeat` refuses a negative repeat, and `Shape.flip` an axis of a
  scalar, with `Invalid_argument`.
- A kernel that calls a sink no longer runs its loops nested wrongly where the
  body has a loop of the size and identity of one of its own
  (`Codegen.to_program`).
- A value staged into local memory under a thread index gets a row of the
  buffer per thread, where the threads of a group shared one and overwrote each
  other's elements (`Codegen.full_rewrite_to_sink`).
- `Render.render_uir` writes a range's identity as words, `LOOP 1 2`, where it
  wrote `LOOP r1_2`.
- An integer `Op.Pow` computes by wrapping products (`Transcendental.xpow_int`,
  `Ops.exec_alu`), not float logarithms; a negative exponent gives 0, but `±1`
  for a base of 1 or -1, and is the one `Ops.pow` refuses for an integer base.
- **Breaking:** tolk compiles for one device. The multi-device rewrite is
  gone: `Multi`, `Allreduce`, their operations and settings, `Axis_type.Device`
  and the sharding functions of `Ops` and `Shape`; the type `Ops.device` is a
  device's name. nx partitions a value placed over several devices.
- **Breaking:** `Tolk_engine.link ~bound` binds one buffer per storage node, and
  `Tolk_engine.run`, `time` and `slots` take one buffer per slot, `None` where
  no parameter takes it (`Rig.Buffer.t option array`).
- Two variables of one name, such as one read under two bounds, take one
  parameter slot.
- **Breaking:** `Ops.Axis_type` loses `Reduce` and `Unroll` and orders `Global`,
  `Local`, `Warp`, `Weak`, `Loop`, `Upcast`, `Placeholder`: a range is a reduce
  axis when a reduction closes it. Its argument prints the type first.
- **Breaking:** `Opt.target` loses `Unroll`: an `Upcast` of a reduce axis
  unrolls it. `Axis_type.position`, `Scheduler.ranges_of`, `reduce_axes`,
  `unrollable_dims` and `upcasted` give way to `Scheduler.reduce_ranges`,
  `axes_of ~reduce` and `upcastable_dims ~reduce`.
- **Breaking:** a tensor core's `upcast_axes` (`Ops.wmma`) are `Ops.wmma_axis`
  values, which name each axis's type; `Ops.compare_range_arg` orders ranges.
- **Breaking:** `Gpudims.add_gpudims` and `pm_add_gpudims` become
  `group_gpudims` and `pm_group_gpudims`, whose hardware indices are global and
  local ranges lowered as other indices are; `Gpudims.pm_range_to_special`
  makes them `Special` nodes once barriers are placed.
- **Breaking:** `Op.Group` is gone: `Ops.group` is the empty stack ordered after
  its effects (`Ops.after`), and only a sink splices the effects of its sources
  (`Symbolic.pm_clean_up_group_sink`).
- **Breaking:** `Dtype.name` is a data type's short name (`i8`, `u32`, `f16`,
  `bf16`, `f32`, `fp8e4m3`) and `Dtype.pp` prints `dtypes.<name>`, where both
  used C names (`signed char`, `half`). `Dtype.of_string` reads the short names
  and the older ones. WMMA functions are named by them (`WMMA_8_8_8_f16_f32`).
- **Breaking:** a parameter, buffer or allocation holds its number of elements
  as its first source (`Ops.size_src`), a weak integer constant or the empty
  stack of a scalar; `Ops.param_arg` has no `size` field or `?size`.
- **Breaking:** `Ops.range` takes no `?src`: a range has one source, its bound,
  and a loop ordered after effects has its bound ordered after them
  (`Ops.after`).
- **Breaking:** `Shape.placeholder` takes no `?tag` or `?volatile`: a named or
  volatile parameter is made with `Shape.param`.
- `Shape.param_like` takes an optional `~name` for the parameter it makes.
- `Prepare.contiguous_view` finds a view of a constant or a computed value:
  the value itself, at offset 0. It found none before.
- `Cstyle.hip` for RDNA 4 (`gfx1200`, `gfx1201`) has native 8-bit floats and
  their tensor cores (`Tc.amd_rdna4`), whose operands it passes as 32-bit
  words (`Tc.pm_wmma_fp8`).
- **Breaking:** `Render.render_uir` replaces `Render.pp_uops`: it writes a
  graph one node a line, `%i = op srcs : arg`, constants inline, as the
  `DEBUG` and `BEAM_DEBUG` prints now do.
- **Breaking:** `Helpers.colored` takes 24-bit colours (`Helpers.Rgb`).
- **Breaking:** `Realize.get_call_written_bufs` is removed: nothing read it
  but its tests.
- A kernel can call a sink: `Codegen.to_program` lowers its body alone,
  unoptimised, and `Cstyle.clang` writes it as a function of the program, each
  body once, before the kernel.
- A call of an external function calls its symbol, which `Cstyle.clang`
  declares and the host links against, where it called a pointer the
  function's body held: a kernel can call `sqrt` or `frexp` of the C library.
- **Breaking:** A call has the type and shape of its body: `Ops.call` loses
  `?ret_dtype`, and an external function states the type it returns
  (`Ops.custom_function ~dtype`, the `Custom_function` argument).
- `Shape.program_info_of_sink` reads no parameter of a call's body.
- A jitted chain of 4 float32 `Nx.betainc` over 4 values compiles in 2.9 s
  instead of 172 s: `Shape.symbolic` leaves a selection whose condition or
  branch reads a load as it leaves one that reads an index, so kernels that
  load their inputs as vectors no longer copy a chain's earlier links into
  each later one.
- An exception raised inside `Ops.v`, as a test runner's timeout or Ctrl-C's
  `Sys.Break` is at an allocation, no longer leaves a lock of its node table
  held, after which every later node failed with "Resource deadlock avoided".
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
  by the points the run's steps report, so neither a kernel that another
  domain runs meanwhile nor a profile taken around a search changes what it
  measures or the kernels a search picks. A submission of several kernels
  counts once, and one of copies alone counts nothing.
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
- Compiled kernels that pad or concatenate an integer quotient no longer die
  with SIGFPE on x86, as `Norn.Nuts`'s compiled transition did: the padding
  divided by the zeros it loads, and now divides by 1.
- `Search.beam_search` takes `~link` and `~time`, which
  `Tolk_engine.link_call`, `Tolk_engine.slots` and `Tolk_engine.time` provide,
  and accepts positional scalar arguments, so `Rune.jit ~beam` searches a loop
  that reads its index. A search allocates its buffers once, and not at all
  when its result is cached, links each candidate once, runs each sample once
  and silently, compiles each kernel once, and compiles no candidate past
  `BEAM_UOPS_MAX` (`Codegen.linearize`). A candidate's setup on Metal takes
  about 1.2 ms, from about 3.9 ms.
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
  the calls: storage (`Shape.param`, `Shape.alloc`, ...), the bindings of
  variables, `Shape.call_with_outputs` and `Shape.program_info_of_sink`.
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
  runs one kernel, from six, and computes its chain once. Slices of one
  computation, stored through two views of it, run as kernels of their own.
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
- `Tolk_engine.time` raises `Rig.Lost` when a device that runs the
  timed kernels is lost during the run, where it raised `Invalid_argument`
  "no span": a search on NV that a GPU co-tenant held past the hang timeout
  died naming a missing span instead of the lost device.
- `Search.beam_search` compiles each source once: candidates of distinct
  kernels that render one source each ran the compiler, a rejected one again
  for every domain waiting on it. A domain asking for a source another is
  compiling now waits for its binary or rejection.
- `Compiler_cpu.clang` runs Clang as a process of its own, waited for with
  the runtime released, instead of through the C library's `system`, which
  macOS runs one at a time: 16 compiles on 8 domains take 0.11 s, from
  0.66 s.
- Lowering a kernel allocates 22% less again: a graph walk pushes its work
  without allocating, and looking a node up builds a smaller probe. 8 domains
  lowering lorenz_simple's candidates collect 25% less often.
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
- `Search.beam_search` times its kernel before its first round and starts
  from it, and a round progresses only when every sample of its fastest
  candidate beats every sample of the beam's first kernel by
  `BEAM_MIN_PROGRESS`, so a search never answers a kernel slower than the one
  it was given. The least of noisy timings is biased low, and searches ran
  rounds on gains within the noise, as PR #235 found.
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
- Scheduling a large graph no longer grows with the square of its kernels:
  weighing whether a stage could be inlined walked every kernel upstream of
  it, and the where-closure rule searched branches its INDEX gate rejects.
  An unrolled training step of 2031 kernels schedules in 5.3 s, from 27 s.
  `Ops.reaches` stops at the node it looks for.
- A beam search is read back from the disk cache only for the candidates it
  chose among, so `TC_OPT=0` no longer reads back a search made with `TC_OPT`
  unset. The search's `BEAM_*` variables are declared, and `NO_MEMORY_PLANNER`
  reaches results, so the caches key on them too.
- A function with two loops or reductions of one length, such as a scan body
  of `sum (mul x (matmul x c))`, schedules again: numbering its ranges for the
  schedule cache's key raised "a bottom-up rewrite cycles" when two ranges
  traded numbers, and merged them when one took the other's (#236).
- A compiled binary is read back from the disk cache only for the compiler,
  version and options that made it: `Renderer.Compiler.v` takes the table as a
  function, which Clang's, Metal's, NVRTC's and comgr's compilers derive from
  their toolchain. A changed flag or an upgraded toolchain was answered with a
  stale binary. Beam searches are kept under their settings and compiler too.
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
- A process reads back the schedules of a function with loops, such as a
  staged scan, that an earlier process kept on disk. The loop's range was
  numbered from a counter that depends on what the process did before, so a
  warm run missed the schedules made after its first miss or hit.
- A program cached under one host compiler (`CC`) is compiled again under
  another, in the compile cache and the program cache: the object the first
  compiler made was served.
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
  `Ops.kernel_info.split` is the loop a launch splits.
- A compiled matrix-vector product whose vector is converted, or whose matrix
  is decoded, such as `Nx_quant.apply`'s, spreads each row's reduction over a
  group of threads: it ran each row in one thread.
- tolk's exact integers, the values of integer constants, dtype limits and
  bounds, are `Bigint.t`, an arbitrary-precision integer in pure OCaml, so
  tolk needs no GMP on the system.
- Add `tolk`, a port of tinygrad's compiler at the tinygrad commit its test
  generator pins, in the libraries `tolk` and `tolk.engine` (modules `Tolk`
  and `Tolk_engine`). `Tolk_engine` runs compiled schedules as programs of
  rig (`Rig_program`) over rig's devices and buffers. rune compiles through
  it.

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

- `Nx.unique` without `~size` numbers keys whose rows take at most as many
  values as there are rows, read from each column's least and greatest order
  keys, in a few passes and no sort: a million int64 keys of 100 values take
  40% fewer instructions. `Nx_ragged.ids` numbers its codes directly, with no
  sort.
- `Nx.betaincinv` and `Nx.betainccinv` take each logistic point's two
  logarithms with one `exp` and one `log1p`, where they took two of each: at
  float64 their program is 6% smaller and their gradient's 4%, which
  `Rune.jit` compiles in proportion. Results keep their bits.
- On nx.cpu, a contraction in lane order against an operand whose
  contracted axis is contiguous, as a decoded token against attention's keys
  or a weight stored `[n × k]`, takes up to 16 of that operand's rows against
  one row of the other and sums each block in registers. On one thread a
  float32 token's scores against 4096 keys of 64 take 0.07 of the
  instructions they took, its 2880-to-4096 projection 0.33, in bfloat16 0.27
  and 0.81. Results keep their bits.
- On nx.cpu, a fused map reads an operand broadcast along its rows, as
  `Nx.softmax`'s and `Nx.log_softmax`'s lane maxima and sums are, where it
  lies instead of copying it: `Nx.softmax` over 8 x 512 x 512 float32 runs
  42% fewer instructions, `Nx.log_softmax` 54%. Results keep their bits.
- `Nx.rshift` computes with integers at every width, where integers of 32
  bits or fewer went through float64: a kernel that shifts needs no float64,
  and a compiled one-token Q6_K product's kernel takes 0.53 ms where it took
  3.3 ms on an RTX 5000 Ada. Results keep their bits.
- Under `Rune.jit`, `Nx.map_segments` over ids the compiler knows are all in
  range, as `Nx.top_k`'s are, gives `f` rows with no reduction inside them, so
  a product in `f` streams its matrix. gpt-oss-20b's 12-token prefill takes
  0.40-0.48 s on an RTX 5000 Ada, from 0.82-0.89 s.
- On Windows, a bytecode program or a toplevel, such as `quill`'s or mdx's,
  loads nx's stub libraries and rig's: their calls into winpthreads, the OCaml
  runtime, `unix` and each other go through import pointers, which flexdll
  links at any distance.
- `Nx.item` allocates 21 fewer words: it fills the index in place, where it
  built lists of positions and windows and two closures on every call.
- An operation that writes its result over a donated operand allocates 5
  fewer words: the option and the array choice it built on every call.
- `Nx.betaincinv`, `Nx.betainccinv`, `Nx.gammaincinv` and `Nx.gammainccinv`
  take one Halley step at float64 where they took two, and a last step of
  fifth order: each is one incomplete beta or gamma shorter. At float64
  `Rune.jit` compiles the beta's quantile in some 40% less time and its
  gradient in 25% less. Results move by an ulp or so.
- A NaN result's sign and payload are unspecified, and it is signalling only
  where an operand is a signalling NaN (`Nx.Dtype`, Stores). Casts and kernels
  no longer promise to quiet a signalling NaN; `Nx.Dtype.of_float` still does,
  and now quiets a `float64` and `complex128` one too, as does a store of an
  OCaml float into such an array, which kept it signalling.
- A fold over a cut axis closes each piece by folding its slots one at a time,
  in C order of their indices, where one `Reduce` read all of them: a device
  holds two slots of a piece at a time, so a fully sharded step's closing fits
  RFC 0005's bound. A float result's association over the slots changes.
- `Nx_quant.apply` sums a Q6_K weight's int8 quants times their scales, and
  reads Q8_0 where it lies at a float64 input, where it decoded both first;
  float64 products decode other weights at float32. Results keep their bits.
- `Nx.rearrange` keeps the cut of a name that is an axis of its own on both
  sides, whatever its extent: `"b f -> 1 b f"` of `[1; 4]` split along `b`
  lay on the new literal axis, as its reshapes paired axes of one element
  front to back.
- `Nx.get`, `Nx.squeeze`, `Nx.unsqueeze` and `Nx.slice`'s `I` and `N` keep
  each kept axis's cut on the axis it becomes, whatever its extent, so a
  batch of one split across devices lies, sums and scans as a batch of two
  does. They went through a reshape that paired axes of one element front to
  back, and so raised, or moved a cut onto another axis.
- **Breaking:** `Nx.squeeze` and `Nx.get` of a cut axis of one element raise
  naming `Nx.place (Placement.on s) x`, as `Nx.get` along a cut axis of two
  elements does: `Nx.squeeze ~axes:[ 0 ]` and `Nx.get [ 0 ]` of `[1; 1; 4]`
  split 0 computed, moving the cut onto the remaining axis of one element.
- `Nx_array.Move.Permute` is an axis map: it may leave out axes of one
  element and add new ones, written as negative entries. It stays a view.
- `Nx_quant.dequant`, `apply` and `apply_experts` raise at the call, naming
  `Nx.place (Placement.on s) w`, where the weight lies cut along `k` inside its
  blocks or with MXFP4 scales apart from their codes; the raise came from an
  operation of the decoding.
- `Nx_quant.apply_experts` computes with its weight cut along `k`, where it
  raised: each device sums its window of every row and the sums close by
  partials, as `Nx.contract`'s do. The cut keeps whole blocks where the devices
  divide them; an MXFP4 weight read in place, by a float32, bfloat16 or float16
  input, may instead cut its codes where the devices divide `k / 2`, its scales
  whole on each device.
- **Breaking:** `Nx.reduce_ranges` with `Add`, `Max` or `Min` raises on rows
  cut across devices, naming the `Nx.place` that keeps the other cuts; eight
  rows in two tiles used to compute. `Set` computes where the rows lie.
- On a set of devices of several architectures, each window that several
  devices hold, of a result, a fold's partials or a part of its closing, is
  computed by the devices of its first holder's architecture and copied to the
  others, so replicas hold equal bits.
- `Nx_quant.apply` reads a Q8_0 weight where it lies at float32,
  bfloat16 and float16 inputs, as it reads MXFP4: it decoded the whole
  weight at float32 on every call. `Nx_quant.quantize_q8_0` stores a float
  weight as Q8_0, its bytes ggml's `quantize_row_q8_0_ref`'s outside a
  block holding a NaN, which makes its block's scale NaN.
- nx.cpu reads a scaled contraction operand whose axes make several runs
  where it lies, scaling each run of a block once: it copied the operand
  whole, scaled, on every call.
- On Linux, `Nx.fft` and `Nx.ifft` of a complex line, and `Nx.rfft` and
  `Nx.irfft` of an even one, of at least 32 MiB and past the largest cache
  keep the four-step grid in their result. They took scratch that the C
  library mapped and faulted in afresh on every call.
- `Nx.Prim.Consumer.take` takes `~piece`, the device of the operand's
  placement whose memory a result takes, so a compiler that runs an operation
  per device lends each device's memory apart. `Nx.Repr.unsafe_of_shards`
  makes a value cut across devices that owns its arrays' memory.
- `Nx.Rng.split`, `split_batch`, `fold_in` and `fold_in_tensor` of a key on
  a set without kernels, such as a GPU's outside `Rune.jit`, raise naming
  the host, where such keys are derived, and `Nx.Rng.place Nx.Host.on key`.
  `Nx.Rng`'s docs show the idiom: derive on the host, place at the call.
- `Nx.donate` of a value cut across devices lets the operation that
  consumes it write its result over each device's window where the result
  lies as the donor does, as it did for a value on one device.
  `Nx.Prim.Consumer.value` gives a consumer's operand as a value that
  owns its memory where a donation of it may write there.
- A reduction, `Nx.matmul`, `Nx.contract`, `Nx.take` or `Nx.scatter`'s
  `Add` over an axis cut across devices reduces each device's share of the
  result in pieces of at most 256 KiB of every device's partials, its own
  included, with the same bits. An `Nx.sum` whose result of `r` bytes is
  whole on `n` devices holds `r + r/n` and at most `256 (1 + 1/n)` KiB
  beside its operand on each device, where it held `2r`; `Nx.argmax`,
  `Nx.var` and `Nx.scatter`'s `Add` hold their partials once, not twice.
- `Nx.place` onto devices and the moves of operations across devices return
  as `Rig.Buffer.submit_copy` returns, where they waited for each device's
  earlier work, and a window gathered into a device moves in one copy.
- Until a wait, a placement's copy may read the memory a value wraps
  (`Nx.Repr.of_array`), and a device lost meanwhile raises at the next use of
  what it wrote.
- `Nx_array.transfer` refuses views of one memory that share bytes off the
  host too, where a device copied them in an order that could read a written
  element, and a transfer of an array onto itself does nothing.
- `Nx.check` on another machine's device copies its result to the host at
  issue, so settling it no longer waits for the work submitted after it.
- Add `Nx.betaincinv` and `Nx.betainccinv`, the beta distribution's
  quantiles from either tail, within `4 + κ (64 (1 + |log m|) + 4)` ulps for
  shapes up to `2^20`. Rune differentiates them in the probability and in both
  shapes.
- An operation on a traced value allocates fewer words: `Nx.Prim.results`
  builds no route where its operands lie at one placement that cuts no axis,
  and `Nx.Prim.iteri` of a contraction or a transform allocates nothing.
- `Rng.beta` draws inside `(0, 1)`: a draw that rounded to `0.` or `1.` is the
  dtype's least positive value or the largest below `1.`.
- `Nx_ragged.take` costs one int64 write, one scan and one gather per value
  it copies, where it built each value's position from `Nx.positions`, an
  `arange`, an add and a second gather. Talon's joins, which take their
  string columns this way, allocate 11% fewer words.
- `Nx.place` of a value on one device onto one other device takes half the
  time it did and a quarter of the words; `Nx.place` onto several devices and
  `Nx.item` of a split value allocate a fifth less.
- An operation over several devices runs nothing on a device whose windows
  of its results are all empty, as an uneven split leaves where an axis has
  fewer elements than devices; it called the kernels there on empty arrays.
- `Nx.Prim.iteri` applies a function to an operation's operands without
  building their list, for interpreters that walk many operations.
- `Nx.softplus`'s gradient at `0.` is `0.5`, where it was `0.`.
- The second derivative of `Nx.hypot x y` and `Nx.magnitude` computed at
  float64 is right where `|x| = |y|`, where it was zero, and the tangent is
  right where both magnitudes are subnormal. The gradient at a NaN operand is
  NaN, as the forward tangent is, where it was zero. Values are unchanged
  except where both magnitudes are subnormal, where about one result in 120
  differs by one subnormal unit.
- Eager operations allocate fewer words per call: `Nx.reshape` and
  `Nx.slice` about a third of what they did, `Nx.sum`, `Nx.cumsum` and
  `Nx.take` about half, composites such as `Nx.unique` about two thirds.
- `Nx.Prim.owner` and `Nx.Prim.payload` allocate nothing.
- `Nx.Rng.with_key` takes any key, traced included, so code that draws
  without `~key` runs inside `Rune.jit` and a loop's step in a scope at a key
  the function receives. Such a scope draws only on its key's set, as `~key`
  does. `Nx.Rng.next_key` raises in a scope at a traced key or one off the
  host.
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

- Kaun's argument errors write each float as the shortest decimal that reads
  back, the same text on every system: `got -1000000` where they wrote
  `got -1e+06`, or `-1e+006` under Windows.
- The gpt-oss example packs its attention projections and untied head as
  Q8_0 when asked (`Gpt_oss.of_hf ~pack:Q8_0`, `main.exe --q8_0`), as GGUF
  files of the model do: 53% of bfloat16's bytes, about twice its logit
  error. A projection is a `Gpt_oss.projection`, a block's attention a
  `Gpt_oss.attn`.
- The Llama example reads GGUF checkpoints (`Llama.of_gguf`,
  `Llama.config_of_gguf`, `main.exe --gguf`), decoding Q8_0, Q4_K and Q6_K
  weights with `Nx_quant.dequant`. `validate.exe` matches TinyLlama's Q8_0,
  Q6_K and Q4_K_M files to the reference's logits at float32's tolerance.
- The Llama example's `main.exe` decodes the prompt with its continuation:
  with a SentencePiece tokenizer the continuation lost its first space.
- A compiled step whose key-value pools are donated, or carried by a loop,
  stores each call's tokens alone into a pool of 8 bits or more per element
  (`Cache_index.extend`): up to 16 tokens a call rewrote the whole pool.
- The gradients of `Fn.softplus` and `Loss.sigmoid_bce` at `0.` are `0.5`
  and `0.5 - y`, where they were `-0.5` and `-y`: both now compute through
  `Nx.softplus`. `Loss.sigmoid_bce` keeps its precision where the loss is
  small: at a logit of 14 and a target of 1 it was 0.35% off at float32.
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

- The pretoken cache of BPE, WordPiece and Unigram models no longer sends the
  words that agree in their first five bytes to one place, where `▁the`,
  `▁they`, `▁them` and `▁their` evicted each other: Mistral merged 169k words
  over 5 MB of OpenWebText, now 62k.
- `encode_batch_ids` moves each row's ids into its result with one block copy
  rather than an id at a time.
- Byte-level BPE (GPT-2 and the cl100k and o200k pipelines) and SentencePiece
  BPE encode with a fifth to a quarter fewer instructions: the fused kernels walk
  64 pretokens ahead of their cache probes, test text sixteen bytes at a time on
  x86-64, and spell a character without a piece through byte fallback in C.
- On a big-endian machine brot encodes through its OCaml code: the C
  kernels' walkers read native words as little-endian and cut text wrong
  there.
- An added token matched against normalized text whose content normalizes to
  the empty text is refused with `Invalid_argument`, as Hugging Face refuses
  it; it used to be dropped silently.
- WordPiece and Unigram tokenizers keep a pretoken cache per domain, as BPE
  ones do, so a word seen before is not cut again: about a fifth fewer
  instructions for BERT-base and T5-base on OpenWebText. The cache starts at
  144 KB and grows to `~cache_capacity` entries (`wordpiece`, `unigram`;
  default 262144, 8 MB; 0 disables it).
- Added tokens that match the same text, such as two normalized ones whose
  contents normalize alike, are refused with `Invalid_argument`, as Hugging
  Face refuses them; the last of them used to win.
- `Normalizer.replace` with a one-byte pattern, the Bert normalizer and the
  Bert pre-tokenizer go through ASCII text eight bytes or a run at a time:
  SentencePiece's `▁` rewrite costs a quarter of what it did, and BERT-base
  encodes with about 23% fewer instructions.
- Added tokens are found by one walk of a trie of all of them, which seeks the
  last byte of the prefix they share: tokens such as SentencePiece's `▁<s>`,
  matched against normalized text, no longer stop the scan at every word.
  Llama 2 encodes with about a third fewer instructions.
- `decode_stream`, `decode_step` and `decode_finish` decode ids one at a time,
  as a model generates them, giving each piece of text once no later id can
  change it. Their texts make up `decode` of the ids, and none ends inside a
  character or an open run of byte-fallback tokens.
- `encode` and the other encode functions take `~match_special_tokens:false`,
  which encodes the text of special tokens in the input as ordinary text, as
  Hugging Face's `encode_special_tokens`. Text from a user must be encoded with
  it; the model can still give a special token's id when its vocabulary holds
  that text as a piece.
- `decode` on a tokenizer without a decoder joins the tokens with spaces for
  every model, as Hugging Face does; it joined them with nothing except for
  word-level models.
- `Pre_tokenizer.unicode_scripts` reads scripts from Unicode 17.0.0, the data
  of `\p{..}` in `split_regex`, rather than from uucp's Unicode 18.0.0: a
  character assigned in 18.0.0, such as U+0558, now joins the piece around it
  as Hugging Face's tokenizers has it.
- Kimi K2's pattern loads: `Pre_tokenizer.split_regex` and
  `Normalizer.replace_regex` accept nested bracket classes, their
  intersection with `&&`, and scripts as `\p{Han}` (Unicode 17.0.0).
- Fix `split_regex` and `replace_regex` matching alternatives that open alike
  as one: `x?x|x?y` matched `"xy"` on `"xy"` where Hugging Face matches `"x"`,
  and o200k's pattern took a mark and the capitals after it as one word.
- GPT-4o, gpt-oss (o200k), Mistral NeMo and Ministral (Tekken) tokenizers
  pre-tokenize with a walker for their pattern and encode through the fused C
  kernel, in place of the regular expression.
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

- `Expr.is_in` over at most eight texts compares the rows' bytes with each
  text instead of coding the column, which sorted it: decision-support
  queries 12 and 19 take less than half the instructions. Parquet reads
  unpack bit-packed dictionary indices with one load per value, aggregates
  take counts and means from their groups' sizes, and nullable integer keys
  are numbered without a sort.
- A comparison of a text column with a text of one row gives its null rows
  the bits a comparison of two columns gives them, so that a result's layout
  no longer depends on how its input is batched.
- `Str.trim` removes Unicode white space at both ends of text, and
  `Str.concat` joins texts with a separator.
- `Temporal.field` of a datetime within a day of its least tick read a day
  near its greatest: the first instant of a datetime[ns] read as 2262.
  `Temporal.floor` there gave a time of that day; a period that starts below
  the least tick is now a data error.
- `Expr.cast` converts between dates and datetimes without a zone, a datetime
  becoming the day it falls on, and between clocks and durations.
- `Expr.cut` gives the bin of a value among ascending edges, as an index,
  null outside them.
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

- **Breaking:** `Quill_top.load_package` returns `Not_installed` or `Failed`
  instead of raising or printing. A raven package that does not load is
  reported in one line, without its printers' or dependents' errors.
- `quill run` names the notebook as given on the command line in its errors
  and its `Updated` line, as `quill clean` does, where it named the absolute
  path.
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
