# Norn

Probabilistic inference for OCaml, built on [Nx](../../nx/doc/index.md),
[Rune](../../rune/doc/index.md) and [Jera](../../jera/doc/index.md). Norn turns a log density over a structure of your own
type into draws and diagnostics. A model written as one generative function
gives that density; samplers run many chains at once; draws and their
diagnostics come back as values of your structure, so
`(Norn.Diag.rhat params draws).loc` is the R-hat of `loc`.

## Quick start

<!-- $MDX skip -->
```ocaml
module D = Norn.Dist
module M = Norn_model

(* The unknowns, as a record of your own. *)
type 'a params = { loc : 'a; scale : 'a }

module Params = struct
  type 'a t = 'a params

  let walk c { loc; scale } =
    let open Nx.Ptree.Walk in
    let loc = field c "loc" leaf loc in
    let scale = field c "scale" leaf scale in
    { loc; scale }
end

let params : Nx.float64_t params Nx.Ptree.t =
  Nx.Ptree.instantiate (module Params)

let s x = Nx.scalar Nx.float64 x
let y = Nx.create Nx.float64 [| 6 |] [| 4.1; 5.3; 3.8; 6.0; 4.9; 5.5 |]

(* The model draws every random variable and returns them. *)
let model =
  M.v Nx.float64 params Nx.Ptree.tensor @@ fun () ->
  let loc = M.sample (D.normal ~loc:(s 0.) ~scale:(s 10.)) in
  let scale = M.sample (D.half_normal ~scale:(s 5.)) in
  let y = M.sample (D.iid [| 6 |] (D.normal ~loc ~scale)) in
  ({ loc; scale }, y)

let () =
  let u = M.coords model and lp = M.log_density model y in
  let key = Nx.Rng.key 0 in
  let state = Norn.Nuts.init u lp (M.init model y ~chains:4 key) in
  let state = Norn.Nuts.warmup u lp key ~steps:500 state in
  let _, coords, stats = Norn.Nuts.sample u lp key ~draws:500 state in
  let draws = Norn.Draws.map u params (M.constrain model) coords in
  Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v params ~stats draws)
```

The [examples](../examples/01-distributions/README.md) cover each part on small problems.

## Libraries

- `norn` holds distributions, bijectors, samplers, draws and diagnostics,
  over densities written as plain functions.
- `norn.model` turns a generative function into those densities, simulates
  data from it, and maps draws back to values.

## What's inside

- **Densities**: a density maps a position, your structure with a leading
  chain axis, to one log density per chain. `Norn.with_gradient` states a
  density's gradient when rune cannot compute it, such as an adjoint
  solver's.
- **Distributions**: `Dist` has continuous, discrete and vector families
  (normal to Student's t, gamma, beta, Dirichlet, multivariate normal,
  Poisson, negative binomial, categorical), with `iid`, `sorted`,
  `transform` and `mixture`. Parameters are tensors and broadcast; a
  parameter outside its domain raises naming its index and value.
- **Supports and bijectors**: `Support` names each family's set of values,
  and `Bij` maps unconstrained coordinates onto it with its log-determinant:
  `exp`, `interval`, `simplex`, `ordered`, `cholesky_corr`, `sum_to_zero`
  and their compositions.
- **Samplers**: `Nuts` runs the No-U-Turn sampler with a step size and
  geometry per chain, tuned in warmup; `Hmc` shares one step size, a tuned
  trajectory length and one geometry across chains; `Ensemble` uses the
  stretch move and needs no gradient. Each evaluates the density on many
  chains at once.
- **Evidence**: `Nested` (nested sampling with slice moves) and `Smc`
  (tempered sequential Monte Carlo with Hamiltonian or slice moves) estimate
  the log evidence with a standard error, and give weighted posterior draws
  (`Weighted`).
- **Gaussians**: `Gaussian` is diagonal plus low rank over your structure,
  built from a covariance or a precision: a Laplace approximation, or a
  sampler's geometry.
- **Draws and diagnostics**: `Draws` holds draws with `[chain; draw]` axes;
  `Diag` computes rank-normalised split R-hat, nested R-hat, bulk and tail
  effective sample sizes, Monte Carlo errors, E-BFMI, divergence locations,
  and simulation-based calibration ranks with an exact uniformity test.
  `Summary` tabulates them and lists its findings as data.
- **Models**: `Norn_model.v` reads one function that draws every random
  variable with `sample`. From it come the posterior, prior and likelihood
  densities over coordinates, starting points, prior draws, simulation,
  prediction, pointwise likelihoods, and the `reparam`, `noncentre` and
  `fix` transformations.

The samplers are formulas of tensor operations: a whole run, warmup
included, compiles under `Rune.jit` as one program whose key is an argument.
