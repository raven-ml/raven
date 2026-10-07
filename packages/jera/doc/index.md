# Jera

Numerical methods for OCaml, built on [Nx](../../nx/doc/index.md) and
[Rune](../../rune/doc/index.md). Jera solves linear and nonlinear systems, finds minima,
integrates, interpolates, and solves ordinary, stiff, differential-algebraic,
delay and stochastic differential equations. Every solve meets a stated
tolerance and reports a status per lane. Rune differentiates every answer,
compiles it and batches it.

## Quick start

```ocaml
open Jera

let () =
  (* The cube root of each c by bracketing, and its derivative in c. *)
  let cube_root c =
    Root.bracket ~tol:(Tol.ulps 4.)
      (fun x -> Nx.sub (Nx.mul x (Nx.square x)) c)
      ~lo:(Nx.zeros_like c) ~hi:(Nx.full_like c 10.)
    |> Solution.get
  in
  let c = Nx.create Nx.float64 [| 3 |] [| 2.; 8.; 27. |] in
  Nx.print (cube_root c);
  Nx.print (Rune.grad' (fun c -> Nx.sum (cube_root c)) c);

  (* A pendulum q'' = -sin q, solved to t = 10 within a tolerance. *)
  let pendulum _t (q, p) = (p, Nx.neg (Nx.sin q)) in
  let s x = Nx.scalar Nx.float64 x in
  let q, _ =
    Ode.solve Nx.Ptree.(pair tensor tensor) Ode.tsit5
      ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12) ~budget:1000 pendulum
      ~t0:(s 0.) ~t1:(s 10.) (s 1., s 0.)
    |> Solution.get
  in
  Nx.print q
```

The [examples](../examples/01-roots/README.md) cover each family on small problems.

## Methods by problem

| Problem | Methods |
| --- | --- |
| Linear systems | `Linear.dense`, `banded`, `cg`, `gmres`, given the operator as a function |
| Zeros | `Root.bracket`, `Root.newton`, elementwise |
| Systems | `System.newton`, `broyden`, `anderson`; `System.lanes` for many small systems |
| Minima | `Minimize.bfgs`, `lbfgs`, `newton`, `levenberg_marquardt`, `nelder_mead`, with boxes through `~within`; `Minimize.bracket` in one variable |
| Integrals | `Quad.fixed`, `cumulative`, `adaptive`, `tanh_sinh` for endpoint singularities and infinite ranges; `cubature` and `qmc` over boxes |
| Approximation | `Piecewise` splines, Steffen and Hermite interpolants, Chebyshev fits to a tolerance, derivatives and integrals; `Grid` over several axes |
| ODEs | `Ode.march` with fixed steps; `solve`, `sample`, `path`, `event`, `delay` with adaptive steps, from `euler` to `tsit5` and `dopri5` |
| Stiff equations and DAEs | `Ode.kvaerno5`, with a mass matrix for algebraic constraints |
| SDEs | `Sde.march` over a `Brownian` path: Euler–Maruyama, Milstein, SRA1, reversible Heun |
| Hamiltonians | `Split` leapfrog to `yoshida8`, symplectic over long times |

## Conventions

- **Problems are closures.** A problem is an OCaml function over tensors, and
  a state is any structure of tensors (`Nx.Ptree`).
- **Solves return a `Solution.t`.** A solve whose answer can miss, on a
  tolerance or a budget, returns its answer with a status per lane:
  `Converged`, `Budget_spent`, `Not_bracketed`, `Not_finite` or `Stalled`.
  `Solution.get` reads an answer that converged everywhere and otherwise
  raises a report naming the lane, its data and what to change.
- **Tolerances are stated.** `Tol.v ~rel ~abs`, `Tol.rel`, `Tol.abs` and
  `Tol.ulps` say when an error estimate is small enough, and each solve says
  what its estimate measures.
- **Derivatives are the answer's.** A solve searches on detached values,
  then states its answer by its equation: a zero by `f x = 0`, a minimum by
  a zero gradient, an integral by its rule over the final partition, a flow
  by its accepted steps. Rune differentiates that statement through every
  value the problem's function reads.
- **Batching.** Elementwise families treat every element as its own
  problem. `Rune.vmap` gives each lane of a structured solve its own
  problem and status, and `Rune.jit` compiles any of them.
