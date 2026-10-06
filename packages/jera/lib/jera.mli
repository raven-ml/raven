(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Numerical methods.

    Jera computes quantities defined as limits (integrals, solutions of
    differential equations, approximations of functions) over {!Nx} tensors.
    Every program is eager unless it says otherwise and runs unchanged under
    {!Rune.val-jit}, {!Rune.val-vmap} and the derivatives of {!Rune}; jera never
    compiles anything itself. A random source, a key or a Brownian path, is an
    argument of a compiled function: one it captures is a constant, whose draws
    it refuses.

    {1:index Methods by problem}

    {table
      {tr {th Problem } {th Regime } {th Method } }
      {tr {td Zero of a function } {td derivative given } {td {!Root.newton} } }
      {tr {td  } {td a bracket } {td {!Root.bracket} } }
      {tr
        {td Minimum }
        {td one variable, a bracket }
        {td {!Minimize.bracket} }
      }
      {tr
        {td Integral, one dimension }
        {td smooth }
        {td {!Quad.adaptive}; {!Quad.fixed}, {!Quad.cumulative} }
      }
      {tr
        {td  }
        {td of samples }
        {td {!Piecewise.integral} of an interpolant }
      }
      {tr
        {td  }
        {td endpoint singularity, infinite range }
        {td {!Quad.tanh_sinh} }
      }
      {tr
        {td Integral, several dimensions }
        {td up to ten }
        {td {!Quad.cubature} }
      }
      {tr {td  } {td high } {td {!Quad.qmc} } }
      {tr
        {td Approximation }
        {td samples }
        {td {!Piecewise.linear}, {!Piecewise.cubic}, {!Piecewise.hermite} }
      }
      {tr {td  } {td monotone samples } {td {!Piecewise.steffen} } }
      {tr {td  } {td a function, to a tolerance } {td {!Piecewise.adapt} } }
      {tr
        {td  }
        {td a function, fixed resolution }
        {td {!Piecewise.chebyshev}, {!Grid.chebyshev} }
      }
      {tr {td  } {td samples on a grid } {td {!Grid.linear}, {!Grid.cubic} } }
      {tr
        {td Differential equation }
        {td non-stiff }
        {td
          {!Ode.solve}, {!Ode.sample} with {!Ode.tsit5} and kin; {!Ode.march}
          with an explicit method, {!Ode.euler} to {!Ode.dopri5}
        }
      }
      {tr {td  } {td randomness } {td {!Sde.march} } }
      {tr {td  } {td separable Hamiltonian, long times } {td {!Split} } }
    }

    {1:conventions Conventions}

    - {b Problems are closures.} A problem is an OCaml function and the data it
      is computed over. Every tracked value the function reads, argument or
      capture, reaches the derivative.
    - {b Batching.} Elementwise families ({!Root}, {!Quad}'s integrals,
      evaluation) treat every element as its own problem; their function must
      not reduce or mix along any axis. Structured families solve one problem,
      and a state's tensors share its steps. {!Rune.val-vmap} gives each lane
      its own.
    - {b Dtypes.} The working dtype is the data's, with no default. Times have
      their own dtype. Float leaves of a state are its vector; other leaves are
      carried unchanged.
    - {b Formula or solve.} A function whose answer can miss on tensor data (a
      tolerance, a budget) is a solve and returns a {!Solution.t}, with a status
      per lane; one that cannot is a formula and returns its value, and its
      derivative is the composition's.
    - {b Searches and answers.} A solve searches on detached values, then states
      its answer from the search's decisions: a zero or a minimum by its
      equation through {!Rune.root}, an integral by its rule over the final
      partition, a flow by its accepted steps taken again, a fit by its final
      pieces. The derivative is the answer's: no search is differentiated, and a
      lane that did not converge has a zero derivative.
    - {b Tags.} A method's tag says which drivers take it: [`Formula] runs in a
      formula, [`Embedded] estimates its error and drives a solve. An impossible
      pairing is a type error.
    - {b Budgets} are arguments only where no bound is derivable, with no
      default.
    - {b Derivatives that steer.} A method that needs a derivative only to steer
      its search takes it from the caller ([slope]); it changes the speed only.
    - {b Errors.} A broken precondition on static data raises [Invalid_argument]
      naming the function, at once. A missed tolerance is a status. A broken
      precondition on a formula's tensor data raises [Invalid_argument] through
      {!Nx.check}: at once eagerly, when the compiled call returns under
      {!Rune.val-jit}.
    - {b Devices.} Constants are computed on the host in float64 and rounded
      once to the working dtype. *)

module Tol = Tol
(** Tolerances. *)

module Solution = Solution
(** Answers of solves, with a status per lane. *)

module Root = Root
(** Zeros of functions of one variable. *)

module Minimize = Minimize
(** Minima. *)

module Quad = Quad
(** Integrals. *)

module Piecewise = Piecewise
(** Piecewise Chebyshev series: splines, interpolants and fits. *)

module Grid = Grid
(** Tensor-product series over grids. *)

module Ode = Ode
(** Ordinary differential equations. *)

module Sde = Sde
(** Stochastic differential equations. *)

module Split = Split
(** Splitting methods for separable Hamiltonians. *)
