(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Numerical methods.

    Jera computes quantities defined as limits (integrals, solutions of
    differential equations, approximations of functions) over {!Nx} tensors.
    Every program is eager unless it says otherwise and runs unchanged under
    {!Rune.val-jit}, {!Rune.val-vmap} and the derivatives of {!Rune}; jera never
    compiles anything itself.

    {1:index Methods by problem}

    {table
      {tr {th Problem } {th Regime } {th Method } }
      {tr {td Separable Hamiltonian } {td long times } {td {!Split} } }
    }

    {1:conventions Conventions}

    - {b Problems are closures.} A problem is an OCaml function and the data it
      is computed over. Every tracked value the function reads, argument or
      capture, reaches the derivative.
    - {b Batching.} A state's tensors are one problem: they share its steps.
      {!Rune.val-vmap} gives each lane its own.
    - {b Dtypes.} The working dtype is the data's, with no default. Times have
      their own dtype. Float leaves of a state are its vector; other leaves are
      carried unchanged.
    - {b Formulas.} A function whose answer cannot miss on tensor data is a
      formula and returns its value; its derivative is the composition's.
    - {b Errors.} A broken precondition on static data raises [Invalid_argument]
      naming the function, at once. One on tensor data raises [Invalid_argument]
      through {!Nx.check}: at once eagerly, when the compiled call returns under
      {!Rune.val-jit}.
    - {b Devices.} Constants are computed on the host in float64 and rounded
      once to the working dtype. *)

module Split = Split
(** Splitting methods for separable Hamiltonians. *)
