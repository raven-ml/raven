(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Symbolic simplification rules for {!Kernel} IR.

    Rules have type [Kernel.t -> Kernel.t option] and compose with
    {!Kernel.first_match}.

    {b Layering.} [gep_pushing] is a self-contained subset. [sym] includes
    [gep_pushing] plus algebraic simplifications. Callers pick the layer
    they need and compose it with their own rules. *)

val gep_pushing : Kernel.t -> Kernel.t option
(** GEP (get-element-pointer) simplification rules.

    - [Gep(Vectorize(a,b,c,...), i)] → lane [i]
    - [Gep(Const(c), _)] → [Const(c)]
    - [Gep(void, _)] → source
    - [Cat(a, b)] → [Vectorize(Gep(a,0), ..., Gep(b,0), ...)] *)

val sym : Kernel.t -> Kernel.t option
(** Full symbolic simplification. Includes {!gep_pushing} plus:

    - [ALU(Vectorize(x,...), Vectorize(y,...))] → [Vectorize(ALU(x,y), ...)]
    - Constant folding for unary/binary ops on {!Const} nodes
    - Identity removal: [x + 0], [x * 1], etc. *)
