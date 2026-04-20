(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Symbolic simplification rules for {!Kernel} IR.

    Rules have type [Kernel.t -> Kernel.t option] and compose with
    {!Kernel.first_match}.

    {b Layering.} Three phases:

    - {!symbolic_simple} (phase 1): generic folding — constant folding,
      identity removal, self-folding, where folding, divmod reconstitution.
    - {!symbolic} (phase 2): adds algebraic rules (combine terms, associative
      folding, lt folding, range collapse) and {!Divandmod.div_and_mod_symbolic}.
    - {!sym} (phase 3): adds GEP pushing, vectorize reordering, POW
      decomposition.

    Callers pick the layer they need and compose it with their own rules. *)

(* CR: documentation is not following our guidelines, need to update with /ocaml-doc skill *)

val gep_pushing : Kernel.t -> Kernel.t option
(** GEP (get-element-pointer) simplification rules.

    - [Gep(Vectorize(a,b,c,...), i)] → lane [i]
    - [Gep(Const(c), _)] → [Const(c)]
    - [Gep(void, _)] → source
    - [Vcat(a, b)] → [Vectorize(Gep(a,0), ..., Gep(b,0), ...)] *)

val symbolic_simple : Kernel.t -> Kernel.t option
(** Phase 1: generic folding. Constant folding, identity removal,
    self-folding (x//x→1, x%x→0, x^x→0), where folding, idempotent ops,
    divmod reconstitution. *)

val symbolic : Kernel.t -> Kernel.t option
(** Phase 2: algebraic simplification. Includes {!symbolic_simple} plus:

    - Combine terms: [x*c0 + x*c1 → x*(c0+c1)], [x+x → x*2]
    - Associative folding: [(x + c1) + c2 → x + (c1 + c2)]
    - Nested div: [(x // c1) // c2 → x // (c1*c2)]
    - Range self-div/mod: [Range(n) % n → Range(n)]
    - Max folding via vmin/vmax
    - Range/ALU collapse when vmin==vmax
    - Lt constant folding: [c0 + x < c1 → x < c1 - c0]
    - {!Divandmod.div_and_mod_symbolic}

    Use this in substitution contexts. *)

val sym : Kernel.t -> Kernel.t option
(** Phase 3: full symbolic simplification. Includes {!symbolic} plus:

    - GEP pushing and vectorize reordering
    - [ALU(Vectorize(x,...), Vectorize(y,...))] → [Vectorize(ALU(x,y), ...)]
    - POW → [exp2(exponent * log2(base))] via {!Decomposition.xpow} *)

val split_and : Kernel.t -> Kernel.t list
(** [split_and node] flattens an AND tree into a list of conjuncts. *)

val is_irreducible : Kernel.t -> bool
(** [is_irreducible node] is [true] for Const, Vconst, Define_var,
    Special, and Range nodes. *)

val parse_valid : Kernel.t -> (Kernel.t * bool * int) option
(** [parse_valid v] parses a comparison into [(expr, is_upper_bound, c)].
    Returns [(X, true, c)] for [X <= c], [(X, false, c)] for [X >= c]. *)

val uop_given_valid : ?try_simplex:bool -> Kernel.t -> Kernel.t -> Kernel.t
(** [uop_given_valid valid uop] simplifies [uop] given that [valid] is true.
    Parses bound constraints from [valid], substitutes bounded expressions
    with proxies, simplifies, and substitutes back. *)

val pm_move_where_on_load : Kernel.t -> Kernel.t option
(** [pm_move_where_on_load node] moves WHERE conditions from around
    loads into the INDEX validity gate when the condition's ranges are
    a subset of the index's ranges. *)
