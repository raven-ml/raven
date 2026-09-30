(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Simplifying integer division and remainder.

    Rewrites of {!Op.Floordiv} and {!Op.Floormod} in index arithmetic into forms
    with the same values, from the bounds of the operands ({!Ops.vmin},
    {!Ops.vmax}) and their known divisors ({!Ops.const_factor}). Division rounds
    towards negative infinity, and a remainder has the sign of its divisor. *)

val div_and_mod_symbolic : (unit, Ops.t) Ops.Pattern_matcher.t
(** [div_and_mod_symbolic] rewrites divisions and remainders. With [x] and [y]
    nodes and [a], [c] and [d] constants:

    - [(x // c + a) // d] is [(x + a * c) // (c * d)] when [d] is positive;
    - for a weak integer [x] and [c % d] other than [c], [(x + c) // d] is
      [(x + c % d) // d + c // d], and [(x + c) % d] is [(x + c % d) % d].

    Any other weak integer [x // y] or [x % y] takes the first of the following
    forms that applies. When the quotient has one possible value [q], [x // y]
    is [q] and [x % y] is [x - q * y]. When [x] is a parameter declared a
    multiple of [m], and [m] is a multiple of the constant [y], [x % y] is [0]
    and [x // y] stays as it is.

    When [y] is a positive constant [c], with [x] a sum of terms and a constant:

    - [(z % (k * c)) // c] is [(z // c) % k] for a positive [k];
    - in [x % c], a term [z % m], with [m] a multiple of [c], is [z];
    - when replacing each term's factor by one of its residues modulo [c] leaves
      a sum within one period of [c], that sum gives the remainder, and the
      factors less their residues give the quotient;
    - a divisor common to [c] and every term's factor is divided out of both,
      when the quotient stays non-negative;
    - for each factor [f] of a term that divides [c], [x // c] is
      [(x // f) // (c / f)], and [x % c] its reconstruction, keeping the
      smallest result.

    Otherwise:

    - a common divisor of [y] and of the terms of [x] is divided out of both;
    - for [x] and [y] non-negative, the terms of [x] that are multiples of [y]
      leave the division.

    Raises [Division_by_zero] if a weak integer division or remainder has a
    divisor that is always [0], or a positive constant divisor and a dividend
    with a term that is the constant [0] or a product by it. The symbolic rules
    fold such a term away before these rules see it. *)
