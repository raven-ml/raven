(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Prime factorisation.

    Factoring is exact and deterministic: the same integer always yields the
    same factorisation. *)

val factor : int -> (int * int) list
(** [factor n] is [n]'s prime factorisation as [(p, k)] pairs, [p] ascending and
    [k >= 1]. [factor 1] is [[]]. Requires [n >= 1].

    It divides by the primes below 2{^ 16}; a cofactor below 2{^ 32} is then
    prime, and a larger one is tested by Miller–Rabin with the first twelve
    primes as bases, which is exact below 3.3·10{^ 24}, and split by
    Pollard–Brent. *)

val factor_smooth : Nat.t -> (int * int) list option
(** [factor_smooth n] is [Some f], with [f] [n]'s prime factorisation as
    {!factor} writes it, when [n] divided by its prime factors below 2{^ 24} is
    below 2{^ 62}, and [None] otherwise. Requires [n >= 1]. Its time is bounded
    by [n]'s bit length. *)
