(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Splittable RNG keys and implicit key management.

    Keys are deterministic integers that can be split to derive independent
    subkeys. {!run} and {!with_key} install an effect handler that provides
    implicit key threading via {!next_key}; outside any handler a domain-local
    auto-seeded generator is used as a convenient fallback. *)

(** {1:keys Keys} *)

type key = int
(** The type for RNG keys. *)

val key : int -> key
(** [key seed] is a normalized 31-bit non-negative key derived from [seed]. *)

val split : ?n:int -> key -> key array
(** [split ?n k] deterministically derives [n] subkeys from [k].

    [n] defaults to [2]. *)

val fold_in : key -> int -> key
(** [fold_in k data] mixes [data] into [k] and returns the derived key. *)

val to_int : key -> int
(** [to_int k] is [k] as an integer. *)

(** {1:implicit Implicit key management} *)

val next_key : unit -> key
(** [next_key ()] returns a fresh subkey from the current RNG scope.

    Inside a {!run} or {!with_key} block, each call returns a deterministically
    derived key. Outside any scope, falls back to a domain-local auto-seeded
    generator (convenient but non-reproducible).

    Two calls to [next_key ()] always return different keys. *)

val run : seed:int -> (unit -> 'a) -> 'a
(** [run ~seed f] executes [f] in an RNG scope seeded by [seed].

    Every {!next_key} call within [f] returns a deterministically derived key.
    The same [seed] and the same sequence of [next_key] calls produce the same
    keys. Scopes nest: an inner [run] replaces the outer scope for its duration.
*)

val with_key : key -> (unit -> 'a) -> 'a
(** [with_key k f] executes [f] in an RNG scope initialized from [k].

    This is the explicit-key equivalent of [run]: useful when you have an
    existing key from a split and want to establish a scope for a
    sub-computation (e.g. in layer composition). *)
