(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A lazily built deterministic automaton of a {!Regex} program, answering
    only whether a row matches.

    A DFA holds the buffers of one kernel; its states are a bounded cache of
    sets of NFA states, built as rows ask for them. When the cache fills it is
    cleared and matching continues, so a row's time stays linear and the
    answer never changes. A DFA covers only a program for which
    {!Regex.dfa_handles} is [true]. *)

type t

val create : ?limit:int -> Regex.t -> t
(** [create ?limit r] is an empty DFA of [r], whose cache holds at most [limit]
    states (by default as many as a transition table of one mebiword holds).

    Raises [Invalid_argument] if [Regex.dfa_handles r] is [false] or
    [limit < 2]. *)

val matches : t -> Scalars.buf -> first:int -> stop:int -> bool
(** [matches d v ~first ~stop] is [true] iff the row [\[first, stop)] of [v]
    has a match, with the answer of a {!Regex.run} of the same program. *)
