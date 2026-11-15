(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The blocks of a float sum's running sums.

    Code generation adds a float sum's terms into running sums of at most
    {!chain} terms each, then adds those sums in order ({!Codegen}). Staging
    chooses how many terms a trip of a tensor core's loop takes
    ({!Stage.tile}) among the trips that leave those blocks as they are, so
    that it never moves the order of a sum. Both read the blocks here. *)

val chain : int
(** [chain] is the most terms a float sum adds into one running sum: [256]. *)

val held : int
(** [held] is the most bytes of a GPU thread's accumulators that are blocked
    into running sums of {!chain} terms: [320]. A sum whose accumulators take
    more keeps one running sum, since a nested sum holds two of each and its
    thread's registers would spill. *)

val serial : Ops.t -> bool
(** [serial r] is [true] iff the range [r] runs its trips in one thread, one
    after another: a weak range, a loop or an upcast's lanes. *)

val innermost : Ops.t list -> Ops.t list
(** [innermost rs] is the serial ranges of a sum's ranges [rs], innermost
    first: ranges nest by axis type, then by identity. *)

val cut : lanes:int -> int option list -> (int * int) option
(** [cut ~lanes trips] is where a sum whose trips add [lanes] terms each,
    over ranges of [trips] trips innermost first ([None] for a count no
    literal states), passes {!chain}: [Some (k, room)] where its [k]
    innermost ranges fit and the next fits [room] trips of theirs, [None]
    where they all fit. *)

val divisor : int -> int -> int
(** [divisor n room] is the trips of a block of a range of [n] trips whose
    blocks take at most [room] of them: [n]'s greatest divisor up to [room]
    where it is not [1] and fills at least half of [room], else [room], the
    last block padded with trips that add [+0]. *)

val block : lanes:int -> int -> int
(** [block ~lanes n] is the trips of a block of a sum's innermost range of [n]
    trips, each adding [lanes] terms ({!cut}): [n] where they all fit, [1]
    where a trip's lanes leave room for fewer than two, and {!divisor}
    otherwise. *)
