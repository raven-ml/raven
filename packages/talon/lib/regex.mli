(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Regular expressions over UTF-8 text: the language of [Expr.Str.regex], a
    compiled program of it and a Pike VM over that program.

    A program is immutable. A {!run} holds the buffers of one search and is
    used by one kernel at a time, so a program may be shared between domains
    while each kernel has its own {!run}. Matching a row of [n] bytes takes
    time at most proportional to [m * n], for a program of [m] instructions. *)

type t
(** A compiled regular expression. *)

val compile : string -> t
(** [compile r] is the program of the pattern [r], in the language of
    [Expr.Str.regex].

    Raises [Invalid_argument], naming a byte of [r], if [r] is not valid UTF-8
    or not a pattern of the language. *)

val source : t -> string
(** [source r] is the pattern text [r] was compiled from. *)

val quote : string -> string
(** [quote s] is a pattern that matches the text [s] literally. *)

val groups : t -> int
(** [groups r] is the number of capture groups of [r], the whole match not
    counted. *)

(** The instructions below are the compiled form of [t], read by the lazy DFA
    of {!Regex_dfa} and by {!sparse_next}. *)

type assertion =
  | Start_text
  | End_text
  | Start_line
  | End_line
  | Word_boundary
  | Not_word_boundary

type inst =
  | Sparse of int array
      (** Ascending [(lo, hi, next)] triples: a byte in [\[lo, hi\]] goes to
          [next]. An empty array matches nothing, as an empty class must. *)
  | Split of int * int  (** Prefer the first branch. *)
  | Save of int * int  (** Save the position in a slot, then go on. *)
  | Assert of assertion * int  (** Go on where the assertion holds. *)
  | Accept

val program : t -> inst array
(** [program r] is [r]'s instructions. *)

val sparse_next : int array -> int -> int
(** [sparse_next a b] is the instruction the ranges [a] of a {!Sparse} send the
    byte [b] to, or [-1]. *)

val start : t -> int
(** [start r] is the instruction a search begins at: the match, or a skip of
    whole scalars ahead for a pattern that may start anywhere. *)

val dfa_handles : t -> bool
(** [dfa_handles r] is [true] iff [r] uses neither [\b], [\B] nor the
    [(?m)] anchors, which the lazy DFA does not resolve. *)

val required : t -> string
(** [required r] is a literal every match of [r] contains, or [""] where none
    is known. *)

(** {1 Matching} *)

type run
(** The buffers of a search of one program. *)

val run : t -> slots:int -> run
(** [run r ~slots] is a run keeping the positions of the first [slots / 2]
    groups, the whole match counted first; [slots] is even, at most
    [2 * (groups r + 1)]. A run with [slots = 0] reports only whether a match
    exists. *)

val find : run -> Scalars.buf -> first:int -> from:int -> stop:int -> bool
(** [find run v ~first ~from ~stop] is [true] iff the leftmost-first match of
    the program lies in bytes [\[from, stop)] of the row [\[first, stop)] of
    [v], [from] a scalar boundary of the row. Assertions see the whole row, and
    the match starts at or after [from]. Allocates nothing. *)

val group : run -> int -> int
(** [group run i] is the byte the group slot [i] holds after a successful
    {!find}, or [-1] where the slot's group did not participate. Slot [0] is
    the start of the whole match, slot [1] its end, slot [2 * g] the start of
    group [g]. *)

(** {1 Templates} *)

type template
(** A replacement text: literal text with group references. *)

val template : t -> string -> template
(** [template r by] is [by] read as a replacement text for [r]: [$n],
    [${n}], [$name], [${name}] for a group, [$0] for the whole match, [$$] for
    a dollar. A digit run is a group number and an identifier a name, so
    [$1_y] is group 1 and then [_y].

    Raises [Invalid_argument] naming the text if [by] holds a [$] that starts
    none of these, or names a group [r] does not have. *)

val literal_template : string -> template
(** [literal_template by] writes the text [by], whatever the match. *)

val template_slots : template -> int
(** [template_slots t] is the slots a {!run} must keep for [t]. *)

val expand_size : template -> int array -> int -> int
(** [expand_size t spans base] is the bytes [t] writes for the match whose slots
    start at index [base] of [spans]. *)

val expand :
  template -> int array -> int -> Scalars.buf -> Scalars.buf -> int -> int
(** [expand t spans base v b at] writes [t], reading its groups from [spans]
    at [base] and the bytes of [v], to [b] from [at], and is the next byte. *)
