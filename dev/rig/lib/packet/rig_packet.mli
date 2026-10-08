(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packets: the 32-bit words a queue reads, around values known later.

    A GPU's queue reads packets, runs of 32-bit words. An encoder describes them
    over values of any type ['v]: integers for a driver that writes its ring
    now, or values a compiler or a loader computes later. A word the layout
    knows is a constant; a word that holds a value, or a computation on one, is
    a {!term}. The caller interprets the description: {!encode} with every value
    known, {!template} with some left for later, {!load} as a C template a
    writer fills, a compiler as its own nodes.

    Packets concatenate with [@]: the words of [p @ q] are [p]'s, then [q]'s. *)

(** {1:words Terms and words} *)

(** The type for computations on a value, as 64-bit unsigned integers.

    A term lists each operation a layout applies, in the order it applies them:
    the address of a piece of a copy is an [Add] node, an address held from its
    bit 8 a [Shift] node, and a shift by [0] is a node too. An interpreter
    applies every node as it stands, never omitting one or merging two, so that
    what it builds of a term mirrors the term. *)
type 'v term =
  | Value of 'v  (** The value. *)
  | Add of 'v term * int64
      (** [Add (t, n)] is [t + n], modulo [2{^64}], [n] read as unsigned. *)
  | Shift of 'v term * int
      (** [Shift (t, n)] is [t] shifted right by [n], with zeros shifted in,
          [0 <= n <= 63]. *)
  | Or of 'v term * int64  (** [Or (t, n)] is the bitwise or of [t] and [n]. *)

(** The type for the words of a packet. *)
type 'v word =
  | Dword of int  (** A word the layout knows: the integer's low 32 bits. *)
  | W32 of 'v term  (** A term's low 32 bits. *)
  | W64 of 'v term  (** A term's 64 bits, as two words, low first. *)

type 'v t = 'v word list
(** The type for packets, and for sequences of packets, in queue order. *)

(** {1:interpreting Interpreting} *)

val size : 'v t -> int
(** [size p] is the number of 32-bit words of [p]: one per [Dword] and [W32],
    two per [W64]. *)

val encode : ('v -> int64) -> 'v t -> string
(** [encode value p] is the [4 * size p] bytes of [p], each word little-endian,
    with each value [v] taken as the 64-bit unsigned integer [value v].

    Raises [Invalid_argument] if a shift is outside \[[0];[63]\]. *)

val template : ('v -> int64 option) -> 'v t -> string * (int * 'v word) list
(** [template known p] is [(b, holes)] where [b] is {!encode} of [p] with every
    word that holds a value [v] with [known v = None] left zero, and [holes]
    those words, in order, each with its word index: the index of its first
    32-bit word in [b], whose bytes start at [4 * i]. A word whose values are
    all known is in [b] and not in [holes].

    A caller that fills the holes later, such as an eager launch, copies [b] and
    writes each hole's word at byte [4 * i].
    [template (fun v -> Some (value v)) p] is [(encode value p, [])].

    Raises [Invalid_argument] if a known word's shift is outside \[[0];[63]\].
*)

(** {1:c C templates}

    A writer in C places the same packets many times with new values: a release
    with each new value of a timeline, a copy with each piece's addresses. It
    holds each packet once as a [struct rig_template] of [rig_packet.h], which
    {!load} writes, and places it with [rig_fill (t, args, w)]: the template's
    words into [w], each hole filled from the three arguments [args], its count
    of words answered. *)

val load : nativeint -> int t -> unit
(** [load at p] writes [p] into the [struct rig_template] at address [at], each
    term's [Value i] reading argument [i] of [rig_fill]. [at] points to a
    template the caller owns, aligned for it, and no [rig_fill] reads it
    meanwhile.

    Raises [Invalid_argument] if [p] has more than 16 words, more than 6 words
    that hold a term, a term with more than 3 operations, a [Value i] with [i]
    outside \[[0];[2]\], or a shift outside \[[0];[63]\]. *)
