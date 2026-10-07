(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Packets: the 32-bit words a channel runs, around the caller's values.

    A channel runs pushbuffers, sequences of 32-bit words ({!Method}). An
    encoder describes them over values of any type ['v]: integers for a driver
    that writes its ring now, or values a compiler computes later. A word the
    layout knows is a constant; a word that holds a value, or a computation on
    one, is a {!term}. The caller interprets the description: {!encode} with
    integers, {!template} with some values left for later, a compiler as its own
    nodes.

    Packets concatenate with [@]: the words of [p @ q] are [p]'s, then [q]'s. *)

(** {1:words Terms and words} *)

(** The type for computations on a value, as 64-bit unsigned integers.

    A term lists each operation a layout applies, in the order it applies them:
    an address shifted for its field, then shifted again for its high word, is
    two [Shift] nodes, and a shift by [0] is a node too. An interpreter applies
    every node as it stands, never omitting one or merging two, so that what it
    builds of a term mirrors the term. *)
type 'v term =
  | Value of 'v  (** The value. *)
  | Add of 'v term * int64
      (** [Add (t, n)] is [t + n], modulo [2{^64}], [n] read as unsigned. *)
  | Shift of 'v term * int
      (** [Shift (t, n)] is [t] shifted right by [n], with zeros shifted in,
          [0 <= n < 64]. *)

(** The type for the words of a packet. *)
type 'v word =
  | Dword of int  (** A word the layout knows: the integer's low 32 bits. *)
  | W32 of 'v term  (** A term's low 32 bits. *)
  | W64 of 'v term  (** A term's 64 bits, as two words, low first. *)

type 'v t = 'v word list
(** The type for packets, and for sequences of packets, in channel order. *)

(** {1:interpreting Interpreting} *)

val eval : ('v -> int64) -> 'v term -> int64
(** [eval value t] is the integer [t] computes, each value [v] taken as the
    64-bit unsigned integer [value v].

    Raises [Invalid_argument] if a shift is outside \[[0];[63]\]. *)

val size : 'v t -> int
(** [size p] is the number of 32-bit words of [p]: one per [Dword] and [W32],
    two per [W64]. *)

val encode : ('v -> int64) -> 'v t -> string
(** [encode value p] is the [4 * size p] bytes of [p], each word little-endian,
    with each value [v] taken as the 64-bit unsigned integer [value v].

    Raises [Invalid_argument] if {!eval} does. *)

val template : ('v -> int64 option) -> 'v t -> string * (int * 'v word) list
(** [template known p] is [(b, holes)] where [b] is {!encode} of [p] with every
    word that holds a value [v] with [known v = None] left zero, and [holes]
    those words, in order, each with the index of its first 32-bit word in [b].
    A word whose values are all known is in [b] and not in [holes].

    A caller that fills the holes later, such as a driver's ring writer, copies
    [b] and writes each hole's word at byte [4 * i]: a writer that releases a
    new value on each submission takes the template of {!Method.release} with
    the value unknown, once, and fills its hole each time.
    [template (fun v -> Some (value v)) p] is [(encode value p, [])].

    Raises [Invalid_argument] if {!eval} does on a known word. *)
