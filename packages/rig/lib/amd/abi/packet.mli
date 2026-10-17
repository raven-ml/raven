(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packets, the 32-bit words a GPU's queue reads.

    An encoder ({!Pm4}, {!Aql}, {!Sdma}) describes a packet
    whose values are of any type ['v], such as integers for a driver that
    writes its ring now, or values a compiler computes later. A word the
    layout fixes is a constant. A word that holds a value, or a computation
    on one, is a {!term}. The caller interprets the description with
    {!encode} for integers, with {!template} to leave some values for later,
    or as a compiler's own nodes.

    Packets concatenate with [@]: the words of [p @ q] are [p]'s, then
    [q]'s. *)

(** {1:words Terms and words} *)

type 'v term = {
  value : 'v;  (** The value. *)
  add : int64;  (** Added to the value, modulo [2{^64}], read as unsigned. *)
  shift : int;
      (** The sum shifted right by it, with zeros shifted in, for
          [0 <= shift < 64]. *)
  set : int64;  (** The bits set in the shifted sum. *)
}
(** The type for computations on a value, as 64-bit unsigned integers: a
    term is [((value + add) lsr shift) lor set]. Every word a layout derives
    from a value takes this form: the address of a piece of a copy adds to
    it, an address a register holds from its bit 8 shifts it, and a flag
    beside a count sets bits in it. *)

val value : 'v -> 'v term
(** [value v] is [v] itself: [{ value = v; add = 0L; shift = 0; set = 0L }]. *)

(** The type for the words of a packet. *)
type 'v word =
  | Dword of int  (** A word the layout fixes: the integer's low 32 bits. *)
  | W32 of 'v term  (** A term's low 32 bits. *)
  | W64 of 'v term  (** A term's 64 bits, as two words, low first. *)

type 'v t = 'v word list
(** The type for packets, and for sequences of packets, in queue order. *)

(** The type for the comparisons of a wait. The bits a wait reads are equal
    to the reference, or at least the reference, as unsigned integers. *)
type comparison = Equal | Greater_equal

(** The type for the scope of a cache operation. [Agent] is work on this
    GPU, for which the L2 keeps memory coherent. [System] is any reader or
    writer, the host and other devices included. *)
type scope = Agent | System

(** {1:interpreting Interpreting} *)

val size : 'v t -> int
(** [size p] is the number of 32-bit words of [p]: one per [Dword] and
    [W32], two per [W64]. *)

val encode : ('v -> int64) -> 'v t -> string
(** [encode value p] is the [4 * size p] bytes of [p], each word
    little-endian, with each value [v] taken as the 64-bit unsigned integer
    [value v].

    Raises [Invalid_argument] if a shift is outside \[[0];[63]\]. *)

val template : ('v -> int64 option) -> 'v t -> string * (int * 'v word) list
(** [template known p] is [(b, holes)]. [b] is {!encode} of [p] with zeros
    in every word that holds a value [v] for which [known v] is [None], and
    [holes] lists those words in order, each with the index of its first
    32-bit word in [b]. A word whose values are all known is in [b] and not
    in [holes]. [template (fun v -> Some (value v)) p] is
    [(encode value p, [])].

    Raises [Invalid_argument] if a known word's shift is outside
    \[[0];[63]\]. *)
