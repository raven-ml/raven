(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bytes around the caller's values.

    A channel's words ({!Method}, {!Gpfifo}) and a launch's descriptor and
    parameters ({!Qmd}) are bytes whose layout the formats know, holding values
    the caller gives as {!operand}s. An integer is written in place. A {e slot},
    a value the caller numbers and knows only later, leaves a {!patch}: where
    its field lies, and how the field takes the value. A caller that knows every
    value writes the bytes as they are; one that keeps the bytes for later fills
    the patches each time ({!fill}). *)

(** The type for the values of a template. *)
type operand =
  | Int of int
      (** A value known now, an unsigned integer: an encoder raises
          [Invalid_argument] for a negative one. *)
  | Slot of int  (** The caller's value number [i], known later. *)

type patch = {
  at : int;  (** The offset of the field's word. *)
  bits : int;
      (** The field's width from bit [0] of its word, in \[[1];[64]\]. *)
  slot : int;  (** The value number that fills it. *)
  add : int64;  (** A constant added to the value, modulo [2{^64}]. *)
  shift : int;  (** A right shift of the sum, in \[[0];[63]\]. *)
}
(** The type for patches. A patch's {e word} is the little-endian word of the
    narrowest of 1, 2, 4 and 8 bytes that holds its [bits], at [at]. Filling it
    with the value [v], a 64-bit unsigned integer, replaces the word's low
    [bits] bits with those of [(v + add) lsr shift] and keeps the word's other
    bits, which belong to the fields beside it.

    For example, an address's high 32 bits in the 4-byte word at [8] are the
    patch [{at = 8; bits = 32; slot; add = 0L; shift = 32}]. *)

type t = {
  bytes : string;  (** The bytes, with zeros in the bits the patches fill. *)
  patches : patch list;
      (** The patches, by increasing [at]. Their words lie in [bytes] and do not
          overlap. *)
}
(** The type for templates. *)

val concat : t list -> t
(** [concat ts] is the bytes of [ts] one after another, with their patches moved
    with them. *)

val fill : (int -> int64) -> t -> string
(** [fill value t] is [t.bytes] with each patch filled with [value slot]. *)

(** {1:words Words} *)

val u32 : int -> operand -> t
(** [u32 s v] is a 4-byte word: [v] shifted right by [s], in \[[0];[63]\], then
    its low 32 bits.

    Raises [Invalid_argument] if [v] is a negative [Int]. *)

val u64 : int64 -> operand -> t
(** [u64 n v] is an 8-byte word: [v + n], modulo [2{^64}].

    Raises [Invalid_argument] if [v] is a negative [Int]. *)
