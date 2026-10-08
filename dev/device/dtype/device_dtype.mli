(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Storage formats of buffer elements. *)

(** Storage formats. *)
module Scalar : sig
  (** The type for storage formats. *)
  type t =
    | Float16
    | Float32
    | Float64
    | BFloat16
    | Float8_e4m3
    | Float8_e5m2
    | Float8_e4m3fnuz
        (** float8 with 4 exponent and 3 fraction bits, exponent bias 8, no
            infinities, no negative zero and one NaN, [0x80]. *)
    | Float8_e5m2fnuz
        (** float8 with 5 exponent and 2 fraction bits, exponent bias 16, no
            infinities, no negative zero and one NaN, [0x80]. *)
    | Int4
    | UInt4
    | Int8
    | UInt8
    | Int16
    | UInt16
    | Int32
    | UInt32
    | Int64
    | UInt64
    | Complex64
    | Complex128
    | Bool
    | Bit

  val of_bigarray_kind : ('a, 'b) Bigarray.kind -> t option
  (** [of_bigarray_kind k] is the format of [k]'s elements: {!UInt8} for [Char].
      It is [None] for [Int] and [Nativeint], whose width depends on the
      platform. *)

  val bitsize : t -> int
  (** [bitsize s] is the size in bits of one element of [s]: [1] for {!Bit}, [4]
      for {!Int4} and {!UInt4}, and [8] for {!Bool}. *)

  val to_string : t -> string
  (** [to_string s] is the lowercase name of [s], its constructor's name. *)

  val equal : t -> t -> bool
  (** [equal s0 s1] is [true] iff [s0] and [s1] are the same format. *)
end
