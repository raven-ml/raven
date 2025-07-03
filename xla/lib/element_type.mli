(** XLA Element Types

    These correspond to the primitive types supported by XLA. *)

type t =
  | Invalid
  | Pred  (** Boolean/predicate type *)
  | S8  (** Signed 8-bit integer *)
  | S16  (** Signed 16-bit integer *)
  | S32  (** Signed 32-bit integer *)
  | S64  (** Signed 64-bit integer *)
  | U8  (** Unsigned 8-bit integer *)
  | U16  (** Unsigned 16-bit integer *)
  | U32  (** Unsigned 32-bit integer *)
  | U64  (** Unsigned 64-bit integer *)
  | F16  (** 16-bit floating point *)
  | F32  (** 32-bit floating point *)
  | F64  (** 64-bit floating point *)
  | BF16  (** Brain 16-bit floating point *)
  | C64  (** Complex 64-bit (2x F32) *)
  | C128  (** Complex 128-bit (2x F64) *)
  | Tuple  (** Tuple type *)
  | Opaque_type  (** Opaque type *)
  | Token  (** Token type *)

val to_int : t -> int
(** Convert element type to XLA's internal integer representation *)

val of_int : int -> t
(** Convert from XLA's internal integer representation *)

val to_string : t -> string
(** String representation of element type *)
