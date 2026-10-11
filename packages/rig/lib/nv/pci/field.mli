(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fields of NVIDIA's structures and registers, as [Defs] gives them.

    A structure's field is its (offset, bytes), little-endian; a register's
    field is its (lowest bit, bits). *)

(** {1:structures Structures} *)

val set : Bytes.t -> int * int -> int -> unit
(** [set b f x] writes the low bytes of [x], sign-extended to 64 bits, to the
    field [f] of [b], of 1 to 8 bytes: a negative [x] fills an 8-byte field's
    top byte with [0xff]. *)

val get : string -> int * int -> int
(** [get s f] reads the field [f] of [s], of 1 to 8 bytes: unsigned for fewer
    than 8, and for 8 the value [Int64.to_int] gives, its top bit dropped. *)

val pset : Rig_nv.params -> int * int -> int -> unit
(** [pset p f x] is {!set} on the parameters [p] of an RM call. *)

val pget : Rig_nv.params -> int * int -> int
(** [pget p f] is {!get} on the parameters [p] of an RM call. *)

val at : int * int -> int * int -> int * int
(** [at s f] is the field [f] of the structure [s], itself a field. *)

val record : int -> (Bytes.t -> unit) -> string
(** [record n fill] is [n] zeroed bytes that [fill] wrote to. *)

(** {1:registers Registers} *)

val mask : int * int -> int
(** [mask f] is the bits of the field [f]. *)

val put : int * int -> int -> int
(** [put f x] is [x] in the field [f], cut to its bits. *)
