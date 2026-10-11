(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fields of NVIDIA's structures and registers, as [Defs] gives them.

    A structure's field is its (offset, bytes), little-endian; a register's
    field is its (lowest bit, bits). *)

(** {1:structures Structures} *)

val set : Bytes.t -> int * int -> int -> unit
(** [set b f x] writes [x] to the field [f] of [b], of 1, 2, 4 or 8 bytes. *)

val get : string -> int * int -> int
(** [get s f] reads the field [f] of [s], of 1, 2, 4 or 8 bytes, unsigned but
    for 8. *)

val at : int * int -> int * int -> int * int
(** [at s f] is the field [f] of the structure [s], itself a field. *)

val record : int -> (Bytes.t -> unit) -> string
(** [record n fill] is [n] zeroed bytes that [fill] wrote to. *)

(** {1:registers Registers} *)

val mask : int * int -> int
(** [mask f] is the bits of the field [f]. *)

val put : int * int -> int -> int
(** [put f x] is [x] in the field [f], cut to its bits. *)
