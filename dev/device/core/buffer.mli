(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffers, documented in device_core.mli. *)

open Def

type t = buffer
type memory = Device | Pinned | Mapped
type access = Read | Read_write

val check_live : string -> t -> unit
(* [check_live fn b] raises [Invalid_argument] naming [fn] if [b] is dead. *)

val of_memory : Def.memory -> Scalar.t -> int -> t
val create : ?memory:memory -> device -> Scalar.t -> int -> t
val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
val borrow : device -> t -> t option
val wait : t -> access -> unit
val device : t -> device
val dtype : t -> Scalar.t
val length : t -> int
val nbytes : t -> int
val is_borrowed : t -> bool
val view : t -> offset:int -> Scalar.t -> int -> t
val spans : t -> bool
val overlaps : t -> t -> bool

val bigarray :
  ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

val address : t -> int
val handle : t -> nativeint
val offset : t -> int
