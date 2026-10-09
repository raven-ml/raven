(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Buffers: the implementation of {!Rig.Buffer}, but for {!Rig.Buffer.copy}, in
    {!Copy}. *)

open Def

type t = buffer
type memory = Rig_edge.memory = Device | Pinned | Mapped
type access = Def.access = Read | Read_write

val is_live : t -> bool
val dead : t -> string option
val access : t -> access

val check_live : string -> t -> unit
(** [check_live fn b] raises [Invalid_argument] if [b] is dead, naming the
    public function [fn], such as ["Buffer.view"]. *)

val of_memory : Def.memory -> int -> t
(** [of_memory m n] is a live buffer over the first [n] bytes of [m]. Unchecked:
    [n <= m.bytes]. *)

val create : ?memory:memory -> device -> int -> t
val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
val io : t -> 'r Type.Id.t -> 'r option
val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
val borrow : device -> t -> t option
val wait : t -> access -> unit

val share : string -> t -> unit
(** [share fn b] is {!Rig.Claim.share} under the name of the public function
    [fn]. *)

val wait_points : t -> access -> unit
(** [wait_points b access] is {!wait} for a live [b] whose memory and device
    the caller found not lost. *)

val device : t -> device
val length : t -> int
val is_borrowed : t -> bool
val view : t -> first:int -> length:int -> t
val spans : t -> bool
val overlaps : t -> t -> bool

val bigarray :
  ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

val of_string : string -> t
val blit_from_string : string -> int -> t -> int -> int -> unit
val blit_to_bytes : t -> int -> bytes -> int -> int -> unit

val blit_string : string -> int -> int -> int -> unit
(** [blit_string s i a n] copies the [n] bytes of [s] from [i] to the host
    address [a]. Unchecked. *)

val address : t -> int
val handle : t -> nativeint
val offset : t -> int
