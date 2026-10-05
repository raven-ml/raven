(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Measured constants, as {!Ymir_units.Constant} documents them. *)

type t

val v : name:string -> string -> Unit.t -> t
val name : t -> string
val unit : t -> Unit.t
val quantity : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t Quantity.t
val uncertainty : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t Quantity.t
val pp : Format.formatter -> t -> unit
