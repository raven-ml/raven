(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Layouts: the implementation of {!Nx_array.Layout}.

    A layout is the bytes of [nx_layout] in [nx_array.h], which C reads in
    place: the representation is the C contract. *)

type t

val max_rank : int
val contiguous : int array -> t
val v : ?offset:int -> strides:int array -> int array -> t
val rank : t -> int
val dim : t -> int -> int
val stride : t -> int -> int
val offset : t -> int
val numel : t -> int
val shape : t -> int array
val strides : t -> int array
val span : t -> int * int
val is_contiguous : t -> bool
val is_distinct : t -> bool
val move : Move.t -> t -> t option
val coalesce : t array -> t array
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
