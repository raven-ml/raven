(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** COCO's run-length encoding: the implementation of {!Mynd.Rle}. *)

type t = private { height : int; width : int; counts : int array }

val v : height:int -> width:int -> int array -> t
val encode : Nx.host Nx.bool_t -> t
val decode : t -> Nx.host Nx.bool_t
val area : t -> int
val to_string : t -> string
val of_string : height:int -> width:int -> string -> t
