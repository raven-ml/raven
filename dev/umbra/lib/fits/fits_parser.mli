(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(**/**)

(** Internal FITS parser. *)

type keyword = { key : string; value : string; comment : string }

type header = {
  keywords : keyword list;
  xtension : string;
  bitpix : int;
  naxis : int array;
  data_bytes : int;
}

type col_desc = {
  name : string;
  tform : char;
  repeat : int;
  width : int;
  tnull : int64 option;
  tscal : float;
  tzero : float;
}

val read_headers : In_channel.t -> header list
val seek_to_data : In_channel.t -> header list -> int -> int
val parse_bintable_cols : header -> col_desc list
val find_keyword : keyword list -> string -> string option
val find_keyword_int : keyword list -> string -> int option
val trim_right : string -> string
val block_size : int
val swap16 : bytes -> int -> unit
val swap32 : bytes -> int -> unit
val swap64 : bytes -> int -> unit

(**/**)
