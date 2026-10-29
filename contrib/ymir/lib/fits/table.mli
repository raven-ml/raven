(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Binary and ASCII tables: the implementation of
    {!Ymir_fits.Fits.Table}. *)

(** {!Ymir_fits.Fits.Table.layout}. *)
type layout = Array of int array | Lists | Text of int array

type column = {
  name : string;
  element : Nx.Dtype.any;
  layout : layout;
  scaled : bool;
  cards : Header.t;
}
(** {!Ymir_fits.Fits.Table.column}. *)

type t
(** {!Ymir_fits.Fits.Table.t}. *)

val of_hdu : Hdu.t -> (t, string) result
val rows : t -> int
val columns : t -> column list
val pp : Format.formatter -> t -> unit

val raw :
  ?rows:int * int ->
  ('a, 'b) Nx.dtype ->
  string ->
  Hdu.t ->
  (('a, 'b, Nx.host) Nx.t, string) result

val values :
  ?rows:int * int ->
  (float, 'b) Nx.dtype ->
  string ->
  Hdu.t ->
  ((float, 'b, Nx.host) Nx.t, string) result

val ragged :
  ?rows:int * int ->
  ('a, 'b) Nx.dtype ->
  string ->
  Hdu.t ->
  (('a, 'b, Nx.host) Nx_ragged.t, string) result

val validity :
  ?rows:int * int -> string -> Hdu.t -> (Nx.host Nx.bit_t option, string) result

val unit :
  scope:string -> string -> Hdu.t -> (Ymir_units.Unit.t option, string) result
(** [unit ~scope name hdu] is {!Ymir_fits.Fits.Table.unit}[ name hdu], its
    data-set symbols scoped by [scope]. *)

(** {!Ymir_fits.Fits.Table.data}. *)
type data =
  | Array of { values : Nx.host Nx.packed; validity : Nx.host Nx.bit_t option }
  | Lists : {
      values : ('a, 'b, Nx.host) Nx_ragged.t;
      validity : Nx.host Nx.bit_t option;
    }
      -> data
  | Text of (int, Nx.int8_unsigned_elt, Nx.host) Nx_ragged.t

val read :
  ?rows:int * int -> Hdu.t -> ((string * Header.t * data) list, string) result

val hdu : Header.t -> (string * Header.t * data) list -> (Hdu.t, string) result
