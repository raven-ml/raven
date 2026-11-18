(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS world coordinates: the implementation of {!Ymir.Wcs}. *)

type keywords = {
  float : string -> (float option, string) result;
  int : string -> (int option, string) result;
  text : string -> (string option, string) result;
}
(** {!Ymir.Wcs.keywords}. *)

(** {!Ymir.Wcs.edit}. *)
type edit =
  | Float of string * float
  | Int of string * int
  | Text of string * string
  | Remove of string

val read :
  ?alt:char ->
  at:'d Nx.Placement.t ->
  ('w, 'd) Transform.endpoint ->
  keywords ->
  (('d Transform.plane, 'w, 'd) Transform.t, string) result

val write :
  ?alt:char ->
  ?window:(int * int) array ->
  ('d Transform.plane, 'w, 'd) Transform.t ->
  keywords ->
  (edit list, string) result
