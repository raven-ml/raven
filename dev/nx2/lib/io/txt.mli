(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Delimited text. {!Nx_io.save_txt} and {!Nx_io.load_txt} state the format. *)

val save :
  sep:string ->
  newline:string ->
  header:string option ->
  footer:string option ->
  comments:string ->
  string ->
  ('v, 's, 'd) Nx.t ->
  unit

val load :
  sep:string ->
  comments:string ->
  skiprows:int ->
  max_rows:int option ->
  string ->
  ('v, 's) Nx_array.Dtype.t ->
  ('v, 's, Nx.host) Nx.t
