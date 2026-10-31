(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Delimited text. {!Nx_io.save_txt} and {!Nx_io.load_txt} state the format. *)

val save : sep:string -> string -> ('v, 's, 'd) Nx.t -> unit
val encode : sep:string -> ('v, 's, 'd) Nx.t -> string

val load :
  sep:string ->
  comments:string ->
  skiprows:int ->
  ('v, 's) Nx_array.Dtype.t ->
  string ->
  ('v, 's, Nx.host) Nx.t
