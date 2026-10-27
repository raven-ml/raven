(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Delimited text. {!Nx_io.save_txt} and {!Nx_io.load_txt} state the format. *)

val save : sep:string -> string -> ('v, 's, 'd) Nx.t -> unit

val load :
  sep:string ->
  comments:string ->
  skiprows:int ->
  string ->
  ('v, 's) Nx_array.Dtype.t ->
  ('v, 's, Nx.host) Nx.t
