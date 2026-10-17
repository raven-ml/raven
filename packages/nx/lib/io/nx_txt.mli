(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Delimited text of tensors, the implementation of {!Nx_io.save_txt} and
    {!Nx_io.load_txt}.

    Each function returns its errors as {!Error.t}, where {!Nx_io} raises
    them. A dtype or shape that text does not hold is [Unsupported_dtype]
    or [Unsupported_shape], text that does not parse or a bad argument a
    [Format_error], and a failed system call an [Io_error]. *)

val save :
  ?sep:string ->
  ?append:bool ->
  ?newline:string ->
  ?header:string ->
  ?footer:string ->
  ?comments:string ->
  out:string ->
  ('a, 'b) Nx.t ->
  (unit, Error.t) result
(** [save ~out t] is {!Nx_io.save_txt}[ out t]. *)

val load :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ?max_rows:int ->
  ('a, 'b) Nx.dtype ->
  string ->
  (('a, 'b) Nx.t, Error.t) result
(** [load dtype path] is {!Nx_io.load_txt}[ path dtype]. *)
