(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HTTP helpers implemented with the [curl] command-line tool. *)

val download :
  ?show_progress:bool ->
  ?headers:(string * string) list ->
  url:string ->
  dest:string ->
  unit ->
  unit
(** [download ?show_progress ?headers ~url ~dest ()] downloads [url] to [dest],
    creating parent directories when needed.

    Redirects are followed.

    Raises [Failure] if [curl] is unavailable or the transfer fails. *)

val get : ?headers:(string * string) list -> string -> string
(** [get ?headers url] fetches [url] and returns the response body.

    Redirects are followed.

    Raises [Failure] if [curl] is unavailable or the request fails. *)
