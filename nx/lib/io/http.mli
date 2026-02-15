(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HTTP client for downloading files via the [curl] CLI.

    All functions require the [curl] binary to be available on PATH. If curl is
    not found, functions raise [Failure] with a descriptive message. *)

val download :
  ?show_progress:bool ->
  ?headers:(string * string) list ->
  url:string ->
  dest:string ->
  unit ->
  unit
(** [download ~url ~dest ()] downloads the resource at [url] to the local file
    [dest]. Creates parent directories as needed. Follows redirects.

    @param show_progress
      if [true], curl displays a progress bar on stderr (default: [false]).
    @param headers optional HTTP headers to include in the request.
    @raise Failure if the download fails or curl is not available. *)

val get : ?headers:(string * string) list -> string -> string
(** [get url] fetches [url] and returns the response body as a string.

    Follows redirects. Fails loudly on non-zero exit codes.

    @param headers optional HTTP headers to include in the request.
    @raise Failure if the request fails or curl is not available. *)
