(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HTTPS through curl: JSON requests, and files read by byte ranges.

    A request runs the [curl] program on the [PATH], without a shell, and
    holds its answer in memory; nothing here speaks TLS or HTTP. Only
    [https://] URLs are requested, curl is told to speak HTTPS alone, and it
    takes the URL as the value of [--url], so no URL is read as an option.

    {b Retries.} An answer of status 429 or 5xx, or a transient failure of
    curl's own (no connection, a timeout, a reset), is followed by the same
    request again, after the [Retry-After] seconds the answer states, or else
    1, 2, 4 and 8 seconds: five attempts in all. The waits of one request
    total at most a minute: a [Retry-After] that would pass it ends the
    retries, and the answer stands. A transfer slower than one byte a second
    for a minute counts as a timeout; a slow transfer has no other bound. *)

(** {1:sessions Sessions} *)

type session
(** The type for sessions: the SAS tokens of Microsoft's Planetary Computer
    (MPC) a program holds. MPC keeps its data on Azure Blob Storage, readable
    with a shared access signature (SAS) token for the blob's storage account
    and container. A URL whose host is one of MPC's blob hosts,
    [sentinel2l2a01.blob.core.windows.net] and
    [sentinel1euwestrtc.blob.core.windows.net], compared exactly, is read
    with the token MPC's token endpoint gives for its account and container,
    signed again on each attempt. The session keeps a token until a minute
    before its [msft:expiry] by the local clock, and asks again after an
    answer of status 403. A session may be shared between domains. *)

val session : unit -> session
(** [session ()] holds no token. *)

val token :
  session -> now:float -> string -> (unit -> string * float) -> string
(** [token s ~now key fetch] is the token [s] holds for [key] that expires
    more than a minute after [now] (Unix time), or else [t] of
    [fetch () = (t, expiry)], which [s] then holds for [key]. If meanwhile
    another domain stored a token for [key] valid at [now], that one is
    returned. *)

(** {1:json JSON} *)

val get_json : string -> Jsont.json
(** [get_json url] is the JSON [url] answers to a GET.

    Raises [Failure] naming [url] if [url] is not [https://], no attempt
    succeeds, the last answer's status is not 2xx, or its body is not
    JSON. *)

val post_json : string -> Jsont.json -> Jsont.json
(** [post_json url body] is the JSON [url] answers to a POST of [body], as
    [application/json].

    Raises [Failure] as {!get_json} does. *)

(** {1:ranges Ranges} *)

exception Changed of string
(** The exception for a file that changed while it was read, with its
    href. *)

type file
(** The type for files read by byte ranges, from the version the first
    range came from. *)

val open_file : session -> string -> file * Nx_io.Partial.t
(** [open_file s href] is the file at [href], read with [s]'s tokens, and
    the partial file of its first 65536 bytes, or of all of it if it is
    shorter, named [href]. 64 KiB holds the IFDs of the cloud-optimised
    GeoTIFFs MPC serves, whose headers measure 3 to 33 KB.

    Raises [Failure] naming [href] if [href] is not [https://], no attempt
    succeeds, or the answer is neither bytes 0 onward of a stated length,
    the whole file, nor the answer to an empty file. *)

val fill : file -> Nx_io.Partial.t -> (int * int) list -> Nx_io.Partial.t
(** [fill f p rs] is [p] holding as well the ranges [rs] of [f]: [(offset,
    length)] pairs sorted by offset, as {!Nx_io.decode_tiff} and
    {!Nx_io.Tiff.window} return them. Ranges less than 16 KiB apart are
    fetched as one, the bytes between them included, so a window's
    neighbouring tiles, which GDAL's layout separates by 8 bytes, cost one
    request. If the first answer stated an ETag, each request carries it as
    [If-Match], so it reads the version the first answer did.

    Raises [Changed] if the file changed since {!open_file} (status 412 or
    another ETag), and [Failure] naming [f]'s href if a request fails or an
    answer is not exactly the range asked for, of [f]'s length. *)
