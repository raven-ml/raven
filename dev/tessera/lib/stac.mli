(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Microsoft's Planetary Computer (MPC): the items its STAC API lists, and
    the windows of their rasters that a warp reads.

    TESSERA's inputs are MPC's [sentinel-2-l2a] and [sentinel-1-rtc]
    collections. {!search} lists a collection's items over an area and a span
    of days, as {!Jord_stac.item}s, and chooses none of them. {!window} reads,
    from one asset of an item, the block of its raster that {!Jord.Warp}
    reads to put it on a grid, by byte ranges: it fetches the tiles the block
    covers, never the whole file.

    Every request runs the [curl] program on the [PATH], to [https://] URLs
    only, and is sent again, with backoff, on a status 429 or 5xx or a
    transient failure, waiting at most a minute in all. MPC's data live on
    Azure Blob Storage, read with SAS tokens that MPC's token endpoint gives
    anyone; a {!session} holds them. Failures name the href, never the
    token. STAC search needs no token, so {!search} takes no session. *)

(** {1:search Search} *)

type date = int * int * int
(** The type for days of UTC's calendar: year, month and day. *)

val search :
  collection:string ->
  bbox:float * float * float * float ->
  date * date ->
  Jord_stac.item list
(** [search ~collection ~bbox (first, last)] is every item of [collection],
    such as ["sentinel-2-l2a"], whose footprint meets [bbox] and whose time
    lies in the days [first] to [last], both included, in the order MPC's
    STAC API returns them. [bbox] is (west, south, east, north) in degrees
    of longitude and latitude (WGS 84); west above east crosses the
    antimeridian. The API pages its answer: each page's [next] link is
    followed, with the body it states, merged into the request's when it
    says [merge], until a page has none.

    Raises [Invalid_argument] if a date is not a day of the calendar,
    [last] precedes [first], or [bbox] holds a number that is not finite or
    south above north; and [Failure] naming the request if an answer is not
    JSON, a page is not a FeatureCollection {!Jord_stac.items} reads, or a
    [next] link repeats a request of the search. *)

(** {1:windows Windows} *)

type session
(** The type for sessions: the tokens a program holds. An href whose host
    is one of MPC's blob hosts, [sentinel2l2a01.blob.core.windows.net] and
    [sentinel1euwestrtc.blob.core.windows.net], compared exactly, is read
    with the SAS token for its account and container, signed again on each
    attempt. The session keeps a token until a minute before it expires by
    the local clock, and asks for a new one after an answer of status 403.
    Any other href is read unsigned. A session may be shared between
    domains. *)

val session : unit -> session
(** [session ()] holds no token. *)

exception Changed of string
(** The exception for a file that changed while it was read, with its href.
    Reading the window again reads the new version. *)

type window = {
  row : int;
  col : int;  (** The block's first cell in the raster. *)
  samples : Nx.host Nx.packed;
      (** The block, [[rows; cols; c]] at the raster's dtype. *)
  points : Nx.host Nx.float64_t;
      (** The points as pixel points of [samples], of the points' shape. *)
}
(** The type for windows of a raster. *)

val window :
  session ->
  Jord.Warp.rule ->
  Jord_stac.asset ->
  Jord.Crs.t ->
  Nx.host Nx.float64_t ->
  window
(** [window s rule asset crs c] is the block of [asset]'s raster that [rule]
    reads at the points [c] of [crs], a grid's centres
    ({!Jord.Warp.centres}[ ~at:Nx.Host.on crs g]), read with [s]'s tokens:
    [rule]'s sampler on [samples] at [points] gives the values and the mask
    it gives on the whole raster at [c], bit for bit ({!Jord.Warp.window}).
    [c] depends only on the grid and [crs], so one serves every asset in
    that system. The raster is the file's first IFD, a cloud-optimised
    GeoTIFF's full resolution, on the grid its GeoTIFF tags state
    ({!Jord_geotiff.grid}), whose system must be [crs].

    It reads the file's first 64 KiB, which hold the IFDs of every file MPC
    was measured to serve, then the ranges {!Nx_io.decode_tiff} and
    {!Nx_io.Tiff.window} lack, until the block decodes: whole tiles, those
    less than 16 KiB apart in one request. Every range after the first is
    asked of the version the first was read from ([If-Match]).

    Raises [Changed] if the file changes while it is read, [Invalid_argument]
    if [c]'s last axis is not 2, and [Failure] naming the href if the href
    is not [https://], a request fails, an answer is not the range asked for,
    the file's tags state no grid or one in another system than [crs], or it
    does not decode as {!Nx_io.Tiff.window} would. *)

(**/**)

val token :
  session -> now:float -> string -> (unit -> string * float) -> string
(** [token s ~now key fetch] is the token [s] holds for [key] that expires
    more than a minute after [now] (Unix time), or else [t] of
    [fetch () = (t, expiry)], which [s] then holds for [key]. If meanwhile
    another domain stored a token for [key] valid at [now], that one is
    returned. For tests. *)
