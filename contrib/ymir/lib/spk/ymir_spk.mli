(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SPK files: the positions of the Sun, the Moon and the planets from JPL's
    ephemerides.

    An {e SPK} file, NAIF's format for ephemerides (SPK and DAF Required
    Reading), is a list of {e segments}. A segment gives the state, the
    position and velocity, of a {e target} body relative to a {e centre}
    body over an interval of TDB, on the axes of a frame. Bodies are NAIF
    ids: 0 the solar system's barycentre, 1 to 9 the barycentres of Mercury
    to Pluto's systems, 10 the Sun, 199 Mercury, 299 Venus, 301 the Moon,
    399 the Earth. JPL's planetary ephemerides, such as DE440 and DE441,
    hold one segment per body, the Moon and the Earth from their barycentre
    3 and the barycentres from 0.

    {!Spk.read} reads a file's segments of types 2 and 3, Chebyshev
    polynomials of the position, and of the position and velocity, which
    JPL's planetary ephemerides use. At each instant a body's state comes
    from the latest segment in the file that names it as target and covers
    the instant, as NAIF orders them, whatever its centre. The state of a
    target from a centre chains such segments: each body to its segment's
    centre, up to a body both chains reach, so the Moon from the Earth is
    the Moon from 3 less the Earth from 3.

    Each state is a formula of nx operations over the instants' seconds,
    batched over them and differentiated by rune, its velocity the
    position's derivative within 1e-12 relative for type 2. It is within
    1e-15 of the vector's norm, per component, of the state NAIF's CSPICE
    toolkit reads from the same file. States are geometric: no light time
    and no aberration.

    The Moon's position from the Earth at TDB instants [tdb], in km, read
    from the file once:

    {[
    let de440 =
      match Spk.read ~at:Nx.Host.on "de440.bsp" with
      | Ok t -> t
      | Error e -> failwith e

    let moon tdb = Spk.position de440 ~target:301 ~center:399 tdb
    ]} *)

module Spk : sig
  type segment = {
    name : string;  (** The segment's name. *)
    target : int;  (** The body whose state it gives. *)
    center : int;  (** The body it gives the state from. *)
    frame : int;  (** The frame's NAIF id: 1 is J2000. *)
    kind : int;  (** The SPK type of its data. *)
    start : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
        (** The first instant it covers. *)
    stop : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
        (** The last instant it covers. *)
  }
  (** The type for segments' descriptors. *)

  type 'd t
  (** The type for SPK files' segments, their data on the device set ['d]. *)

  val read : at:'d Nx.Placement.t -> string -> ('d t, string) result
  (** [read ~at path] is the segments of the SPK file [path], in file order,
      the data of those of types 2 and 3 placed at [at]. A segment of
      another type reads as its descriptor.

      The result is [Error] starting with [path] if the file cannot be read;
      if its ID word, ND and NI, or binary format differ from an SPK DAF's
      ["DAF/SPK "], 2 and 6, ["LTL-IEEE"] or ["BIG-IEEE"]; if its summary
      records, from FWARD on, do not form a list of the file's records, each
      of 0 to 25 summaries; or if a segment's words lie outside the file, or
      its directory, INIT, INTLEN above 0, RSIZE and N, does not fit them,
      or its epochs are not within 2{^62} s of J2000. *)

  val of_string :
    at:'d Nx.Placement.t -> name:string -> string -> ('d t, string) result
  (** [of_string ~at ~name s] is {!read} of the file whose bytes are [s].
      [name] starts its errors. *)

  val segments : 'd t -> segment list
  (** [segments t] is [t]'s segments' descriptors, in file order. *)

  val state :
    'd t ->
    target:int ->
    center:int ->
    (Ymir.Time.tdb, 'd) Ymir.Time.t ->
    (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t
    * (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t
  (** [state t ~target ~center tdb] is [(p, v)], [target]'s position in km
      and velocity in km/s from [center] at the instants [tdb], each
      [[...; 3]] on ICRS's axes, which J2000's are in JPL's ephemerides.
      A NaN instant gives NaN.

      Each choice of a centre at each body of the two chains is evaluated,
      so the work grows as the product of the numbers of centres per body:
      one per body in JPL's ephemerides, more in a file that layers
      segments of one body from several centres.

      Raises [Invalid_argument] if no segment names [target] or [center],
      as in ["Spk.state: de440.bsp has no segment of body 499"], if the two
      chain to no common body, if a body's segments chain back to it, even
      over disjoint intervals, which NAIF reads, or if a segment of a body
      on the chains is of a type other than 2 and 3 or on a frame other
      than 1, and where the segments serving an instant chain the two to no
      common body. *)

  val position :
    'd t ->
    target:int ->
    center:int ->
    (Ymir.Time.tdb, 'd) Ymir.Time.t ->
    (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t
  (** [position t ~target ~center tdb] is {!state}'s position. *)
end
