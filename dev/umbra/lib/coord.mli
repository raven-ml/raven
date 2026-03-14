(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Celestial coordinates with frame transforms and catalog matching.

    Positions are stored as longitude/latitude pairs in 1D {!Unit.angle}
    quantities and can be converted between {!ICRS}, {!Galactic},
    {!Ecliptic_j2000}, and {!Supergalactic} frames via 3x3 rotation matrices.

    {[
      let c = Coord.of_radec ~ra:(Unit.Angle.of_deg ra) ~dec:(Unit.Angle.of_deg dec) in
      let gal = Coord.galactic c
    ]} *)

(** {1:types Types} *)

(** The type for celestial reference frames. *)
type frame =
  | ICRS  (** International Celestial Reference System. *)
  | Galactic  (** IAU Galactic coordinates. *)
  | Ecliptic_j2000  (** Ecliptic coordinates at J2000.0 epoch. *)
  | Supergalactic  (** Supergalactic coordinates. *)

type t
(** The type for celestial coordinates. A pair of 1D angle quantities
    (longitude, latitude), tagged with a {!frame}. *)

(** {1:constructors Constructors}

    All constructors require 1D angle quantities of equal length.

    Raises [Invalid_argument] if the tensors are not 1D or differ in length. *)

val of_radec : ra:Unit.angle Unit.t -> dec:Unit.angle Unit.t -> t
(** [of_radec ~ra ~dec] is a coordinate in the ICRS frame. [ra] and [dec] must
    be scalar or 1-D angle quantities with matching sizes. *)

val of_galactic : l:Unit.angle Unit.t -> b:Unit.angle Unit.t -> t
(** [of_galactic ~l ~b] is a coordinate in the Galactic frame. [l] and [b] must
    be scalar or 1-D angle quantities with matching sizes. *)

val of_ecliptic_j2000 : lon:Unit.angle Unit.t -> lat:Unit.angle Unit.t -> t
(** [of_ecliptic_j2000 ~lon ~lat] is a coordinate in the ecliptic frame at the
    J2000.0 mean obliquity (23.4392911 degrees). [lon] and [lat] must be scalar
    or 1-D angle quantities with matching sizes. *)

val of_supergalactic : sgl:Unit.angle Unit.t -> sgb:Unit.angle Unit.t -> t
(** [of_supergalactic ~sgl ~sgb] is a coordinate in the Supergalactic frame.
    [sgl] and [sgb] must be scalar or 1-D angle quantities with matching sizes.
*)

(** {1:accessors Accessors} *)

val frame : t -> frame
(** [frame c] is the reference frame of [c]. *)

val size : t -> int
(** [size c] is the number of positions in [c]. *)

val lon : t -> Unit.angle Unit.t
(** [lon c] is the longitude component of [c]. *)

val lat : t -> Unit.angle Unit.t
(** [lat c] is the latitude component of [c]. *)

val ra : t -> Unit.angle Unit.t
(** [ra c] is the ICRS right ascension of [c]. Converts to ICRS first if [c] is
    in another frame. *)

val dec : t -> Unit.angle Unit.t
(** [dec c] is the ICRS declination of [c]. Converts to ICRS first if [c] is in
    another frame. *)

(** {1:transforms Frame transforms} *)

val to_frame : frame -> t -> t
(** [to_frame f c] is [c] converted to frame [f]. Returns [c] unchanged if [c]
    is already in [f]. All conversions go through ICRS as the pivot frame. Not
    differentiable (scalar-level rotation matrices). *)

val icrs : t -> t
(** [icrs c] is [to_frame ICRS c]. *)

val galactic : t -> t
(** [galactic c] is [to_frame Galactic c]. *)

val ecliptic_j2000 : t -> t
(** [ecliptic_j2000 c] is [to_frame Ecliptic_j2000 c]. *)

val supergalactic : t -> t
(** [supergalactic c] is [to_frame Supergalactic c]. *)

(** {1:separation Angular separation} *)

val separation : t -> t -> Unit.angle Unit.t
(** [separation a b] is the angular separation between corresponding positions
    of [a] and [b], computed with the Vincenty formula. Both coordinates are
    converted to ICRS before computation. Not differentiable (scalar-level
    trigonometry).

    Raises [Invalid_argument] if [a] and [b] differ in {!size}. *)

val position_angle : t -> t -> Unit.angle Unit.t
(** [position_angle a b] is the position angle from [a] to [b], measured North
    through East, in \[0, 2{e pi}). Both coordinates are converted to ICRS
    before computation. Not differentiable (scalar-level trigonometry).

    Raises [Invalid_argument] if [a] and [b] differ in {!size}. *)

(** {1:offsets Offset operations} *)

val offset_by :
  position_angle:Unit.angle Unit.t -> separation:Unit.angle Unit.t -> t -> t
(** [offset_by ~position_angle ~separation c] is the coordinate obtained by
    moving each position in [c] along bearing [position_angle] (North through
    East) by angular distance [separation]. The result is in the same frame as
    [c]. Not differentiable (scalar-level trigonometry). *)

val spherical_offsets_to : t -> t -> Unit.angle Unit.t * Unit.angle Unit.t
(** [spherical_offsets_to a b] is [(dlon, dlat)] where
    [dlon = (lon_b - lon_a) * cos(lat_a)] and [dlat = lat_b - lat_a]. Both
    coordinates must be in the same frame. Not differentiable (scalar-level
    trigonometry).

    Raises [Invalid_argument] if [a] and [b] differ in {!size} or {!frame}. *)

(** {1:matching Catalog cross-matching}

    Matches positions between catalogs using a 3D kd-tree built from unit-sphere
    Cartesian coordinates. All indices in results are 0-based.

    {b Warning.} Cross-matching is not differentiable: it produces integer
    indices and uses discrete tree search. *)

type coord = t
(** Alias for {!t}, used inside {!Index} to avoid shadowing. *)

type result = {
  indices : Nx.int32_t;  (** 0-based indices into the catalog. *)
  separations : Unit.angle Unit.t;  (** Angular distances. *)
}
(** The type for nearest-match results. For each query position, {!indices}
    gives the index of the nearest catalog entry and {!separations} gives the
    angular distance to it. Both have the same length as the query. *)

type within_result = {
  indices_a : Nx.int32_t;  (** 0-based indices into the query. *)
  indices_b : Nx.int32_t;  (** 0-based indices into the catalog. *)
  separations : Unit.angle Unit.t;  (** Angular distances. *)
}
(** The type for within-radius match results. Each entry represents one matched
    pair. The three fields have equal length. *)

(** {2:index Reusable index}

    Build a kd-tree once and query it many times. *)

module Index : sig
  type t
  (** The type for a prebuilt spatial index over a catalog. *)

  val of_coord : coord -> t
  (** [of_coord c] builds a kd-tree index from the positions in [c]. Coordinates
      are converted to ICRS internally. *)

  val nearest : t -> coord -> result
  (** [nearest idx query] finds, for each position in [query], the nearest
      position in the indexed catalog. *)

  val within : t -> coord -> max_sep:Unit.angle Unit.t -> within_result
  (** [within idx query ~max_sep] finds all pairs where a position in [query] is
      within [max_sep] of a position in the indexed catalog. *)
end

val nearest : t -> t -> result
(** [nearest query catalog] finds, for each position in [query], the nearest
    position in [catalog].

    Raises [Invalid_argument] if [catalog] is empty. *)

val within : t -> t -> max_sep:Unit.angle Unit.t -> within_result
(** [within a b ~max_sep] finds all pairs of positions where the separation is
    at most [max_sep]. Builds a kd-tree on [b]. *)
