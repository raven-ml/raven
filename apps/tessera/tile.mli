(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tiles: the observations of a grid's cells that TESSERA's preprocessing
    keeps, and the directory of files that holds them.

    A tile has [h × w] cells and holds {e frames}, each the observations of one
    sensor on one day, in day order:
    - a Sentinel-2 frame gives each cell the ten bands B04 B02 B03 B08 B8A B05
      B06 B07 B11 B12, in that order, as surface reflectance times 10⁴,
      negatives kept, and a mask, nonzero where the cell was observed;
    - a Sentinel-1 frame, of one orbit direction, gives each cell VV and VH as
      TESSERA's code of γ⁰ ({!Sentinel1.code}), both 0 where the cell was not
      observed.

    Each frame has its day of year, 1 to 366, of its UTC date.

    {b Files.} A tile's directory holds TESSERA's preprocessing's files, which
    its model reads:
    - [bands.npy], int16 [[t; h; w; 10]];
    - [masks.npy], uint8 [[t; h; w]], and [doys.npy], uint16 [[t]];
    - [sar_ascending.npy] and [sar_descending.npy], int16 [[t'; h; w; 2]], with
      [sar_ascending_doy.npy] and [sar_descending_doy.npy], int16 [[t']];
    - [roi.tiff], the tile's {!place}: the region of interest, uint8 [[h; w]],
      as a GeoTIFF of the grid's tags ({!Jord_geotiff.tags}), its nodata value
      0, and the catalogue in its ImageDescription (270).

    TESSERA's own preprocessing writes [bands.npy] as uint16, and no
    [roi.tiff]; {!load} reads its tiles too. They hold TESSERA's values: on
    the dates its rule harmonises, a DN below the offset kept as it is (DN in
    place of DN − 1000), so their values are not to be joined with this
    module's. {!Model} reads both alike. *)

type place = {
  grid : Jord.Grid.t;
  roi : Nx.host Nx.uint8_t;
      (** [[h; w]], nonzero at the cells observations are kept of. *)
  source : string;  (** The catalogue the observations come from, as ["mpc"]. *)
}
(** The type for where a tile lies: its grid, its region of interest and its
    catalogue, which a model's statistics are bound to. *)

type t = {
  place : place option;
  bands : Nx.host Nx.int16_t;
  masks : Nx.host Nx.uint8_t;
  doys : Nx.host Nx.uint16_t;
  ascending : Nx.host Nx.int16_t;
  ascending_doys : Nx.host Nx.int16_t;
  descending : Nx.host Nx.int16_t;
  descending_doys : Nx.host Nx.int16_t;
}
(** The type for tiles, each array of the shape its file has. *)

val day_of_year : int * int * int -> int
(** [day_of_year (y, m, d)] is the day of year, 1 to 366, of the day [d] of
    month [m] of year [y] of the Gregorian calendar, as a frame states it:
    {!Jord.Time.day_of_year} of its first instant.

    Raises [Invalid_argument] if it is not a day of the calendar. *)

val shape : t -> int * int
(** [shape t] is [t]'s cells [(h, w)], from [bands]. *)

val empty : ?place:place -> int * int -> t
(** [empty ?place (h, w)] is the tile of [h × w] cells with no frame.

    Raises [Invalid_argument] if [h] or [w] is negative, or [place]'s grid is
    not [h × w] cells or its [roi] not [[h; w]]. *)

val shard :
  ?place:place ->
  int * int ->
  int * int * int ->
  s2:(Nx.host Nx.int16_t * Nx.host Nx.uint8_t) option ->
  ascending:Nx.host Nx.int16_t option ->
  descending:Nx.host Nx.int16_t option ->
  t
(** [shard ?place (h, w) date ~s2 ~ascending ~descending] is the tile of
    [h × w] cells of one day's frames, each of day {!day_of_year} [date]: the
    Sentinel-2 frame [s2], its bands [[h; w; 10]] and mask [[h; w]], and the
    Sentinel-1 frame of each orbit, its codes [[h; w; 2]], none where [None].
    TESSERA's preprocessing writes a tile a day, which its stackers join
    ({!concat}).

    Raises [Invalid_argument] if a frame has another shape, and as {!empty}
    and {!day_of_year} do. *)

val save : string -> t -> unit
(** [save dir t] writes [t]'s files to [dir], made if missing, [roi.tiff] only
    if [t] has a place.

    Raises [Invalid_argument] if an array's shape is not its file's, as {!load}
    checks it. *)

val load : string -> t
(** [load dir] is the tile [dir] holds, its arrays mapping the files. A uint16
    [bands.npy] is read as int16, by its bits. A value above 32767 there is a
    saturated DN, which TESSERA's preprocessing keeps (ESA's SATURATED, 65535,
    or 64535 once its offset is subtracted): its observation is read as
    invalid, mask 0 and bands 0, since a saturated DN measures nothing
    ({!Jord.Band.decode}) and this module's own frames never hold one. A
    missing [masks.npy] gives masks of 0, a missing [doys.npy] days 0, a
    missing orbit's files no frame of it, and a missing [roi.tiff] no place.
    Cost: a pass over a uint16 [bands.npy], and a copy of its bands and masks
    when it holds a saturated DN.

    Raises [Failure] naming the file if [bands.npy] is missing, if a file's
    dtype or shape disagrees with the above, an orbit has frames but no days,
    or [roi.tiff] states no grid or another shape, and as {!Nx_io.load_npy}
    does. *)

val concat : place:place option -> string -> t list -> unit
(** [concat ~place dir ts] writes to [dir] the tile of the frames of [ts] in
    order, at [place]: each file joins its arrays along the frames' axis
    ({!Nx_io.save_npy_concat}), holding one tile's at a time, so tiles larger
    than memory are joined from their files ({!load}). A frame keeps its day, so
    [ts] are given in day order.

    Raises [Invalid_argument] if [ts] is empty, or one of its tiles has other
    cells than the first or another place than [place]. *)

val of_landmask :
  source:string -> Jord.Grid.t -> ('a, 'b, Nx.host) Nx.t -> place
(** [of_landmask ~source g mask] is the place of TESSERA's tile of the
    landmask [mask], of any dtype, [[g.rows; g.cols]] on the north-up grid
    [g]: cells of 10 m in [g]'s system, [⌈width / 10⌉] columns and
    [⌈height / 10⌉] rows of [g]'s extent, from [g]'s top-left corner moved
    west and north to multiples of 10 m, computed in float64; the region of
    interest, by index, is [mask] read onto 10 m cells laid from [g]'s own
    corner, each cell holding [1] where the cell of [mask] that holds its
    centre is above [0], as TESSERA reads a landmask, and [0] elsewhere, NaN
    included. On a grid of 10 m cells that is [mask] itself, cell for cell.
    On another, a centre on a border between two cells of [mask] takes the
    one east or south of it, and a centre beyond [mask] holds [0].

    That is where TESSERA's preprocessing puts a landmask's data
    (github.com/ucam-eo/tessera, [tessera_preprocessing/s2_fast_processor.py],
    [load_roi]): it lays 10 m cells over the landmask's bounds, reads the
    mask onto them by GDAL's nearest, and reads the bands with stackstac on
    those bounds snapped out to multiples of 10 m (stackstac, [stack],
    [snap_bounds]), keeping the first rows and columns. GeoTessera's tiles
    hold the data of this grid (github.com/ucam-eo/geotessera, pull request
    433); its GeoTIFFs and the landmasks of its npy tiles label the same
    pixels with [g], up to a cell east and south of their data.

    Raises [Invalid_argument] if [g] is not north up, or [mask] is not
    [[g.rows; g.cols]]. *)

val load_landmask : source:string -> string -> place
(** [load_landmask ~source file] is {!of_landmask} [~source g mask] of
    GeoTessera's landmask, the GeoTIFF [file]: [g] the grid its tags state
    ({!Jord_geotiff.grid}) and [mask] its first band, of any dtype, as
    TESSERA reads a landmask.

    Raises [Failure] naming [file] if its tags state no grid or
    {!of_landmask} refuses them, and as {!Nx_io.load_tiff} does. *)

val of_polygons : source:string -> Jord.Polygons.t -> place
(** [of_polygons ~source p] is the place TESSERA makes of the regions of [p], of
    any system: the cells of 10 m in the zone the UTM grid gives the centre
    of [p]'s bounds in longitude and latitude ({!Jord.Crs.utm_grid_at}, as
    TESSERA's ROI script applies the grid's exceptions), whose bounds are
    those of [p]'s points in that zone,
    moved out to multiples of 10 m; the region of interest is the cells whose
    centre lies in a region ({!Jord.Rasterize.centres}), so that it lies on
    the polygons.

    Raises [Invalid_argument] if [p] has no point. *)
