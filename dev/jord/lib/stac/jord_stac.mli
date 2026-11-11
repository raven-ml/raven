(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** STAC items: the acquisitions a catalogue lists, their files, grids and
    bands.

    A STAC {e item} (SpatioTemporal Asset Catalog, item specification 1.1) is
    one acquisition: its time, its footprint, and its {e assets}, the files
    that hold its data, each with a link ([href]) and roles. An item is a
    GeoJSON Feature, so its footprint is [Jord_geojson.polygons] of the same
    JSON, and a page of items, a FeatureCollection, gives every footprint in
    one call. This module reads the rest from the JSON the caller decoded
    ([Jsont_bytesrw.decode_string Jsont.json]) and does no I/O.

    {b Grids.} The projection extension states an asset's grid: its system by
    [proj:code] (["EPSG:32633"]) or, before projection 1.2, [proj:epsg]
    ([32633]); [proj:shape], [[rows, cols]]; and [proj:transform], the map
    from a pixel's column and row to the system's plane,

    {[
      [ x ]   [ a0 a1 a2 ]   [ col ]
      [ y ] = [ a3 a4 a5 ] · [ row ]
      [ 1 ]   [ 0  0  1  ]   [  1  ]
    ]}

    as 6 numbers or 9, the last row included. Pixel (0, 0) is the corner of
    the first cell, as in {!Jord.Grid}, so the grid has origin (a2, a5), row
    step (a1, a4) and column step (a0, a3).

    A field holds where it is stated most specifically: a band's over its
    asset's, an asset's over the item's properties (projection 2.1, "If the
    transform is defined in Item Properties, it is used as the default
    transform for all assets that don't have an overriding transform"), and
    a [null] field states that there is none. [proj:code] and [proj:epsg]
    are one field in two spellings, so both are read from the most specific
    object that holds either: an asset's [proj:code: null] states that it
    has no system, and so no grid. The item's fields apply to every asset,
    previews and other files that are not rasters included, as the
    extension says; a catalogue that states no override for them, as
    Microsoft's Planetary Computer for its [rendered_preview] and
    [tilejson], gives them the data's grid, so a caller selects data by its
    [data] role.

    {b Bands.} An asset's bands are read into {!Jord.Band.t}, in both of
    STAC's spellings. STAC 1.1 lists them in [bands], with [name], [nodata],
    [unit], [eo:common_name], [raster:scale] and [raster:offset], and a field
    the asset states holds for each band that does not state its own. An
    asset with no [bands] that states one of those fields is read as one
    band; STAC does not state that count, so a caller pairs it with the
    file's. STAC
    1.0 lists them twice, in [eo:bands] ([name], [common_name]) and
    [raster:bands] ([nodata], [unit], [scale], [offset]), merged by index.
    STAC states no saturated value, so none is read. A band with
    [classification:bitfields], or in an asset with them, is a [Bitfield]
    band. Microsoft's Planetary Computer, for one, states no scale or offset
    for Sentinel-2 L2A: its bands read here decode to DNs, and the product's
    own metadata states its offsets ([Jord_safe.l2a],
    {!Jord.Sentinel2.reflectance}). *)

(** {1:time Time} *)

type time = {
  date : int * int * int;  (** Year, month and day of UTC's calendar. *)
  seconds : float;
      (** Since that day's midnight, in [[0, 86401)]: a leap second, 60,
          reads past 86400. *)
}
(** The type for instants of UTC. *)

val time : string -> (time, string) result
(** [time s] is the instant [s] states in RFC 3339's [date-time] (§5.6),
    [YYYY-MM-DDTHH:MM:SS], an optional fraction of a second, and [Z] or an
    offset [±HH:MM], which is taken off. The separator may be [T], [t] or a
    space, as RFC 3339 allows (§5.6, NOTE), and Microsoft's Planetary
    Computer writes Sentinel-1's [start_datetime] and [end_datetime].
    [Error] for any other text, or a field out of its range. *)

(** {1:items Items} *)

type asset = {
  href : string;
  media_type : string option;  (** Its [type], as ["image/tiff"]. *)
  roles : string list;  (** As ["data"], ["metadata"] or ["thumbnail"]. *)
  grid : (Jord.Grid.t, string) result option;
      (** [None] unless a shape, a transform and a system are stated. [Error]
          names the field where they are stated and do not make a grid: a
          system {!Jord.Crs.of_epsg} refuses, a [proj:code] outside EPSG,
          [proj:code] and [proj:epsg] naming different systems, a shape that
          is not two counts, a transform of another length or whose last row
          is not [0, 0, 1], or a map that is not invertible. An asset whose
          grid is an [Error], such as a preview in Web Mercator, leaves the
          item and its other assets readable. *)
  bands : (Jord.Band.t * (Jord.Grid.t, string) result option) list;
      (** Each band with its grid: a STAC 1.1 band's own projection fields
          over its asset's, as for [grid], else the asset's [grid]. *)
}
(** The type for assets. *)

type item = {
  id : string;
  collection : string option;
  datetime : time option;  (** [None] where the item states [null]. *)
  interval : (time * time) option;
      (** [start_datetime] and [end_datetime], inclusive. *)
  assets : (string * asset) list;  (** By key, in the item's order. *)
  properties : Jsont.json;
      (** Every property, for the extensions' fields, as
          [sat:orbit_state]. *)
}
(** The type for items. An item states [datetime], [interval] or both. *)

val item : Jsont.json -> (item, string) result
(** [item json] is the STAC item [json]. [Error] names the field and the
    asset for JSON that is not a Feature, lacks [id], [properties] or
    [assets], an asset without [href], a [datetime] that is [null] without
    an interval, a [start_datetime] without [end_datetime] or the reverse, a
    time {!time} refuses, a field of the wrong type, a [nodata] text other
    than [nan], [inf] or [-inf], or [eo:bands] and [raster:bands] of different
    lengths. A grid that cannot be read is its asset's [Error], never the
    item's. *)

val items : Jsont.json -> (item list, string) result
(** [items json] is the items of a FeatureCollection, as a STAC API search
    returns a page of them, in order. [Error] names the item as {!item}
    does. *)
