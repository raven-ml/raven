(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The STAC items that list a catalogue's acquisitions, with their files,
    grids and bands.

    A STAC {e item} (SpatioTemporal Asset Catalog, item specification 1.1) is
    one acquisition, with its time, its footprint, and its {e assets}, the
    files that hold its data, each under a {e key}, such as [B04], with a
    link ([href]) and roles. An item is
    a GeoJSON Feature, so its footprint is [Jord_geojson.polygons] of the
    same JSON, and a page of items, a FeatureCollection, gives every
    footprint in one call. This module reads the rest from the JSON the
    caller decoded ([Jsont_bytesrw.decode_string Jsont.json]) and does no
    I/O. *)

(** {1:grids Grids}

    The projection extension states an asset's grid. Its system is
    [proj:code] (["EPSG:32633"]) or, before projection 1.2, [proj:epsg]
    ([32633]), or, where no code is stated, [proj:wkt2], which
    {!Jord.Crs.of_wkt} reads (the extension sets the code to [null] when the
    system has none). A WKT 1 text in [proj:wkt2] is read as WKT 1, since
    its top keyword tells the two apart. [proj:shape] is [[rows, cols]], and
    [proj:transform] the map from a pixel's column and row to the system's
    plane,

    {[
      [ x ]   [ a0 a1 a2 ]   [ col ]
      [ y ] = [ a3 a4 a5 ] · [ row ]
      [ 1 ]   [ 0  0  1  ]   [  1  ]
    ]}

    as 6 numbers or 9, the last row included. Pixel (0, 0) is the corner of
    the first cell, as in {!Jord.Grid}, so the grid has origin (a2, a5), row
    step (a1, a4) and column step (a0, a3). *)

(** {2:precedence Where a field holds}

    A field holds where it is stated most specifically, a band's over its
    asset's and an asset's over the item's properties (projection 2.1), and
    a [null] field states that there is none. The system's three fields are
    read from the most specific object that holds any of them
    ({!section-system_fields}). The item's fields apply to every asset,
    previews and other files that are not rasters included, as the
    extension says, so a catalogue that states no override for them gives
    them the data's grid. A program must select the assets it reads by
    their keys, such as [B04] or [qa_pixel], which each catalogue chooses.
    A selection by the [data] role drops a quality layer whose roles leave
    it out. *)

(** {3:system_fields The system's fields}

    The system is one fact stated by three fields, [proj:code], [proj:epsg]
    and [proj:wkt2], so all three are read from the most specific object
    that holds any. That object's code holds where it states one, and its
    [proj:wkt2] is then not read; else its [proj:wkt2] holds. So an asset's
    [proj:wkt2] wins over its item's code, and an asset's
    [proj:code: null] alone states that it has no system, and so no
    grid. *)

(** {1:bands Bands}

    An asset's bands are read into {!Jord.Band.t}, in both of STAC's
    spellings ({!section-spellings}). STAC states no saturated value, so
    none is read. A band whose catalogue states no scale or offset decodes
    to its DNs; a Sentinel-2 product's own metadata states its offsets
    ([Jord_safe.l2a], {!Jord.Sentinel2.val-reflectance}). A band's
    wavelengths are read as the catalogue states them ({!Jord.Band}), in
    micrometres. *)

(** {2:spellings Spellings}

    STAC 1.1 lists an asset's bands in [bands], with [name], [nodata],
    [unit], [eo:common_name], [eo:center_wavelength],
    [eo:full_width_half_max], [raster:scale] and [raster:offset], and a
    field the asset states holds for each band that states none of its own.
    An asset with no [bands] that states one of those fields is read as one
    band; STAC does not state that count, so a caller pairs it with the
    file's. STAC 1.0 lists them twice, in [eo:bands] ([name],
    [common_name], [center_wavelength], [full_width_half_max]) and
    [raster:bands] ([nodata], [unit], [scale], [offset]), merged by
    index. *)

(** {2:classes Classes}

    A band with [classification:classes], or in an asset with them, is a
    [Classes] band (classification extension 1.0 to 2.0, "Class Object"),
    with at least one class, each with its [value], [name], [title],
    [description] and [nodata], in 1.x's spelling, which requires a
    description, and 2.0's, which requires a name. jord adds two rules to
    STAC's. A band states classes or bit fields and never both, since a band
    has one kind, and a value stands for one class. A catalogue's classes
    are read as it states them ({!Jord.Band}), and for a product without a
    module in jord they are the only classes stated. The program reads such
    a band's codes itself, comparing them with each class's [value] as
    {!Jord.Band} compares codes, and masks its [nodata] with
    {!Jord.Band.val-nodata}. *)

(** {2:bitfields Bit fields}

    A band with [classification:bitfields], or in an asset with them, is a
    [Bitfield] band, and its bit fields, at least one, are read beside it,
    as the catalogue's statement of its bits ({!bitfield}, {!Jord.Band}).
    The program masks such a band's [nodata] with {!Jord.Band.val-nodata},
    then reads its fields itself, with {!Nx}'s bitwise operations at each
    field's [offset] and [length] ({!Jord.Band.section-bits}). *)

(** {1:items Items} *)

type view = {
  off_nadir : float option;
      (** [view:off_nadir], "The angle from the sensor between nadir
          (straight down) and the scene center", 0 to 90. *)
  incidence_angle : float option;
      (** [view:incidence_angle], between "the vertical (normal) to the
          intercepting surface and the line of sight back to the satellite
          at the scene center", 0 to 90. *)
  azimuth : float option;
      (** [view:azimuth], "measured from the sub-satellite point (point on
          the ground below the platform) between the scene center and true
          north", clockwise from north, 0 to 360. *)
  sun_azimuth : float option;
      (** [view:sun_azimuth], the Sun's, clockwise from north at the scene
          centre, 0 to 360. *)
  sun_elevation : float option;
      (** [view:sun_elevation], the Sun's above the horizon at the scene
          centre, -90 to 90. *)
}
(** The type for view angles, in degrees, as the view extension states them
    for the scene centre, [None] where not stated. View 1.0's text names the
    fields for an item's properties, and 1.1's for its assets too, where
    1.0's schema already allows them. An asset's view is read field by
    field, the asset's over the properties', as for a grid, and a [null]
    states that there is none. Products state angles per pixel or on grids
    ({!Jord.Hls}'s angle layers, [Jord_safe.tile]); {!Jord.Sun.angles} gives
    the Sun's anywhere from the time. *)

type bitfield = {
  offset : int;  (** Its first bit, counted from the least significant, 0. *)
  length : int;  (** Its number of bits, at least 1. *)
  name : string option;  (** A short name for machines. *)
  description : string option;
  classes : Jord.Band.class_ list;
      (** What the field's values stand for, the field read as bits
          ({!Jord.Band.section-bits}). *)
}
(** The type for the bit fields a catalogue states of a band, STAC
    classification's Bit Field Object. *)

type band = {
  band : Jord.Band.t;  (** The band, as {!Jord.Band} reads it. *)
  grid : (Jord.Grid.t, string) result option;
      (** A STAC 1.1 band's own projection fields over its asset's, as for
          the asset's [grid], else the asset's [grid]. *)
  bitfields : bitfield list;
      (** Its [classification:bitfields], or its asset's, as stated: [[]]
          exactly when [band] is not a [Bitfield] band. *)
}
(** The type for an asset's bands. *)

type asset = {
  href : string;  (** The file's link, as the item writes it. *)
  media_type : string option;  (** Its [type], as ["image/tiff"]. *)
  roles : string list;  (** As ["data"], ["metadata"] or ["thumbnail"]. *)
  grid : (Jord.Grid.t, string) result option;
      (** [None] unless a shape, a transform and a system are stated
          ({!section-grids}). It is [Error], naming the field, where they are
          stated and make no grid, for a system {!Jord.Crs.of_epsg} or
          {!Jord.Crs.of_wkt} refuses, a [proj:code] outside EPSG, [proj:code]
          and [proj:epsg] naming different systems, a shape that is not two
          counts, a transform of another length or whose last row is not
          [0, 0, 1], or a map that is not invertible. An asset whose grid is
          an [Error], such as a preview in Web Mercator, leaves the item and
          its other assets readable. *)
  bands : band list;  (** Its bands, in order. *)
  view : view;
      (** Its view angles, each the asset's over the item's properties'. *)
}
(** The type for assets. *)

(** The states of a satellite's orbit, as the satellite extension (1.2)
    names them in [sat:orbit_state]. *)
type orbit_state =
  | Ascending  (** ["ascending"], a polar orbit moving north. *)
  | Descending  (** ["descending"], moving south. *)
  | Geostationary  (** ["geostationary"]. *)
  | Crossing
      (** ["crossing"], an acquisition that "spans a polar crossing (i.e.,
          the satellite transitions between ascending and descending during
          the acquisition)", which the extension adds in 1.2. *)

type item = {
  id : string;  (** Its [id], which STAC asks be unique in its collection. *)
  collection : string option;
      (** Its [collection], [None] where absent or [null]. *)
  datetime : float option;
      (** A {!Jord.Time} instant, [None] where the item states [null]. *)
  interval : (float * float) option;
      (** [start_datetime] and [end_datetime], inclusive. *)
  orbit_state : orbit_state option;
      (** [sat:orbit_state], [None] where it is absent or not one of the
          four texts the extension names. It is a field of the item alone,
          so an unknown text leaves the item and the other items
          readable. *)
  relative_orbit : int option;
      (** [sat:relative_orbit], "A count of orbits from 1 to the number of
          orbits contained in a repeat cycle", at the start of the
          acquisition, as Sentinel-1's and Sentinel-2's items state it. *)
  view : view;  (** The view angles its properties state. *)
  center_frequency : float option;
      (** [sar:center_frequency], "The center frequency of the instrument",
          in gigahertz (SAR extension 1.0), as 5.405 for Sentinel-1's C
          band. A radar states its band by this frequency. *)
  assets : (string * asset) list;  (** By key, in the item's order. *)
  properties : Jsont.json;
      (** Every property, for the extensions' other fields, as
          [s2:mgrs_tile]. *)
}
(** The type for items. An item states [datetime], [interval] or both. *)

val item : Jsont.json -> (item, string) result
(** [item json] is the STAC item [json].

    The result is [Error], naming the field and the asset, for:
    - JSON that is not a Feature, or lacks [id], [properties] or [assets];
    - an asset without [href];
    - a [datetime] that is [null] without an interval, a [start_datetime]
      without [end_datetime] or the reverse, or a time
      {!Jord.Time.of_rfc3339} refuses;
    - a field of the wrong type, an integer an [int] cannot hold, a
      [nodata] text other than [nan], [inf] or [-inf], or [eo:bands] and
      [raster:bands] of different lengths;
    - no classes, or a class without a value or with neither a name nor a
      description;
    - no bit fields, or a bit field without an offset, a length or classes,
      of a negative offset or no bits, or with a class whose value its bits
      cannot hold;
    - a value outside the range its extension's schema allows, as a
      [sat:relative_orbit] below 1 or a view angle outside its field's;
    - by jord's rules ({!section-classes}), a value stated twice in one
      band's classes, or a band that states both classes and bit fields. *)

val items : Jsont.json -> (item list, string) result
(** [items json] is the items of a FeatureCollection, as a STAC API search
    returns a page of them, in order.

    The result is [Error] for JSON that is not a FeatureCollection or lacks
    [features], and, naming the item, for an item {!val-item} refuses. *)
