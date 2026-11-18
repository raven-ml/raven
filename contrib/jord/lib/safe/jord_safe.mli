(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ESA's Sentinel-2 product metadata, from the bytes of its XML.

    A Sentinel-2 product in ESA's SAFE format (Standard Archive Format for
    Europe) is a folder of images and XML metadata. The images are JPEG
    2000, which nx.io does not read. A program reads them with a reader of
    its own, or reads with [Nx_io.load_tiff] the Cloud Optimized GeoTIFF
    copies that Microsoft's Planetary Computer serves. Its main metadata file,
    [MTD_MSIL1C.xml] for a Level-1C product and [MTD_MSIL2A.xml] for a
    Level-2A one, states in [General_Info/Product_Image_Characteristics] the
    values that decode its digital numbers (ESA SentiWiki, "S2 Products",
    {{:https://sentiwiki.copernicus.eu/web/s2-products}}), which are the
    special values NODATA and SATURATED, a quantification value and an
    offset for each band, whose names and rule
    {!Jord.Sentinel2.section-reflectance} gives. An offset names its band by
    an index, [band_id], which the file's [Spectral_Information] list maps
    to the band's name. That list also states each band's wavelengths, those
    of the product's satellite ({!Jord.Sentinel2.type-wavelength}).
    Catalogues serve the file beside the images, as Microsoft's Planetary
    Computer does under the asset [product-metadata].

    ESA's Products Specification Document defines the files by XML schemas,
    [User_Product_Level-1C.xsd] and [User_Product_Level-2A.xsd] with their
    [dimap.xsd] and [dimap2A.xsd]; SNAP, ESA's toolbox, ships those of PSD 15
    ({{:https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-reader/src/main/resources/schemas/PSD150}}).
    This module relies on their rules. An offset is a non-positive integer or
    ["Not applicable"] ([RADIO_OFFSET_TYPE]), at most 13 of them, each with a
    [band_id] from ["0"] to ["12"] ([A_BAND_NUMBER]); the special values are
    exactly two, NODATA and SATURATED, each an [xs:int] ([A_SPECIAL_VALUES]);
    and the quantification value is an [xs:double]. The list of offsets entered
    the schemas at version 14.7, so an older product states none.

    A product's tile metadata states when its tile was sensed and the
    angles of the Sun and of the view, which {!val-tile} reads.

    This module reads the files from their bytes and does no I/O. *)

(** {1:products Products}

    {!l1c} and {!l2a} read a product's [PROCESSING_BASELINE], its
    quantification value, its special values, the offset of each band in
    the order the file states them, and the [Wavelength] of each band whose
    [Spectral_Information] states one. [offsets] is empty for a product that
    states no offsets, as before baseline 04.00, and leaves out an offset
    stated as ["Not applicable"]. Both refuse a text that is not their
    product's well-formed metadata, that lacks or repeats what they read, or
    whose band, offset, wavelength or quantification value is malformed
    ({!section-refusals}). *)

val l1c : string -> (Jord.Sentinel2.metadata, string) result
(** [l1c s] is the metadata of the Level-1C product whose [MTD_MSIL1C.xml] is
    [s]. That file's root element is [Level-1C_User_Product], its
    quantification value [QUANTIFICATION_VALUE] and its offsets
    [RADIO_ADD_OFFSET].

    The result is [Error] for what both readers refuse
    ({!section-refusals}). *)

val l2a : string -> (Jord.Sentinel2.metadata, string) result
(** [l2a s] is the metadata of the Level-2A product whose [MTD_MSIL2A.xml] is
    [s]. That file's root element is [Level-2A_User_Product], its
    quantification value [BOA_QUANTIFICATION_VALUE] and its offsets
    [BOA_ADD_OFFSET], [B10]'s included, though Level-2A holds no [B10]
    image.

    The result is [Error] for what both readers refuse
    ({!section-refusals}). *)

(** {2:refusals Refusals}

    Both readers {e refuse}, with an [Error] that names the element, attribute
    or text at fault, or the line and column of a syntax error, an [s] that:
    - is not well-formed XML, or whose root element is not the one its
      reader's doc names;
    - lacks [PROCESSING_BASELINE], the quantification value, NODATA or
      SATURATED, or states one of them, the spectral list, the list of
      offsets, a band's offset or a band's [Wavelength] twice;
    - names a band in [Spectral_Information] that is not Sentinel-2's, or
      numbers one other than {!Jord.Sentinel2.index} does;
    - states an offset that is not an integer, or whose [band_id] the list
      does not map;
    - states a wavelength that is not a positive number of nanometres, or a
      [CENTRAL] outside [MIN] and [MAX];
    - states a quantification value that is not a plain decimal (digits, an
      optional fraction and exponent) whose value and inverse are positive
      and finite. *)

(** {1:tiles Tiles}

    A product's tile metadata, [MTD_TL.xml] in its [GRANULE] folder
    (Microsoft's Planetary Computer's asset [granule-metadata]), states when
    the tile was sensed and the angles of the Sun and of the view at
    {e nodes} in the tile's system, as {e angle arrays} of one value per
    node. ESA places the nodes every 5000 m from the tile's top left corner
    and measures the angles at them ({!section-nodes}). *)

type tile = {
  sensing_time : float;
      (** SENSING_TIME, an instant ({!Jord.Time}), the one instant ESA
          states for the tile. A catalogue's [datetime] can be the
          datatake's start instead, minutes earlier. *)
  nodes : Jord.Grid.t;
      (** The nodes, as the centres of its cells. Node
          [(i, j)] lies at (ULX + j·COL_STEP, ULY − i·ROW_STEP), where
          (ULX, ULY) is the tile's top left corner ([Tile_Geocoding]), so the
          grid's origin is half a step before the corner on each axis. Its
          system is the tile's [HORIZONTAL_CS_CODE]. *)
  sun : Nx.host Nx.float64_t * Nx.host Nx.float64_t;
      (** [Sun_Angles_Grid], the Sun's zenith and azimuth in degrees,
          [[rows; cols]] on [nodes], as {!Jord.Sun.angles} gives them at
          [nodes]' centres and [sensing_time]. The file's [Mean_Sun_Angle],
          their mean, is not read. *)
  view : Nx.host Nx.float64_t * Nx.host Nx.float64_t;
      (** [Viewing_Incidence_Angles_Grids], the zenith and azimuth in degrees
          of the view from the ground to the sensor, [[rows; cols; 13; 12]]
          on [nodes], by band, at its {!Jord.Sentinel2.index}, and by
          detector, detector [d] at [d - 1]. ESA does not state the
          azimuth's direction. The value is NaN where ESA computes no
          angle, as the file states, and for a band and detector the file
          states no angles for. *)
}
(** The type for a tile's metadata. *)

val tile : string -> (tile, string) result
(** [tile s] is the metadata of the tile whose [MTD_TL.xml] is [s], of a
    Level-1C or a Level-2A product, whose schemas share its angles.

    The result is [Error] if [s] is not well-formed XML or its root element
    is neither [Level-1C_Tile_ID] nor [Level-2A_Tile_ID], if it lacks
    SENSING_TIME, the tile's system, corner or [Sun_Angles_Grid], or states
    one twice, if SENSING_TIME is not RFC 3339's, the system is not an EPSG
    code {!Jord.Crs.of_epsg} reads, or the [Geoposition]s state different
    corners, if an angle array's steps are not positive metres, its values
    are neither decimals nor [NaN], or its steps and size are not the Sun's,
    or if a [Viewing_Incidence_Angles_Grids]' [bandId] is not from 0 to 12,
    its [detectorId] not from 1 to 12, or it is stated twice. *)

(** {2:nodes Nodes and angles}

    SentiWiki states where ESA places the nodes ("S2 Processing",
    §"Computation of Sun and viewing angles",
    {{:https://sentiwiki.copernicus.eu/web/s2-processing}}). "The top left
    node of the grid has the same coordinates as the top left corner of the
    tile footprint", and on a 109.8 km tile there are "23 by 23 nodes, each
    node being separated by 5000 m", the last row and column 200 m past the
    tile's edges. A zenith is measured from the normal to the ellipsoid, and
    "the azimuth Sun and viewing angles are computed clockwise, based on the
    North direction at the node coordinates". *)

(** {2:detectors Detectors}

    The Sun's angles are one pair of angle arrays. The view's are a pair for
    each band and each of the instrument's 12 detectors, numbered 01 to 12
    (Products Specification Document, §2.5, "MSI Detectors"), and "only
    nodes lying inside the detector/band have their viewing angles
    computed", so a view's angle array is NaN past its detector's
    footprint, and a node where two detectors overlap has an angle from
    each. ESA states no rule that merges them. Which detector sees a pixel
    is its band's detector footprint mask, MSK_DETFOO, which this module
    does not read. *)

(** {2:direction A view's direction}

    A view's direction is the line of sight's offset from the vertical per
    unit of height, east and north, tan θ · (sin φ, cos φ) for zenith θ and
    azimuth φ. Seen from one point, it is affine on the ground and
    continuous through nadir and across north, where the azimuth turns fast
    or wraps. The zenith is the arctangent of its length, and the azimuth
    atan2 of its east and north. *)

(** {2:reading Reading a view}

    An angle array is read at a pixel by {!Jord.Warp.bilinear} on the
    tile's [nodes], one band and detector at a time with that array's
    finite nodes valid, and a view by its direction ({!section-direction}).
    This function reads B04's view from detector [d] of [t] at the cells of
    [dst], with [Nx.isfinite z] marking the nodes the detector sees:

    {[
      let view (t : Jord_safe.tile) ~d (dst : Jord.Grid.t) =
        let open Jord in
        let rad x = Nx.mul_s x (Float.pi /. 180.) in
        let slice x =
          Nx.slice Nx.[ A; A; I (Sentinel2.index B04); I (d - 1) ] x
        in
        let z = rad (slice (fst t.view)) and a = rad (slice (snd t.view)) in
        let v =
          Nx.(stack ~axis:2 [ mul (tan z) (sin a); mul (tan z) (cos a) ])
        in
        let at = Warp.centres ~at:Nx.Host.on t.nodes.crs dst in
        let p = Warp.pixel t.nodes at in
        let v, ok = Warp.bilinear ~valid:(Nx.isfinite z) t.nodes p v in
        (v, ok)
    ]}

    It returns the direction at each cell and where it was read ([ok]). *)

(** {2:seams Seams}

    One call over all of [view] would give NaN next to every node the
    detector does not see. Even one detector at a time, a pixel near the
    seam between two detectors can lie in a node cell outside its own
    detector's nodes, and gets no value ([ok] is false), in strips up to a
    node wide. ESA states no rule for that strip, and the caller decides how
    to fill it. *)
