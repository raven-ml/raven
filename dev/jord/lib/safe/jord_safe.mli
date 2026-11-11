(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ESA's Sentinel-2 product metadata, from the bytes of its XML.

    A Sentinel-2 product in ESA's SAFE format (Standard Archive Format for
    Europe) is a folder of images and XML metadata. Its main metadata file,
    [MTD_MSIL2A.xml] for a Level-2A product, states in
    [General_Info/Product_Image_Characteristics] the values that decode its
    digital numbers (ESA SentiWiki, "S2 Products",
    {{:https://sentiwiki.copernicus.eu/web/s2-products}}): the special values
    NODATA and SATURATED, the quantification value
    [BOA_QUANTIFICATION_VALUE], and, from processing baseline 04.00,
    [BOA_ADD_OFFSET] for each band. An offset names its band by an index,
    [band_id], which the file's [Spectral_Information] list maps to the
    band's name. Catalogues serve the file beside the images: Microsoft's
    Planetary Computer as the asset [product-metadata].

    ESA's Products Specification Document defines the file by XML schemas,
    [User_Product_Level-2A.xsd] with its [dimap.xsd] and [dimap2A.xsd]; SNAP,
    ESA's toolbox, ships those of PSD 15
    ({{:https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-reader/src/main/resources/schemas/PSD150}}).
    This module relies on their rules: an offset is a non-positive integer
    or ["Not applicable"] ([RADIO_OFFSET_TYPE]), at most 13 of them, each
    with a [band_id] from ["0"] to ["12"] ([A_BAND_NUMBER]); the special
    values are exactly two, NODATA and SATURATED, each an [xs:int]
    ([A_SPECIAL_VALUES]); and the quantification value is an [xs:double].
    The list of offsets entered the schemas at version 14.7, so an older
    product states none.

    This module reads the file from its bytes and does no I/O. *)

val l2a : string -> (Jord.Sentinel2.metadata, string) result
(** [l2a s] is the metadata of the Level-2A product whose [MTD_MSIL2A.xml] is
    [s]: its [PROCESSING_BASELINE], [BOA_QUANTIFICATION_VALUE], special
    values, and the [BOA_ADD_OFFSET] of each band in the order the file
    states them. [offsets] is empty for a product that states no offsets, as
    before baseline 04.00. An offset stated for B10, which Level-2A omits,
    or as ["Not applicable"], which ESA's schema allows, is left out.

    The result is [Error] if [s] is not well-formed XML, or not a Level-2A
    product's metadata, or lacks one of these values or states one twice, or
    if the quantification value is not a plain decimal (digits, an optional
    fraction and exponent) whose value and inverse are positive and finite.
    *)
