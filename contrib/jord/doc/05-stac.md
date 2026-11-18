# STAC items

A STAC catalogue lists acquisitions as items: each one's time, footprint
and assets, the files that hold its data. `jord.stac` reads an item's time,
assets, grids and bands from the `Jsont.json` the caller decoded
([`jord_stac.mli`](../lib/stac/jord_stac.mli)), and does no I/O, so a
program fetches pages of items with the HTTP client it likes. Add
`jord.stac`, `jsont` and, to decode text, `jsont.bytesrw` to an
executable's `libraries`. The toplevel prints each system as `Crs.pp`
does, by its EPSG name:

```ocaml
# open Jord;;
# #install_printer Crs.pp;;
```

## An item and its grids

This item is cut from Microsoft's Planetary Computer's Sentinel-2 Level-2A
item of tile 33UUU on 2024-06-29, which jord's tests hold
(`test/stac/mpc_sentinel2_l2a.json`), down to its footprint, its time, its
system and one asset. It is STAC 1.0, so its properties state the system
as `proj:epsg`, and its asset states its shape, its transform and its band
in `eo:bands`:

```ocaml
let text =
  {|{"type": "Feature", "stac_version": "1.0.0",
     "id": "S2A_MSIL2A_20240629T102021_R065_T33UUU_20240629T161811",
     "collection": "sentinel-2-l2a",
     "geometry": {"type": "Polygon",
                  "coordinates": [[[12.0047609, 53.2119778],
                                   [13.6483415, 53.2419635],
                                   [13.6785441, 52.2551507],
                                   [12.0715951, 52.2262109],
                                   [12.0047609, 53.2119778]]]},
     "bbox": [12.0047609, 52.2262109, 13.6785441, 53.2419635],
     "properties": {"datetime": "2024-06-29T10:20:21.024000Z",
                    "proj:epsg": 32633},
     "assets": {"B04": {"href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/33/U/UU/2024/06/29/S2A_MSIL2A_20240629T102021_N0510_R065_T33UUU_20240629T161811.SAFE/GRANULE/L2A_T33UUU_A047109_20240629T102556/IMG_DATA/R10m/T33UUU_20240629T102021_B04_10m.tif",
                        "type":
                          "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["data"],
                        "proj:shape": [10980, 10980],
                        "proj:transform":
                          [10.0, 0.0, 300000.0, 0.0, -10.0, 5900040.0],
                        "eo:bands": [{"name": "B04", "common_name": "red",
                                      "center_wavelength": 0.665,
                                      "full_width_half_max": 0.038}]}}}|}
```

The program decodes the text, then reads the item:

```ocaml
# let item =
    let json = Result.get_ok (Jsont_bytesrw.decode_string Jsont.json text) in
    Result.get_ok (Jord_stac.item json);;
val item : Jord_stac.item =
  {Jord_stac.id = "S2A_MSIL2A_20240629T102021_R065_T33UUU_20240629T161811";
   collection = Some "sentinel-2-l2a"; datetime = Some 1719656421.024;
   interval = None; orbit_state = None; relative_orbit = None;
   view =
    {Jord_stac.off_nadir = None; incidence_angle = None; azimuth = None;
     sun_azimuth = None; sun_elevation = None};
   center_frequency = None;
   assets =
    [("B04",
      {Jord_stac.href =
        "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/33/U/UU/2024/06/29/S2A_MSIL2A_20240629T102021_N0510_R065_T33UUU_20240629T161811.SAFE/GRANULE/L2A_T33UUU_A047109_20240629T102556/IMG_DATA/R10m/T33UUU_20240629T102021_B04_10m.tif";
       media_type =
        Some "image/tiff; application=geotiff; profile=cloud-optimized";
       roles = ["data"];
       grid =
        Some
         (Ok
           {Grid.crs = WGS 84 / UTM zone 33N; rows = 10980; cols = 10980;
            origin = (300000., 5900040.); row_step = (0., -10.);
            col_step = (10., 0.)});
       bands =
        [{Jord_stac.band =
           {Band.name = Some "B04"; common_name = Some "red";
            center_wavelength = Some 0.665; full_width_half_max = Some 0.038;
            unit = None; kind = Band.Measurement; scale = 1.; offset = 0.;
            nodata = None; saturated = None};
          grid =
           Some
            (Ok
              {Grid.crs = WGS 84 / UTM zone 33N; rows = 10980; cols = 10980;
               origin = (300000., 5900040.); row_step = (0., -10.);
               col_step = (10., 0.)});
          bitfields = []}];
       view =
        {Jord_stac.off_nadir = None; incidence_angle = None; azimuth = None;
         sun_azimuth = None; sun_elevation = None}})];
   properties =
    Jsont.Object
     ([(("datetime", <abstr>),
        Jsont.String ("2024-06-29T10:20:21.024000Z", <abstr>));
       (("proj:epsg", <abstr>), Jsont.Number (32633., <abstr>))],
      <abstr>)}
```

The asset states no system of its own, so it takes the item's (the Where
a field holds section of [`jord_stac.mli`](../lib/stac/jord_stac.mli)),
and its grid and its band's grid are complete. The transform's six
numbers map a pixel's column and row to the plane (the Grids section), so
the grid's origin is `(300000, 5900040)`, its rows run 10 m south and its
columns 10 m east. The band states no scale, offset or nodata, so it
decodes to the DNs themselves and masks no DN, DN 0 included. The
[Sentinel-2 metadata](06-safe.md) states the offsets and the NODATA value, and
`Sentinel2.reflectance` makes the band that decodes the DNs into
reflectance and masks NODATA.

## Times

An item's `datetime` and `interval` are instants of
[`Time`](../lib/time.mli): the seconds of UTC since 1970-01-01T00:00:00Z.
`Time.of_rfc3339` reads them from RFC 3339's date-times, with the space
that some catalogues write in place of `T` (Microsoft's Planetary Computer
in Sentinel-1's `start_datetime` and `end_datetime`), and takes an offset
off (`of_rfc3339` in [`time.mli`](../lib/time.mli)); `Time.date` gives an
instant's day:

```ocaml
# Time.of_rfc3339 "2024-06-29 10:20:21+02:00";;
- : (float, string) result = Ok 1719649221.
# Time.date 1719656421.024;;
- : int * int * int = (2024, 6, 29)
```

## What catalogues state

`Jord_stac` reads a catalogue's statement of a product's bands as the
catalogue states it, and a product module states its producer's (the
preamble of [`band.mli`](../lib/band.mli)). These are the differences
Microsoft's Planetary Computer's items show. `test/stac/` holds an item
of most of these collections.

- Its ESA WorldCover items state their classes, a product jord has no
  module for, so the catalogue's classes are all a program has.
- Its Landsat items state QA_PIXEL's bits, and describe bit 6 by the
  guides' text, "Cloud and dilated cloud bits are not set", which the
  guides' own values contradict (`Clear` in the Quality bits section of
  [`landsat.mli`](../lib/landsat.mli)).
- Its Sentinel-2 items state centres that are the midpoints of a
  product's `MIN` and `MAX`, 0.665 µm for B04 from 646 to 684 nm, whose
  `CENTRAL` is 664.6 nm, and widths that are `MAX` - `MIN`, 0.038 µm for
  B04 (`Sentinel2.wavelength` in the Bands section of
  [`sentinel2.mli`](../lib/sentinel2.mli)).
- Its HLS S30 items state one set of wavelengths on every satellite, whose
  centres are near Sentinel-2A MSI's (SentiWiki, "S2 Mission", Table 3):
  1.6137 µm for B11, S2A's 1613.7 nm, and 2.22024 µm for B12, S2A's
  2202.4 nm with a digit repeated. Most of its widths are the extent of the
  response that a Sentinel-2 product's metadata tabulates, its `MAX` -
  `MIN`, as 0.242 µm for B12, whose width SentiWiki states as 180 nm. The
  HLS guide's S30 bands are "Adjusted to OLI-like (except red edge, water
  vapor and cirrus bands)" (Table 2), and OLI's B6 is centred at
  1.6088 µm (`Landsat.band` in [`landsat.mli`](../lib/landsat.mli)). The
  HLS guide states nominal ranges and no centre or width (the preamble of
  [`hls.mli`](../lib/hls.mli)).
- Its Sentinel-2 Level-2A items state no scale, offset or nodata, so their
  bands decode to the DNs themselves and mask no DN, and the product's
  metadata states the offsets and NODATA (the Reflectance section of
  [`sentinel2.mli`](../lib/sentinel2.mli)). Its HLS items state no scale or
  fill either.
- Its HLS items state no common name for S30's red edge, B8A and B09.
- Its Landsat items are named by the product's identifier without its
  processing date, as `LC09_L2SP_193024_20240629_02_T1`, and call both
  TM and ETM+'s band 4 and OLI's band 5 `nir08`. Its Landsat 9 items state
  band 10 as 10.9 µm, 0.59 µm wide, which are Landsat 8's TIRS's values
  (10.896 and 0.588 by `Landsat.band`), and its QA_RADSAT bits follow the
  guides' tables.
- Its MOD09A1 items state the ranges' midpoints as centres and ten times
  the ranges as widths, 0.5 µm for band 1. They name band 2 `nir08`, where
  STAC's own table lists MODIS's band 2 as `nir`, and band 5 `lwir12`,
  11.5 to 12.5 µm.
- Its MODIS items state a tile's corners rounded to the micrometre, as
  1111950.519667, and their grid from them, which is within a micrometre of
  `Modis.tile`'s at every corner and not equal to it by `Grid.equal`.
- Its older MODIS items hold WKT 1 in `proj:wkt2`, which `Jord_stac` reads
  as WKT 1 (the Grids section of
  [`jord_stac.mli`](../lib/stac/jord_stac.mli)).
- It states no grid of its own for an item's `rendered_preview` and
  `tilejson` assets, so they take the data's grid, and a program selects
  the assets it reads by key (the Where a field holds section of
  [`jord_stac.mli`](../lib/stac/jord_stac.mli)).
- Its Landsat items give the quality layers roles of their own, none of
  them `data`. `qa_pixel`, and TM's and ETM+'s `cloud_qa`, have `cloud`,
  `cloud-shadow`, `snow-ice` and `water-mask`, `qa_radsat` has
  `saturation`, and OLI's `qa_aerosol` has `data-mask` and `water-mask`.
- In item `HLS.S30.T21JVF.2026306T134221.v2.0`, the identifier's stamp is
  13:42:21 and the item's `datetime` 13:51:24.358 (the Identity section of
  [`hls.mli`](../lib/hls.mli) says what the stamp is).

The Copernicus Data Space Ecosystem's Sentinel-2 items state each band's
incidence angle on its asset, which `Jord_stac` reads as the asset's view.
