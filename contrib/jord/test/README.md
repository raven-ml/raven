# Test data

Each fixture's header, or the dune file of its directory, records its
source and the command that made it. The generators are in `gen/`, run
through `uv run` (each one's docstring says how). This file gives each
fixture's terms. Where a term is marked unverified, the source's terms page
could not be read when the file was written (2026-11-18), and the source
alone is given.

## Computed from jord's own inputs

`albers.dat`, `mercator_sinusoidal.dat` and `plane_areas.dat` (exact values
by mpmath and Python's Fractions), `warp_gdal.dat` and the GDAL cases of
`warp_area.dat` (GDAL's output on test_warp.ml's formulas), and
`geotiff/cog_overviews.tif` (written by rasterio from a formula). They hold
no third party's data.

## tm_coords.dat

- Source: Karney's transverse Mercator test set, TMcoords.dat
  (GeographicLib), <https://zenodo.org/records/32470>.
- Terms: CC0 1.0.

## wkt.dat, and lib/crs.ml's parameters

- Source: the EPSG Dataset's WKT exports (lines `epsg`), PROJ's WKT2 of the
  same systems (lines `proj`), and a Planetary Computer item's `proj:wkt2`
  (line `mpc`; see `stac/` below). `lib/crs.ml`'s codes, parameters and
  datum steps are a subset of the EPSG Dataset.
- Terms: the EPSG Dataset is owned by IOGP, under its terms of use,
  <https://epsg.org/terms-of-use.html>: "Ownership of the EPSG Dataset by
  IOGP must be acknowledged in any publication or transmission (by whatever
  means) thereof (including permitted modifications)", and a recipient is
  to be told of those terms, which this paragraph does.

## esri_26910.prj

- Source: Esri's Projection Engine WKT for wkid 26910, from its published
  lists.
- Terms: unverified.

## areas.dat

- Source: the `belle-ile` parcels of BreizhCrops
  (<https://github.com/dl4sits/BreizhCrops>, its 2017 shapefile,
  `gen/areas.py` gives the URL), which are parcels of the Registre
  Parcellaire Graphique (RPG) 2017, the parcels declared in 2017 as they
  stood on 1 January 2018, cut by BreizhCrops on 2020-06-08. The
  geometries are the RPG's, moved onto a turned grid.
- Terms: Licence Ouverte 2.0
  (<https://www.etalab.gouv.fr/licence-ouverte-open-licence/>). Grantor:
  IGN, the Institut national de l'information géographique et forestière,
  for the RPG, <https://www.data.gouv.fr/datasets/rpg> (its record last
  updated 2026-09-29).

## warp_area.dat, case xzone

- Source: a window of band B04 of the Sentinel-2 L2A product
  `S2A_MSIL2A_20250702T103701_R008_T31UGR_20250702T124303`, read from
  Microsoft's Planetary Computer (`gen/warp_area.py`).
- Terms: contains modified Copernicus Sentinel data 2025, under the
  Copernicus Sentinel data legal notice
  (<https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice>),
  which allows reproduction, distribution and modification with that
  notice.

## safe/

- Source: Sentinel-2 product metadata, as `safe/dune` records:
  `mtd_n0300.xml` (2021) and `mtd_n0510.xml` (2024) unchanged, and
  `mtd_l1c_n0510.xml` and `mtd_tl_n0510.xml` (2024) cut.
- Terms: Copernicus Sentinel data 2021, 2024 for the unchanged files;
  contains modified Copernicus Sentinel data 2024 for the two cut files.
  The legal notice is the one above.

## stac/

Items of Microsoft's Planetary Computer (`mpc_*.json`) and of the
Copernicus Data Space Ecosystem (`cdse_sentinel2_l2a.json`), fetched
unchanged, as `stac/dune` records. The Planetary Computer's own terms
(<https://planetarycomputer.microsoft.com/terms>) are unverified: the page
gives no text without a browser. Each collection states its licence:

- `mpc_sentinel2_l2a.json`: Copernicus Sentinel data 2024;
  `cdse_sentinel2_l2a.json`: Copernicus Sentinel data 2026. The legal
  notice is the one above.
- `mpc_sentinel1_rtc.json`: the collection's licence is CC BY 4.0; it is
  derived from Copernicus Sentinel data 2024.
- `mpc_landsat_c2_l2.json`, `mpc_landsat_c2_l2_le07.json`: USGS Landsat
  Collection 2; the collection links the USGS data policy
  (<https://www.usgs.gov/core-science-systems/hdds/data-policy>), whose
  terms are unverified (the page refused the fetch).
- `mpc_hls2_s30.json`, `mpc_modis_13q1.json`, `mpc_modis_09a1.json`: NASA's
  HLS and MODIS products. LP DAAC's policy
  (<https://lpdaac.usgs.gov/data/data-citation-and-policies/>): "Unless the
  content is marked with a use restriction or license, data provided from a
  NASA-led mission are licensed as Creative Commons Zero (CC0)", and users
  are "very strongly urged to cite the data". HLS S30 is derived from
  Copernicus Sentinel data 2024.
- `mpc_esa_worldcover.json`: ESA WorldCover 2021, under CC BY 4.0
  (<https://esa-worldcover.org/en/data-access>), with the attribution
  "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel
  data (2021) processed by ESA WorldCover consortium".

## landsat_rsr.dat

- Source: the USGS's band-average relative spectral responses, from the
  Spectral Characteristics Viewer
  (<https://landsat.usgs.gov/spectral-characteristics-viewer>).
- Terms: unverified (usgs.gov's policy pages refused the fetch).

## sun_horizons.dat

- Source: JPL Horizons (<https://ssd.jpl.nasa.gov/horizons/>), the Solar
  System Dynamics Group of the Jet Propulsion Laboratory.
- Terms: unverified; no statement on the use of Horizons' output was found.
  JPL's Solar System Dynamics site asks that its data be cited as "NASA/JPL
  Solar System Dynamics. Data retrieved on 2026-11-16", with the page's
  title and <https://ssd.jpl.nasa.gov> (<https://ssd.jpl.nasa.gov/about/>).
