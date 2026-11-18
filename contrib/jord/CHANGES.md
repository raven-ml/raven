# Changelog

All notable changes to jord are documented in this file.

## Unreleased

- `Crs` is a datum (WGS 84, NAD83, ETRS89-FIN, NZGD2000, GDA94) and an
  optional projection (transverse Mercator, Albers, Pseudo-Mercator, the
  sinusoid), read by `Crs.of_epsg` and `Crs.of_wkt`, each with its error.
- `Grid` places an array's first two axes in a system by an affine map, and
  `Polygons` holds features in GeoArrow's layout under the even-odd rule.
- `Rasterize` gives the cells features cover, by centre, by touch, and with
  each cell's covered share (`Rasterize.areas`); `Polygons.area` is a
  feature's area in its system's plane.
- `Warp` moves arrays between grids: `Warp.nearest` and `Warp.bilinear`
  sample a source at pixel points, with a mask, and `Warp.window` is the
  block they read.
- `Warp.average`, `Warp.mode`, `Warp.span`, `Warp.area_window` and
  `Warp.corners` read a source over each cell's exact footprint. They are
  provisional: their names and the fraction may change with a first consumer.
- `Band` is STAC's band, decoded by `Band.decode`. `Sentinel2`, `Sentinel1`,
  `Landsat`, `Hls` and `Modis` state their products' bands, scales, fills
  and quality bits.
- `Time` holds instants of UTC as POSIX seconds, and `Sun` the Sun's zenith,
  azimuth and distance from a point at an instant.
- `jord.geotiff`, `jord.geojson`, `jord.shapefile`, `jord.stac` and
  `jord.safe` read GeoTIFF tags, GeoJSON, ESRI Shapefiles, STAC items and
  Sentinel-2 product metadata from bytes or values.
