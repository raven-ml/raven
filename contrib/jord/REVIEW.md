# Reviewing jord

## Design

- File I/O in `jord`, or a format library reached from it: `jord` takes
  values, and each format is a library of its own that takes bytes.
- An operation written with its own loop where nx operations compose it;
  rune can't differentiate a loop jord hides.
- A producer's fact (a band's scale, fill or quality bit) stated twice, or
  computed where the producer's guide states it.

## Semantics

References: the EPSG Dataset and IOGP's Guidance Note 7-2 for `Crs`'s
methods, with Karney's papers and GeographicLib for the transverse Mercator
and Albers; OGC GeoTIFF 1.1 (19-008r4); RFC 7946; STAC 1.1 and its
extensions; ESA's Sentinel-2 product specification; and the USGS's and
NASA's product guides each product module cites.

Failure here is a value placed at the wrong spot on the Earth, or read as
the wrong quantity: a transform off by metres, a grid off by half a cell, a
footprint read on the wrong side of a seam, a fill read as data.

- Transforms in float32, which err by metres.
- `Grid.equal` on computed grids, which compares exactly.
- The antimeridian, a geographic source's 360° seam, and the poles.
- Fill, saturated and masked values read as data; bitfields and classes
  decoded as numbers.
