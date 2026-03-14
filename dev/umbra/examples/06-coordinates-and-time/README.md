# `06-coordinates-and-time`

Celestial coordinates, astronomical time scales, and survey selection.
Demonstrates frame transforms (ICRS, Galactic), angular separation, time scale
conversions (UTC, TAI, TT, TDB), altitude-azimuth coordinates, airmass, and
a practical survey selection function.

```bash
dune exec dev/umbra/examples/06-coordinates-and-time/main.exe
```

## What You'll Learn

- Creating celestial coordinates in ICRS and converting to Galactic frame
- Computing angular separations between objects
- Parsing ISO 8601 dates and converting between time scales
- Computing horizontal coordinates for a ground-based observer
- Building a survey selection function from airmass, altitude, and magnitude cuts

## Key Functions

| Function                     | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `Coord.of_radec`             | Create ICRS coordinates from RA/Dec           |
| `Coord.galactic`             | Convert to Galactic coordinates               |
| `Coord.separation`           | Angular separation between positions          |
| `Time.of_iso`                | Parse ISO 8601 date-time as UTC               |
| `Time.utc_to_tai`            | Convert UTC to TAI                            |
| `Time.tai_to_tt`             | Convert TAI to Terrestrial Time               |
| `Time.tt_to_tdb`             | Convert TT to Barycentric Dynamical Time      |
| `Time.to_jd`, `Time.to_mjd`  | Extract Julian Date / Modified Julian Date    |
| `Altaz.make_observer`        | Create a ground-based observer location       |
| `Altaz.of_coord`             | Convert celestial to horizontal coordinates   |
| `Altaz.alt`, `Altaz.az`      | Extract altitude and azimuth                  |
| `Altaz.airmass`              | Compute airmass at given altitude              |
| `Filters.rubin_r`            | Pre-built Rubin/LSST r-band filter            |

## How It Works

**Coordinates**: Positions are stored as (longitude, latitude) pairs in typed
angle quantities. Frame transforms use 3x3 rotation matrices to convert
between ICRS, Galactic, Ecliptic, and Supergalactic systems. Angular separation
uses the Vincenty formula for numerical stability.

**Time**: Julian Dates carry phantom type tags (UTC, TAI, TT, TDB) that
enforce correct scale conversions at compile time. UTC-TAI uses the IERS
leap-second table; TT = TAI + 32.184s exactly; TDB-TT uses the Fairhead &
Bretagnon series.

**Altaz**: Converts ICRS to horizontal coordinates using IAU 2006 precession
and the Earth Rotation Angle. Airmass uses the Pickering (2002) formula.

**Selection**: Combines altitude (above horizon), airmass (atmospheric
extinction), and magnitude limit into a boolean selection function -- a
building block for survey simulations.

## Try It

1. Add atmospheric refraction with `Altaz.of_coord ~refraction:true`.
2. Compute the position angle from Vega to Deneb with `Coord.position_angle`.
3. Use `Coord.of_galactic` to create coordinates in the Galactic plane and
   convert to ICRS.
4. Change the observer location and time to see how visibility changes.

## Next Steps

Explore the other Umbra examples for more advanced topics: catalog
cross-matching with `Coord.nearest`, cosmological power spectra with
`Cosmo.linear_power`, and Fisher matrix forecasts.
