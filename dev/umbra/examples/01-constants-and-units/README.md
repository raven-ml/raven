# `01-constants-and-units`

Introduction to Umbra's type-safe unit system and physical constants. Creates
quantities in different units, converts between them, and demonstrates how
phantom types prevent mixing incompatible dimensions at compile time.

```bash
dune exec dev/umbra/examples/01-constants-and-units/main.exe
```

## What You'll Learn

- Creating quantities with scalar constructors (`Length.pc`, `Angle.deg`, `Mass.solar_mass`)
- Converting between units (`Length.in_ly`, `Angle.in_arcsec`)
- Adding quantities of the same dimension (`Unit.(+)`)
- Using physical constants (`Const.c`, `Const.h_si`, `Const.k_b_si`)
- Cross-dimension conversions (`parallax_to_distance`, `wavelength_to_frequency`)
- Batch operations on tensor-valued quantities

## Key Functions

| Function                    | Purpose                                      |
| --------------------------- | -------------------------------------------- |
| `Length.pc`, `Length.au`     | Create length quantities in parsecs, AU       |
| `Length.in_m`, `Length.in_ly`| Extract values in metres, light-years         |
| `Angle.deg`, `Angle.arcsec` | Create angles in degrees, arcseconds          |
| `Temperature.kelvin`        | Create temperature quantities                 |
| `Mass.solar_mass`           | Create mass in solar masses                   |
| `Const.c`, `Const.h_si`    | Speed of light, Planck constant               |
| `parallax_to_distance`      | Convert stellar parallax to distance          |
| `wavelength_to_frequency`   | Convert wavelength to frequency via c/lambda  |

## Try It

1. Compute the Schwarzschild radius of the Sun using `Const.g_si`, `Const.solar_mass`, and `Const.c`.
2. Add `Length.ly 4.246` (Proxima Centauri) and check it matches the parallax-derived distance.
3. Use `Unit.doppler_optical` to compute the observed wavelength of H-alpha at a radial velocity of 100 km/s.

## Next Steps

Continue to [02-cosmological-distances](../02-cosmological-distances/) to compute
distances and times in an expanding universe.
