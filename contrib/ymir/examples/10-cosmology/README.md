# `10-cosmology`

A cosmology is a record of tensors, and distances, volumes and times are
functions of it and of redshifts. This example evaluates a published model,
varies it by record update, and evaluates many models in one call.

```bash
dune exec contrib/ymir/examples/10-cosmology/main.exe
```

## What You'll Learn

- The Planck 2018 model from `Cosmology.planck2018`
- Distances, times and the distance modulus, read in Mpc and Gyr
- Density parameters with `Cosmology.density_parameter`
- Other models as record updates: dark energy's `w0`, curvature
- Batching: leaves of shape `[k; 1]` against `n` redshifts give `[k; n]`

## Key Functions

| Function                             | Purpose                         |
| ------------------------------------ | ------------------------------- |
| `Cosmology.planck2018 ~codata dtype` | A published flat model          |
| `Cosmology.comoving_distance c z`    | Line-of-sight comoving distance |
| `Cosmology.luminosity_distance c z`  | Luminosity distance             |
| `Cosmology.age c z`, `lookback_time` | Times                           |
| `Cosmology.distance_modulus c z`     | 5 log10(D_L / 10 pc)            |
| `Cosmology.density_parameter c i z`  | Omega of component `i` at `z`   |

## Next Steps

Continue to [11-derivatives](../11-derivatives/).
