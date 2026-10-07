# `09-observations`

An observation is data on a grid with a variance and a mask of valid samples.
This example measures a synthetic source with a circular aperture and a
background annulus.

```bash
dune exec contrib/ymir/examples/09-observations/main.exe
```

## What You'll Learn

- Building an observation with `Observation.v`, its variance and validity
- Data per cell with `Grid.cell` in the unit
- Sums, areas, propagated variances and coverage from `Observation.integrate`
- Background subtraction with quantity arithmetic, and with `Observation.sub`
  of a model on the same grid
- Windows of static shape with `Observation.around`, and the error for one that
  clips the region

## Key Functions

| Function                                | Purpose                                          |
| --------------------------------------- | ------------------------------------------------ |
| `Observation.v ?variance ?valid g data` | Data on grid `g`                                 |
| `Observation.integrate r o`             | Sum, variance, area and coverage over region `r` |
| `Observation.sub a b`                   | `a` less `b` on agreeing grids, variances added  |
| `Observation.around x ~shape o`         | The block of cells about a point                 |
| `Grid.cell`                             | The unit symbol counting cells                   |

## Next Steps

Continue to [10-cosmology](../10-cosmology/).
