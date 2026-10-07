# `08-grids-and-regions`

A grid is an image's cells seen through a transform; a region is a shape placed
on the grid's world. Each cell weighs the exact fraction of its area inside the
shape. This example works on a pixel plane and on the sky.

```bash
dune exec contrib/ymir/examples/08-grids-and-regions/main.exe
```

## What You'll Learn

- Grids with `Grid.pixels`, and each cell's area or solid angle with
  `Grid.measure`
- Circles and annuli placed with `Transform.shift` in a plane
- Exact overlap weights: they sum to the shape's area
- Caps on the sky placed with `Transform.about`

## Key Functions

| Function                         | Purpose                                |
| -------------------------------- | -------------------------------------- |
| `Grid.pixels ~shape dtype t`     | The cells of an image seen through `t` |
| `Grid.measure g`                 | Each cell's area or solid angle        |
| `Region.circle p ~radius`        | A disc in the plane of placement `p`   |
| `Region.annulus p ~inner ~outer` | A ring                                 |
| `Region.weights r g`             | Each cell's covered fraction           |
| `Transform.about c`              | Angular offsets about direction `c`    |

## Next Steps

Continue to [09-observations](../09-observations/).
