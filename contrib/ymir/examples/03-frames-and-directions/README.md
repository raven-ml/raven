# `03-frames-and-directions`

A direction is a batch of unit vectors in a celestial frame, and the frame is
part of its type. This example builds directions, rotates them between frames,
and measures angles between them.

```bash
dune exec contrib/ymir/examples/03-frames-and-directions/main.exe
```

## What You'll Learn

- Directions from longitudes and latitudes with `Direction.lonlat`
- Rotating into another frame with `Direction.rotate`
- Angular separations and position angles, broadcast over batches
- The rotation matrix between two frames

## Key Functions

| Function                         | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `Direction.lonlat f ~lon ~lat`   | Directions in frame `f`                  |
| `Direction.lon`, `Direction.lat` | Angles of each direction, as quantities  |
| `Direction.rotate g d`           | `d` in frame `g`                         |
| `Direction.separation a b`       | Angle between the rays                   |
| `Direction.position_angle a b`   | Bearing of `b` from `a`, east of north   |
| `Frame.matrix a b`               | The rotation from frame `a` to frame `b` |

## Next Steps

Continue to [04-transforms](../04-transforms/).
