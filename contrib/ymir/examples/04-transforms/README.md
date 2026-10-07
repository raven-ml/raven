# `04-transforms`

A transform maps points through a list of stages: pixel coordinates to a plane,
a plane to the sky. This example builds a pixel-to-sky map by hand, applies it,
inverts it, and asks where it is defined.

```bash
dune exec contrib/ymir/examples/04-transforms/main.exe
```

## What You'll Learn

- Composing stages with `>>`: `axes`, `shift`, `linear` and a projection
- `Transform.gnomonic` and its inverse, a tangent-plane projection about a
  direction
- Mapping a batch of pixels with `Transform.apply`, and back with
  `Transform.inverse`
- The domain of a projection: `Transform.covers` and the error `apply` raises
  outside it

## Key Functions

| Function                            | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| `Transform.( >> )`                  | Apply one transform, then another         |
| `Transform.axes`, `shift`, `linear` | Planar stages                             |
| `Transform.gnomonic c`              | Directions to the tangent plane about `c` |
| `Transform.inverse t`               | `t` run backwards                         |
| `Transform.apply t x`               | Map points through `t`                    |
| `Transform.covers t x`              | Where `apply t x` is defined              |

## Next Steps

Continue to [05-fits-images](../05-fits-images/).
