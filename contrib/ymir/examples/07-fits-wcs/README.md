# `07-fits-wcs`

`Wcs.read` turns a header's celestial keywords into a transform from pixels to the sky, asking for each keyword's value as `ymir.fits` decodes it. This example reads one, maps pixels through it, composes an image and its error into an observation, and writes a cutout's header.

```bash
dune exec contrib/ymir/examples/07-fits-wcs/main.exe
```

## What You'll Learn

- Reading a header from text with `Fits.Header.of_string`
- Building a pixel-to-sky transform with `Wcs.read` over the header's decoded keywords
- The error for a header in another frame than the one expected
- Composing an image and its error into an `Observation`
- Writing a cutout's world coordinates as the edits of `Wcs.write ~window`

## Key Functions

| Function                       | Purpose                                    |
| ------------------------------ | ------------------------------------------ |
| `Fits.Header.of_string`        | A header from 80-byte records or lines     |
| `Wcs.read f kw`                | Pixel indices to directions in frame `f`   |
| `Wcs.write ?window t kw`       | The keyword edits that spell `t`           |
| `Grid.pixels`, `Observation.v` | An image's grid and data as an observation |

## Next Steps

Continue to [08-grids-and-regions](../08-grids-and-regions/).
