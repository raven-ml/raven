# `07-fits-wcs`

`Fits.Wcs.read` turns a header's celestial keywords into a transform from pixels to the sky. This example reads one, maps pixels through it, and writes a cutout's header.

```bash
dune exec contrib/ymir/examples/07-fits-wcs/main.exe
```

## What You'll Learn

- Reading a header from text with `Fits.Header.of_string`
- Building a pixel-to-sky transform with `Fits.Wcs.read`
- The error for a header in another frame than the one expected
- Writing a cutout's world coordinates with `Fits.Wcs.write ~window`

## Key Functions

| Function                     | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `Fits.Header.of_string`      | A header from 80-byte records or lines   |
| `Fits.Wcs.read f h`          | Pixel indices to directions in frame `f` |
| `Fits.Wcs.write ?window t h` | `h` with keywords spelling `t`           |

## Next Steps

Continue to [08-grids-and-regions](../08-grids-and-regions/).
