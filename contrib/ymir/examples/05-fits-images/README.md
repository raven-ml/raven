# `05-fits-images`

A FITS file is a list of HDUs, each a header and a data unit. This example
writes an image with its keywords, plain and tile-compressed, and reads it back.

```bash
dune exec contrib/ymir/examples/05-fits-images/main.exe
```

## What You'll Learn

- Building a header with `Fits.Header.set` and typed `Fits.Value`s
- Writing image HDUs with `Fits.Image.hdu`, tile-compressed with `~tiles`
- Reading a file, finding an HDU by name and verifying its checksums
- Reading keywords, the `BUNIT` unit, and pixels through a window

## Key Functions

| Function                              | Purpose                           |
| ------------------------------------- | --------------------------------- |
| `Fits.Header.set v k x h`             | Set keyword `k`                   |
| `Fits.Header.get v k h`               | Read keyword `k` as a `v`         |
| `Fits.Image.hdu ?tiles h t`           | An image HDU from a tensor        |
| `Fits.write`, `Fits.read`             | Write and read files              |
| `Fits.get name hdus`                  | The HDU whose `EXTNAME` is `name` |
| `Fits.Image.values ?window dtype hdu` | Physical pixel values             |
| `Fits.unit hdu`                       | `BUNIT` as a unit                 |

## Next Steps

Continue to [06-fits-tables](../06-fits-tables/).
