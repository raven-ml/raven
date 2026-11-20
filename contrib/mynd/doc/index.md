# mynd

mynd computes on images, positions in them and their annotations, as nx
arrays. An image is an `Nx.t` whose last three axes are rows, columns and
channels. mynd defines what an image's numbers mean, converts them
between dtypes, reads colours, samples and resizes images, filters them,
and holds boxes, masks and their run-length encoding. Rune
differentiates, batches and compiles its functions on arrays, with the
exceptions the preamble of [`Mynd`](../lib/mynd.mli) names.

## Install

mynd is a contrib package of raven. Install the `mynd` package and add
`mynd` to your executable's `libraries`, with `nx.io` to read and write
image files and `rune` to differentiate, as the other pages do:

<!-- $MDX skip -->
```sh
$ opam install mynd
```

```dune
(executable
 (name main)
 (libraries mynd nx.io rune))
```

## A first program

The program below reads a photograph, makes it grey and writes the grey
image. So that it runs without files of its own, this page first draws
the photograph, a 32 by 48 pixel pattern of 8-bit RGB codes:

```ocaml
let () =
  let pattern (i : int array) = ((i.(0) * 8) + (i.(1) * 5 * (i.(2) + 1))) mod 256 in
  Nx_io.save_png "photo.png" (Nx.init Nx.uint8 [| 32; 48; 3 |] pattern)
```

The program:

```ocaml
# let () =
    let img = Nx_io.load_image "photo.png" in
    let x = Mynd.convert Nx.float32 img in
    let grey = Mynd.Colour.luma x in
    Nx_io.save_png "grey.png" (Mynd.convert Nx.uint8 grey);
    Printf.printf "%d rows, %d columns, %d channel\n" (Nx.dim 0 grey)
      (Nx.dim 1 grey) (Nx.dim 2 grey);;
32 rows, 48 columns, 1 channel
```

Each call, in order:

- `Nx_io.load_image "photo.png"` reads the PNG file `photo.png` as an
  image of `uint8` values, `[32; 48; 3]`, on the host. Each value is a
  code from `0` to `255`, and the file's colours are encoded sRGB, as
  most photographs' are. A grey file loads with one channel, a file with
  alpha with a channel more, and a 16-bit sample keeps its high byte.
- `Mynd.convert Nx.float32 img` is the image in `float32`, each code `k`
  read as the fraction `k / 255` of full scale, the range `[0, 1]` that
  `0` and `255` stand for. `Mynd.Colour` takes floats, and `convert` is
  the one function that moves between codes and fractions. The preamble
  of [`Mynd`](../lib/mynd.mli) defines codes and fractions, and its
  Conversions section states `convert`'s rounding.
- `Mynd.Colour.luma x` is each pixel's luma, `[32; 48; 3]` to
  `[32; 48; 1]`, the weighted sum `0.299 r + 0.587 g + 0.114 b` of the
  encoded values, which is the grey that image files carry. It raises
  `Invalid_argument` for a last axis other than three channels, so an
  alpha channel is split off first ([`Mynd.Colour`](../lib/mynd.mli)).
- `Mynd.convert Nx.uint8 grey` turns the fractions back into codes,
  `k = 255 f` rounded to even, after a fraction outside `[0, 1]` is
  clamped.
- `Nx_io.save_png "grey.png" ...` writes the one-channel image as a grey
  PNG file.

The image keeps its encoded values throughout, so luma weighs encoded
numbers. The light a colour emits is its luminance, which
`Mynd.Colour.luminance` computes from linear values, after
`Mynd.srgb_to_linear` (the preamble of [`Mynd`](../lib/mynd.mli)).

## A batch of images

Axes before the last three are batch axes, and each index of them is an
independent image. The same program makes every image of a batch grey,
and the result for each image is the one it gets alone:

```ocaml
let () =
  let x = Mynd.convert Nx.float32 (Nx_io.load_image "photo.png") in
  let batch = Nx.stack [ x; Nx.flip ~axes:[ 1 ] x ] in
  let grey = Mynd.Colour.luma batch in
  assert (Nx.shape grey = [| 2; 32; 48; 1 |]);
  assert (Nx.to_array (Nx.slice [ Nx.I 0 ] grey) = Nx.to_array (Mynd.Colour.luma x))
```

## Further pages

- [Adjusting colour](colour.md): brightness, contrast, saturation, hue
  and Lab relative to D50.
- [Filtering images](filters.md): box blurs, unsharp masks, the
  Laplacian, rank and bilateral filters, and the discrete Gaussian.
- [Gradients and edges](edges.md): Canny's edges, wider stencils,
  thresholds and the structure tensor.
- [Morphology](morphology.md): openings and closings, hit-or-miss, holes,
  small objects and distances.
- [Moving annotations with an image](annotations.md): boxes, masks and
  keypoints under flips, crops, resizes and warps, the boxes that
  survive, training recipes and GIoU as a loss.

Each rule of mynd's that these pages use is stated in
[`mynd.mli`](../lib/mynd.mli), and each rule of nx's in
[`nx.mli`](../../../packages/nx/lib/nx.mli), such as the reads and
writes of positions outside an axis in its section Indexing and slicing.
The pages link each where they use it.
