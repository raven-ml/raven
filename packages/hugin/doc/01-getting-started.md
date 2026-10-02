# Getting Started

This page draws a first figure, explains how a mark turns tensors into rows,
saves figures to files, shows them in a notebook, and reads the errors and
warnings Hugin reports.

## Installation

<!-- $MDX skip -->
```bash
opam install hugin
```

Add the library to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries hugin))
```

Hugin needs no system library: it draws text with its bundled Inter faces
and writes PNG, SVG and PDF itself.

## A First Figure

```ocaml
open Hugin

let x = Nx.linspace Nx.float64 0. 6.28 100

let () = line ~x:(num x) ~y:(num (Nx.sin x)) () |> save "sine.png"
```

`line` is a mark. Its labelled arguments are its *roles*, the slots it reads
data from, and each takes a *channel*: here `num x` and `num (Nx.sin x)`,
the elements of two tensors read as quantities. A mark is already a figure,
so it can be saved as it is. `save` picks the format from the file's
extension.

Wrappers take the figure last, so they chain with `|>`:

```ocaml
let () =
  line ~x:(num ~title:(Text.v "x") x) ~y:(num ~title:(Text.v "sin x") (Nx.sin x)) ()
  |> title (Text.v "A sine wave")
  |> save "sine-titled.svg"
```

The `title` of a channel titles the axis or legend of the scale it reads.
Text is a `Text.t`, so a title can mix styles:
`Text.concat [ Text.v "x"; Text.sup (Text.v "2") ]` writes x².

## Channels

A channel says how a mark reads its data:

| Channel | Reads | Example |
|---|---|---|
| `num t` | the elements of a tensor, as quantities | `num losses` |
| `cat ~labels t` | the integer codes of a tensor, as categories | `cat ~labels:[\| "cat"; "dog" \|] ys` |
| `strings a` | the strings of an array, as categories | `strings [\| "adam"; "sgd" \|]` |
| `floats a` | the floats of an array, as quantities | `floats [\| 0.5 \|]` |
| `dim k` | the index along axis `k` of the mark's shape, as a category | `dim 0` |
| `index k` | the index along axis `k`, as a quantity | `index (-1)` |
| `const v` | one value for every row | `const Color.red` |

`num` and `cat` read nothing when the channel is made: the tensor stays where
it lives until the figure is rendered. A constant takes the range of its
role, so `~stroke:(const Color.red)` is a colour and `~x:(const 0.5)` is the
middle of the panel.

## Rows and Shapes

The channels of a mark broadcast together, as the operands of an nx
operation do, into the mark's *shape*. Each element of the shape is one
*row*. One tensor holding five curves of 200 points is 1,000 rows of one
`line`:

```ocaml
let steps = Nx.linspace Nx.float64 0. 1. 200
let rates = Nx.create Nx.float64 [| 5; 1 |] [| 2.; 2.5; 3.; 3.5; 4. |]
let curves = Nx.exp (Nx.neg (Nx.mul rates steps))

let () =
  line ~x:(num steps) ~y:(num curves) ~stroke:(dim ~title:(Text.v "rate") 0) ()
  |> save "curves.png"
```

`steps` has shape `[200]` and `curves` shape `[5; 200]`, so the mark has
shape `[5; 200]`. A line draws one curve per index of its leading axes, in
order along the last axis. `dim 0` reads the index along axis 0, the curve,
as a category: each curve takes its own colour and the legend lists them.
Without `~x`, a line reads `index (-1)`, the position along the last axis.

A value is *missing* if it is NaN or infinite, if the `valid` mask of its
channel is false there, or if its scale cannot place it, such as a value
that is not positive on a log scale. A row with a missing value is dropped,
and a line breaks there:

```ocaml
let wave = Nx.sin (Nx.mul_s steps 12.)

let () =
  line ~y:(num ~valid:(Nx.greater_s wave 0.) wave) ()
  |> save "positive-half.png"
```

## Saving

`save` writes PNG for `.png`, SVG for `.svg` and PDF for `.pdf`. A figure is
360 by 240 points by default, a point being 1/72 inch, and raster output is
drawn at 2 pixels per point. `Size` changes the page:

```ocaml
let () =
  line ~y:(num curves) ()
  |> save ~size:(Size.figure (Size.mm 85.) (Size.mm 60.)) "column.pdf"

let () =
  line ~y:(num curves) ()
  |> save ~size:(Size.panels 200. 120.) ~density:(Size.dpi 300.) "panel.png"
```

`Size.figure w h` fixes the whole page. `Size.panels w h` fixes each data
area instead, and the page grows to hold the axes, titles and legends around
it. `Size.dpi 300.` is a density of 300 pixels per inch.

## Notebooks

In a Quill notebook, a cell whose value is a figure shows the figure:

<!-- $MDX skip -->
```ocaml
let fig = Hugin.(line ~y:(num curves) ())
```

`Hugin.pp` renders the figure as SVG and hands it to the notebook. Printed
anywhere else, such as in a terminal toplevel, it is a one-line summary:
`hugin figure`.

## Errors and Warnings

Mistakes in a figure's structure raise `Invalid_argument` as early as they
can be found: a tensor of the wrong dtype when the channel is made, channels
that do not broadcast when the mark is made, two marks putting one panel on
two different x scales when the figure is resolved. Messages name the part
of the figure at fault.

Problems with data values are *warnings*. Nothing is substituted for the
values at fault, and the figure is still drawn. `save` prints each warning
on standard error, or gives it to `~warn`:

```ocaml
let warnings = ref []

let () =
  let codes = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; 7l |] in
  dot ~x:(index 0) ~y:(num (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]))
    ~fill:(cat ~labels:[| "train"; "test" |] codes) ()
  |> save ~warn:(fun w -> warnings := w :: !warnings) "codes.png"
```

```ocaml
# List.iter (Format.printf "%a@." pp_warning) !warnings;;
root: fill: 1 code is outside its 2 labels
- : unit = ()
```

A warning names the node of the figure it concerns by its id, a path from
the root of the figure: here the root, since the figure is one mark.
