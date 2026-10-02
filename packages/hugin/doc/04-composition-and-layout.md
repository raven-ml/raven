# Composition and Layout

Figures combine into figures. This page covers drawing figures over one
another, arranging them in grids and facets, deciding which parts share a
scale, and the titles, sizes and themes of the result.

```ocaml
open Hugin

let t = Nx.linspace Nx.float64 0. 10. 200
let signal k = Nx.mul (Nx.exp (Nx.mul_s t (-0.1 *. k))) (Nx.sin t)
```

## Layers

`layer` draws its figures over one another, the first at the bottom. The
children of a layer share every scale, so they share their axes and
legends:

```ocaml
let () =
  layer
    [
      area ~x:(num t) ~y:(num (signal 1.)) ~opacity:(const 0.2) ();
      line ~x:(num t) ~y:(num (signal 1.)) ();
      rule ~y:(floats [| 0. |]) ();
    ]
  |> save "layers.png"
```

## Grids

`grid` arranges figures in rows of cells, from the top and from the left.
The data areas of a column share their left and right edges, and those of a
row their top and bottom edges, whatever the width of each cell's tick
labels:

```ocaml
let decay = line ~x:(num t) ~y:(num (signal 0.5)) ()
let fast = line ~x:(num t) ~y:(num (Nx.mul_s (signal 2.) 1000.)) ()
let hist = rect ~x:(strings [| "a"; "b" |]) ~y:(floats [| 3.; 5. |]) ()

let () =
  grid [ [ decay; fast ]; [ hist; layer [] ] ]
  |> save ~size:(Size.figure 420. 300.) "grid.png"
```

`layer []` is an empty cell. `~widths` and `~heights` weigh the columns and
rows, and `span ~cols:2 f` makes `f` a cell two columns wide:

```ocaml
let () =
  grid ~heights:[ 2.; 1. ] [ [ span ~cols:2 decay ]; [ fast; hist ] ]
  |> save ~size:(Size.figure 420. 300.) "spans.png"
```

## Which Scales Are Shared

The cells of a grid keep their own x and y scales, and every other scale is
shared by the whole figure, so a figure has one colour legend however it is
arranged. `share` changes that for one name at one node:

```ocaml
let () =
  grid [ [ decay ]; [ line ~x:(num t) ~y:(num (signal 1.)) () ] ]
  |> share [ ("x", `Shared) ]
  |> save "shared-x.png"
```

With one x scale, the two cells line up on one domain and the axis is
labelled once, below the bottom cell. `` share [ ("color", `Independent) ] ``
gives each cell of a grid its own colour scale and legend.

## Facets

The facet channels `~fx` and `~fy` split one mark into panels: each row is
drawn in the column of its `fx` category and the row of its `fy` category.
Facet panels share every scale:

```ocaml
let damping = Nx.create Nx.float64 [| 4; 1 |] [| 0.1; 0.2; 0.4; 0.8 |]
let signals = Nx.mul (Nx.exp (Nx.neg (Nx.mul damping t))) (Nx.sin t)

let () =
  line ~x:(num t) ~y:(num signals)
    ~fx:(dim ~title:"damping" ~labels:[| "0.1"; "0.2"; "0.4"; "0.8" |] 0)
    ()
  |> save ~size:(Size.figure 480. 200.) "facets.png"
```

A band scale with `~wrap` wraps the panels into rows of that many:
`~fx:(dim ~scale:(Scale.band ~wrap:2 ()) 0)` puts the four panels in two rows.
A mark without facet channels, layered with a faceted one, is drawn in every
panel, so a reference line appears in each.

## Titles and Coordinates

`title` sets a bold title above a figure, aligned with the left edge of its
data areas by default. Titles nest, and a grid of titled cells labels each
panel:

```ocaml
let () =
  grid [ [ title "(a)" decay; title "(b)" fast ] ]
  |> title ~align:`Center "Two decays"
  |> save ~size:(Size.figure 480. 220.) "titled.png"
```

`coord` sets the coordinate system of a figure's panels.
`Coord.cartesian ~aspect:1. ()` makes one unit of y as long as one unit of x,
which keeps a heatmap's cells square or a trajectory's shape true:

```ocaml
let () =
  let angle = Nx.linspace Nx.float64 0. 6.28 100 in
  line ~x:(num (Nx.cos angle)) ~y:(num (Nx.sin angle)) ()
  |> coord (Coord.cartesian ~aspect:1. ())
  |> save "circle.png"
```

## Sizes

A size fixes one length and lets the layout derive the rest.
`Size.figure w h` fixes the page, and the panels share what the axes,
titles and legends leave. `Size.panels w h` fixes each data area, and the
page grows to hold the rest. In a grid, `Size.panels` gives each column of
weight `k` a data area `k *. w` points wide. Lengths are in points;
`Size.mm` converts millimetres.

If the page is too small for what it must hold, layout raises
`Invalid_argument` and names a size at which the figure fits.

## Themes

A theme holds what a figure does not state: colours of ink, paper and
marks, the faces text is set in, the base size, the default colour schemes
and the locale numbers are written in. Every length derives from the base
size, so `~size:8.` sets a figure for a printed column:

```ocaml
let print = Theme.v ~size:8. ~palette:Scheme.okabe_ito ()

let () =
  grid [ [ title "(a)" decay; title "(b)" fast ] ]
  |> save ~theme:print ~size:(Size.figure (Size.mm 85.) (Size.mm 40.)) "column.pdf"
```

`Theme.dark` sets light ink on dark paper, and a scheme for continuous
colour scales that starts at viridis's middle, since its darkest colours
would fade into the paper. `Theme.talk` and `Theme.poster` enlarge text and
every length for slides and posters. Presets are themes like any other,
combined through their accessors:

```ocaml
let slides =
  Theme.(v ~ink:(ink dark) ~paper:(paper dark) ~scheme:(scheme dark) ~size:(size talk) ())

let () = decay |> save ~theme:Theme.dark "dark.png"
let () = decay |> save ~theme:slides ~size:(Size.figure 576. 384.) "slide.png"
```

The same figure renders in any theme, since the theme is given at
rendering.
