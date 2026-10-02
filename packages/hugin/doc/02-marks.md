# Marks

A mark draws rows of data. Each built-in mark takes its channels as labelled
arguments named after its roles, and every mark takes `~opacity` (except
`image`) and the facet channels `~fx` and `~fy`. This page goes through the
marks and the figures each one makes.

```ocaml
open Hugin

let key k = Nx.Rng.key k
```

## Roles

A role fixes what a constant means and which scale a channel reads:

| Role | A constant is | Scale |
|---|---|---|
| `x`, `y`, `x2`, `y2` | a position in the panel, `0.` to `1.` | `"x"`, `"y"` |
| `fill`, `stroke` | a `Color.t` | `"color"` |
| `opacity` | an opacity, `0.` to `1.` | `"opacity"` |
| `size` | an area in square points | `"size"` |
| `width` | a line width in points | `"width"` |
| `symbol` | a `Symbol.t` | `"symbol"` |
| `dash` | a `Dash.t` | `"dash"` |
| `text` | a `Text.t` | none |
| `fx`, `fy` | the panel to draw in | `"fx"`, `"fy"` |

A mark without a colour channel paints with the theme's accent colour, and
`rule` and `text` with its ink.

## Dots

`dot` draws a symbol at each row's point. A categorical `~fill` colours the
dots by category, and `~size` scales their area:

```ocaml
let () =
  let n = 200 in
  let keys = Nx.Rng.split ~n:3 (key 1) in
  let group = Nx.cast Nx.int32 (Nx.div_s (Nx.arange Nx.int32 0 n 1) 100l) in
  let px = Nx.Rng.normal keys.(0) Nx.float64 [| n |] in
  let py = Nx.add (Nx.mul_s (Nx.cast Nx.float64 group) 2.) (Nx.Rng.normal keys.(1) Nx.float64 [| n |]) in
  let labels = [| "control"; "treated" |] in
  dot ~x:(num px) ~y:(num py)
    ~fill:(cat ~title:"group" ~labels group)
    ~size:(num ~title:"weight" (Nx.Rng.uniform keys.(2) Nx.float64 [| n |]))
    ()
  |> save "dots.png"
```

`~fill` reads the `"color"` scale and `~size` the `"size"` scale, and each
yields a legend. `~symbol` changes the dots' shape, by category or with a
constant such as `const Symbol.diamond`. A dot mark with tens of thousands of
rows in a panel is drawn as one image of its rows, so a large scatter stays
cheap to render and to store.

## Lines and Areas

`line` draws a curve through the points of each *series* of its rows. A
series is one index along every axis but the last, split further by the
categories of its colour channels. `~curve` says how the curve passes
through the points:

```ocaml
let t = Nx.linspace Nx.float64 0. 10. 11
let level = Nx.floor (Nx.mul_s (Nx.sin t) 3.)

let () =
  layer
    [
      line ~x:(num t) ~y:(num level) ~curve:Curve.step_after ();
      line ~x:(num t) ~y:(num (Nx.mul_s (Nx.sin t) 3.)) ~curve:Curve.natural
        ~stroke:(const (Color.gray 0.6)) ();
    ]
  |> save "steps.png"
```

`~dash` breaks a line into a dash pattern. A constant dashes every series,
and a categorical channel gives each category a pattern of its own. Reading
one band scale for both `~stroke` and `~dash` tells the series apart by
colour and pattern in one legend. A channel serves roles of one range, a
colour or a pattern, so `by_run` makes one for each:

```ocaml
let () =
  let curves = Nx.sin (Nx.add (Nx.reshape [| 3; 1 |] (Nx.linspace Nx.float64 0. 1. 3)) t) in
  let run = Scale.band ~name:"run" () in
  let by_run () = dim ~scale:run ~title:"run" ~labels:[| "a"; "b"; "c" |] 0 in
  line ~x:(num t) ~y:(num curves) ~stroke:(by_run ()) ~dash:(by_run ()) ()
  |> save "dashes.png"
```

`Dash` has the patterns `solid`, `dashed`, `dotted` and `dash_dot`, and
`Dash.v` makes one from alternating dash and gap lengths, in multiples of the
line's width so that a pattern keeps its look at every width.

`area` fills the region between a curve and a baseline. Without `~y2` the
baseline is zero; with it, the area is a band between two curves:

```ocaml
let () =
  let runs =
    Nx.add (Nx.reshape [| 1; 11 |] (Nx.sin t))
      (Nx.mul_s (Nx.Rng.normal (key 2) Nx.float64 [| 8; 11 |]) 0.2)
  in
  let lo = Nx.min ~axes:[ 0 ] runs and hi = Nx.max ~axes:[ 0 ] runs in
  layer
    [
      area ~x:(num t) ~y:(num lo) ~y2:(num hi) ~opacity:(const 0.3) ();
      line ~x:(num t) ~y:(num (Nx.mean ~axes:[ 0 ] runs)) ();
    ]
  |> save "band.png"
```

`layer` draws figures over one another on shared scales; [Composition and
Layout](04-composition-and-layout.md) covers it.

## Rectangles

`rect` draws the rectangle each row spans along x and y. What it spans
depends on its channels, so one mark draws bars, heatmaps, histograms and
spans.

A position on a categorical scale spans its band, and a quantity alone spans
from zero to its value. Bars are categories along x and lengths along y:

```ocaml
let () =
  let models = [| "small"; "base"; "large" |] in
  rect
    ~x:(strings ~title:"model" models)
    ~y:(num ~title:"accuracy" (Nx.create Nx.float64 [| 3 |] [| 0.71; 0.78; 0.83 |]))
    ()
  |> save "bars.png"
```

A matrix is a heatmap: its rows and columns read through `dim`, its values
through `~fill`, and a continuous colour scale yields a colour bar:

```ocaml
let () =
  let z = Nx.Rng.normal (key 3) Nx.float64 [| 12; 16 |] in
  rect ~x:(dim 1) ~y:(dim 0) ~fill:(num ~title:"z" z) ()
  |> save "heatmap.png"
```

With `~x2`, a row spans from `x` to `x2`. `Stats.histogram` counts values
into bins and gives each bin's edges, so a histogram is a `rect` from edge to
edge:

```ocaml
let () =
  let h = Stats.histogram ~bins:30 (Nx.Rng.normal (key 4) Nx.float64 [| 2_000 |]) in
  rect ~x:(num h.x) ~x2:(num h.x2) ~y:(num ~title:"count" h.count) ()
  |> save "histogram.png"
```

A rect without a y channel spans the whole height of its panel, which
shades a range of x:

```ocaml
let () =
  layer
    [
      rect ~x:(floats [| 2. |]) ~x2:(floats [| 4. |])
        ~fill:(const (Color.gray 0.9)) ();
      line ~x:(num t) ~y:(num (Nx.sin t)) ();
    ]
  |> save "span.png"
```

## Rules

`rule` draws a straight segment per row. With `~y` alone it crosses the
panel horizontally, with `~x` alone vertically, and with both `~x` and `~y`
it is a stem from zero:

```ocaml
let () =
  layer
    [
      rule ~y:(floats [| 0. |]) ();
      rule ~x:(num t) ~y:(num (Nx.sin t)) ~stroke:(const (Color.gray 0.5)) ();
      dot ~x:(num t) ~y:(num (Nx.sin t)) ();
    ]
  |> save "stems.png"
```

A rule from `~y` to `~y2` at each `~x` is an error bar:

```ocaml
let () =
  let names = strings [| "a"; "b"; "c" |] in
  let mean = Nx.create Nx.float64 [| 3 |] [| 0.6; 0.75; 0.7 |] in
  let err = Nx.create Nx.float64 [| 3 |] [| 0.05; 0.03; 0.08 |] in
  layer
    [
      rule ~x:names ~y:(num (Nx.sub mean err)) ~y2:(num (Nx.add mean err)) ();
      dot ~x:names ~y:(num mean) ();
    ]
  |> save "errorbars.png"
```

`~dash` breaks rules as it breaks lines, so
`rule ~y:(floats [| 0. |]) ~dash:(const Dash.dashed) ()` is a dashed
reference line.

`abline` draws the line `y = slope × x + intercept` across each panel's x
domain. Its slope and intercept read no scale, so the line never widens the
domains the data fits:

```ocaml
let () =
  let xs = Nx.Rng.uniform (key 6) Nx.float64 [| 50 |] in
  let ys = Nx.add (Nx.mul_s xs 2.) (Nx.mul_s (Nx.Rng.normal (key 7) Nx.float64 [| 50 |]) 0.2) in
  layer
    [
      dot ~x:(num xs) ~y:(num ys) ();
      abline ~slope:(const 2.) ~intercept:(const 0.) ~dash:(const Dash.dashed) ();
    ]
  |> save "abline.png"
```

With channels of several values, as `~slope:(num slopes)`, it draws one line
per row. `frame ()` outlines each panel's data area, under the axes:
`layer [ f; frame () ]` boxes the panels of `f`.

## Text

`text` writes each row's text centred on its point, moved `~dx` points right
and `~dy` points up. The numbers of a mark share their decimals, as many as
the most precise of them needs:

```ocaml
let () =
  let scores = Nx.create Nx.float64 [| 3 |] [| 0.71; 0.78; 0.834 |] in
  let x = strings [| "small"; "base"; "large" |] in
  layer
    [
      rect ~x ~y:(num scores) ();
      text ~x ~y:(num scores) ~text:(num scores) ~dy:6. ();
    ]
  |> save "labelled-bars.png"
```

`~text` also takes `strings`, `cat ~labels` or `const (Text.v "...")`.

## Images

`image` draws a tensor's pixels. A rank-2 tensor is a grey image; otherwise
the last three axes are height, width and 1, 3 or 4 channels. Values range
over 0 to 255 for `uint8` and over 0 to 1 for floats. Leading axes hold
several images, which the facet channels put in panels of their own:

```ocaml
let () =
  let batch = Nx.Rng.uniform (key 5) Nx.float32 [| 6; 16; 16; 3 |] in
  image ~fx:(dim ~scale:(Scale.band ~wrap:3 ()) 0) batch
  |> save "images.png"
```

An image covers `[0, w]` on x and `[0, h]` on y with its first row at the
top, and has no axes, so marks layered over it draw in pixel coordinates.

## Contours

`contour` fills the regions between levels of a field sampled on a grid.
The last two axes of its shape are the grid's rows and columns. `~x` gives
each column's position and `~y` each row's, and the levels are the ticks of
the colour bar:

```ocaml
let () =
  let xs = Nx.linspace Nx.float64 (-2.) 2. 60 in
  let ys = Nx.reshape [| 50; 1 |] (Nx.linspace Nx.float64 (-1.5) 1.5 50) in
  let field = Nx.add (Nx.square xs) (Nx.mul_s (Nx.square ys) 2.) in
  contour ~x:(num xs) ~y:(num ys) ~fill:(num ~title:"f" field) ()
  |> save "contour.png"
```
