# Rendering and Extending

`save` hides three stages. This page opens them up: what each stage
computes, how to inspect and reuse its output, how a viewer's settings reach
a figure, and how to write a mark of your own.

```ocaml
open Hugin

let steps = Nx.linspace Nx.float64 0. 100. 101
let loss = Nx.add_s (Nx.exp (Nx.mul_s steps (-0.05))) 0.1
let fig = line ~x:(num steps) ~y:(num loss) ()
```

## Three Stages

```text
           view                   size, theme             density
             │                         │                     │
figure ── resolve ──▶ Resolved.t ── layout ──▶ Layout.t ── draw ──▶ Drawing.t
                     fitted scales            panels, ticks        picture, warnings
```

- `resolve` fits every scale to the data and checks the figure's
  composition. It reads neither size nor theme.
- `layout` measures text in the theme's faces, chooses ticks, and places the
  panels and guides on a page of the given size.
- `draw` reads the data, reduces marks with more rows than the output has
  pixels, and paints the figure as a picture.

`render` runs the three; `save` renders and writes a file. Each stage is a
pure function of its arguments, so its output can be inspected:

```ocaml
let resolved = resolve fig
let y = Resolved.scale resolved (Scale.linear ~name:"y" ())
```

```ocaml
# Scale.domain y;;
- : float Scale.domain = Scale.Floats (0.1, 1.1)
```

```ocaml
let laid = layout (Size.figure 360. 240.) resolved
```

```ocaml
# Layout.size laid;;
- : float * float = (360., 240.)
```

Each stage takes its previous output as `~prev` and returns it unchanged when
its inputs are equal, so a program that redraws a figure on every event pays
only for what changed:

```ocaml
let again = resolve ~prev:resolved fig
```

```ocaml
# again == resolved;;
- : bool = true
```

## Output Without Files

`Drawing.renderable` is the picture on its page. The renderers of
`hugin.vg.raster`, `hugin.vg.svg` and `hugin.vg.pdf` turn it into pixels or
documents, here an RGBA tensor and an SVG string:

```ocaml
let drawing = draw ~density:2. laid
let pixels = Hugin_vg_raster.render ~density:2. (Drawing.renderable drawing)
let svg = Hugin_vg_svg.render (Drawing.renderable drawing)
```

```ocaml
# Nx.shape pixels;;
- : int array = [|480; 720; 4|]
```

`Drawing.warnings` lists the warnings of all three stages.

## Views

A *view* holds values that a viewer sets on a figure, such as a slider's
position or a zoom. A key names a value, and `bind` builds the part of a
figure that reads it:

```ocaml
let smoothing = View.number "smoothing" ~init:0.5

let noisy =
  Nx.mul loss (Nx.exp (Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 0) Nx.float64 [| 101 |]) 0.2))

let live =
  bind smoothing (fun alpha -> line ~y:(num (Nx.ewma ~alpha noisy)) ())

let () = save ~view:View.(set smoothing 0.1 empty) "smoothed.png" live
```

Every continuous scale has a zoom key without declaring one. Its value sets
the scale's domain in place of the fitted one:

```ocaml
let x_zoom = View.zoom (Scale.linear ~name:"x" ())
let () = save ~view:View.(set x_zoom (Some (0., 20.)) empty) "zoomed.png" fig
```

Views live outside figures and keys are identified by their names, so a
figure rebuilt by the same code reads the values a viewer set on the
previous one.

## Writing a Mark

A mark is a list of channel bindings and a function that draws one panel's
rows as a picture. `Mark.v` makes one, and the built-in marks are made the
same way. This mark draws a needle at each point, in the direction of an
angle that reads no scale:

```ocaml
let angle = Role.value ~name:"angle"

let needles ~x ~y ~angle:a () =
  Mark.v ~name:"needles"
    [ Mark.bind Role.x x; Mark.bind Role.y y; Mark.bind angle a ]
    (fun rows ->
      let xs, ys = Mark.points rows in
      let th = Option.get (Mark.get rows angle) in
      let ink = Theme.ink (Mark.theme rows) in
      let needle i =
        Picture.stroke (Stroke.v 1.) ink
          (Path.polyline
             [| xs.(i); xs.(i) +. (10. *. cos th.(i)) |]
             [| ys.(i); ys.(i) -. (10. *. sin th.(i)) |])
      in
      Picture.group (List.init (Mark.length rows) needle))

let () =
  let gx = Nx.reshape [| 1; 12 |] (Nx.linspace Nx.float64 0. 1. 12) in
  let gy = Nx.reshape [| 8; 1 |] (Nx.linspace Nx.float64 0. 1. 8) in
  let theta = Nx.mul_s (Nx.add gx gy) 3. in
  needles ~x:(num gx) ~y:(num gy) ~angle:(num theta) () |> save "needles.png"
```

`Mark.points` gives each row's point on the page, in points, with the y
axis pointing down, which is why the needle's tip subtracts the sine. A
dropped row, or one outside the domain, is at NaN, and a path through NaN
leaves a gap, so the mark needs no check of its own. The needles are a figure
like any other: they layer, facet and fit their x and y scales.
