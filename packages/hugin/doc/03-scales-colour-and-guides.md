# Scales, Colour and Guides

A scale maps data values to the interval `[0, 1]`, and a role maps that
interval to what it draws: a position across the panel, a colour, an area.
This page shows how channels find their scales, how to configure them, how
colours are chosen, and how axes and legends follow from the scales.

```ocaml
open Hugin

let steps = Nx.linspace Nx.float64 1. 1000. 200
let loss = Nx.add_s (Nx.div (Nx.ones_like steps) (Nx.sqrt steps)) 0.02
```

## Finding a Scale

Every channel reads a scale by name. Without a name, it reads the scale of
its role: a channel on `~y` reads `"y"`, one on `~fill` or `~stroke` reads
`"color"`. A scale specification given to a channel configures that scale,
so this line is drawn on a logarithmic y:

```ocaml
let () =
  line ~x:(num steps) ~y:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss) ()
  |> save "log.png"
```

The marks of a panel read one x and one y scale, and the scale's domain is
fitted to the values of every channel that reads it. A `layer` of a line and
a dot mark therefore shares its axes.

## Scale Constructors

`Hugin.Scale` makes specifications of each kind:

| Constructor | Normalises |
|---|---|
| `Scale.linear ()` | quantities, in proportion |
| `Scale.log ()` | positive quantities, by their logarithm |
| `Scale.symlog ()` | quantities of either sign, logarithmic away from zero |
| `Scale.pow ~exponent ()` | quantities, by a power |
| `Scale.band ()` | categories, one band each |

Each constructor's optional arguments set properties, and an unset property
keeps its default:

- `~domain` fixes the domain. The fitted domain is the extent of the data
  otherwise.
- `~nice:false` keeps the fitted domain at the data's extent instead of
  rounding it outward to round numbers.
- `~zero:true` widens the domain to include zero.
- `~reverse:true` runs the scale from 1 to 0.
- `~clamp:true` clamps values outside the domain to its ends.
- `~padding` (band scales) is the fraction of each band left empty.
- `~ticks` fixes the values of the scale's axis ticks, legend entries and
  colour bar. On a band scale it limits the labels of an axis and the
  entries of a legend to the categories named; facet headers name every
  panel whatever the ticks.
- `~notation` writes tick labels in `Number.Plain`, `Exponent`, `Si` or
  `Percent` notation.
- `~stepped:true` (quantitative scales) gives roles a range that is constant
  between consecutive ticks, so a colour scale paints bands.
- `~dashes` and `~symbols` (band scales) set the patterns and symbols its
  categories are drawn with.

```ocaml
let () =
  let acc = Nx.create Nx.float64 [| 5 |] [| 0.42; 0.61; 0.7; 0.74; 0.76 |] in
  line ~y:(num ~scale:(Scale.linear ~domain:(0., 1.) ()) ~title:(Text.v "accuracy") acc) ()
  |> save "domain.png"
```

Ticks are chosen to be round numbers whose labels do not overlap. `~ticks`
replaces that choice, and `~notation` the way labels are written:

```ocaml
let () =
  let lr = Nx.create Nx.float64 [| 5 |] [| 1e-4; 3e-4; 1e-3; 3e-3; 1e-2 |] in
  let acc = Nx.create Nx.float64 [| 5 |] [| 0.62; 0.71; 0.78; 0.74; 0.55 |] in
  line
    ~x:(num ~scale:(Scale.log ~ticks:[| 1e-4; 1e-3; 1e-2 |] ~notation:Number.Exponent ()) lr)
    ~y:(num ~scale:(Scale.linear ~notation:Number.Percent ()) acc)
    ()
  |> save "ticks.png"
```

The labels of the x axis read 10⁻⁴, 10⁻³ and 10⁻², and those of the y axis
are percentages. A tick label names a value; a label that is not the value,
such as the name of a category, belongs to a band scale.

Marks imply properties too: a bar implies `~zero:true` on its length's scale
and padding on its band scale. A property set on a channel overrides what a
mark implies, and two channels setting one property to different values
raise `Invalid_argument` when the figure is resolved.

## Sharing a Scale by Name

A name of your own denotes one scale wherever it appears in the figure. Two
panels that colour by the same quantity read the same scale, and the figure
has one colour bar:

```ocaml
let () =
  let rates = Nx.create Nx.float64 [| 5; 1 |] [| 0.1; 0.2; 0.3; 0.4; 0.5 |] in
  let curves = Nx.exp (Nx.neg (Nx.mul rates (Nx.reshape [| 1; 200 |] (Nx.div_s steps 100.)))) in
  let final = Nx.slice [ Nx.A; Nx.I 199 ] curves in
  let lr = Scale.linear ~name:"lr" () in
  grid
    [
      [
        line ~x:(num steps) ~y:(num curves) ~stroke:(num ~scale:lr ~title:(Text.v "rate") rates) ();
        dot ~x:(num ~scale:lr (Nx.flatten rates)) ~y:(num final)
          ~fill:(num ~scale:lr (Nx.flatten rates)) ();
      ];
    ]
  |> save "shared-scale.png"
```

The second panel reads `"lr"` on its x axis as well, so its axis and the
colour bar show the same domain.

## Colours

`Color.t` is an sRGB colour with an opacity:

```ocaml
let teal = Color.v 0.1 0.5 0.55
let faded = Color.with_alpha 0.4 teal
let blend = Color.mix 0.5 teal Color.white
let brand = Result.get_ok (Color.of_hex "#e4572e")
```

`Color.mix` interpolates in Oklab, a space where equal steps look equally
different, and `Color.contrast c` is black or white, whichever reads better
on `c`.

A colour scale paints with a *scheme*. A continuous scale reads the scheme's
continuous gradient, and a band scale gives each category the next colour of
the scheme's palette. `Hugin.Scheme` holds sequential schemes (`viridis`,
`magma`, `blues`, ...), diverging ones (`rdbu`, `brbg`, ...) and qualitative
palettes (`tableau10`, `okabe_ito`, `set2`, ...):

```ocaml
let () =
  let z = Nx.Rng.normal (Nx.Rng.key 1) Nx.float64 [| 10; 10 |] in
  rect ~x:(dim 1) ~y:(dim 0)
    ~fill:(num ~scale:(Scale.linear ~domain:(-3., 3.) ~scheme:Scheme.rdbu ()) z)
    ()
  |> save "diverging.png"
```

Without a scheme, a continuous scale takes the theme's scheme, viridis by
default, and a band scale the theme's palette, tableau10 by default.

`map_range` applies a function to a channel's values after its scale and
role have mapped them. Text drawn over coloured cells reads on each cell
with the contrasting colour:

```ocaml
let () =
  let counts = Nx.create Nx.int32 [| 2; 2 |] [| 48l; 2l; 5l; 45l |] in
  let share = num (Nx.div_s (Nx.cast Nx.float64 counts) 50.) in
  layer
    [
      rect ~x:(dim 1) ~y:(dim 0) ~fill:share ();
      text ~x:(dim 1) ~y:(dim 0) ~text:(num counts) ~fill:(map_range Color.contrast share) ();
    ]
  |> save "contrast.png"
```

## Axes and Legends

Each position scale gets an axis on its panel, and each other scale a
legend: a colour bar for a continuous colour scale, and one entry per
category or tick otherwise. A guide is titled by the `~title`s of the
channels that read its scale.

`axis` and `legend` replace a generated guide. Layered with the figure, they
draw nothing themselves and configure the guide of the named scale:

```ocaml
let () =
  let codes = Nx.create Nx.int32 [| 6 |] [| 0l; 0l; 1l; 1l; 2l; 2l |] in
  let values = Nx.create Nx.float64 [| 6 |] [| 1.; 2.; 2.5; 1.5; 3.; 2.8 |] in
  layer
    [
      dot ~x:(index 0) ~y:(num values) ~fill:(cat ~labels:[| "a"; "b"; "c" |] codes) ();
      axis ~grid:true "y";
      axis ~show:false "x";
      legend ~side:`Bottom "color";
    ]
  |> save "guides.png"
```

`~grid:true` draws lines across the panel at the axis's ticks, `~show:false`
hides a guide, and `~side` moves it.

A legend can also lie inside a corner of its panels, with
`` ~side:(`Inside `Top_right) ``. It is drawn under the marks and no corner
is chosen for being empty, so pick the one the data leaves empty:

```ocaml
let () =
  let t = Nx.linspace Nx.float64 0. 1. 50 in
  let curves = Nx.exp (Nx.neg (Nx.mul (Nx.create Nx.float64 [| 2; 1 |] [| 2.; 5. |]) t)) in
  layer
    [
      line ~x:(num t) ~y:(num curves) ~stroke:(dim ~labels:[| "slow"; "fast" |] 0) ();
      legend ~side:(`Inside `Top_right) "color";
    ]
  |> save "inside-legend.png"
```
