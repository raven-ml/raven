# Porting from the First Hugin

The first hugin built a plot from mark functions with options per mark,
then set titles, labels, limits and scales on the plot with functions of
their own. The current hugin moves those settings onto the channels and
scales they concern: an axis label is the title of the channel that reads
the axis's scale, a limit is the domain of that scale. This page maps each
function of the first hugin to its counterpart.

```ocaml
open Hugin

let x = Nx.linspace Nx.float64 0. 6.28 100
let y = Nx.sin x
```

## A Ported Plot

A line with a label, axis labels, a limit, a log scale and a legend in a
corner:

<!-- $MDX skip -->
```ocaml
(* The first hugin *)
Hugin.line ~x ~y ~label:"sin" ~line_style:`Dashed ()
|> Hugin.xlabel "x" |> Hugin.ylabel "y" |> Hugin.ylim (-1.) 1.
|> Hugin.legend ~loc:Upper_right
|> Hugin.render_png "sin.png"
```

```ocaml
let () =
  layer
    [
      line
        ~x:(num ~title:"x" x)
        ~y:(num ~scale:(Scale.linear ~domain:(-1., 1.) ()) ~title:"y" y)
        ~stroke:(strings [| "sin" |]) ~dash:(const Dash.dashed) ();
      legend ~side:(`Inside `Top_right) "color";
    ]
  |> save "sin.png"
```

A legend entry comes from a category of a scale, so the label becomes a
categorical channel: `strings [| "sin" |]` is one category, which the colour
scale gives its first colour and its legend lists.

## Marks

| First hugin | Current hugin |
|---|---|
| `line ~x ~y ()` | `line ~x:(num x) ~y:(num y) ()` |
| `line ~color c` | `line ~stroke:(const c)` |
| `line ~line_width w` | `line ~width:(const w)` |
| `line ~line_style` | `line ~dash:(const Dash.dashed)`, `Dash.dotted`, `Dash.dash_dot` |
| `` line ~step:`Post ``, `` `Pre ``, `` `Mid `` | `line ~curve:Curve.step_after`, `step_before`, `step_mid` |
| `line ~marker` | `layer [ line ...; dot ~symbol:(const Symbol.square) ... ]` |
| `~label` on any mark | a categorical colour channel, as `~stroke:(strings [\| "train" \|])` |
| `~alpha a` on any mark | `~opacity:(const a)` |
| `point ~x ~y ()` | `dot ~x:(num x) ~y:(num y) ()` |
| `point ~color_by v`, `~size_by v` | `dot ~fill:(num v)`, `~size:(num v)` |
| `point ~marker` | `dot ~symbol:(const Symbol.diamond)` |
| `bar ~x ~height ()` | `rect ~x:(strings names) ~y:(num height) ()` |
| `bar ~bottom b` | `rect ~y2:(floats [\| b \|])` |
| `bar ~width` | `Scale.band ~padding` on the bars' x scale |
| `` hist ~x ~bins:(`Num n) ~density `` | `Stats.histogram ~bins:n x`, then `rect ~x:(num h.x) ~x2:(num h.x2) ~y:(num h.density) ()` |
| `text ~x ~y s ()` | `text ~x:(floats [\| x \|]) ~y:(floats [\| y \|]) ~text:(const (Text.v s)) ()` |
| `text ~font_size` | `Text.scale k (Text.v s)` |
| `hline ~y ()`, `vline ~x ()` | `rule ~y:(floats [\| y \|]) ()`, `rule ~x:(floats [\| x \|]) ()` |
| `abline ~slope ~intercept ()` | `abline ~slope:(const a) ~intercept:(const b) ()` |
| `fill_between ~x ~y1 ~y2 ()` | `area ~x:(num x) ~y:(num y1) ~y2:(num y2) ()` |
| `fill_between ~where m` | `area ~y:(num ~valid:m y1)` |
| `hspan ~y0 ~y1 ()` | `rect ~y:(floats [\| y0 \|]) ~y2:(floats [\| y1 \|]) ()` |
| `vspan ~x0 ~x1 ()` | `rect ~x:(floats [\| x0 \|]) ~x2:(floats [\| x1 \|]) ()` |
| `errorbar ~x ~y ~yerr ()` | `layer [ rule ~x ~y:(num lo) ~y2:(num hi) (); dot ~x ~y () ]` |
| `errorbar ~xerr` | `rule ~y ~x:(num lo) ~x2:(num hi) ()` |
| `heatmap ~data ()` | `rect ~x:(dim 1) ~y:(dim 0) ~fill:(num data) ()` |
| `heatmap ~annotate:true` | layer `text ~x:(dim 1) ~y:(dim 0) ~text:(num data) ()` over it |
| `heatmap ~vmin ~vmax`, `~cmap` | `~fill:(num ~scale:(Scale.linear ~domain:(vmin, vmax) ~scheme:Scheme.magma ()) data)` |
| `imshow ~data ()` of a field | the heatmap above, drawn as one image when its cells are many |
| `imshow ~stretch` | the fill scale's constructor: `Scale.log`, `Scale.pow`, `Scale.symlog` |
| `image data` | `image data` |
| `contour ~data ~filled:true ()` | `contour ~fill:(num data) ()` |
| `` contour ~levels:(`Values vs) `` | `contour ~fill:(num ~scale:(Scale.linear ~ticks:vs ()) data)` |

## Decorations

| First hugin | Current hugin |
|---|---|
| `layers fs` | `layer fs` |
| `title s` | `title s` |
| `xlabel s`, `ylabel s` | `~title:s` on the channel given to `~x` or `~y`, or `layer [ f; axis ~title:s "x" ]` |
| `xlim lo hi`, `ylim lo hi` | `~scale:(Scale.linear ~domain:(lo, hi) ())` on that channel |
| `` xscale `Log ``, `` `Sqrt ``, `` `Symlog c `` | `Scale.log ()`, `Scale.pow ~exponent:0.5 ()`, `Scale.symlog ~constant:c ()` |
| `` xscale `Asinh `` | `Scale.custom ~transform:"asinh" ~forward:Float.asinh ~inverse:Float.sinh ()` |
| `xinvert`, `yinvert` | `~reverse:true` on the scale |
| `xticks [ (v, label); ... ]` | `~ticks:[\| v; ... \|]` on the scale |
| `xtick_format f` | `~notation:Number.Percent`, `Number.Si`, `Number.Exponent` on the scale |
| `grid_lines true` | `layer [ f; axis ~grid:true "x"; axis ~grid:true "y" ]` |
| `legend ~loc:Upper_right` | `` layer [ f; legend ~side:(`Inside `Top_right) "color" ] ``, and likewise for the other corners |
| `legend ~loc:Right` | the default, `` legend ~side:`Right "color" `` |
| `frame true` | `layer [ f; frame () ]` |
| `no_axes` | `layer [ f; axis ~show:false "x"; axis ~show:false "y" ]`; `image` shows no axes already |
| `with_theme th` | `save ~theme:th` |

Tick labels name their values. `xticks` with labels other than the values
and `xtick_format` with a function of your own have no counterpart: a label
that names something other than a value, such as a category, belongs to a
band scale (`strings`, `cat ~labels`, `dim ~labels`). `legend ~ncol` and the
legend locations `Center`, `Upper_center` and `Lower_center` have none either;
a legend set below its panels with `` ~side:`Bottom `` wraps its entries into
rows.

## Layout and Output

| First hugin | Current hugin |
|---|---|
| `hstack fs` | `grid [ fs ]` |
| `vstack fs` | `grid (List.map (fun f -> [ f ]) fs)` |
| `grid rows` | `grid rows`; gaps come from the theme |
| `render_png file f`, `render_svg`, `render_pdf` | `save file f`, the format from the extension |
| `~width`, `~height` | `save ~size:(Size.figure w h)`, in points |
| `render_svg_to_string f` | `Hugin_vg_svg.render (Drawing.renderable (render size f))` |
| `render_to_buffer f` | `Hugin_vg_raster.png ~density:2. (Drawing.renderable (render size f))` |
| `pp` | `pp` |

## Themes and Colours

| First hugin | Current hugin |
|---|---|
| `Theme.default` | `Theme.default` |
| `Theme.dark` | `Theme.dark` |
| `Theme.minimal` | `Theme.default`, which draws no grid lines |
| `Theme.paper th` | `Theme.default`, whose base size is 10 points |
| `Theme.notebook th` | `Theme.v ~size:13. ()` |
| `Theme.talk th`, `Theme.poster th` | `Theme.talk`, `Theme.poster` |
| `Color.rgb ~r ~g ~b ()` | `Color.v r g b` |
| `Color.hex s` | `Color.of_hex s`, a `result` |
| `Color.oklch ~l ~c ~h ()` | `Color.of_oklch l c h`, with the hue in radians |
| `Cmap.viridis`, `Cmap.coolwarm`, ... | `Scheme.viridis`, `Scheme.rdbu`, ... given to a scale's `~scheme` |
| `Cmap.gray` for an image | none needed: `image` draws one channel as grey |

A context preset of the first hugin scaled a theme it was given; a preset
now is a theme, and `Theme.v` combines presets through their accessors, as
[Composition and Layout](04-composition-and-layout.md) shows.
