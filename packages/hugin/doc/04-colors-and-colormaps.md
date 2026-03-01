# Colors and Colormaps

Hugin uses the OKLCH color space for perceptually uniform color operations and ships with colorblind-friendly palettes and scientific colormaps.

## Colors

### OKLCH Color Space

Colors are represented internally in [OKLCH](https://bottosson.github.io/posts/oklab/), a perceptually uniform color space. Operations like `lighten`, `darken`, and `mix` produce visually consistent results: equal numerical steps yield equal perceived differences.

OKLCH components:

| Component | Range | Description |
|-----------|-------|-------------|
| Lightness (L) | [0, 1] | Black to white |
| Chroma (C) | [0, ~0.4] | Gray to saturated |
| Hue (H) | [0, 360) | Color wheel angle |
| Alpha (A) | [0, 1] | Transparency |

### Constructors

<!-- $MDX skip -->
```ocaml
(* From OKLCH components *)
Color.oklch ~l:0.7 ~c:0.15 ~h:145. ()
Color.oklcha ~l:0.7 ~c:0.15 ~h:145. ~a:0.5 ()

(* From sRGB [0, 1] *)
Color.rgb ~r:0.2 ~g:0.6 ~b:0.8 ()
Color.rgba ~r:0.2 ~g:0.6 ~b:0.8 ~a:0.5 ()

(* From hex string *)
Color.hex "#3399CC"
Color.hex "#3399CCAA"  (* with alpha *)
```

All constructors convert to OKLCH on creation. The reverse conversion (`to_rgba`) is called at render time.

### Accessors

<!-- $MDX skip -->
```ocaml
Color.lightness c    (* OKLCH lightness *)
Color.chroma c       (* OKLCH chroma *)
Color.hue c          (* OKLCH hue in degrees *)
Color.alpha c        (* alpha channel *)
Color.to_rgba c      (* sRGB (r, g, b, a) tuple, clamped to gamut *)
```

### Operations

<!-- $MDX skip -->
```ocaml
Color.lighten 0.1 c      (* increase lightness by 0.1, clamped to [0, 1] *)
Color.darken 0.1 c       (* decrease lightness by 0.1, clamped to [0, 1] *)
Color.with_alpha 0.5 c   (* set alpha *)
Color.mix 0.5 a b        (* blend a and b: 0.0 = a, 1.0 = b *)
```

`mix` interpolates all OKLCH components. Hue follows the shortest arc on the color wheel.

### Named Colors

The default named colors follow the [Okabe-Ito palette](https://jfly.uni-koeln.de/color/), designed to be distinguishable under all forms of color-vision deficiency:

| Color | Value |
|-------|-------|
| `Color.orange` | Okabe-Ito orange |
| `Color.sky_blue` | Okabe-Ito sky blue |
| `Color.green` | Okabe-Ito bluish green |
| `Color.yellow` | Okabe-Ito yellow |
| `Color.blue` | Okabe-Ito blue |
| `Color.vermillion` | Okabe-Ito vermillion |
| `Color.purple` | Okabe-Ito reddish purple |
| `Color.black` | Black |
| `Color.white` | White |
| `Color.gray` | Neutral gray |

### Formatting

`Color.pp` formats as `oklch(L C H / A)` for debugging.

## Colormaps

A colormap is a continuous mapping from [0, 1] to `Color.t`. Internally stored as a 256-entry lookup table with OKLCH interpolation.

### Evaluation

<!-- $MDX skip -->
```ocaml
let c = Cmap.eval Cmap.viridis 0.5  (* color at midpoint *)
```

Values are clamped to [0, 1].

### Predefined Colormaps

Perceptually uniform sequential colormaps from the [viridis family](https://bids.github.io/colormap/):

| Colormap | Description |
|----------|-------------|
| `Cmap.viridis` | Purple-teal-yellow (default) |
| `Cmap.plasma` | Purple-orange-yellow |
| `Cmap.inferno` | Black-purple-orange-yellow |
| `Cmap.magma` | Black-purple-pink-yellow |
| `Cmap.cividis` | Optimized for color-vision deficiency |

Other colormaps:

| Colormap | Description |
|----------|-------------|
| `Cmap.coolwarm` | Blue-white-red diverging |
| `Cmap.gray` | Black to white |
| `Cmap.gray_r` | White to black (standard for astronomy) |
| `Cmap.hot` | Black-red-yellow-white |

### Custom Colormaps

`Cmap.of_colors` creates a colormap by interpolating linearly through an array of color stops in OKLCH space:

<!-- $MDX skip -->
```ocaml
let my_cmap = Cmap.of_colors [|
  Color.hex "#000080";
  Color.hex "#FFFFFF";
  Color.hex "#800000";
|]
```

Stops are evenly spaced from 0 to 1. Requires at least 2 colors.

## Using Colors with Marks

### Uniform Color

Set `~color` on any mark:

<!-- $MDX skip -->
```ocaml
line ~x ~y ~color:Color.vermillion ()
bar ~x ~height ~color:(Color.hex "#336699") ()
```

### Data-Driven Color

`point` supports `~color_by` to map per-point values through the theme's sequential colormap:

<!-- $MDX skip -->
```ocaml
point ~x ~y ~color_by:temperature ~marker:Circle ()
```

A colorbar is displayed automatically.

### Colormaps on 2-D Data

`heatmap`, `imshow`, and `contour` accept `~cmap` to override the default:

<!-- $MDX skip -->
```ocaml
heatmap ~data ~cmap:Cmap.coolwarm ()
imshow ~data ~cmap:Cmap.inferno ~stretch:`Log ()
contour ~data ~x0 ~x1 ~y0 ~y1 ~filled:true ~cmap:Cmap.plasma ()
```

## Next Steps

- [Matplotlib Comparison](/docs/hugin/matplotlib-comparison/) — side-by-side with Python
- [Marks and Styling](/docs/hugin/marks-and-styling/) — full mark catalog
