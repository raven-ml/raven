# Marks and Styling

Every visualization in Hugin starts with one or more marks. A mark constructor takes data arrays and optional visual properties and returns an immutable plot specification.

## Mark Catalog

### Line Plots

`line ~x ~y ()` connects points `(x.(i), y.(i))` with straight segments.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~x` | `Nx.float32_t` | required | X coordinates |
| `~y` | `Nx.float32_t` | required | Y coordinates |
| `~color` | `Color.t` | theme palette | Line color |
| `~line_width` | `float` | theme line width | Stroke width |
| `~line_style` | `` `Solid \| `Dashed \| `Dotted \| `Dash_dot `` | `` `Solid `` | Dash pattern |
| `~step` | `` `Pre \| `Post \| `Mid `` | none | Staircase interpolation |
| `~marker` | `marker` | none | Marker at each point |
| `~label` | `string` | none | Legend entry |
| `~alpha` | `float` | 1.0 | Opacity |

Step modes: `Post` holds each value until the next x-point, `Pre` steps at the current x-point, `Mid` steps at the midpoint.

### Scatter Plots

`point ~x ~y ()` places individual markers at data coordinates.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~x` | `Nx.float32_t` | required | X coordinates |
| `~y` | `Nx.float32_t` | required | Y coordinates |
| `~color` | `Color.t` | theme palette | Uniform color |
| `~color_by` | `Nx.float32_t` | none | Per-point values mapped through sequential colormap |
| `~size` | `float` | theme marker size | Uniform marker size |
| `~size_by` | `Nx.float32_t` | none | Per-point values for variable marker area |
| `~marker` | `marker` | `Circle` | Marker shape |
| `~label` | `string` | none | Legend entry |
| `~alpha` | `float` | 1.0 | Opacity |

When `~color_by` is set, a colorbar is displayed showing the value-to-color mapping.

### Bar Charts

`bar ~x ~height ()` draws vertical bars centered on `x` values.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~x` | `Nx.float32_t` | required | Bar center positions |
| `~height` | `Nx.float32_t` | required | Bar heights |
| `~width` | `float` | 0.8 | Bar width |
| `~bottom` | `float` | 0.0 | Baseline y-value |
| `~color` | `Color.t` | theme palette | Fill color |
| `~label` | `string` | none | Legend entry |
| `~alpha` | `float` | 1.0 | Opacity |

### Histograms

`hist ~x ()` bins the values in `x` and draws a bar chart.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~x` | `Nx.float32_t` | required | Data values |
| `~bins` | `` `Num of int \| `Edges of float array `` | `` `Num 10 `` | Number of bins or explicit edges |
| `~density` | `bool` | false | Normalize so total area equals 1.0 |
| `~color` | `Color.t` | theme palette | Fill color |
| `~label` | `string` | none | Legend entry |

### Reference Lines and Spans

`hline ~y ()` draws a horizontal line across the full plot width. `vline ~x ()` draws a vertical line across the full height. Both accept `~color`, `~line_width`, `~line_style`, `~label`, and `~alpha`.

`hspan ~y0 ~y1 ()` shades a horizontal band. `vspan ~x0 ~x1 ()` shades a vertical band. Both accept `~color`, `~alpha` (default 0.2), and `~label`.

### Fill Between

`fill_between ~x ~y1 ~y2 ()` fills the area between two curves. `~alpha` defaults to 0.3.

### Error Bars

`errorbar ~x ~y ~yerr ()` draws error bars at each point.

- `~yerr`: `` `Symmetric e `` draws y ± e, `` `Asymmetric (lo, hi) `` draws [y - lo, y + hi]
- `~xerr`: optional horizontal error bars with the same format
- `~cap_size`: cap width (defaults to half the theme marker size)

### Text

`text ~x ~y "label" ()` places a string at data coordinates `(x, y)`. Accepts `~color` and `~font_size`.

### Image

`image data` displays an Nx uint8 array as an image. `data` must have shape `[|h; w; 3|]` (RGB) or `[|h; w; 4|]` (RGBA).

### Colormapped Image

`imshow ~data ()` displays a 2-D float array through a colormap.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~data` | `Nx.float32_t` | required | 2-D array of shape `[|rows; cols|]` |
| `~stretch` | `` `Linear \| `Log \| `Sqrt \| `Asinh \| `Power of float `` | `` `Linear `` | Transfer function before colormap lookup |
| `~cmap` | `Cmap.t` | theme sequential | Colormap |
| `~vmin` | `float` | data min | Lower bound of color range |
| `~vmax` | `float` | data max | Upper bound of color range |

### Heatmap

`heatmap ~data ()` displays a 2-D array as a grid of colored cells. Row 0 appears at the top.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~data` | `Nx.float32_t` | required | 2-D array of shape `[|rows; cols|]` |
| `~annotate` | `bool` | false | Show cell values |
| `~cmap` | `Cmap.t` | theme sequential | Colormap |
| `~vmin` | `float` | data min | Lower bound |
| `~vmax` | `float` | data max | Upper bound |
| `~fmt` | `float -> string` | `Printf.sprintf "%.2g"` | Cell value formatter (when annotate is true) |

### Contour

`contour ~data ~x0 ~x1 ~y0 ~y1 ()` draws iso-level contour lines through a 2-D grid.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `~data` | `Nx.float32_t` | required | 2-D array of shape `[|rows; cols|]` |
| `~x0`, `~x1`, `~y0`, `~y1` | `float` | required | Data-space rectangle |
| `~levels` | `` `Num of int \| `Values of float array `` | `` `Num 8 `` | Number of levels or explicit values |
| `~filled` | `bool` | false | Fill regions between levels |
| `~cmap` | `Cmap.t` | theme sequential | Per-level colormap |
| `~color` | `Color.t` | none | Single stroke color (unfilled contours) |
| `~line_width` | `float` | theme line width | Stroke width |
| `~label` | `string` | none | Legend entry |
| `~alpha` | `float` | 1.0 | Opacity |

## Marker Shapes

Five built-in shapes:

| Marker | Description |
|--------|-------------|
| `Circle` | Filled circle |
| `Square` | Filled square |
| `Triangle` | Filled triangle |
| `Plus` | Plus sign (+) |
| `Star` | Five-pointed star |

Use with `line ~marker:Triangle` or `point ~marker:Star`.

## Auto-Coloring

When you omit `~color`, marks are colored automatically from the theme's categorical palette. The first mark in a spec gets `palette.(0)`, the second gets `palette.(1)`, and so on. Explicitly setting `~color` takes precedence.

## Next Steps

- [Layout and Decorations](/docs/hugin/layout-and-decorations/) — axes, scales, themes, multi-panel layouts
- [Colors and Colormaps](/docs/hugin/colors-and-colormaps/) — OKLCH color space, palettes, and colormap reference
