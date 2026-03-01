# Layout and Decorations

Decorations add metadata and styling to a plot specification. Layout functions arrange multiple specs into multi-panel figures.

## Decorations

All decoration functions take a `t` and return a new `t`, designed for the `|>` pipeline:

<!-- $MDX skip -->
```ocaml
line ~x ~y ()
|> title "Frequency Response"
|> xlabel "Frequency (Hz)"
|> ylabel "Magnitude (dB)"
|> xscale `Log
|> ylim (-60.) 0.
|> grid_lines true
```

### Titles and Labels

| Function | Description |
|----------|-------------|
| `title s t` | Plot title |
| `xlabel s t` | X-axis label |
| `ylabel s t` | Y-axis label |

### Axis Limits

| Function | Description |
|----------|-------------|
| `xlim lo hi t` | Fix x-axis range to [lo, hi] |
| `ylim lo hi t` | Fix y-axis range to [lo, hi] |

When omitted, axis ranges are computed automatically from the data with 5% padding.

### Axis Scales

| Function | Description |
|----------|-------------|
| `xscale s t` | Set x-axis scale |
| `yscale s t` | Set y-axis scale |

Available scales:

| Scale | When to use |
|-------|-------------|
| `` `Linear `` | Default. Uniform spacing. |
| `` `Log `` | Data spanning multiple orders of magnitude. All values must be positive. |
| `` `Sqrt `` | Moderate compression of large values. Handles zero. |
| `` `Asinh `` | Like log but handles zero and negative values. Transitions smoothly from linear near zero to logarithmic at large magnitudes. |
| `` `Symlog linthresh `` | Linear within [-linthresh, linthresh], logarithmic outside. Good for data with both small and large values centered around zero. |

### Axis Direction

| Function | Description |
|----------|-------------|
| `xinvert t` | X-axis values increase right-to-left |
| `yinvert t` | Y-axis values increase top-to-bottom |

Useful for conventions like right ascension in sky charts (xinvert) or magnitude axes in HR diagrams (yinvert).

### Ticks

| Function | Description |
|----------|-------------|
| `xticks ticks t` | Explicit tick positions and labels as `(float * string) list` |
| `yticks ticks t` | Same for y-axis |
| `xtick_format fmt t` | Custom tick label formatter (preserves auto-generated positions) |
| `ytick_format fmt t` | Same for y-axis |

Example with explicit ticks:

<!-- $MDX skip -->
```ocaml
line ~x ~y ()
|> xticks [ (0., "Jan"); (1., "Feb"); (2., "Mar"); (3., "Apr") ]
```

Example with custom formatting:

<!-- $MDX skip -->
```ocaml
line ~x ~y ()
|> xtick_format (Printf.sprintf "%.1f%%")
```

### Grid and Legend

| Function | Description |
|----------|-------------|
| `grid_lines visible t` | Show or hide grid lines |
| `legend ?loc t` | Show legend at `loc` (default `Upper_right`) |

Legend locations: `Upper_right`, `Upper_left`, `Lower_right`, `Lower_left`, `Center`.

The legend is populated from marks that have a `~label`. Marks without labels are excluded.

### Theme Override

`with_theme theme t` renders with `theme` instead of the default.

## Layout

### Grid

`Layout.grid rows` arranges specs in a grid where each inner list is a row:

<!-- $MDX skip -->
```ocaml
let p1 = line ~x ~y:(Nx.sin x) () |> title "sin" in
let p2 = line ~x ~y:(Nx.cos x) () |> title "cos" in
let p3 = line ~x ~y:(Nx.tan x) () |> title "tan" |> ylim (-5.) 5. in
let p4 = hist ~x:(Nx.rand Nx.float32 [|500|]) () |> title "random" in
Layout.grid [ [ p1; p2 ]; [ p3; p4 ] ] |> render_png "grid.png"
```

`~gap` controls spacing between panels as a fraction of total size (default 0.05).

### Stack

| Function | Description |
|----------|-------------|
| `Layout.hstack specs` | Single row of panels |
| `Layout.vstack specs` | Single column of panels |

Both accept `~gap`.

## Themes

A theme controls every non-data visual element: background, typography, axes, grid, spacing, and data palettes.

### Predefined Themes

| Theme | Description |
|-------|-------------|
| `Theme.default` | Light background, subtle grid, Okabe-Ito palette |
| `Theme.dark` | Dark background |
| `Theme.minimal` | No grid, thin axes |

<!-- $MDX skip -->
```ocaml
line ~x ~y () |> with_theme Theme.dark |> render_png "dark.png"
```

### Context Scaling

Context functions scale all visual elements (fonts, line widths, spacing) for different output media:

| Function | Scale factor | Use case |
|----------|-------------|----------|
| `Theme.paper` | 1.0 | Journal figures |
| `Theme.notebook` | 1.3 | Quill notebooks |
| `Theme.talk` | 1.6 | Slides and presentations |
| `Theme.poster` | 2.0 | Conference posters |

<!-- $MDX skip -->
```ocaml
let theme = Theme.dark |> Theme.talk in
line ~x ~y () |> with_theme theme |> render_png "slide.png"
```

### Theme Fields

The `Theme.t` record is fully public. You can create custom themes by modifying fields:

| Field | Type | Description |
|-------|------|-------------|
| `background` | `Color.t` | Background color |
| `palette` | `Color.t array` | Categorical color palette |
| `sequential` | `Cmap.t` | Default sequential colormap |
| `diverging` | `Cmap.t` | Default diverging colormap |
| `font_title` | `Theme.font` | Title font |
| `font_label` | `Theme.font` | Axis label font |
| `font_tick` | `Theme.font` | Tick label font |
| `axis` | `Theme.line` | Axis line style |
| `grid` | `Theme.line option` | Grid line style (None to hide) |
| `tick_length` | `float` | Tick mark length |
| `padding` | `float` | Plot area padding |
| `title_gap` | `float` | Gap below title |
| `label_gap` | `float` | Gap between label and axis |
| `scale_factor` | `float` | Global size multiplier |
| `line_width` | `float` | Default line width |
| `marker_size` | `float` | Default marker size |

## Next Steps

- [Colors and Colormaps](/docs/hugin/colors-and-colormaps/) — OKLCH color space, operations, and colormap reference
- [Matplotlib Comparison](/docs/hugin/matplotlib-comparison/) — side-by-side with Python
