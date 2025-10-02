# Hugin Developer Guide

## Architecture

Hugin is a publication-quality plotting library built on Nx tensors. It provides a matplotlib-like API with a Cairo+SDL rendering backend.

### Core Components

- **[lib/figure.ml](lib/figure.ml)**: Figure and subplot management
- **[lib/axes.ml](lib/axes.ml)**: Coordinate system, ticks, labels, and axes rendering
- **[lib/plotting.ml](lib/plotting.ml)**: High-level plotting functions (plot, scatter, bar, etc.)
- **[lib/artist.ml](lib/artist.ml)**: Low-level drawing primitives (lines, shapes, text)
- **[lib/cairo_sdl_backend/](lib/cairo_sdl_backend/)**: Cairo rendering with SDL display

### Key Design Principles

1. **Layered API**: High-level plotting functions → Axes → Artist → Cairo backend
2. **Data from Nx**: All plotting data comes from Nx tensors
3. **Immediate mode**: Render on demand, no retained scene graph
4. **Backend abstraction**: Cairo+SDL on native, extensible to other backends

## Data Flow

```
User code
  ↓
Plotting functions (plot, scatter, etc.)
  ↓
Axes (coordinate transform, tick calculation)
  ↓
Artist (primitive shapes, text, paths)
  ↓
Cairo backend (rendering to surface)
  ↓
SDL display / PNG export
```

## Development Workflow

### Building and Testing

```bash
# Build hugin
dune build hugin/

# Run example plots
dune build hugin/example/plot_line.exe && _build/default/hugin/example/plot_line.exe

# Test interactively
dune exec hugin/example/plot_scatter.exe
```

### Creating Examples

Examples live in [example/](example/) and serve as both tests and documentation:

```ocaml
open Nx
open Hugin

let () =
  let x = linspace 0. 10. 100 in
  let y = sin x in

  let fig = Figure.create () in
  let ax = Figure.subplot fig 1 1 0 in
  plot ~x ~y ax;
  Figure.save fig "sine.png"
```

## Coordinate Systems

### Data Coordinates

User data lives in **data coordinates** (actual X/Y values):

```ocaml
(* Data points: (0, 0), (1, 1), (2, 4) *)
let x = create float32 [|3|] [|0.; 1.; 2.|] in
let y = create float32 [|3|] [|0.; 1.; 4.|] in
```

### Display Coordinates

Rendered to **display coordinates** (pixels on screen/image):

- Origin at top-left
- X increases right
- Y increases down

### Axes Transform

Axes map data → display coordinates:

1. Calculate data limits (xmin, xmax, ymin, ymax)
2. Map to display region (accounting for margins)
3. Transform all data points for rendering

**Auto-scaling:**
- Default: Fit all data with 5% margin
- Manual: `Axes.set_xlim`, `Axes.set_ylim`

## Adding Plot Types

### High-Level Function

Add to [lib/plotting.ml](lib/plotting.ml):

```ocaml
let errorbar ~x ~y ~yerr ax =
  (* 1. Convert Nx tensors to float arrays *)
  let x_data = Nx.to_float_array x in
  let y_data = Nx.to_float_array y in
  let yerr_data = Nx.to_float_array yerr in

  (* 2. Get axes transform *)
  let transform = Axes.get_transform ax in

  (* 3. Draw using Artist primitives *)
  for i = 0 to Array.length x_data - 1 do
    let xi, yi = transform (x_data.(i), y_data.(i)) in
    let err = yerr_data.(i) in
    (* Draw error bars *)
    Artist.line ctx xi (yi -. err) xi (yi +. err);
    (* Draw caps *)
    Artist.line ctx (xi -. 2.) (yi -. err) (xi +. 2.) (yi -. err);
    Artist.line ctx (xi -. 2.) (yi +. err) (xi +. 2.) (yi +. err);
  done
```

### Artist Primitives

Use low-level drawing in [lib/artist.ml](lib/artist.ml):

- `line`: Draw line segment
- `path`: Draw polyline
- `rectangle`: Draw rectangle
- `circle`: Draw circle
- `text`: Render text
- `set_color`, `set_line_width`: Styling

### Backend Operations

Cairo operations in [lib/cairo_sdl_backend/](lib/cairo_sdl_backend/):

- Surface creation
- Context management
- Path construction
- Stroking and filling

**Adding backend support:**
1. Implement backend interface
2. Provide surface creation
3. Map Artist operations to backend primitives

## Styling and Appearance

### Colors

Colors specified as RGB tuples:

```ocaml
Artist.set_color ctx (1.0, 0.0, 0.0);  (* Red *)
Artist.set_color ctx (0.0, 0.5, 1.0);  (* Blue *)
```

**Color cycling:**
- Default palette in `plotting.ml`
- Cycles through colors for multiple series

### Line Styles

```ocaml
Artist.set_line_width ctx 2.0;
Artist.set_dash ctx [|5.; 3.|];  (* Dashed *)
```

### Markers

Point markers in scatter plots:
- Circle, square, triangle, etc.
- Size controlled by parameter

### Fonts

Text rendering:

```ocaml
Artist.set_font_size ctx 12.0;
Artist.text ctx x y "Label";
```

## Layout and Spacing

### Figure Layout

Figure manages:
- Overall size (width × height)
- Subplot grid (rows × columns)
- Spacing between subplots

### Axes Margins

Axes calculate margins for:
- Tick labels (measure text width/height)
- Axis labels
- Title
- Legend

**Dynamic sizing:**
- Measure text before layout
- Adjust margins to fit labels
- Recalculate on resize

## Common Pitfalls

### Data Shape Mismatches

Plotting requires 1D tensors:

```ocaml
(* Wrong: 2D tensor *)
let x = create float32 [|10; 1|] ... in
plot ~x ...  (* Error *)

(* Correct: flatten or reshape *)
let x = reshape x [|10|] in
plot ~x ...
```

### Coordinate Transform Timing

Apply transform **after** calculating limits:

```ocaml
(* Wrong: transform before limits known *)
let xi, yi = transform (x, y) in
Axes.set_xlim ...  (* Too late! *)

(* Correct: set limits, then transform *)
Axes.set_xlim ...
let xi, yi = transform (x, y) in
```

### Memory Management

Cairo surfaces hold resources:

```ocaml
(* Save and destroy *)
Figure.save fig "plot.png";
Figure.destroy fig;  (* Free Cairo surface *)
```

### Text Measurement

Measure text **before** positioning:

```ocaml
let w, h = Artist.measure_text ctx "Label" in
let x = center_x -. w /. 2. in
Artist.text ctx x y "Label";
```

## Performance

- **Batch drawing**: Minimize Cairo state changes
- **Limit data points**: Decimate dense data before plotting
- **Reuse surfaces**: Don't recreate for animations
- **Profile**: Check Cairo vs. data processing time

## Testing

No automated tests yet—examples serve as visual tests:

1. Run example
2. Visually verify output
3. Compare to matplotlib equivalent

**Future:** Add image comparison tests (render to PNG, compare pixels).

## Code Style

- **Naming**: `snake_case` for functions, `capitalized_case` for modules
- **Errors**: `"function_name: error description"`
- **Documentation**: Explain coordinate systems and transforms
- **Examples**: Keep concise, demonstrate single feature

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Nx tensor operations
- Cairo documentation for backend reference
