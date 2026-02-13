# hugin Documentation

Hugin is our Matplotlib. It creates publication-quality plots from your nx arrays.

## What hugin Does

Hugin gives you 2D and 3D plotting with a functional API designed for OCaml. Line plots, scatter plots, histograms, heatmaps, all rendered with Cairo and displayed via SDL. The API feels familiar if you know Matplotlib, but embraces OCaml's pipeline style.

Unlike Matplotlib's object-oriented approach, hugin uses functional composition. You build plots by piping data through transformations, which feels natural in OCaml.

## Current Status

Hugin works today for essential plotting needs:

- 2D plots (line, scatter, bar, etc.)
- Basic 3D visualization  
- Image display with `imshow`
- Multiple subplots
- PNG export

What's missing: animations, interactive backends, advanced plot types (contour, surface), and automatic legends. These are on the roadmap.

## System Requirements

Hugin needs Cairo and SDL2 for rendering:

```bash
# macOS
brew install cairo sdl2

# Ubuntu/Debian
apt install libcairo2-dev libsdl2-dev
```

## Learn More

- [Getting Started](/docs/hugin/getting-started/) - Installation and first plots
- [Matplotlib Comparison](/docs/hugin/matplotlib-comparison/) - Coming from Python
- [API Reference](https://ocaml.org/p/hugin/latest/doc/Hugin/index.html) - Complete API docs (coming soon)