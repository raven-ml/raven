# Hugin Examples

A collection of OCaml programs (with `Hugin` + `Nx`) showcasing common plotting tasks, alongside equivalent Python/Matplotlib scripts for comparison.

## Directory Layout

Each numbered folder contains a self-contained example. For most, you'll find both an OCaml implementation and a Python counterpart:

1. **01-plot2d**  
   Basic 2D line plots of sine & cosine with legend, grid, labels.  

2. **02-imshow**  
   Load & display an image via `Nx_io.load_image` + `Hugin.imshow`.

3. **02-plot3d**  
   Generate a 3D helix and render with `Hugin.plot3d`.  

4. **04-subplot**  
   A 4×3 grid showing line, scatter, bar, histogram, step, fill, error bars, imshow, matshow, 3D, and combined plots.  

## Building & Running

### OCaml

From the root of the Raven repository:

```bash
dune exec hugin/example/01-plot2d/plot2d.exe
dune exec hugin/example/02-imshow/imshow.exe path/to/image.png
dune exec hugin/example/02-plot3d/plot3d.exe
dune exec hugin/example/04-subplot/subplots.exe
```

Or cd into a folder:

```bash
cd hugin/example/01-plot2d
dune exec plot2d.exe
```
