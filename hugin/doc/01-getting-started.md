# Getting Started with hugin

This guide shows you how to create plots with hugin.

## Installation

First, install the system dependencies:

```bash
# macOS
brew install cairo sdl2

# Ubuntu/Debian  
apt install libcairo2-dev libsdl2-dev
```

Then install hugin:

```bash
opam install hugin
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build hugin
```

## Your First Plot

Here's a working example that creates a simple line plot:

```ocaml
open Hugin
open Nx

let () =
  (* Create data *)
  let x = linspace float32 0. (2. *. Float.pi) 100 in
  let y = Nx.map (fun x -> Float.sin x) x in
  
  (* Create figure and plot *)
  let fig = figure ~width:800 ~height:600 () in
  let ax = subplot fig in
  let _ = 
    ax
    |> Plotting.plot ~x ~y ~color:Artist.Color.blue ~label:"sin(x)"
    |> Axes.set_xlabel "x"
    |> Axes.set_ylabel "y" 
    |> Axes.set_title "Sine Wave"
  in
  
  (* Display the plot *)
  show fig
```

## Key Concepts

**Pipeline style.** Hugin embraces OCaml's `|>` operator. You build plots by piping axes through transformations:

```ocaml
subplot fig
|> Plotting.plot ~x ~y
|> Axes.set_title "My Plot"
|> Axes.grid true
```

**Module organization.** Functions are organized by purpose:
- `Plotting` - plot functions (plot, scatter, bar, etc.)
- `Axes` - axis manipulation (set_xlabel, set_xlim, etc.)
- `Artist` - colors and styles

**Colors are records.** Instead of strings, use predefined colors:
```ocaml
Artist.Color.red
Artist.Color.blue
Artist.Color.(rgba 0.5 0.5 0.5 1.0)  (* custom RGBA *)
```

## Common Plots

```ocaml
(* Line plot with style *)
Plotting.plot ~x ~y 
  ~color:Artist.Color.red 
  ~linestyle:Artist.Dashed 
  ~linewidth:2.0
  ax

(* Scatter plot *)
Plotting.scatter ~x ~y 
  ~color:Artist.Color.green
  ~marker:Artist.Circle
  ~size:5.0
  ax

(* Multiple lines *)
let ax = subplot fig in
let _ = 
  ax
  |> Plotting.plot ~x ~y1 ~label:"sin(x)"
  |> Plotting.plot ~x ~y2 ~label:"cos(x)"
  |> Axes.set_xlabel "x"
in ()

(* Subplots *)
let ax1 = subplot ~nrows:2 ~ncols:1 ~index:1 fig in
let ax2 = subplot ~nrows:2 ~ncols:1 ~index:2 fig in
(* plot on ax1 and ax2 separately *)

(* Save to file *)
savefig fig "plot.png"
```

## Display Images

```ocaml
(* Load and display an image *)
let img = Nx_io.load_image "photo.jpg" in
let fig = imshow ~title:"My Image" img in
show fig
```

## Next Steps

Check out the [Matplotlib Comparison](/docs/hugin/matplotlib-comparison/) to see how hugin's functional approach differs from Matplotlib's object-oriented style.

The examples in `hugin/example/` show more complex plots including 3D visualization.