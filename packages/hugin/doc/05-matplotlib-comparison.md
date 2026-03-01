# Hugin vs Matplotlib

Side-by-side examples comparing Hugin (OCaml) with Matplotlib (Python). Hugin uses a declarative, pipeline-oriented API while Matplotlib uses an imperative, object-oriented approach.

## Key Differences

| | Hugin | Matplotlib |
|---|---|---|
| Style | Declarative, immutable specs | Imperative, mutable state |
| Composition | `\|>` pipeline | Method calls on axes |
| State | No global state | `plt` global state |
| Colors | OKLCH color space | sRGB strings |
| Output | `render_png`, `render_svg`, `show` | `plt.savefig`, `plt.show` |

## Line Plot

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  layers [
    line ~x ~y:(Nx.sin x) ~label:"sin(x)" ~color:Color.blue ();
    line ~x ~y:(Nx.cos x) ~label:"cos(x)" ~color:Color.vermillion
      ~line_style:`Dashed ();
  ]
  |> title "Trigonometric Functions"
  |> xlabel "Angle (radians)"
  |> ylabel "Value"
  |> ylim (-1.2) 1.2
  |> grid_lines true
  |> legend
  |> render_png "trig.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)

plt.figure()
plt.plot(x, np.sin(x), label="sin(x)", color="blue")
plt.plot(x, np.cos(x), label="cos(x)", color="red", linestyle="--")
plt.title("Trigonometric Functions")
plt.xlabel("Angle (radians)")
plt.ylabel("Value")
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.legend()
plt.savefig("trig.png")
```

## Scatter Plot

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.rand Nx.float32 [| 200 |] in
  let y = Nx.rand Nx.float32 [| 200 |] in
  let c = Nx.add x y in
  point ~x ~y ~color_by:c ~size:8. ~marker:Circle ()
  |> title "Random Scatter"
  |> render_png "scatter.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(200)
y = np.random.rand(200)
c = x + y

plt.figure()
plt.scatter(x, y, c=c, s=64, marker="o")
plt.title("Random Scatter")
plt.colorbar()
plt.savefig("scatter.png")
```

## Bar Chart

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let h = Nx.create Nx.float32 [| 4 |] [| 3.; 7.; 2.; 5. |] in
  bar ~x ~height:h ~color:Color.orange ()
  |> title "Quarterly Revenue"
  |> xticks [ (1., "Q1"); (2., "Q2"); (3., "Q3"); (4., "Q4") ]
  |> ylabel "Revenue ($M)"
  |> render_png "bar.png"
```

**Matplotlib:**

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
h = [3, 7, 2, 5]

plt.figure()
plt.bar(x, h, color="orange")
plt.title("Quarterly Revenue")
plt.xticks(x, ["Q1", "Q2", "Q3", "Q4"])
plt.ylabel("Revenue ($M)")
plt.savefig("bar.png")
```

## Histogram

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let data = Nx.randn Nx.float32 [| 1000 |] in
  hist ~x:data ~bins:(`Num 30) ~density:true ~color:Color.sky_blue ()
  |> title "Normal Distribution"
  |> xlabel "Value"
  |> ylabel "Density"
  |> render_png "hist.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)

plt.figure()
plt.hist(data, bins=30, density=True, color="skyblue")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("hist.png")
```

## Multi-Panel Layout

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  let p1 = line ~x ~y:(Nx.sin x) () |> title "sin" in
  let p2 = line ~x ~y:(Nx.cos x) () |> title "cos" in
  let p3 = line ~x ~y:(Nx.tan x) () |> title "tan" |> ylim (-5.) 5. in
  let p4 = hist ~x:(Nx.rand Nx.float32 [| 500 |]) () |> title "random" in
  Layout.grid [ [ p1; p2 ]; [ p3; p4 ] ]
  |> render_png "grid.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(x, np.sin(x)); axes[0, 0].set_title("sin")
axes[0, 1].plot(x, np.cos(x)); axes[0, 1].set_title("cos")
axes[1, 0].plot(x, np.tan(x)); axes[1, 0].set_title("tan")
axes[1, 0].set_ylim(-5, 5)
axes[1, 1].hist(np.random.rand(500)); axes[1, 1].set_title("random")
plt.tight_layout()
plt.savefig("grid.png")
```

## Heatmap

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let data = Nx.init Nx.float32 [| 8; 10 |] (fun idx ->
    let i = Float.of_int idx.(0) and j = Float.of_int idx.(1) in
    Float.sin (i *. 0.5) *. Float.cos (j *. 0.4))
  in
  heatmap ~data ~annotate:true ~cmap:Cmap.viridis ()
  |> title "Heatmap"
  |> render_png "heatmap.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.fromfunction(
    lambda i, j: np.sin(i * 0.5) * np.cos(j * 0.4), (8, 10)
)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap="viridis")
for i in range(8):
    for j in range(10):
        ax.text(j, i, f"{data[i, j]:.2g}", ha="center", va="center")
ax.set_title("Heatmap")
plt.colorbar(im)
plt.savefig("heatmap.png")
```

## Styling

**Hugin:**

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 50 in
  line ~x ~y:(Nx.sin x)
    ~color:Color.vermillion
    ~line_style:`Dashed
    ~line_width:2.5
    ~marker:Triangle
    ~alpha:0.7
    ()
  |> render_png "styled.png"
```

**Matplotlib:**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)

plt.figure()
plt.plot(x, np.sin(x), color="red", linestyle="--",
         linewidth=2.5, marker="^", alpha=0.7)
plt.savefig("styled.png")
```

## Themes

Hugin provides built-in themes with context scaling. Matplotlib uses style sheets.

**Hugin:**

<!-- $MDX skip -->
```ocaml
(* Dark theme scaled for a presentation *)
let theme = Theme.dark |> Theme.talk in
line ~x ~y () |> with_theme theme |> render_png "slide.png"
```

**Matplotlib:**

```python
plt.style.use("dark_background")
plt.rcParams.update({"font.size": 14})
plt.plot(x, y)
plt.savefig("slide.png")
```

## Save and Export

**Hugin:**

<!-- $MDX skip -->
```ocaml
let spec = line ~x ~y () |> title "My Plot" in
spec |> render_png "plot.png";
spec |> render_svg "plot.svg";
spec |> render_pdf "plot.pdf";
spec |> show  (* interactive SDL window *)
```

**Matplotlib:**

```python
plt.plot(x, y)
plt.title("My Plot")
plt.savefig("plot.png")
plt.savefig("plot.svg")
plt.savefig("plot.pdf")
plt.show()
```

In Hugin, the spec is an immutable value. You can render the same spec to multiple formats without rebuilding it. In Matplotlib, the figure is mutable state that `savefig` and `show` consume.

## Interactive Display

**Hugin:**

<!-- $MDX skip -->
```ocaml
show ~width:1600. ~height:1200. spec
```

The SDL window is resizable. The plot re-renders at the new dimensions. Press Escape or Q to close.

**Matplotlib:**

```python
plt.show()
```
