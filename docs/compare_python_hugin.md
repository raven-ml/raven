# Hugin vs Matplotlib Comparison

This document compares the Hugin visualization library (OCaml) with Matplotlib (Python), highlighting similarities, differences, and providing equivalent code examples.

- [Hugin vs Matplotlib Comparison](#hugin-vs-matplotlib-comparison)
  - [1. Overview](#1-overview)
  - [2. Basic Image Display](#2-basic-image-display)
    - [Image Display with imshow](#image-display-with-imshow)
  - [3. 2D Plotting](#3-2d-plotting)
    - [Basic Line Plots](#basic-line-plots)
  - [4. 3D Plotting](#4-3d-plotting)
    - [3D Line/Curve Plots](#3d-linecurve-plots)
  - [5. Creating Multiple Subplots](#5-creating-multiple-subplots)
  - [6. Example: Creating and Displaying a Figure](#6-example-creating-and-displaying-a-figure)
  - [7. Example: Customizing Plot Appearance](#7-example-customizing-plot-appearance)

## 1. Overview

Hugin is a visualization library for OCaml inspired by Matplotlib's API and functionality. Just as Ndarray provides NumPy-like functionality for OCaml, Hugin provides Matplotlib-like visualization capabilities for the OCaml ecosystem.

Key characteristics of Hugin vs Matplotlib:

- **Cairo with SDL Backend:** Hugin uses Cairo for rendering with SDL as the display backend. It has been designed to support multiple backends, with plans to add more in the future.
- **Integration with Ndarray:** Hugin is designed to work seamlessly with Ndarray, just as Matplotlib works with NumPy arrays.
- **Functional Style:** Hugin follows OCaml's functional paradigm, with a pipeline-oriented API that uses the `|>` operator for chaining operations, making the code flow naturally in OCaml.
- **Limited Feature Set:** While Hugin covers the core visualization needs, it's not yet as feature-rich as Matplotlib, focusing on the fundamentals with more features planned for future development.
- **API Structure:** While Matplotlib is object-oriented, Hugin uses a more functional approach with modules and functions that align with OCaml's programming model.

## 2. Basic Image Display

### Image Display with imshow

**Hugin:**
```ocaml
open Hugin

let () =
  if Array.length Sys.argv < 2 then (
    Printf.printf "Usage: %s <image_path>\n" Sys.executable_name;
    exit 1);

  let image_path = Sys.argv.(1) in
  let img = Ndarray_io.load_image image_path in
  let fig = imshow ~title:"Image" img in
  show fig
```

**Matplotlib:**
```python
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
img = cv2.imread(image_path)
# Convert BGR (OpenCV default) to RGB for proper display in matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.show()
```

## 3. 2D Plotting

### Basic Line Plots

**Hugin:**
```ocaml
open Hugin
open Ndarray

let x = linspace float32 0. (2. *. Float.pi) 100
let y1 = map Float.sin x
let y2 = map Float.cos x

let () =
  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.plot ~x ~y:y1 ~color:Artist.Color.blue ~label:"sin(x)"
    |> Plotting.plot ~x ~y:y2 ~color:Artist.Color.red ~linestyle:Dashed
         ~label:"cos(x)"
    |> Axes.set_title "Trigonometric Functions"
    |> Axes.set_xlabel "Angle (radians)"
    |> Axes.set_ylabel "Value"
    |> Axes.set_ylim ~min:(-1.2) ~max:1.2
    |> Axes.grid true
  in
  show fig
```

**Matplotlib:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y1, color='blue', label='sin(x)')
plt.plot(x, y2, color='red', linestyle='--', label='cos(x)')
plt.title('Trigonometric Functions')
plt.xlabel('Angle (radians)')
plt.ylabel('Value')
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.legend()
plt.show()
```

## 4. 3D Plotting

### 3D Line/Curve Plots

**Hugin:**
```ocaml
open Ndarray

let create_helix_data () =
  let t = Ndarray.linspace float32 0. (4. *. Float.pi) 100 in
  let x = Ndarray.map (fun t -> Float.cos t) t in
  let y = Ndarray.map (fun t -> Float.sin t) t in
  let z = Ndarray.map (fun t -> t /. (4. *. Float.pi)) t in
  (x, y, z)

let () =
  let x, y, z = create_helix_data () in
  let fig =
    Hugin.plot3d ~title:"3D Helix" ~xlabel:"x" ~ylabel:"y" ~zlabel:"z" x y z
  in
  Hugin.show fig
```

**Matplotlib:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_helix_data():
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi)
    return x, y, z

# Generate data
x, y, z = create_helix_data()

# Create figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D helix
ax.plot(x, y, z, color='b', linewidth=1.0)

# Set labels and title
ax.set_title("3D Helix")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Display the plot
plt.show()
```

## 5. Creating Multiple Subplots

**Hugin:**
```ocaml
let fig = Figure.create ~width:1600 ~height:1200 () in
let ax1 = Figure.add_subplot ~nrows ~ncols ~index:1 fig in
```

**Matplotlib:**
```python
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(nrows, ncols, 1)
```

## 6. Example: Creating and Displaying a Figure

**Hugin:**
```ocaml
let fig = figure ~width:800 ~height:600 () in
let _ = subplot fig |> (* plotting commands *) in
show fig
```

**Matplotlib:**
```python
plt.figure(figsize=(8, 6))
# plotting commands
plt.show()
```

## 7. Example: Customizing Plot Appearance

**Hugin:**
```ocaml
subplot fig
|> Plotting.plot ~x ~y ~color:Artist.Color.blue ~linestyle:Dashed
   ~marker:Artist.Circle ~label:"Data"
|> Axes.set_title "Plot Title"
|> Axes.grid true
```

**Matplotlib:**
```python
ax = plt.subplot()
ax.plot(x, y, color='blue', linestyle='--', marker='o', label='Data')
ax.set_title("Plot Title")
ax.grid(True)
```
