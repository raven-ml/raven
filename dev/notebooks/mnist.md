# Training a Simple MLP on MNIST using Rune

This notebook demonstrates how to build, train, and evaluate a simple Multi-Layer Perceptron (MLP) on the MNIST dataset using the Rune OCaml library.
We will also visualize some of the MNIST data using the Hugin plotting library.

**Libraries Used:**

* `Rune`: For tensor operations, automatic differentiation, and neural network building blocks.
* `Ndarray`: Rune's underlying library for multi-dimensional arrays (often used implicitly via Rune).
* `Ndarray_datasets`: For easily loading standard datasets like MNIST.
* `Hugin`: For plotting and data visualizationd.

## 1. Setup and Imports

First, let's open the necessary libraries.

```ocaml
open Rune
module Nd = Ndarray
module Hg = Hugin
```

## 2. Load MNIST Data

We use `Ndarray_datasets` to load the MNIST dataset. It conveniently provides training and testing splits.

```ocaml
let (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = Ndarray_datasets.load_mnist ()

let () =
  Printf.printf "Dataset loaded.\n%!";
  Printf.printf "Raw Training Data Shape: %s\n" (Nd.shape_to_string (Nd.shape x_train_raw));
  Printf.printf "Raw Training Labels Shape: %s\n" (Nd.shape_to_string (Nd.shape y_train_raw));
  Printf.printf "Raw Test Data Shape: %s\n" (Nd.shape_to_string (Nd.shape x_test_raw));
  Printf.printf "Raw Test Labels Shape: %s\n%!" (Nd.shape_to_string (Nd.shape y_test_raw));
```

## 2. Visualize MNIST Data

Before preprocessing the data for the neural network, let's visualize some of the training images using Hugin to get a feel for the dataset. We'll display the first 16 images in a 4x4 grid.

```ocaml
let fig =
  Printf.printf "Visualizing first 16 MNIST training images...\n%!";

  let fig = Hg.Figure.create ~width:800 ~height:800 () in
  let nrows, ncols = (4, 4) in

  for i = 0 to (nrows * ncols) - 1 do
    let ax = Hg.Figure.add_subplot ~nrows ~ncols ~index:(i + 1) fig in

    (* Extract the i-th image (28x28) *)
    let img_data = Nd.slice [| i; 0; 0 |] [| i + 1; 28; 28 |] x_train_raw in
    let img_data = Nd.reshape [| 28; 28 |] img_data in

    (* Extract the i-th label *)
    let label = Nd.get [| i |] y_train_raw |> int_of_char in

    (* Display the image using imshow *)
    let _ax =
      ax
      |> P.imshow ~data:img_data ~cmap:A.Colormap.gray_r ~origin:`upper
      |> Hg.Axes.set_title (Printf.sprintf "Label: %d" label)
      |> Hg.Axes.set_xticks [] (* Hide x-axis ticks *)
      |> Hg.Axes.set_yticks [] (* Hide y-axis ticks *)
    in
    ()
  done;

  (* Add a main title to the figure *)
  let _ = Hg.Figure.suptitle fig "Sample MNIST Training Images" in

  fig
```
