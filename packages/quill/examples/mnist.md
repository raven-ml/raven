<!-- quill:cell id="c_mnist_title" -->
# MNIST Digit Classification

In this notebook we train a neural network to recognize handwritten digits from
the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset. Each image is a 28x28
grayscale picture of a digit (0--9), and the model learns to predict which digit
it is.

We use three raven packages:
- **Nx** -- n-dimensional arrays
- **Kaun** -- neural network layers, optimizers and training
- **Hugin** -- plotting and visualization


<!-- quill:cell id="c_mnist_load" -->
## 1. Loading the dataset

`Kaun_datasets.mnist` downloads MNIST the first time and caches it locally.
It returns two data pipelines (train and test), each yielding `(image, label)`
pairs. We collect them into full tensors for batching and shuffling during
training. Labels are cast to int32 for `cross_entropy_sparse`.

```ocaml
open Kaun

let collect ds =
  let a = Data.to_array ds in
  Data.stack_batch (Array.map fst a),
  Nx.cast Nx.int32 (Data.stack_batch (Array.map snd a))

let () = Printf.printf "Loading MNIST...\n%!"
let train_ds, test_ds = Kaun_datasets.mnist ()
let x_train, y_train = collect train_ds
let x_test, y_test = collect test_ds

let () =
  let s = Nx.shape x_train in
  Printf.printf "train: %d images  shape: [%d; %d; %d]\n" s.(0) s.(1) s.(2) s.(3);
  Printf.printf "test:  %d images\n" (Nx.shape x_test).(0)
```

<!-- quill:cell id="c_mnist_viz_text" -->
## 2. Visualizing the data

Let's look at the first 10 training images and their labels.

<!-- quill:cell id="c_mnist_viz_code" -->
```ocaml
let () =
  let fig = Hugin.figure ~width:800 ~height:120 () in
  for i = 0 to 9 do
    let img = Nx.get [i; 0] x_train |> Nx.reshape [|28; 28|] in
    let label = Nx.item [i] y_train in
    let ax = Hugin.subplot ~nrows:1 ~ncols:10 ~index:(i + 1) fig in
    let ax = Hugin.Axes.imshow ~cmap:Hugin.Artist.Colormap.gray ~data:img ax in
    let ax = Hugin.Axes.set_title (Printf.sprintf "%ld" label) ax in
    let ax = Hugin.Axes.set_xticks [] ax in
    ignore (Hugin.Axes.set_yticks [] ax)
  done;
  ignore fig
```

<!-- quill:cell id="c_mnist_model_text" -->
## 3. Defining the model

We use a simple multi-layer perceptron (MLP): flatten the 1x28x28 image into a
784-element vector, pass through a hidden layer with 128 units and ReLU
activation, then project to 10 output logits (one per digit class).

<!-- quill:cell id="c_mnist_model_code" -->
```ocaml
let model =
  Layer.sequential [
    Layer.flatten ();
    Layer.linear ~in_features:784 ~out_features:128 ();
    Layer.relu ();
    Layer.linear ~in_features:128 ~out_features:10 ();
  ]
```

<!-- quill:cell id="c_mnist_trainer_text" -->
## 4. Setting up the trainer

A `Train.t` pairs the model with an optimizer. We use Adam with a constant
learning rate of 0.001. `Train.init` creates initial random weights.

<!-- quill:cell id="c_mnist_trainer_code" -->
```ocaml
let batch_size = 64

let trainer =
  Train.make ~model
    ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.001) ())

let st = ref (Nx.Rng.run ~seed:42 @@ fun () -> Train.init trainer ~dtype:Nx.float32)
```

<!-- quill:cell id="c_mnist_train_text" -->
## 5. Training

`Train.fit` iterates over the data, computing the forward pass, loss, gradients,
and optimizer update on each batch. The `~report` callback prints the current
loss after every batch -- you should see it decrease in real time.

<!-- quill:cell id="c_mnist_train_code" -->
```ocaml
let epochs = 3

let () =
  let n_train = (Nx.shape x_train).(0) in
  let num_batches = n_train / batch_size in
  let test_batches = Data.prepare ~batch_size (x_test, y_test) in
  for epoch = 1 to epochs do
    let train_data =
      Nx.Rng.run ~seed:(42 + epoch) @@ fun () ->
      Data.prepare ~shuffle:true ~batch_size (x_train, y_train)
      |> Data.map (fun (x, y) ->
          (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    let tracker = Metric.tracker () in
    st :=
      Train.fit trainer !st
        ~report:(fun ~step ~loss _st ->
          Metric.observe tracker "loss" loss;
          Printf.printf "\r  epoch %d  batch %d/%d  loss: %.4f%!" epoch step num_batches loss)
        train_data;
    Printf.printf "\n%!";

    Data.reset test_batches;
    let test_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in
    Printf.printf "  train_loss: %.4f  test_acc: %.2f%%\n%!"
      (Metric.mean tracker "loss") (test_acc *. 100.)
  done
```

<!-- quill:cell id="c_mnist_eval_text" -->
## 6. Evaluating predictions

Let's look at the model's predictions on some test images. For each image we
show the true label and the predicted label.

<!-- quill:cell id="c_mnist_eval_code" -->
```ocaml
let () =
  let fig = Hugin.figure ~width:800 ~height:120 () in
  for i = 0 to 9 do
    let img = Nx.get [i; 0] x_test |> Nx.reshape [|28; 28|] in
    let true_label = Nx.item [i] y_test in
    let logits = Train.predict trainer !st (Nx.get [i] x_test |> Nx.expand_dims 0) in
    let pred_label = Nx.item [] (Nx.argmax ~axis:1 logits) in
    let ax = Hugin.subplot ~nrows:1 ~ncols:10 ~index:(i + 1) fig in
    let ax = Hugin.Axes.imshow ~cmap:Hugin.Artist.Colormap.gray ~data:img ax in
    let title = Printf.sprintf "%ld->%ld" true_label pred_label in
    let ax = Hugin.Axes.set_title title ax in
    let ax = Hugin.Axes.set_xticks [] ax in
    ignore (Hugin.Axes.set_yticks [] ax)
  done;
  ignore fig
```
