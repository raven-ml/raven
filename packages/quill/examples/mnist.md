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
It returns `(x_train, y_train, x_test, y_test)` — images as float32
in [0, 1] with shape `[N; 1; 28; 28]`, labels as int32 with shape `[N]`.

```ocaml
open Kaun

let () = Printf.printf "Loading MNIST...\n%!"
let x_train, y_train, x_test, y_test = Kaun_datasets.mnist ()

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
let _fig =
  List.init 10 (fun i ->
    let img = Nx.get [i; 0] x_train |> Nx.reshape [|28; 28|] in
    let label = Nx.item [i] y_train in
    Hugin.imshow ~data:img ~cmap:Hugin.Cmap.gray ()
    |> Hugin.title (Printf.sprintf "%ld" label)
    |> Hugin.no_axes)
  |> Hugin.hstack ~gap:0.
```

<!-- quill:cell id="c_mnist_model_text" -->
## 3. Defining the model

We use a simple multi-layer perceptron (MLP): flatten the 1x28x28 image into a
784-element vector, pass through a hidden layer with 128 units and ReLU
activation, then project to 10 output logits (one per digit class).

In Kaun, a model is a plain record of layers with a traversal over its tensor
leaves — no special layer type:

<!-- quill:cell id="c_mnist_model_code" -->
```ocaml
module Model = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x =
    let x = Nx.reshape [| (Nx.shape x).(0); 784 |] x in
    Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end
```

<!-- quill:cell id="c_mnist_trainer_text" -->
## 4. Initializing parameters and optimizer

The parameters are just a value we construct; the optimizer state (Adam's
moments) is a record shaped like the parameters themselves.

<!-- quill:cell id="c_mnist_trainer_code" -->
```ocaml
let batch_size = 64

let params =
  Nx.Rng.run ~seed:42 @@ fun () ->
  Model.{ l1 = Linear.init ~inputs:784 ~outputs:128;
          l2 = Linear.init ~inputs:128 ~outputs:10 }

let st = ref (params, Vega.adam_init (module Model) params)
```

<!-- quill:cell id="c_mnist_train_text" -->
## 5. Training

The training step is three lines you own: differentiate the loss with
`Rune.value_and_grad`, then apply the optimizer update. The loop is a plain
fold over shuffled minibatches.

<!-- quill:cell id="c_mnist_train_code" -->
```ocaml
let epochs = 1

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let n_train = (Nx.shape x_train).(0) in
  let num_batches = n_train / batch_size in
  for epoch = 1 to epochs do
    let step = ref 0 in
    Data.batches2 ~shuffle:true ~batch_size (x_train, y_train)
    |> Seq.iter (fun (x, y) ->
        let params, ostate = !st in
        let loss, grads =
          Rune.value_and_grad (module Model)
            (fun p -> Loss.softmax_cross_entropy_sparse (Model.apply p x) y)
            params
        in
        st := Vega.adam_step (module Model) ~lr:0.001 ostate ~params ~grads;
        incr step;
        Printf.printf "\r  epoch %d  batch %d/%d  loss: %.4f%!" epoch !step
          num_batches (Nx.item [] loss));
    Printf.printf "\n%!";
    let test_acc = Metric.accuracy (Model.apply (fst !st) x_test) y_test in
    Printf.printf "  test_acc: %.2f%%\n%!" (test_acc *. 100.)
  done
```

<!-- quill:cell id="c_mnist_eval_text" -->
## 6. Evaluating predictions

Let's look at the model's predictions on some test images. For each image we
show the true label and the predicted label.

<!-- quill:cell id="c_mnist_eval_code" -->
```ocaml
let _fig =
  List.init 10 (fun i ->
    let img = Nx.get [i; 0] x_test |> Nx.reshape [|28; 28|] in
    let true_l = Nx.item [i] y_test in
    let logits = Model.apply (fst !st) (Nx.get [i] x_test |> Nx.expand_dims [0]) in
    let pred_l = Nx.item [0] (Nx.argmax ~axis:1 logits) in
    Hugin.imshow ~data:img ~cmap:Hugin.Cmap.gray ()
    |> Hugin.title (Printf.sprintf "%ld->%ld" true_l pred_l)
    |> Hugin.no_axes)
  |> Hugin.hstack ~gap:0.
```
