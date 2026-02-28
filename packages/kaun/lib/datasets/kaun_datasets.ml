(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let mnist ?(fashion = false) ?(normalize = true) ?(data_format = `NCHW) () =
  let (train_images, train_labels), (test_images, test_labels) =
    Mnist.load ~fashion_mnist:fashion
  in
  let make_tensors images labels =
    let n = Bigarray.Array3.dim1 images in
    let h = Bigarray.Array3.dim2 images in
    let w = Bigarray.Array3.dim3 images in
    let x =
      Nx.of_bigarray (Bigarray.genarray_of_array3 images)
      |> Nx.reshape [| n; h; w; 1 |]
      |> Nx.cast Nx.float32
    in
    let x = if normalize then Nx.div_s x 255.0 else x in
    let x =
      match data_format with
      | `NCHW -> Nx.transpose x ~axes:[ 0; 3; 1; 2 ]
      | `NHWC -> x
    in
    let y =
      Nx.of_bigarray (Bigarray.genarray_of_array1 labels) |> Nx.cast Nx.int32
    in
    (x, y)
  in
  let train = make_tensors train_images train_labels in
  let test = make_tensors test_images test_labels in
  (train, test)

let cifar10 ?(normalize = true) ?(data_format = `NCHW) () =
  let (train_images, train_labels), (test_images, test_labels) =
    Cifar10.load ()
  in
  let make_tensors images labels =
    let x = Nx.of_bigarray images |> Nx.cast Nx.float32 in
    let x = if normalize then Nx.div_s x 255.0 else x in
    let x =
      match data_format with
      | `NCHW -> x
      | `NHWC -> Nx.transpose x ~axes:[ 0; 2; 3; 1 ]
    in
    let y =
      Nx.of_bigarray (Bigarray.genarray_of_array1 labels) |> Nx.cast Nx.int32
    in
    (x, y)
  in
  let train = make_tensors train_images train_labels in
  let test = make_tensors test_images test_labels in
  (train, test)
