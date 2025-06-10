(** Ready-to-use datasets for Kaun *)

let mnist ?(train = true) ?(flatten = false) ?(normalize = true)
    ?(data_format = `NCHW) () =
  (* Load MNIST data from Nx_datasets *)
  let (x_train, y_train), (x_test, y_test) = Nx_datasets.load_mnist () in

  (* Select training or test data *)
  let x, y = if train then (x_train, y_train) else (x_test, y_test) in

  (* Convert from uint8 to float *)
  let dev = Rune.cpu in
  (* Cast to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let dtype = Rune.float32 in
  (* Convert Nx tensors to Rune tensors via bigarray *)
  let x = Rune.of_bigarray dev (Nx.to_bigarray x) in
  let y = Rune.of_bigarray dev (Nx.to_bigarray y) in

  (* Normalize to [0, 1] if requested *)
  let x = if normalize then Rune.div x (Rune.scalar dev dtype 255.0) else x in

  (* Handle data format *)
  let x =
    match data_format with
    | `NCHW ->
        (* Original shape is [N, H, W, 1], convert to [N, 1, H, W] *)
        let shape = Rune.shape x in
        let n, h, w, _ = (shape.(0), shape.(1), shape.(2), shape.(3)) in
        let x_reshaped = Rune.reshape [| n; h; w; 1 |] x in
        Rune.transpose x_reshaped ~axes:[| 0; 3; 1; 2 |]
    | `NHWC ->
        (* Keep original shape [N, H, W, 1] *)
        x
  in

  (* Flatten if requested *)
  let x =
    if flatten then
      let shape = Rune.shape x in
      let n = shape.(0) in
      Rune.reshape [| n; 28 * 28 |] x
    else x
  in

  (* Keep labels as class indices *)
  let y = Rune.squeeze y ~axes:[| 1 |] in

  (* Remove the extra dimension [N, 1] -> [N] *)
  (* Keep as float for now, will be cast to int when needed *)

  (* Create the dataset *)
  Kaun.Dataset.of_xy (x, y)
