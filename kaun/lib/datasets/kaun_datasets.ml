(** Ready-to-use datasets for Kaun *)

(* Cache for loaded datasets to avoid reloading *)
let mnist_cache = ref None

let load_mnist_cached () =
  match !mnist_cache with
  | Some data -> data
  | None ->
      let data = Nx_datasets.load_mnist () in
      mnist_cache := Some data;
      data

let mnist ?(train = true) ?(flatten = false) ?(normalize = true)
    ?(data_format = `NCHW) ~device () =
  (* Load MNIST data from Nx_datasets *)
  let start = Unix.gettimeofday () in
  let (x_train, y_train), (x_test, y_test) = load_mnist_cached () in
  Printf.printf "[Kaun_datasets.mnist] Data loaded in %.3fs\n%!"
    (Unix.gettimeofday () -. start);

  (* Select training or test data *)
  let x, y = if train then (x_train, y_train) else (x_test, y_test) in

  (* Convert from uint8 to float *)
  (* Cast to float32 *)
  let cast_start = Unix.gettimeofday () in
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in
  Printf.printf "[Kaun_datasets.mnist] Cast to float32 in %.3fs\n%!"
    (Unix.gettimeofday () -. cast_start);

  (* Convert to Rune tensors *)
  let dtype = Rune.float32 in
  (* Convert Nx tensors to Rune tensors via bigarray *)
  let convert_start = Unix.gettimeofday () in
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in
  Printf.printf "[Kaun_datasets.mnist] Converted to Rune tensors in %.3fs\n%!"
    (Unix.gettimeofday () -. convert_start);

  (* Normalize to [0, 1] if requested *)
  let norm_start = Unix.gettimeofday () in
  let x =
    if normalize then Rune.div x (Rune.scalar device dtype 255.0) else x
  in
  Printf.printf "[Kaun_datasets.mnist] Normalization in %.3fs\n%!"
    (Unix.gettimeofday () -. norm_start);

  (* Handle data format *)
  let format_start = Unix.gettimeofday () in
  let x =
    match data_format with
    | `NCHW ->
        (* Original shape is [N, H, W, 1], convert to [N, 1, H, W] *)
        let shape = Rune.shape x in
        let n, h, w, _ = (shape.(0), shape.(1), shape.(2), shape.(3)) in
        let x_reshaped = Rune.reshape [| n; h; w; 1 |] x in
        let x_transposed = Rune.transpose x_reshaped ~axes:[| 0; 3; 1; 2 |] in
        Printf.printf
          "[Kaun_datasets.mnist] After transpose, is_c_contiguous: %b\n%!"
          (Rune.is_c_contiguous x_transposed);
        x_transposed
    | `NHWC ->
        (* Keep original shape [N, H, W, 1] *)
        x
  in
  Printf.printf "[Kaun_datasets.mnist] Data format conversion in %.3fs\n%!"
    (Unix.gettimeofday () -. format_start);

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
  let dataset_start = Unix.gettimeofday () in
  let result = Kaun.Dataset.of_xy (x, y) in
  Printf.printf "[Kaun_datasets.mnist] Dataset.of_xy in %.3fs\n%!"
    (Unix.gettimeofday () -. dataset_start);
  result
