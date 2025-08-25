(** Ready-to-use datasets for machine learning *)

(* Set up logging *)
let src = Logs.Src.create "kaun.datasets" ~doc:"Kaun datasets module"

module Log = (val Logs.src_log src : Logs.LOG)

(** {1 Core Types} *)

type ('elt, 'kind, 'dev) tensor_dataset =
  ('elt, 'kind, 'dev) Rune.t Kaun_dataset.t

(** {1 Vision Datasets} *)

let mnist ?(train = true) ?(flatten = false) ?(normalize = true)
    ?(data_format = `NCHW) ?cache_dir:_ ~device () =
  (* Load from nx-datasets *)
  let (x_train, y_train), (x_test, y_test) = Nx_datasets.load_mnist () in

  (* Select training or test data *)
  let x, y = if train then (x_train, y_train) else (x_test, y_test) in

  (* Convert to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in

  (* Normalize to [0, 1] if requested *)
  let x =
    if normalize then Rune.div x (Rune.scalar device Rune.float32 255.0) else x
  in

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

  (* Create the dataset using Kaun_dataset *)
  Kaun_dataset.from_tensors (x, y)

let cifar10 ?(train = true) ?(normalize = true) ?(data_format = `NCHW)
    ?(augmentation = false) ?cache_dir:_ ~device () =
  (* Load from nx-datasets *)
  let (x_train, y_train), (x_test, y_test) = Nx_datasets.load_cifar10 () in

  (* Select training or test data *)
  let x, y = if train then (x_train, y_train) else (x_test, y_test) in

  (* Convert to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in

  (* Normalize with ImageNet stats if requested *)
  let x =
    if normalize then
      let mean_arr =
        Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
          [| 0.485; 0.456; 0.406 |]
      in
      let std_arr =
        Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
          [| 0.229; 0.224; 0.225 |]
      in
      let mean =
        Rune.of_bigarray device (Bigarray.genarray_of_array1 mean_arr)
      in
      let std = Rune.of_bigarray device (Bigarray.genarray_of_array1 std_arr) in
      let x = Rune.div x (Rune.scalar device Rune.float32 255.0) in
      let mean = Rune.reshape [| 1; 3; 1; 1 |] mean in
      let std = Rune.reshape [| 1; 3; 1; 1 |] std in
      Rune.div (Rune.sub x mean) std
    else x
  in

  (* Handle data format *)
  let x =
    match data_format with
    | `NCHW -> x (* CIFAR-10 is already in NCHW format *)
    | `NHWC ->
        (* Convert from [N, C, H, W] to [N, H, W, C] *)
        Rune.transpose x ~axes:[| 0; 2; 3; 1 |]
  in

  (* Keep labels as class indices *)
  let y = Rune.squeeze y ~axes:[| 1 |] in

  (* Create dataset and apply augmentation if requested *)
  let dataset = Kaun_dataset.from_tensors (x, y) in
  if augmentation && train then
    (* TODO: Add augmentation transforms when available *)
    dataset
  else dataset

let fashion_mnist ?(train = true) ?(flatten = false) ?(normalize = true)
    ?(data_format = `NCHW) ?cache_dir:_ ~device () =
  (* Load from nx-datasets *)
  let (x_train, y_train), (x_test, y_test) =
    Nx_datasets.load_fashion_mnist ()
  in

  (* Select training or test data *)
  let x, y = if train then (x_train, y_train) else (x_test, y_test) in

  (* Convert to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in

  (* Normalize to [0, 1] if requested *)
  let x =
    if normalize then Rune.div x (Rune.scalar device Rune.float32 255.0) else x
  in

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

  Kaun_dataset.from_tensors (x, y)

(** {1 Text Datasets} *)

let imdb ?(train = true) ?tokenizer ?(max_length = 512) ?cache_dir:_ ~device ()
    =
  (* TODO: Load actual IMDB data when available in nx-datasets *)
  (* For now, create a placeholder with synthetic data *)
  let num_samples = if train then 25000 else 25000 in
  let texts =
    Array.init num_samples (fun i ->
        if i mod 2 = 0 then
          "This movie was absolutely fantastic great amazing wonderful"
        else "This movie was terrible awful bad horrible worst")
  in

  (* Create labels *)
  let labels =
    Array.init num_samples (fun i ->
        Rune.scalar device Rune.float32 (float_of_int (i mod 2)))
  in

  (* Create text dataset *)
  let text_dataset = Kaun_dataset.from_array texts in

  (* Tokenize if tokenizer provided *)
  let tokenized =
    match tokenizer with
    | Some tok ->
        Kaun_dataset.tokenize tok ~max_length ~truncation:true text_dataset
    | None ->
        (* Use default whitespace tokenizer *)
        Kaun_dataset.tokenize Kaun_dataset.whitespace_tokenizer ~max_length
          ~truncation:true text_dataset
  in

  (* Create label dataset *)
  let label_dataset = Kaun_dataset.from_array labels in

  (* Zip tokenized text with labels *)
  Kaun_dataset.zip tokenized label_dataset

let wikitext ?(dataset_name = `Wikitext2) ?tokenizer ?(sequence_length = 1024)
    ?cache_dir:_ ~device:_ () =
  let _ = dataset_name in
  (* Mark as intentionally unused *)
  (* TODO: Load actual WikiText data when available in nx-datasets *)
  (* For now, create a placeholder with synthetic text *)
  let text =
    String.concat " "
      [
        "The quick brown fox jumps over the lazy dog.";
        "Machine learning is a subset of artificial intelligence.";
        "Neural networks are inspired by biological neurons.";
        "Deep learning has revolutionized computer vision.";
        "Natural language processing enables machines to understand text.";
      ]
  in

  (* Tokenize the entire text *)
  let tokenizer =
    Option.value tokenizer ~default:Kaun_dataset.whitespace_tokenizer
  in
  let tokens = tokenizer text in

  (* Create sliding windows for language modeling *)
  let num_windows =
    max 1 ((Array.length tokens - sequence_length - 1) / sequence_length)
  in

  let windows =
    Array.init num_windows (fun i ->
        let start = i * sequence_length in
        let input_ids = Array.sub tokens start sequence_length in
        let target_ids = Array.sub tokens (start + 1) sequence_length in
        (input_ids, target_ids))
  in

  Kaun_dataset.from_array windows

(** {1 Structured Data} *)

let iris ?(normalize = true) ?(train_split = 0.8) ?shuffle_seed ~device () =
  let _ = train_split in
  (* Mark as intentionally unused - will use train_test_split *)

  (* Load from nx-datasets *)
  let x, y = Nx_datasets.load_iris () in

  (* Convert to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in

  (* Normalize if requested *)
  let x =
    if normalize then
      let mean = Rune.mean x ~axes:[| 0 |] ~keepdims:true in
      let std = Rune.std x ~axes:[| 0 |] ~keepdims:true in
      Rune.div (Rune.sub x mean)
        (Rune.add std (Rune.scalar device Rune.float32 1e-8))
    else x
  in

  (* Create dataset *)
  let dataset = Kaun_dataset.from_tensors (x, y) in

  (* Optionally shuffle *)
  match shuffle_seed with
  | Some seed ->
      let key = Rune.Rng.key seed in
      Kaun_dataset.shuffle ~rng:key dataset
  | None -> dataset

let boston_housing ?(normalize = true) ?(train_split = 0.8) ~device () =
  let _ = train_split in
  (* Mark as intentionally unused - will use train_test_split *)

  (* Use California housing as a replacement since Boston housing is not in
     nx-datasets *)
  let x, y = Nx_datasets.load_california_housing () in

  (* Convert to float32 *)
  let x = Nx.cast Nx.float32 x in
  let y = Nx.cast Nx.float32 y in

  (* Convert to Rune tensors *)
  let x = Rune.of_bigarray device (Nx.to_bigarray x) in
  let y = Rune.of_bigarray device (Nx.to_bigarray y) in

  (* Normalize if requested *)
  let x =
    if normalize then
      let mean = Rune.mean x ~axes:[| 0 |] ~keepdims:true in
      let std = Rune.std x ~axes:[| 0 |] ~keepdims:true in
      Rune.div (Rune.sub x mean)
        (Rune.add std (Rune.scalar device Rune.float32 1e-8))
    else x
  in

  Kaun_dataset.from_tensors (x, y)

(** {1 Dataset Utilities} *)

let download_and_extract ~url ~cache_dir ?(extract = true) () =
  (* Ensure directory exists *)
  if not (Sys.file_exists cache_dir) then
    Sys.command (Printf.sprintf "mkdir -p %s" cache_dir) |> ignore;

  let filename = Filename.basename url in
  let filepath = Filename.concat cache_dir filename in

  (* Download if not exists *)
  if not (Sys.file_exists filepath) then (
    Log.info (fun m -> m "Downloading %s to %s..." url filepath);
    let cmd = Printf.sprintf "curl -L -o %s %s" filepath url in
    match Sys.command cmd with
    | 0 -> Log.info (fun m -> m "Download complete")
    | _ -> failwith (Printf.sprintf "Failed to download %s" url));

  (* Extract if needed *)
  if
    extract
    && (Filename.check_suffix filename ".tar.gz"
       || Filename.check_suffix filename ".zip")
  then (
    let extract_dir = Filename.chop_extension filepath in
    if not (Sys.file_exists extract_dir) then (
      Log.info (fun m -> m "Extracting %s..." filename);
      let cmd =
        if Filename.check_suffix filename ".tar.gz" then
          Printf.sprintf "tar -xzf %s -C %s" filepath cache_dir
        else Printf.sprintf "unzip -q %s -d %s" filepath cache_dir
      in
      match Sys.command cmd with
      | 0 -> Log.info (fun m -> m "Extraction complete")
      | _ -> failwith (Printf.sprintf "Failed to extract %s" filename));
    extract_dir)
  else filepath

let train_test_split ?(test_size = 0.2) ?(shuffle = true) ?seed dataset =
  (* Get dataset length *)
  let total_length =
    match Kaun_dataset.cardinality dataset with
    | Kaun_dataset.Finite n -> n
    | _ -> failwith "Cannot split dataset with unknown or infinite cardinality"
  in

  let test_length = int_of_float (float_of_int total_length *. test_size) in
  let train_length = total_length - test_length in

  (* Optionally shuffle before splitting *)
  let dataset =
    if shuffle then
      match seed with
      | Some s ->
          let key = Rune.Rng.key s in
          Kaun_dataset.shuffle ~rng:key dataset
      | None -> Kaun_dataset.shuffle dataset
    else dataset
  in

  (* Split into train and test *)
  let train_dataset = Kaun_dataset.take train_length dataset in
  let test_dataset =
    dataset |> Kaun_dataset.skip train_length |> Kaun_dataset.take test_length
  in

  (train_dataset, test_dataset)
