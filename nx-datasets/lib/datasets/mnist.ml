(* mnist.ml *)
open Bigarray
open Dataset_utils

(* Config remains the same *)
module Config = struct
  type t = {
    name : string;
    cache_subdir : string;
    train_images_url : string;
    train_labels_url : string;
    test_images_url : string;
    test_labels_url : string;
    image_magic_number : int;
    label_magic_number : int;
  }

  let mnist =
    {
      name = "MNIST";
      cache_subdir = "mnist/";
      train_images_url =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
      train_labels_url =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
      test_images_url =
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
      test_labels_url =
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz";
      image_magic_number = 2051;
      label_magic_number = 2049;
    }

  let fashion_mnist =
    {
      name = "Fashion-MNIST";
      cache_subdir = "fashion-mnist/";
      train_images_url =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz";
      train_labels_url =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz";
      test_images_url =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz";
      test_labels_url =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz";
      image_magic_number = 2051;
      label_magic_number = 2049;
    }
end

let mnist_config = Config.mnist
let fashion_mnist_config = Config.fashion_mnist

(* IDX parsing logic remains specific and local *)
let read_int32_be s pos =
  let b1 = Char.code s.[pos] in
  let b2 = Char.code s.[pos + 1] in
  let b3 = Char.code s.[pos + 2] in
  let b4 = Char.code s.[pos + 3] in
  (b1 lsl 24) lor (b2 lsl 16) lor (b3 lsl 8) lor b4

let ensure_dataset config =
  let dataset_dir = get_cache_dir config.Config.cache_subdir in
  mkdir_p dataset_dir;

  (* Ensure base dir exists *)
  let files_to_process =
    [
      ("train-images-idx3-ubyte", config.Config.train_images_url);
      ("train-labels-idx1-ubyte", config.Config.train_labels_url);
      ("t10k-images-idx3-ubyte", config.Config.test_images_url);
      ("t10k-labels-idx1-ubyte", config.Config.test_labels_url);
    ]
  in
  List.iter
    (fun (base_filename, url) ->
      let gz_filename = base_filename ^ ".gz" in
      let gz_path = dataset_dir ^ gz_filename in
      let path = dataset_dir ^ base_filename in

      if not (Sys.file_exists path) then (
        Printf.printf "File %s not found for %s dataset.\n%!" base_filename
          config.name;
        (* Ensure the .gz file is downloaded *)
        ensure_file url gz_path;
        (* Ensure it's decompressed *)
        if not (ensure_decompressed_gz ~gz_path ~target_path:path) then
          failwith (Printf.sprintf "Failed to obtain decompressed file %s" path))
      else Printf.printf "Found decompressed file %s.\n%!" path)
    files_to_process

let read_idx_file ~read_header ~create_array ~populate_array ~expected_magic
    config filename =
  Printf.printf "Reading %s file: %s\n%!" config.Config.name filename;
  let ic = open_in_bin filename in
  let s =
    try really_input_string ic (in_channel_length ic)
    with exn ->
      close_in_noerr ic;
      failwith
        (Printf.sprintf "Error reading file %s: %s" filename
           (Printexc.to_string exn))
  in
  close_in ic;

  let magic = read_int32_be s 0 in
  if magic <> expected_magic then
    failwith
      (Printf.sprintf "Invalid magic number %d in %s (expected %d)" magic
         filename expected_magic);

  let dimensions, data_offset = read_header s in
  let total_items, data_len =
    match dimensions with
    | [| d1 |] -> (d1, d1)
    | [| d1; d2; d3 |] -> (d1, d1 * d2 * d3)
    | _ -> failwith "Unsupported dimension format"
  in
  let expected_len = data_offset + data_len in
  if String.length s <> expected_len then
    failwith
      (Printf.sprintf
         "File %s has unexpected length: %d vs %d (header offset %d, data len \
          %d)"
         filename (String.length s) expected_len data_offset data_len);

  let arr = create_array dimensions in
  populate_array arr s data_offset total_items;
  arr

(* read_images and read_labels remain largely the same, just use the config
   passed in *)
let read_images config filename =
  let read_header s =
    let num_images = read_int32_be s 4 in
    let num_rows = read_int32_be s 8 in
    let num_cols = read_int32_be s 12 in
    ([| num_images; num_rows; num_cols |], 16)
  in
  let create_array dims =
    Array3.create int8_unsigned c_layout dims.(0) dims.(1) dims.(2)
  in
  let populate_array arr s offset _ =
    let num_images = Array3.dim1 arr in
    let num_rows = Array3.dim2 arr in
    let num_cols = Array3.dim3 arr in
    let img_size = num_rows * num_cols in
    for i = 0 to num_images - 1 do
      let start_pos = offset + (i * img_size) in
      for r = 0 to num_rows - 1 do
        for c = 0 to num_cols - 1 do
          let pos = start_pos + (r * num_cols) + c in
          arr.{i, r, c} <- Char.code s.[pos]
        done
      done
    done
  in
  read_idx_file ~read_header ~create_array ~populate_array
    ~expected_magic:config.Config.image_magic_number config filename

let read_labels config filename =
  let read_header s =
    let num_labels = read_int32_be s 4 in
    ([| num_labels |], 8)
  in
  let create_array dims = Array1.create int8_unsigned c_layout dims.(0) in
  let populate_array arr s offset total_items =
    for i = 0 to total_items - 1 do
      arr.{i} <- Char.code s.[offset + i]
    done
  in
  read_idx_file ~read_header ~create_array ~populate_array
    ~expected_magic:config.Config.label_magic_number config filename

let load ~fashion_mnist =
  let config = if fashion_mnist then Config.fashion_mnist else Config.mnist in
  ensure_dataset config;

  let dataset_dir = get_cache_dir config.Config.cache_subdir in
  let train_images_path = dataset_dir ^ "train-images-idx3-ubyte" in
  let train_labels_path = dataset_dir ^ "train-labels-idx1-ubyte" in
  let test_images_path = dataset_dir ^ "t10k-images-idx3-ubyte" in
  let test_labels_path = dataset_dir ^ "t10k-labels-idx1-ubyte" in

  Printf.printf "Loading %s datasets...\n%!" config.name;
  let train_images = read_images config train_images_path in
  let train_labels = read_labels config train_labels_path in
  let test_images = read_images config test_images_path in
  let test_labels = read_labels config test_labels_path in
  Printf.printf "%s loading complete.\n%!" config.name;
  ((train_images, train_labels), (test_images, test_labels))
