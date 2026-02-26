(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray
open Dataset_utils

let src = Logs.Src.create "kaun.datasets.cifar10" ~doc:"CIFAR-10 dataset loader"

module Log = (val Logs.src_log src : Logs.LOG)

module Config = struct
  let url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
  let cache_subdir = "cifar10/"
  let archive_name = "cifar-10-binary.tar.gz"
  let extracted_subdir = "cifar-10-batches-bin/"
  let height = 32
  let width = 32
  let channels = 3
  let image_size = channels * height * width
  let entry_size = 1 + image_size
  let entries_per_batch = 10000

  let train_batches =
    [
      "data_batch_1.bin";
      "data_batch_2.bin";
      "data_batch_3.bin";
      "data_batch_4.bin";
      "data_batch_5.bin";
    ]

  let test_batches = [ "test_batch.bin" ]
end

let ensure_dataset () =
  let dataset_dir = get_cache_dir Config.cache_subdir in
  mkdir_p dataset_dir;
  let archive_path = dataset_dir ^ Config.archive_name in
  let extracted_dir = dataset_dir ^ Config.extracted_subdir in
  let check_file = extracted_dir ^ "test_batch.bin" in
  if not (Sys.file_exists check_file) then (
    ensure_file Config.url archive_path;
    if
      not
        (ensure_extracted_tar_gz ~tar_gz_path:archive_path
           ~target_dir:dataset_dir ~check_file)
    then
      failwith
        (Printf.sprintf "Failed to extract CIFAR-10 archive to %s" extracted_dir));
  extracted_dir

let read_batch_file ~extracted_dir filename =
  let path = extracted_dir ^ filename in
  Log.debug (fun m -> m "Reading CIFAR-10 batch: %s" path);
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let s = really_input_string ic (in_channel_length ic) in
      let num_entries = String.length s / Config.entry_size in
      if String.length s <> num_entries * Config.entry_size then
        failwith
          (Printf.sprintf
             "CIFAR-10 batch %s has unexpected size %d (expected multiple of \
              %d)"
             filename (String.length s) Config.entry_size);
      (s, num_entries))

let load () =
  let extracted_dir = ensure_dataset () in
  let load_split batch_files expected_total =
    let images =
      Genarray.create int8_unsigned c_layout
        [| expected_total; Config.channels; Config.height; Config.width |]
    in
    let labels = Array1.create int8_unsigned c_layout expected_total in
    let flat = Bigarray.reshape_1 images (expected_total * Config.image_size) in
    let offset = ref 0 in
    List.iter
      (fun filename ->
        let s, num_entries = read_batch_file ~extracted_dir filename in
        for i = 0 to num_entries - 1 do
          let entry_offset = i * Config.entry_size in
          let idx = !offset + i in
          Array1.unsafe_set labels idx (Char.code s.[entry_offset]);
          let img_offset = entry_offset + 1 in
          let base = idx * Config.image_size in
          for p = 0 to Config.image_size - 1 do
            Array1.unsafe_set flat (base + p)
              (Char.code (String.unsafe_get s (img_offset + p)))
          done
        done;
        offset := !offset + num_entries)
      batch_files;
    (images, labels)
  in
  Log.info (fun m -> m "Loading CIFAR-10 datasets...");
  let train_images, train_labels =
    load_split Config.train_batches
      (List.length Config.train_batches * Config.entries_per_batch)
  in
  let test_images, test_labels =
    load_split Config.test_batches
      (List.length Config.test_batches * Config.entries_per_batch)
  in
  Log.info (fun m -> m "CIFAR-10 loading complete");
  ((train_images, train_labels), (test_images, test_labels))
