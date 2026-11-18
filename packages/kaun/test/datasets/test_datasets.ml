(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Everything here runs without network access: each test seeds a temporary
   cache directory with the files a loader would otherwise download. *)

open Windtrap

(* A CIFAR-10 entry is one label byte followed by a 3 x 32 x 32 image. *)
let cifar10_entry = 1 + (3 * 32 * 32)
let cifar10_batch = 10_000

let write_file path n_bytes =
  Out_channel.with_open_bin path (fun oc ->
      output_string oc (String.make n_bytes '\001'))

(* [seed_cifar10 ~entries] fills a fresh cache with the extracted archive, every
   batch empty except [data_batch_1.bin], which holds [entries] entries. *)
let seed_cifar10 ~entries =
  let root = temp_dir () in
  setenv "RAVEN_CACHE_ROOT" (Some root);
  let mkdir parent name =
    let d = Filename.concat parent name in
    Sys.mkdir d 0o755;
    d
  in
  let dir =
    List.fold_left mkdir root [ "datasets"; "cifar10"; "cifar-10-batches-bin" ]
  in
  List.iter
    (fun name -> write_file (Filename.concat dir name) 0)
    [
      "data_batch_2.bin";
      "data_batch_3.bin";
      "data_batch_4.bin";
      "data_batch_5.bin";
      "test_batch.bin";
    ];
  write_file (Filename.concat dir "data_batch_1.bin") (entries * cifar10_entry)

(* [seed_mnist ~rows ~cols images] fills a fresh cache with MNIST's four
   files, each split holding [images], whose pixel [(i, r, c)] is
   [i * rows * cols + r * cols + c], and as many labels, [i]. *)
let seed_mnist ~rows ~cols images =
  let root = temp_dir () in
  setenv "RAVEN_CACHE_ROOT" (Some root);
  let dir = Filename.concat (Filename.concat root "datasets") "mnist" in
  Sys.mkdir (Filename.concat root "datasets") 0o755;
  Sys.mkdir dir 0o755;
  let int32_be n =
    String.init 4 (fun k -> Char.chr ((n lsr (8 * (3 - k))) land 0xff))
  in
  let write name header body =
    Out_channel.with_open_bin (Filename.concat dir name) (fun oc ->
        List.iter (fun n -> output_string oc (int32_be n)) header;
        output_string oc body)
  in
  let pixels =
    String.init (images * rows * cols) (fun p -> Char.chr (p land 0xff))
  in
  let labels = String.init images Char.chr in
  List.iter
    (fun split ->
      write (split ^ "-images-idx3-ubyte") [ 2051; images; rows; cols ] pixels;
      write (split ^ "-labels-idx1-ubyte") [ 2049; images ] labels)
    [ "train"; "t10k" ]

(* Images come channels last, [[N; rows; cols; 1]], each pixel where its
   row and column put it. *)
let test_mnist_channels_last () =
  let rows = 2 and cols = 3 and images = 2 in
  seed_mnist ~rows ~cols images;
  let train_x, train_y, _, _ = Kaun_datasets.mnist ~normalize:false () in
  equal ~msg:"shape" (array int) [| images; rows; cols; 1 |]
    (Nx.shape train_x);
  equal ~msg:"pixels in C order of (image, row, column)" (array float_exact)
    (Array.init (images * rows * cols) Float.of_int)
    (Nx.to_array train_x);
  equal ~msg:"labels" (array int64) [| 0L; 1L |] (Nx.to_array train_y)

let test_cifar10_batch_count entries =
  seed_cifar10 ~entries;
  raises_match (Exn.failure ~sub:"data_batch_1.bin") (fun () ->
      Kaun_datasets.cifar10 ())

let tests =
  [
    group "mnist"
      [ test "images are channels last" test_mnist_channels_last ];
    group "cifar10"
      [
        cases "batch of another size"
          ~id:(Printf.sprintf "refuses a batch of %d entries")
          [ 0; 1; cifar10_batch - 1; cifar10_batch + 1 ]
          test_cifar10_batch_count;
      ];
  ]

let () = exit (run "kaun datasets" tests)
