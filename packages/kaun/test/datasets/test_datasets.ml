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

let test_cifar10_batch_count entries =
  seed_cifar10 ~entries;
  raises_match (Exn.failure ~substring:"data_batch_1.bin") (fun () ->
      Kaun_datasets.cifar10 ())

let tests =
  [
    group "cifar10"
      [
        cases
          ~name:(Printf.sprintf "refuses a batch of %d entries")
          "batch of another size"
          [ 0; 1; cifar10_batch - 1; cifar10_batch + 1 ]
          test_cifar10_batch_count;
      ];
  ]

let () = exit (run "kaun datasets" tests)
