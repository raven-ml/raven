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
  let batches_dir = "cifar-10-batches-bin/"
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

(* The archive, downloaded once into the dataset's cache directory. *)
let ensure_archive () =
  let dataset_dir = get_cache_dir Config.cache_subdir in
  Kaun_download.mkdir_p dataset_dir;
  let archive = dataset_dir ^ Config.archive_name in
  ensure_file Config.url archive;
  archive

let err_batch name what =
  failwith (Printf.sprintf "CIFAR-10 batch %s %s" name what)

(* The images, [[|n; 3; 32; 32|]], and labels of a split of [n] entries. *)
let split n =
  ( Genarray.create int8_unsigned c_layout
      [| n; Config.channels; Config.height; Config.width |],
    Array1.create int8_unsigned c_layout n )

(* Reads the batch [name] of [size] bytes through [take] into [images] and
   [labels] from entry [first] on. *)
let read_batch name size take (images, labels) first =
  if size mod Config.entry_size <> 0 then
    err_batch name
      (Printf.sprintf "has unexpected size %d (expected multiple of %d)" size
         Config.entry_size);
  let entries = size / Config.entry_size in
  if entries <> Config.entries_per_batch then
    err_batch name
      (Printf.sprintf "has %d entries (expected %d)" entries
         Config.entries_per_batch);
  let n = Array1.dim labels in
  let flat = Bigarray.reshape_1 images (n * Config.image_size) in
  for i = 0 to entries - 1 do
    let e = take Config.entry_size in
    let idx = first + i in
    Array1.unsafe_set labels idx (Char.code e.[0]);
    let base = idx * Config.image_size in
    for p = 0 to Config.image_size - 1 do
      Array1.unsafe_set flat (base + p)
        (Char.code (String.unsafe_get e (p + 1)))
    done
  done

let load () =
  let archive = ensure_archive () in
  let per_batch = Config.entries_per_batch in
  let train = split (List.length Config.train_batches * per_batch) in
  let test = split (List.length Config.test_batches * per_batch) in
  (* Each batch's split and first entry. *)
  let slots =
    List.mapi (fun i b -> (b, (train, i * per_batch))) Config.train_batches
    @ List.mapi (fun i b -> (b, (test, i * per_batch))) Config.test_batches
  in
  let seen = Hashtbl.create 8 in
  Log.info (fun m -> m "Loading CIFAR-10 from %s..." archive);
  read_tar_gz archive (fun member size take ->
      let dir = Config.batches_dir in
      let n = String.length dir in
      if String.length member > n && String.sub member 0 n = dir then
        let name = String.sub member n (String.length member - n) in
        match List.assoc_opt name slots with
        | None -> ()
        | Some (dst, first) ->
            read_batch name size take dst first;
            Hashtbl.replace seen name ());
  List.iter
    (fun (name, _) ->
      if not (Hashtbl.mem seen name) then
        err_batch name (Printf.sprintf "is missing from %s" archive))
    slots;
  Log.info (fun m -> m "CIFAR-10 loading complete");
  (train, test)
