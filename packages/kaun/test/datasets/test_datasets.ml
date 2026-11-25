(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nothing here touches the network: each test seeds a temporary cache with the
   files a loader would otherwise download, and runs with no [curl] on [PATH]
   or a stand-in that fails. *)

open Windtrap

(* A cache root of [RAVEN_CACHE_ROOT], and the directory [datasets/name] in
   it. *)
let dataset_dir root name =
  let dir = Filename.concat (Filename.concat root "datasets") name in
  if not (Sys.file_exists (Filename.dirname dir)) then
    Sys.mkdir (Filename.dirname dir) 0o755;
  Sys.mkdir dir 0o755;
  dir

let fresh_root () =
  let root = temp_dir () in
  setenv "RAVEN_CACHE_ROOT" (Some root);
  setenv "PATH" (Some (temp_dir ()));
  root

(* MNIST *)

let rows = 2
let cols = 3
let images = 2
let image_magic = 2051
let label_magic = 2049

(* Pixel [p] in C order of (image, row, column) of the training images:
   distinct, from 0 to 255. The test images are [255 - pixel p]. *)
let pixel p = p * 255 / ((images * rows * cols) - 1)
let test_pixel p = 255 - pixel p

let int32_be n =
  String.init 4 (fun k -> Char.chr ((n lsr (8 * (3 - k))) land 0xff))

let write dir name header body =
  Out_channel.with_open_bin (Filename.concat dir name) (fun oc ->
      List.iter (fun n -> output_string oc (int32_be n)) header;
      output_string oc body)

(* Writes the four files of an MNIST-format dataset into [dir]: the images of
   each split by [pixel] and [test_pixel], the training labels counting from
   [first_label] and the test labels from [first_label + images]. *)
let seed_mnist ?(image_magic = image_magic) ?(label_magic = label_magic)
    ~first_label dir =
  let bytes f = String.init (images * rows * cols) (fun p -> Char.chr (f p)) in
  let labels first = String.init images (fun i -> Char.chr (first + i)) in
  List.iter
    (fun (split, f, first) ->
      write dir
        (split ^ "-images-idx3-ubyte")
        [ image_magic; images; rows; cols ]
        (bytes f);
      write dir
        (split ^ "-labels-idx1-ubyte")
        [ label_magic; images ] (labels first))
    [
      ("train", pixel, first_label); ("t10k", test_pixel, first_label + images);
    ]

let pixels f = Array.init (images * rows * cols) (fun p -> float_of_int (f p))

let channels_last () =
  seed_mnist ~first_label:0 (dataset_dir (fresh_root ()) "mnist");
  let train_x, train_y, test_x, test_y =
    Kaun_datasets.mnist ~normalize:false ()
  in
  List.iter
    (fun (split, f, x) ->
      equal ~msg:(split ^ " shape") (array int)
        [| images; rows; cols; 1 |]
        (Nx.shape x);
      equal
        ~msg:(split ^ " pixels in C order of (image, row, column)")
        (array float_exact) (pixels f) (Nx.to_array x))
    [ ("train", pixel, train_x); ("test", test_pixel, test_x) ];
  equal ~msg:"train labels" (array int64) [| 0L; 1L |] (Nx.to_array train_y);
  equal ~msg:"test labels" (array int64) [| 2L; 3L |] (Nx.to_array test_y)

(* The seeded bytes span 0 to 255 in both splits. *)
let normalizes () =
  seed_mnist ~first_label:0 (dataset_dir (fresh_root ()) "mnist");
  let train_x, _, test_x, _ = Kaun_datasets.mnist () in
  List.iter
    (fun (split, x) ->
      Array.iter
        (satisfies ~msg:split ~claim:"in [0, 1]" float_exact (fun v ->
             0. <= v && v <= 1.))
        (Nx.to_array x))
    [ ("train", train_x); ("test", test_x) ]

let download_error () =
  ignore (fresh_root ());
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ())

(* curl reads ~/.curlrc before its arguments unless [-q] comes first (man curl,
   [-q]); an option there, such as [-o], would act on the download. A [curl] alone on [PATH] writes its arguments to a log, one per
   line, and fails as curl does on a 404. *)
let reads_no_curlrc () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  ignore (fresh_root ());
  let bin = temp_dir ~prefix:"bin" () in
  let log = Filename.concat (temp_dir ~prefix:"curl" ()) "arguments" in
  let curl = Filename.concat bin "curl" in
  Out_channel.with_open_bin curl (fun oc ->
      Printf.fprintf oc "#!/bin/sh\nprintf '%%s\\n' \"$@\" > %s\nexit 22\n"
        (Filename.quote log));
  Unix.chmod curl 0o755;
  setenv "PATH" (Some bin);
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ());
  let first =
    if Sys.file_exists log then
      In_channel.with_open_bin log In_channel.input_line
    else None
  in
  equal ~msg:"curl's first argument" (option string) (Some "-q") first

let selects_fashion () =
  let root = fresh_root () in
  seed_mnist ~first_label:0 (dataset_dir root "mnist");
  seed_mnist ~first_label:5 (dataset_dir root "fashion-mnist");
  let labels fashion =
    let _, train_y, _, _ = Kaun_datasets.mnist ~fashion () in
    Nx.to_array train_y
  in
  equal ~msg:"~fashion:true" (array int64) [| 5L; 6L |] (labels true);
  equal ~msg:"~fashion:false" (array int64) [| 0L; 1L |] (labels false)

let wrong_magic (_, seed) =
  seed (dataset_dir (fresh_root ()) "mnist");
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ())

(* CIFAR-10 *)

(* An entry is one label byte followed by a 3 x 32 x 32 image. *)
let cifar10_entry = 1 + (3 * 32 * 32)
let cifar10_batch = 10_000

let write_bytes path n =
  Out_channel.with_open_bin path (fun oc ->
      output_string oc (String.make n '\001'))

(* The extracted archive, every batch empty but [data_batch_1.bin], which holds
   [entries] entries. *)
let seed_cifar10 ~entries =
  let dir = dataset_dir (fresh_root ()) "cifar10" in
  let dir = Filename.concat dir "cifar-10-batches-bin" in
  Sys.mkdir dir 0o755;
  List.iter
    (fun name -> write_bytes (Filename.concat dir name) 0)
    [
      "data_batch_2.bin";
      "data_batch_3.bin";
      "data_batch_4.bin";
      "data_batch_5.bin";
      "test_batch.bin";
    ];
  write_bytes (Filename.concat dir "data_batch_1.bin") (entries * cifar10_entry)

let refuses_batch_size entries =
  seed_cifar10 ~entries;
  raises_match (Exn.failure ~sub:"data_batch_1.bin") (fun () ->
      Kaun_datasets.cifar10 ())

let () =
  exit
    (run "kaun datasets"
       [
         group "mnist"
           [
             test
               "reads each split's images channels last and its labels as int64"
               channels_last;
             test "gives images in [0, 1] by default" normalizes;
             test "reads Fashion-MNIST when fashion is true" selects_fashion;
             test "raises Failure on a download error" download_error;
             test "runs curl without the user's configuration" reads_no_curlrc;
             cases "raises Failure on a file of the wrong magic number" ~id:fst
               [
                 ( "images",
                   fun dir ->
                     seed_mnist ~image_magic:label_magic ~first_label:0 dir );
                 ( "labels",
                   fun dir ->
                     seed_mnist ~label_magic:image_magic ~first_label:0 dir );
               ]
               wrong_magic;
           ];
         group "cifar10"
           [
             cases
               "raises Failure naming a batch file of another size than 10,000 \
                entries"
               ~id:(Printf.sprintf "%d entries")
               [ 0; 1; cifar10_batch - 1; cifar10_batch + 1 ]
               refuses_batch_size;
           ];
       ])
