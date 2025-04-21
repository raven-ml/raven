(* cifar10.ml *)
open Bigarray
open Dataset_utils

let dataset_name = "cifar-10"
let base_dir = get_cache_dir dataset_name
let archive_dir_name = "cifar-10-batches-py"
let dataset_dir = base_dir ^ archive_dir_name ^ "/"
let url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
let tar_path = base_dir ^ Filename.basename url
let test_batch_rel_path = archive_dir_name ^ "/test_batch"

let ensure_dataset () =
  ensure_extracted_archive ~url ~archive_path:tar_path ~extract_dir:base_dir
    ~check_file:test_batch_rel_path

let read_cifar_batch filename =
  Printf.printf "Reading batch file: %s\n%!" filename;
  let ic = open_in_bin filename in
  let s =
    try really_input_string ic (in_channel_length ic)
    with exn ->
      close_in_noerr ic;
      raise exn
  in
  close_in ic;

  let num_bytes = String.length s in
  let bytes_per_image = 3073 in
  if num_bytes mod bytes_per_image <> 0 then
    failwith
      (Printf.sprintf "File %s has unexpected size %d" filename num_bytes);

  let num_images = num_bytes / bytes_per_image in
  Printf.printf "Found %d images in %s.\n%!" num_images filename;

  let images =
    Genarray.create int8_unsigned c_layout [| num_images; 32; 32; 3 |]
  in
  let labels = Array1.create int8_unsigned c_layout num_images in

  for i = 0 to num_images - 1 do
    let base_offset = i * bytes_per_image in
    labels.{i} <- Char.code s.[base_offset];
    let r_offset = base_offset + 1 in
    let g_offset = r_offset + 1024 in
    let b_offset = g_offset + 1024 in
    for row = 0 to 31 do
      for col = 0 to 31 do
        let plane_idx = (row * 32) + col in
        Genarray.set images [| i; row; col; 0 |]
          (Char.code s.[r_offset + plane_idx]);
        (* Red *)
        Genarray.set images [| i; row; col; 1 |]
          (Char.code s.[g_offset + plane_idx]);
        (* Green *)
        Genarray.set images [| i; row; col; 2 |]
          (Char.code s.[b_offset + plane_idx])
        (* Blue *)
      done
    done
  done;
  (images, labels)

let load () =
  ensure_dataset ();
  Printf.printf "Loading CIFAR-10 dataset...\n%!";

  let train_batches_files =
    List.init 5 (fun i -> dataset_dir ^ Printf.sprintf "data_batch_%d" (i + 1))
  in
  let train_batches_data = List.map read_cifar_batch train_batches_files in

  let total_train_images = 50000 in
  (* Create the final training Genarray *)
  let train_images =
    Genarray.create int8_unsigned c_layout [| total_train_images; 32; 32; 3 |]
  in
  let train_labels = Array1.create int8_unsigned c_layout total_train_images in

  let current_offset = ref 0 in
  List.iter
    (fun (batch_images, batch_labels) ->
      let batch_size = Array1.dim batch_labels in
      let img_slice_dims = [| batch_size; 32; 32; 3 |] in
      let img_slice =
        Genarray.sub_left train_images !current_offset batch_size
      in
      (* Ensure the slice has the expected dimensions before blitting *)
      if Genarray.dims img_slice <> img_slice_dims then
        failwith
          (Printf.sprintf
             "Internal error: train image slice dimension mismatch (expected \
              %s, got %s)"
             (String.concat "x"
                (Array.to_list (Array.map string_of_int img_slice_dims)))
             (String.concat "x"
                (Array.to_list
                   (Array.map string_of_int (Genarray.dims img_slice)))));

      let lbl_slice = Array1.sub train_labels !current_offset batch_size in
      Genarray.blit batch_images img_slice;
      Array1.blit batch_labels lbl_slice;
      current_offset := !current_offset + batch_size)
    train_batches_data;

  let test_batch_file = dataset_dir ^ "test_batch" in
  let test_images, test_labels = read_cifar_batch test_batch_file in

  Printf.printf "CIFAR-10 loading complete.\n%!";
  ((train_images, train_labels), (test_images, test_labels))
