(* Test integration between kaun.datasets and kaun.dataset *)

let test_mnist_streaming () =
  let device = Rune.c in

  (* Load MNIST using the new streaming API *)
  let dataset =
    Kaun_datasets.mnist ~train:true ~normalize:true ~device ()
    |> Kaun_dataset.shuffle ~buffer_size:1000
    |> Kaun_dataset.batch_map 32 (fun batch ->
           (* Stack into batched tensors *)
           let images, labels = Array.split batch in
           let batched_images = Rune.stack ~axis:0 (Array.to_list images) in
           let batched_labels = Rune.stack ~axis:0 (Array.to_list labels) in
           (batched_images, batched_labels))
    |> Kaun_dataset.take 3 (* Just take first 3 batches for testing *)
  in

  (* Iterate through batches *)
  let batch_count = ref 0 in
  Kaun_dataset.iter
    (fun (x_batch, y_batch) ->
      incr batch_count;
      let x_shape = Rune.shape x_batch in
      let y_shape = Rune.shape y_batch in
      Printf.printf "Batch %d: images shape [%s], labels shape [%s]\n"
        !batch_count
        (x_shape |> Array.to_list |> List.map string_of_int
       |> String.concat "; ")
        (y_shape |> Array.to_list |> List.map string_of_int
       |> String.concat "; "))
    dataset;

  assert (!batch_count = 3);
  print_endline "✓ MNIST streaming test passed"

let test_vision_pipeline () =
  let device = Rune.c in

  (* Use pre-configured vision pipeline *)
  let dataset =
    Kaun_datasets.mnist ~train:false ~device ()
    |> Kaun_datasets.vision_pipeline ~batch_size:64 ~shuffle_buffer:5000
    |> Kaun_dataset.take 2
  in

  (* Check cardinality *)
  let card = Kaun_dataset.cardinality dataset in
  (match card with
  | Kaun_dataset.Finite n -> Printf.printf "Dataset has %d batches\n" n
  | _ -> print_endline "Dataset cardinality unknown");

  print_endline "✓ Vision pipeline test passed"

let test_train_test_split () =
  let device = Rune.c in

  (* Load iris dataset and split it *)
  let full_dataset = Kaun_datasets.iris ~device () in
  let train_data, test_data =
    Kaun_datasets.train_test_split ~test_size:0.2 ~seed:42 full_dataset
  in

  (* Count samples in each split *)
  let count_samples dataset =
    let count = ref 0 in
    Kaun_dataset.iter (fun _ -> incr count) dataset;
    !count
  in

  let train_count = count_samples train_data in
  let test_count = count_samples test_data in

  Printf.printf "Train samples: %d, Test samples: %d\n" train_count test_count;
  assert (train_count = 120);
  (* 80% of 150 *)
  assert (test_count = 30);

  (* 20% of 150 *)
  print_endline "✓ Train/test split test passed"

let () =
  print_endline "Testing kaun.datasets with new streaming API...";
  test_mnist_streaming ();
  test_vision_pipeline ();
  test_train_test_split ();
  print_endline "\nAll tests passed! ✓"
