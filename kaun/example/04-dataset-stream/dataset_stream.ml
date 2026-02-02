(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let dtype = Rune.float32
let array_to_list arr = Array.fold_right (fun x acc -> x :: acc) arr []

let sin_stream () =
  let rng = Random.State.make [| 42 |] in
  Seq.unfold
    (fun step ->
      let x = Random.State.float rng (2. *. Float.pi) in
      let y = Float.sin x in
      Some ((x, y), step + 1))
    0

let to_tensor_pair (x, y) =
  let input = Rune.create dtype [| 1 |] [| x |] in
  let target = Rune.create dtype [| 1 |] [| y |] in
  (input, target)

let batch_stack tensors =
  match array_to_list tensors with
  | [] -> failwith "empty batch"
  | first :: rest -> Rune.stack ~axis:0 (first :: rest)

let build_pipeline () =
  Dataset.from_seq (sin_stream ())
  |> Dataset.map to_tensor_pair
  |> Dataset.shuffle ~buffer_size:256
  |> Dataset.batch_map 8 (fun batch ->
      let inputs = Array.map fst batch |> batch_stack in
      let targets = Array.map snd batch |> batch_stack in
      (inputs, targets))
  |> Dataset.prefetch ~buffer_size:2

let shape_to_string shape =
  shape |> Array.map string_of_int |> array_to_list |> String.concat "x"

let print_batch idx (inputs, targets) =
  Printf.printf "Batch %d | shape=%s -> %s\n" idx
    (shape_to_string (Rune.shape inputs))
    (shape_to_string (Rune.shape targets));
  let first_input = Rune.item [ 0; 0 ] inputs in
  let first_target = Rune.item [ 0; 0 ] targets in
  Printf.printf "  sample[0]: sin(%.4f) ≈ %.4f\n" first_input first_target

let () =
  let pipeline = build_pipeline () in
  Printf.printf "Cardinality: %s\n"
    (match Dataset.cardinality pipeline with
    | Dataset.Finite n -> Printf.sprintf "Finite %d" n
    | Dataset.Unknown -> "Unknown"
    | Dataset.Infinite -> "Infinite");
  let preview = Dataset.take 3 pipeline in
  let counter = ref 0 in
  Dataset.iter
    (fun batch ->
      incr counter;
      print_batch !counter batch)
    preview;
  Printf.printf
    "\nResetting and grabbing another batch from the same stream...\n";
  Dataset.reset preview;
  let printed = ref false in
  Dataset.iter
    (fun batch ->
      if not !printed then (
        printed := true;
        print_batch 1 batch))
    preview
