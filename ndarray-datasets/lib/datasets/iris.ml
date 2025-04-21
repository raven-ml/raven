(* iris.ml *)
open Bigarray
open Dataset_utils

let dataset_name = "iris"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "iris.data"
let data_path = dataset_dir ^ data_filename

let url =
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

let ensure_dataset () = ensure_file url data_path
let label_map = Hashtbl.create 3

let () =
  Hashtbl.add label_map "Iris-setosa" 0l;
  Hashtbl.add label_map "Iris-versicolor" 1l;
  Hashtbl.add label_map "Iris-virginica" 2l

let encode_label s =
  try Hashtbl.find label_map s
  with Not_found -> failwith ("Unknown Iris label: " ^ s)

let load () =
  ensure_dataset ();
  Printf.printf "Loading Iris dataset...\n%!";

  let data_rows =
    try
      (* Use Csv.load, no header handling needed *)
      Csv.load ~separator:',' data_path
    with
    | Csv.Failure (r, c, msg) ->
        failwith
          (Printf.sprintf "CSV Parsing Error in %s at row %d, col %d: %s"
             data_path r c msg)
    | Sys_error msg ->
        failwith (Printf.sprintf "Cannot open file %s: %s" data_path msg)
    | ex ->
        failwith
          (Printf.sprintf "Error loading CSV %s: %s" data_path
             (Printexc.to_string ex))
  in

  let data_rows =
    List.filter (function [] | [ "" ] -> false | _ -> true) data_rows
  in
  let num_samples = List.length data_rows in
  let num_features = 4 in

  if num_samples = 0 then failwith "No data loaded from iris.data";
  Printf.printf "Found %d samples.\n%!" num_samples;

  let features = Array2.create float64 c_layout num_samples num_features in
  let labels = Array1.create int32 c_layout num_samples in

  List.iteri
    (fun i row ->
      let row_num = i + 1 in
      if List.length row <> 5 then
        failwith
          (Printf.sprintf "Row %d has %d columns, expected 5" row_num
             (List.length row));

      for j = 0 to num_features - 1 do
        let feature_str = List.nth row j in
        let context () = Printf.sprintf "row %d, col %d" row_num (j + 1) in
        features.{i, j} <- parse_float_cell ~context feature_str
      done;

      let label_str = List.nth row num_features in
      labels.{i} <- encode_label label_str)
    data_rows;

  Printf.printf "Iris loading complete.\n%!";
  (features, labels)
