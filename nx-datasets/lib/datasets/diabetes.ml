open Bigarray
open Dataset_utils

(* Logging source for this loader *)
let src = Logs.Src.create "nx.datasets.diabetes" ~doc:"Diabetes dataset loader"

module Log = (val Logs.src_log src : Logs.LOG)

let dataset_name = "diabetes-sklearn"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "diabetes.tab.txt"
let data_path = dataset_dir ^ data_filename
let url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
let ensure_dataset () = ensure_file url data_path

let load () =
  ensure_dataset ();
  Log.info (fun m -> m "Loading Diabetes (sklearn version) dataset...");

  let header, data_rows_iter =
    try
      let chan =
        Csv.of_channel ~has_header:true ~separator:'\t' (open_in data_path)
      in
      let h = Csv.Rows.header chan in
      (h, chan)
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

  let expected_cols = 11 in
  let num_features = 10 in
  let target_col_name = "Y" in

  let get_col_index name =
    match List.find_index (( = ) name) header with
    | Some idx -> idx
    | None ->
        Csv.close_in data_rows_iter;
        failwith
          ("Required column '" ^ name ^ "' not found in header: "
         ^ String.concat ", " header)
  in
  let feature_indices = List.init num_features (fun i -> i) in
  (* Assuming first 10 *)
  let target_index = get_col_index target_col_name in

  (* Use Fold to process rows *)
  let collected_features = ref [] in
  let collected_labels = ref [] in
  let row_counter = ref 0 in
  (try
     Csv.Rows.iter
       ~f:(fun row ->
         incr row_counter;
         let row_num = !row_counter in
         let row_list = Csv.Row.to_list row in
         if List.length row_list <> expected_cols then
           failwith
             (Printf.sprintf "Row %d has %d columns, expected %d" row_num
                (List.length row_list) expected_cols);

         (* Extract features *)
         let current_features =
           List.map
             (fun feature_idx ->
               let feature_str = List.nth row_list feature_idx in
               let col_name = List.nth header feature_idx in
               let context () =
                 Printf.sprintf "row %d, col %s" row_num col_name
               in
               parse_float_cell ~context feature_str)
             feature_indices
         in
         collected_features := current_features :: !collected_features;

         (* Extract label *)
         let label_str = List.nth row_list target_index in
         let context () =
           Printf.sprintf "row %d, col %s" row_num target_col_name
         in
         let label_float = parse_float_cell ~context label_str in
         collected_labels := label_float :: !collected_labels)
       data_rows_iter
   with
  | Csv.Failure (r, c, msg) ->
      Csv.close_in data_rows_iter;
      failwith
        (Printf.sprintf
           "CSV Parsing Error during iteration at approx row %d, field %d: %s" r
           c msg)
  | ex ->
      Csv.close_in data_rows_iter;
      failwith
        (Printf.sprintf "Error iterating CSV %s: %s" data_path
           (Printexc.to_string ex)));
  Csv.close_in data_rows_iter;

  let features_rev = !collected_features in
  let labels_rev = !collected_labels in
  let num_samples = List.length labels_rev in

  if num_samples = 0 then failwith "No data rows loaded from diabetes.tab.txt";
  Log.info (fun m ->
      m "Found %d samples with %d features and target '%s'." num_samples
        num_features target_col_name);

  let features_ba = Array2.create float64 c_layout num_samples num_features in
  let labels_ba = Array1.create float64 c_layout num_samples in

  List.iteri
    (fun i feature_row ->
      let current_row_idx = num_samples - 1 - i in
      List.iteri
        (fun j feature_val -> features_ba.{current_row_idx, j} <- feature_val)
        feature_row)
    features_rev;

  List.iteri
    (fun i label_val -> labels_ba.{num_samples - 1 - i} <- label_val)
    labels_rev;

  Log.info (fun m -> m "Diabetes loading complete.");
  (features_ba, labels_ba)
