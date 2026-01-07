(* breast_cancer.ml *)
open Bigarray
open Dataset_utils

(* Logging source for this loader *)
let src =
  Logs.Src.create "nx.datasets.breast_cancer"
    ~doc:"Breast Cancer dataset loader"

module Log = (val Logs.src_log src : Logs.LOG)

let dataset_name = "breast-cancer-wisconsin"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "wdbc.data"
let data_path = dataset_dir ^ data_filename

let url =
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

let ensure_dataset () = ensure_file url data_path

let make_parse_context row col =
  let context () = Printf.sprintf "row %d, col %d" row col in
  context

let encode_label label row =
  match label with
  | "M" -> 1l
  | "B" -> 0l
  | _ -> failwith (Printf.sprintf "Unknown label '%s' at row %d" label row)

let load () =
  ensure_dataset ();
  Log.info (fun m -> m "Loading Breast Cancer Wisconsin dataset...");

  let data_rows =
    try
      (* Csv.load directly loads file into string list list. No header
         handling. *)
      Csv.load ~separator:',' data_path
    with
    | Csv.Failure (r, c, msg) ->
        (* Csv.load can still fail on bad format *)
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

  let data_rows = List.filter (fun row -> List.length row > 0) data_rows in
  let num_samples = List.length data_rows in
  if num_samples = 0 then failwith "No data loaded from wdbc.data";
  let expected_cols = 32 in
  let num_features = 30 in

  Log.info (fun m -> m "Found %d samples." num_samples);

  let features = Array2.create float64 c_layout num_samples num_features in
  let labels = Array1.create int32 c_layout num_samples in

  List.iteri
    (fun i row ->
      let row_num = i + 1 in
      if List.length row <> expected_cols then
        failwith
          (Printf.sprintf "Row %d has %d columns, expected %d" row_num
             (List.length row) expected_cols);

      let label_str = List.nth row 1 in
      labels.{i} <- encode_label label_str row_num;

      for j = 0 to num_features - 1 do
        let col_idx = j + 2 in
        let feature_str = List.nth row col_idx in
        let context = make_parse_context row_num (col_idx + 1) in
        features.{i, j} <- parse_float_cell ~context feature_str
      done)
    data_rows;

  Log.info (fun m -> m "Breast Cancer Wisconsin loading complete.");
  (features, labels)
