(* california_housing.ml *)
open Bigarray
open Dataset_utils

let dataset_name = "california-housing"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "housing.csv"
let data_path = dataset_dir ^ data_filename

let url =
  "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

let ensure_dataset () = ensure_file url data_path

let parse_float_or_nan s =
  try float_of_string s with Failure _ | Invalid_argument _ -> nan

let calculate_mean_non_nan column_data =
  let sum = ref 0.0 in
  let count = ref 0 in
  List.iter
    (fun v ->
      if not (Float.is_nan v) then (
        sum := !sum +. v;
        incr count))
    column_data;
  if !count = 0 then 0.0 else !sum /. float_of_int !count

let load () =
  ensure_dataset ();
  Printf.printf "Loading California Housing dataset...\n%!";

  let header, all_data_rows =
    try
      let rows = Csv.Rows.load ~has_header:true ~separator:',' data_path in
      let chan_header =
        Csv.of_channel ~has_header:true ~separator:',' (open_in data_path)
      in
      let h = Csv.Rows.header chan_header in
      Csv.close_in chan_header;
      (h, rows)
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

  let data_rows_str = List.map Csv.Row.to_list all_data_rows in
  let num_samples = List.length data_rows_str in
  if num_samples = 0 then failwith "No data loaded from housing.csv";

  let feature_names =
    [
      "longitude";
      "latitude";
      "housing_median_age";
      "total_rooms";
      "total_bedrooms";
      "population";
      "households";
      "median_income";
    ]
  in
  let target_name = "median_house_value" in
  let num_features = List.length feature_names in

  let get_col_index name =
    match List.find_index (( = ) name) header with
    | Some idx -> idx
    | None ->
        failwith
          ("Required column '" ^ name ^ "' not found in header: "
         ^ String.concat ", " header)
  in
  let feature_indices = List.map get_col_index feature_names in
  let target_index = get_col_index target_name in
  let total_bedrooms_index_opt =
    List.find_index (( = ) "total_bedrooms") header
  in

  Printf.printf "Found %d samples. Loading %d features + target '%s'.\n%!"
    num_samples num_features target_name;

  let parsed_features_temp = Array.make_matrix num_samples num_features nan in
  let parsed_labels_temp = Array.make num_samples nan in
  let total_bedrooms_col_temp = ref [] in

  List.iteri
    (fun i row ->
      if List.length row <> List.length header then
        Printf.eprintf "Warning: Row %d has %d columns, expected %d\n%!" (i + 1)
          (List.length row) (List.length header);

      List.iteri
        (fun j feature_idx ->
          if List.length row > feature_idx then (
            let v_str = List.nth row feature_idx in
            let v_float = parse_float_or_nan v_str in
            parsed_features_temp.(i).(j) <- v_float;
            if Some feature_idx = total_bedrooms_index_opt then
              total_bedrooms_col_temp := v_float :: !total_bedrooms_col_temp)
          else (
            Printf.eprintf
              "Warning: Row %d missing feature column %d ('%s'). Setting NaN.\n\
               %!"
              (i + 1) feature_idx (List.nth feature_names j);
            parsed_features_temp.(i).(j) <- nan;
            if Some feature_idx = total_bedrooms_index_opt then
              total_bedrooms_col_temp := nan :: !total_bedrooms_col_temp))
        feature_indices;

      if List.length row > target_index then
        let label_str = List.nth row target_index in
        parsed_labels_temp.(i) <- parse_float_or_nan label_str
      else (
        Printf.eprintf
          "Warning: Row %d missing target column %d ('%s'). Setting NaN.\n%!"
          (i + 1) target_index target_name;
        parsed_labels_temp.(i) <- nan))
    data_rows_str;

  let total_bedrooms_mean = calculate_mean_non_nan !total_bedrooms_col_temp in
  Printf.printf "Calculated mean for 'total_bedrooms' (for imputation): %f\n%!"
    total_bedrooms_mean;
  let total_bedrooms_feature_index =
    match List.find_index (( = ) "total_bedrooms") feature_names with
    | Some idx -> idx
    | None -> -1
  in

  let features = Array2.create float64 c_layout num_samples num_features in
  let labels = Array1.create float64 c_layout num_samples in

  for i = 0 to num_samples - 1 do
    let row_num = i + 1 in
    for j = 0 to num_features - 1 do
      let v = parsed_features_temp.(i).(j) in
      let col_name = List.nth feature_names j in
      if j = total_bedrooms_feature_index && Float.is_nan v then
        features.{i, j} <- total_bedrooms_mean
      else if Float.is_nan v then
        failwith
          (Printf.sprintf
             "Unexpected NaN/unparseable value (after potential short row \
              handling) in column '%s' at row %d"
             col_name row_num)
      else features.{i, j} <- v
    done;
    let label_v = parsed_labels_temp.(i) in
    if Float.is_nan label_v then
      failwith
        (Printf.sprintf
           "Unexpected NaN/unparseable value (after potential short row \
            handling) in target column '%s' at row %d"
           target_name row_num)
    else labels.{i} <- label_v
  done;

  Printf.printf "California Housing loading complete.\n%!";
  (features, labels)
