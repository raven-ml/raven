(* airline_passengers.ml *)
open Bigarray
open Dataset_utils

let dataset_name = "airline-passengers"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "airline-passengers.csv"
let data_path = dataset_dir ^ data_filename

let url =
  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

(* Logging source for this loader *)
let src =
  Logs.Src.create "nx.datasets.airline_passengers"
    ~doc:"Airline passengers loader"

module Log = (val Logs.src_log src : Logs.LOG)

let ensure_dataset () = ensure_file url data_path

let load () =
  ensure_dataset ();
  Log.info (fun m -> m "Loading Airline Passengers dataset...");

  let header, data_rows =
    try load_csv ~has_header:true ~separator:',' data_path
    with
    | Sys_error msg ->
        failwith (Printf.sprintf "Cannot open file %s: %s" data_path msg)
    | ex ->
        failwith
          (Printf.sprintf "Error loading CSV %s: %s" data_path
             (Printexc.to_string ex))
  in

  let passenger_col_name = "Passengers" in
  let passenger_col_index =
    match List.find_index (( = ) passenger_col_name) header with
    | Some idx -> idx
    | None ->
        failwith
          ("Required column '" ^ passenger_col_name ^ "' not found in header: "
         ^ String.concat ", " header)
  in

  let collected_data =
    List.fold_left
      (fun acc row_list ->
        if List.length row_list <> List.length header then
          Log.warn (fun m ->
              m "Row %d has %d columns, expected %d (header: %s)"
                (List.length acc + 1)
                (List.length row_list) (List.length header)
                (String.concat ", " header));

        if List.length row_list > passenger_col_index then
          let passenger_str = List.nth row_list passenger_col_index in
          let context () =
            Printf.sprintf "row %d, col %s"
              (List.length acc + 1)
              passenger_col_name
          in
          let passenger_int = parse_int_cell ~context passenger_str in
          passenger_int :: acc
        else (
          Log.warn (fun m ->
              m
                "Row %d is shorter than expected (%d < %d), skipping \
                 passenger value. Missing column: %s"
                (List.length acc + 1)
                (List.length row_list) (passenger_col_index + 1)
                passenger_col_name);
          -1 :: acc))
      [] data_rows
  in

  (* Important: Close the channel after iteration *)
  let data_rows_rev = collected_data in
  let num_samples = List.length data_rows_rev in
  if num_samples = 0 then
    failwith "No data rows loaded from airline-passengers.csv";
  Log.info (fun m -> m "Found %d samples." num_samples);

  (* Create Bigarray and populate (data is reversed from fold_left) *)
  let passengers = Array1.create int32 c_layout num_samples in
  List.iteri
    (fun i passenger_val ->
      passengers.{num_samples - 1 - i} <- Int32.of_int passenger_val)
    data_rows_rev;

  Log.info (fun m -> m "Airline Passengers loading complete.");
  passengers
